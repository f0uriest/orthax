r"""
===============================
Three Term Recurrence Relations
===============================

Every family of orthogonal polynomials can be shown to satisfy a three term recurrence
relation of the form

.. math::

    \begin{align}
    p_{-1}(x) &= 0 \\
    p_0(x) &= 1 \\
    p_{i+1}(x) &= (x-a_i) p_i(x) - b_i p_{i-1}(x) \\
    \end{align}

Knowing the coefficients of the recurrence relation :math:`a_i, b_i` along with the
normalization constants :math:`g_i` such that

.. math::

    \int_D p_i(x) p_j(x) w(x) dx = g^2_i \delta_{i,j}


provides all the needed information for performing arithmetic, calculus, and other
manipulations on orthogonal polynomial series.


This module provides a number of classes for storing information about these recurrence
relationships that can be used with any of the functions in the base ``orthax``
namespace that expect an ``AbstractRecurrenceRelation``. These include classes
representing many of the "classical" orthogonal polynomial families, as well as the
function ``generate_recurrence`` for generating recurrence coefficients for orthogonal
polynomials with arbitrary user defined weight functions and domains.


Functions
---------
.. autosummary::
   :toctree: generated/

   generate_recurrence

Base Classes
------------
.. autosummary::
   :toctree: generated/

   AbstractRecurrenceRelation
   TabulatedRecurrenceRelation

Classical Recurrence Relations
------------------------------
.. autosummary::
   :toctree: generated/

   Legendre
   ShiftedLegendre
   ChebyshevT
   ChebyshevU
   ChebyshevV
   ChebyshevW
   Gegenbauer
   Jacobi
   Laguerre
   GeneralizedLaguerre
   Hermite
   HermiteE

"""

import abc
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
from jax.typing import ArrayLike


def _asarray(k, kmax=None, check=True):
    k = jnp.asarray(k)
    if not check:
        return k
    k = eqx.error_if(
        k,
        (k < 0).any(),
        "Negative indices not allowed for recurrence coefficients.",
    )
    if kmax is not None:
        k = eqx.error_if(
            k,
            (k > kmax).any(),
            "Requested recurrence coefficient outside of tabulated range.",
        )

    return k


class AbstractRecurrenceRelation(eqx.Module, abc.ABC):
    """Base class for three term recurrence relations.

    Subclasses should declare attributes and `_domain` and implement methods
    `weight`, `a`, `b`, `g`, `m`
    """

    _domain: tuple[float, float]

    @abc.abstractmethod
    def weight(self, x: ArrayLike) -> jax.Array:
        """Weight function defining inner product."""
        pass

    @property
    def domain(self) -> tuple[jax.Array, jax.Array]:
        """tuple: Lower and upper bounds for inner product defining orthogonality."""
        return self._domain

    @abc.abstractmethod
    def a(self, k: ArrayLike) -> jax.Array:
        """`a` coefficients of the monic three term recurrence relation."""
        pass

    @abc.abstractmethod
    def b(self, k: ArrayLike) -> jax.Array:
        """`b` coefficients of the monic three term recurrence relation."""
        pass

    @abc.abstractmethod
    def g(self, k: ArrayLike) -> jax.Array:
        """Weighted norm of the kth monic orthogonal polynomial."""
        pass

    @abc.abstractmethod
    def m(self, k: ArrayLike) -> jax.Array:
        """Coefficient of x**k in the kth polynomial in the desired normalization."""
        pass


class TabulatedRecurrenceRelation(AbstractRecurrenceRelation):
    """Recurrence relation from tabulated values.

    Parameters
    ----------
    weight : callable
        Weight function.
    domain : tuple
        Lower and upper bounds for inner product defining orthogonality.
    a, b : jax.Array
        Coefficients of the monic three term recurrence relation.
    g : jax.Array
        ``g[k]`` is the weighted norm of the kth monic orthogonal polynomial.
    m : jax.Array
        ``m[k]`` is the coefficient of x**k in the kth orthogonal polynomial in the
        desired normalization. Default is 1 (monic form). For normalized form, set
        m = 1/g
    check : bool
        Whether to check that requested indices are within the tabulated range. Checks
        use a runtime callback which can be slow. If False, indexing outside the
        tabulated range silently returns incorrect values.

    """

    _ak: jax.Array
    _bk: jax.Array
    _gk: jax.Array
    _mk: jax.Array
    _weight: Callable = eqx.field(static=True)
    _check: bool = eqx.field(static=True)

    def __init__(
        self,
        weight: Callable,
        domain: tuple,
        ak: jax.Array,
        bk: jax.Array,
        gk: jax.Array,
        mk: jax.Array | None = None,
        check: bool = True,
    ):
        if mk is None:
            mk = jnp.ones_like(ak)
        self._ak = ak
        self._bk = bk
        self._gk = gk
        self._mk = mk
        self._weight = weight
        self._domain = domain
        self._check = check

    def weight(self, x: ArrayLike) -> jax.Array:
        """Weight function defining inner product."""
        return self._weight(x)

    def a(self, k: ArrayLike) -> jax.Array:
        """`a` coefficients of the monic three term recurrence relation."""
        k = _asarray(k, kmax=len(self._ak) - 1, check=self._check)
        return self._ak[k]

    def b(self, k: ArrayLike) -> jax.Array:
        """`b` coefficients of the monic three term recurrence relation."""
        k = _asarray(k, kmax=len(self._bk) - 1, check=self._check)
        return self._bk[k]

    def g(self, k: ArrayLike) -> jax.Array:
        """Weighted norm of the kth monic orthogonal polynomial."""
        k = _asarray(k, kmax=len(self._gk) - 1, check=self._check)
        return self._gk[k]

    def m(self, k: ArrayLike) -> jax.Array:
        """Coefficient of x**k in the kth polynomial in the desired normalization."""
        k = _asarray(k, kmax=len(self._mk) - 1, check=self._check)
        return self._mk[k]


class ClassicalRecurrenceRelation(AbstractRecurrenceRelation, abc.ABC):
    """Base class for recurrence relations for "classical" orthogonal polynomials.

    Parameters
    ----------
    domain : tuple
        Lower and upper bounds for inner product defining orthogonality.
    scale : {"standard", "monic", "normalized"}
        Most classical orthogonal polynomials have ad-hoc normalizations (ie,
        the common definitions in textbooks are neither monic nor unit norm). This
        is encompassed in the "standard" scale, which should match that in texts such as
        Abramowitz & Stegun. Alternatively, they can be scaled to be monic or unit norm.
    """

    _scale: str = eqx.field(static=True)

    def __init__(self, domain: tuple, scale: str = "standard"):
        assert scale in {"standard", "monic", "normalized"}

        self._domain = domain
        self._scale = scale

    @abc.abstractmethod
    def _std_norm(self, k: ArrayLike) -> jax.Array:
        # norm of the kth polynomial in "standard" scaling (ie, AS, wikipedia, etc)
        pass

    @abc.abstractmethod
    def _std_scale(self, k: ArrayLike) -> jax.Array:
        # coefficient of x**k in "standard" scaling (ie, AS, wikipedia, etc)
        pass

    def g(self, k: ArrayLike) -> jax.Array:
        """Weighted norm of the kth monic orthogonal polynomial."""
        k = _asarray(k)
        return self._std_norm(k) / jnp.abs(self._std_scale(k))

    def m(self, k: ArrayLike) -> jax.Array:
        """Coefficient of x**k in the kth polynomial in the desired normalization."""
        # scaling factor. polynomials are evaluated in monic form then multiplied
        # by this scale factor
        # ie, evaluate monic polynomials then multiply by m to get scaled version
        k = _asarray(k)
        if self._scale == "monic":
            out = jnp.ones_like(k)
        elif self._scale == "standard":
            out = self._std_scale(k)
        else:  # normalized
            out = self._std_scale(k) / self._std_norm(k)
        return out


# General notes:
# ak, bk from Gautschi, Orthogonal Polynomials: Computation and Approximation,
# Table 1.1, ak=alpha_k, bk = beta_k
# std_scale, std_norm from NIST Handbook of Mathematical Functions,
# Table 18.3.1, std_scale = k_n, std_norm = sqrt(h_n)


class Legendre(ClassicalRecurrenceRelation):
    """Recurrence relation for Legendre Polynomials :math:`P_n(x)`

    Legendre polynomials are orthogonal on the interval (-1, 1)
    with the weight function :math:`w(x) = 1`

    Parameters
    ----------
    scale : {"standard", "monic", "normalized"}
        "standard" corresponds to the common scaling found in textbooks such as
        Abramowitz & Stegun. "monic" scales them such that the leading coefficient is 1.
        "normalized" scales them to have a weighted norm of 1.
    """

    def __init__(self, scale: str = "standard"):
        super().__init__(domain=(-1, 1), scale=scale)

    def weight(self, x: ArrayLike) -> jax.Array:
        """Weight function defining inner product."""
        return jnp.ones_like(x)

    def a(self, k: ArrayLike) -> jax.Array:
        """`a` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return jnp.zeros_like(k)

    def b(self, k: ArrayLike) -> jax.Array:
        """`b` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return jnp.where(k == 0, 2, 1 / (4 - 1 / jnp.where(k == 0, 1, k) ** 2))

    def _std_norm(self, k: ArrayLike) -> jax.Array:
        return jnp.sqrt(2 / (2 * k + 1))

    def _std_scale(self, k: ArrayLike) -> jax.Array:
        return jnp.exp(
            k * jnp.log(2) - gammaln(k + 1) + gammaln(k + 0.5) - gammaln(0.5)
        )


class ShiftedLegendre(ClassicalRecurrenceRelation):
    """Recurrence relation for Shifted Legendre Polynomials :math:`P^*_n(x)`

    Shifted Legendre polynomials are orthogonal on the interval (0, 1)
    with the weight function :math:`w(x) = 1`

    Parameters
    ----------
    scale : {"standard", "monic", "normalized"}
        "standard" corresponds to the common scaling found in textbooks such as
        Abramowitz & Stegun. "monic" scales them such that the leading coefficient is 1.
        "normalized" scales them to have a weighted norm of 1.
    """

    def __init__(self, scale: str = "standard"):
        super().__init__(domain=(0, 1), scale=scale)

    def weight(self, x: ArrayLike) -> jax.Array:
        """Weight function defining inner product."""
        return jnp.ones_like(x)

    def a(self, k: ArrayLike) -> jax.Array:
        """`a` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return 0.5 * jnp.ones_like(k)

    def b(self, k: ArrayLike) -> jax.Array:
        """`b` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return jnp.where(k == 0, 1, 0.25 / (4 - 1 / jnp.where(k == 0, 1, k) ** 2))

    def _std_norm(self, k: ArrayLike) -> jax.Array:
        return jnp.sqrt(1 / (2 * k + 1))

    def _std_scale(self, k: ArrayLike) -> jax.Array:
        return jnp.exp(
            2 * k * jnp.log(2) - gammaln(k + 1) + gammaln(k + 0.5) - gammaln(0.5)
        )


class ChebyshevT(ClassicalRecurrenceRelation):
    """Recurrence relation for Chebyshev polynomials of the first kind :math:`T_n(x)`

    Chebyshev polynomials of the first kind are orthogonal on the interval (-1, 1)
    with the weight function :math:`w(x) = (1-x^2)^{-1/2}`

    Parameters
    ----------
    scale : {"standard", "monic", "normalized"}
        "standard" corresponds to the common scaling found in textbooks such as
        Abramowitz & Stegun. "monic" scales them such that the leading coefficient is 1.
        "normalized" scales them to have a weighted norm of 1.
    """

    def __init__(self, scale: str = "standard"):
        super().__init__(domain=(-1, 1), scale=scale)

    def weight(self, x: ArrayLike) -> jax.Array:
        """Weight function defining inner product."""
        x = jnp.asarray(x)
        return 1.0 / jnp.sqrt(1 - x**2)

    def a(self, k: ArrayLike) -> jax.Array:
        """`a` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return jnp.zeros_like(k)

    def b(self, k: ArrayLike) -> jax.Array:
        """`b` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return jnp.where(k == 0, jnp.pi, jnp.where(k == 1, 1 / 2, 1 / 4))

    def _std_norm(self, k: ArrayLike) -> jax.Array:
        return jnp.sqrt(jnp.where(k == 0, jnp.pi, jnp.pi / 2))

    def _std_scale(self, k: ArrayLike) -> jax.Array:
        return jnp.where(k == 0, 1, 2 ** jnp.maximum(0.0, k - 1.0))


class ChebyshevU(ClassicalRecurrenceRelation):
    """Recurrence relation for Chebyshev polynomials of the second kind :math:`U_n(x)`

    Chebyshev polynomials of the second kind are orthogonal on the interval (-1, 1)
    with the weight function :math:`w(x) = (1-x^2)^{1/2}`

    Parameters
    ----------
    scale : {"standard", "monic", "normalized"}
        "standard" corresponds to the common scaling found in textbooks such as
        Abramowitz & Stegun. "monic" scales them such that the leading coefficient is 1.
        "normalized" scales them to have a weighted norm of 1.
    """

    def __init__(self, scale: str = "standard"):
        super().__init__(domain=(-1, 1), scale=scale)

    def weight(self, x: ArrayLike) -> jax.Array:
        """Weight function defining inner product."""
        x = jnp.asarray(x)
        return jnp.sqrt(1 - x**2)

    def a(self, k: ArrayLike) -> jax.Array:
        """`a` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return jnp.zeros_like(k)

    def b(self, k: ArrayLike) -> jax.Array:
        """`b` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return jnp.where(k == 0, jnp.pi / 2, 1 / 4)

    def _std_norm(self, k: ArrayLike) -> jax.Array:
        return jnp.full(k.shape, jnp.sqrt(jnp.pi / 2))

    def _std_scale(self, k: ArrayLike) -> jax.Array:
        return 2 ** jnp.asarray(k).astype(float)


class ChebyshevV(ClassicalRecurrenceRelation):
    """Recurrence relation for Chebyshev polynomials of the third kind :math:`V_n(x)`

    Chebyshev polynomials of the third kind are orthogonal on the interval (-1, 1)
    with the weight function :math:`w(x) = (1-x)^{1/2} (1+x)^{-1/2}`

    Parameters
    ----------
    scale : {"standard", "monic", "normalized"}
        "standard" corresponds to the common scaling found in textbooks such as
        Abramowitz & Stegun. "monic" scales them such that the leading coefficient is 1.
        "normalized" scales them to have a weighted norm of 1.
    """

    def __init__(self, scale: str = "standard"):
        super().__init__(domain=(-1, 1), scale=scale)

    def weight(self, x: ArrayLike) -> jax.Array:
        """Weight function defining inner product."""
        x = jnp.asarray(x)
        return jnp.sqrt((1 + x) / (1 - x))

    def a(self, k: ArrayLike) -> jax.Array:
        """`a` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return jnp.where(k == 0, 0.5, 0)

    def b(self, k: ArrayLike) -> jax.Array:
        """`b` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return jnp.where(k == 0, jnp.pi, 1 / 4)

    def _std_norm(self, k: ArrayLike) -> jax.Array:
        return jnp.full(k.shape, jnp.sqrt(jnp.pi))

    def _std_scale(self, k: ArrayLike) -> jax.Array:
        return 2 ** jnp.asarray(k).astype(float)


class ChebyshevW(ClassicalRecurrenceRelation):
    """Recurrence relation for Chebyshev polynomials of the fourth kind :math:`W_n(x)`

    Chebyshev polynomials of the fourth kind are orthogonal on the interval (-1, 1)
    with the weight function :math:`w(x) = (1-x)^{-1/2} (1+x)^{1/2}`

    Parameters
    ----------
    scale : {"standard", "monic", "normalized"}
        "standard" corresponds to the common scaling found in textbooks such as
        Abramowitz & Stegun. "monic" scales them such that the leading coefficient is 1.
        "normalized" scales them to have a weighted norm of 1.
    """

    def __init__(self, scale: str = "standard"):
        super().__init__(domain=(-1, 1), scale=scale)

    def weight(self, x: ArrayLike) -> jax.Array:
        """Weight function defining inner product."""
        x = jnp.asarray(x)
        return jnp.sqrt((1 - x) / (1 + x))

    def a(self, k: ArrayLike) -> jax.Array:
        """`a` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return jnp.where(k == 0, -0.5, 0)

    def b(self, k: ArrayLike) -> jax.Array:
        """`b` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return jnp.where(k == 0, jnp.pi, 1 / 4)

    def _std_norm(self, k: ArrayLike) -> jax.Array:
        return jnp.full(k.shape, jnp.sqrt(jnp.pi))

    def _std_scale(self, k: ArrayLike) -> jax.Array:
        return 2 ** jnp.asarray(k).astype(float)


class Gegenbauer(ClassicalRecurrenceRelation):
    r"""Recurrence relation for Gegenbauer polynomials :math:`C^\lambda_n(x)`

    Also known as Ultraspherical harmonics.

    Gegenbauer polynomials are orthogonal on the interval (-1, 1)
    with the weight function :math:`w(x) = (1-x^2)^{\lambda - 1/2}`

    Parameters
    ----------
    lmbda : float > -1/2, != 0
        Hyperparameter λ.
    scale : {"standard", "monic", "normalized"}
        "standard" corresponds to the common scaling found in textbooks such as
        Abramowitz & Stegun. "monic" scales them such that the leading coefficient is 1.
        "normalized" scales them to have a weighted norm of 1.
    """

    lmbda: jax.Array

    def __init__(self, lmbda: ArrayLike, scale: str = "standard"):
        lmbda = jnp.asarray(lmbda)
        lmbda = eqx.error_if(lmbda, lmbda <= -0.5, "lmbda must be > -1/2")
        lam_zero_err = """
        Classical Gegenbauer polynomials are undefined for lmbda==0,
        consider using ChebyshevT which has similar orthogonality properties
        but with a well behaved normalization"""
        lmbda = eqx.error_if(lmbda, lmbda == 0.0, lam_zero_err)
        self.lmbda = lmbda
        super().__init__(domain=(-1, 1), scale=scale)

    def weight(self, x: ArrayLike) -> jax.Array:
        """Weight function defining inner product."""
        x = jnp.asarray(x)
        return (1 - x**2) ** (self.lmbda - 0.5)

    def a(self, k: ArrayLike) -> jax.Array:
        """`a` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return jnp.zeros_like(k)

    def b(self, k: ArrayLike) -> jax.Array:
        """`b` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        b0 = jnp.sqrt(jnp.pi) * jnp.exp(
            gammaln(self.lmbda + 0.5) - gammaln(self.lmbda + 1)
        )
        bknum = jnp.where(self.lmbda == 0, 1, k * (k + 2 * self.lmbda - 1))
        bkden = jnp.where(
            self.lmbda == 0, 4, (4 * (k + self.lmbda) * (k + self.lmbda - 1))
        )
        return jnp.where(k == 0, b0, bknum / jnp.where(k == 0, 1, bkden))

    def _std_norm(self, k: ArrayLike) -> jax.Array:
        lognum = (
            (1 - 2 * self.lmbda) * jnp.log(2)
            + jnp.log(jnp.pi)
            + gammaln(k + 2 * self.lmbda)
        )
        sgn = jnp.sign(k + self.lmbda) * jax.scipy.special.gammasgn(k + 2 * self.lmbda)
        logden = (
            jnp.log(jnp.abs(k + self.lmbda)) + 2 * gammaln(self.lmbda) + gammaln(k + 1)
        )
        return sgn * jnp.exp(0.5 * (lognum - logden))

    def _std_scale(self, k: ArrayLike) -> jax.Array:
        return jnp.exp(
            k * jnp.log(2)
            - gammaln(k + 1)
            + gammaln(self.lmbda + k)
            - gammaln(self.lmbda)
        )


class Jacobi(ClassicalRecurrenceRelation):
    r"""Recurrence relation for Jacobi polynomials :math:`P^{(\alpha, \beta)}_n(x)`

    Jacobi polynomials are orthogonal on the interval (-1, 1)
    with the weight function :math:`w(x) = (1-x)^\alpha (1+x)^\beta`

    Parameters
    ----------
    alpha, beta : float > -1
        Hyperparameters α, β.
    scale : {"standard", "monic", "normalized"}
        "standard" corresponds to the common scaling found in textbooks such as
        Abramowitz & Stegun. "monic" scales them such that the leading coefficient is 1.
        "normalized" scales them to have a weighted norm of 1.
    """

    alpha: jax.Array
    beta: jax.Array

    def __init__(self, alpha: ArrayLike, beta: ArrayLike, scale: str = "standard"):
        alpha = jnp.asarray(alpha)
        beta = jnp.asarray(beta)
        self.alpha = eqx.error_if(alpha, alpha <= -1, "alpha must be > -1")
        self.beta = eqx.error_if(beta, beta <= -1, "beta must be > -1")
        super().__init__(domain=(-1, 1), scale=scale)

    def weight(self, x: ArrayLike) -> jax.Array:
        """Weight function defining inner product."""
        x = jnp.asarray(x)
        return (1 - x) ** self.alpha * (1 + x) ** self.beta

    def a(self, k: ArrayLike) -> jax.Array:
        """`a` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        a, b = self.alpha, self.beta
        num = jnp.where(k == 0, b - a, b**2 - a**2)
        den = jnp.where(k == 0, a + b + 2, (2 * k + a + b) * (2 * k + a + b + 2))
        return num / den

    def b(self, k: ArrayLike) -> jax.Array:
        """`b` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        a, b = self.alpha, self.beta
        b0 = jnp.exp(
            (a + b + 1) * jnp.log(2)
            + gammaln(a + 1)
            + gammaln(b + 1)
            - gammaln(a + b + 1)
        ) / (a + b + 1)
        num = jnp.where(
            k > 1, 4 * k * (k + a) * (k + b) * (k + a + b), 4 * k * (k + a) * (k + b)
        )
        den = jnp.where(
            k > 1,
            (2 * k + a + b) ** 2 * (2 * k + a + b + 1) * (2 * k + a + b - 1),
            (2 * k + a + b) ** 2 * (2 * k + a + b + 1),
        )
        return jnp.where(k == 0, b0, num / den)

    def _std_norm(self, k: ArrayLike) -> jax.Array:
        a, b = self.alpha, self.beta
        lognum = (a + b + 1) * jnp.log(2) + gammaln(k + a + 1) + gammaln(k + b + 1)
        logden = jnp.log(2 * k + a + b + 1) + gammaln(k + a + b + 1) + gammaln(k + 1)
        return jnp.exp(0.5 * (lognum - logden))

    def _std_scale(self, k: ArrayLike) -> jax.Array:
        a, b = self.alpha, self.beta
        logm = (
            gammaln(2 * k + a + b + 1)
            - k * jnp.log(2)
            - gammaln(k + 1)
            - gammaln(k + a + b + 1)
        )
        return jnp.exp(logm)


class Laguerre(ClassicalRecurrenceRelation):
    """Recurrence relation for Laguerre polynomials :math:`L_n(x)`

    Laguerre polynomials are orthogonal on the interval (0, inf)
    with the weight function :math:`w(x) = e^{-x}`

    Parameters
    ----------
    scale : {"standard", "monic", "normalized"}
        "standard" corresponds to the common scaling found in textbooks such as
        Abramowitz & Stegun. "monic" scales them such that the leading coefficient is 1.
        "normalized" scales them to have a weighted norm of 1.
    """

    def __init__(self, scale: str = "standard"):
        super().__init__(domain=(0, jnp.inf), scale=scale)

    def weight(self, x: ArrayLike) -> jax.Array:
        """Weight function defining inner product."""
        x = jnp.asarray(x)
        return jnp.exp(-x)

    def a(self, k: ArrayLike) -> jax.Array:
        """`a` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return 2 * k + 1

    def b(self, k: ArrayLike) -> jax.Array:
        """`b` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return jnp.where(k == 0, 1, k**2)

    def _std_norm(self, k: ArrayLike) -> jax.Array:
        return jnp.ones(jnp.asarray(k).shape)

    def _std_scale(self, k: ArrayLike) -> jax.Array:
        return (-1) ** k * jnp.exp(-gammaln(k + 1))


class GeneralizedLaguerre(ClassicalRecurrenceRelation):
    r"""Recurrence relation for Generalized Laguerre polynomials :math:`L^\alpha_n(x)`

    Generalized Laguerre polynomials are orthogonal on the interval (0, inf)
    with the weight function :math:`w(x) = x^\alpha e^{-x}`

    Parameters
    ----------
    alpha : float > -1
        Hyperparameter α.
    scale : {"standard", "monic", "normalized"}
        "standard" corresponds to the common scaling found in textbooks such as
        Abramowitz & Stegun. "monic" scales them such that the leading coefficient is 1.
        "normalized" scales them to have a weighted norm of 1.
    """

    alpha: jax.Array

    def __init__(self, alpha: ArrayLike, scale: str = "standard"):
        alpha = jnp.asarray(alpha)
        self.alpha = eqx.error_if(alpha, alpha <= -1, "alpha must be > -1")
        super().__init__(domain=(0, jnp.inf), scale=scale)

    def weight(self, x: ArrayLike) -> jax.Array:
        """Weight function defining inner product."""
        x = jnp.asarray(x)
        return x**self.alpha * jnp.exp(-x)

    def a(self, k: ArrayLike) -> jax.Array:
        """`a` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return 2 * k + self.alpha + 1

    def b(self, k: ArrayLike) -> jax.Array:
        """`b` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return jnp.where(k == 0, jnp.exp(gammaln(self.alpha + 1)), k * (k + self.alpha))

    def _std_norm(self, k: ArrayLike) -> jax.Array:
        return jnp.exp(-0.5 * gammaln(k + 1) + 0.5 * gammaln(k + self.alpha + 1))

    def _std_scale(self, k: ArrayLike) -> jax.Array:
        return (-1) ** k * jnp.exp(-gammaln(k + 1))


class Hermite(ClassicalRecurrenceRelation):
    """Recurrence relation for (physicists) Hermite polynomials :math:`H_n(x)`

    Hermite polynomials are orthogonal on the interval (-inf, inf)
    with the weight function :math:`w(x) = e^{-x^2}`

    Parameters
    ----------
    scale : {"standard", "monic", "normalized"}
        "standard" corresponds to the common scaling found in textbooks such as
        Abramowitz & Stegun. "monic" scales them such that the leading coefficient is 1.
        "normalized" scales them to have a weighted norm of 1.
    """

    def __init__(self, scale: str = "standard"):
        super().__init__(domain=(-jnp.inf, jnp.inf), scale=scale)

    def weight(self, x: ArrayLike) -> jax.Array:
        """Weight function defining inner product."""
        x = jnp.asarray(x)
        return jnp.exp(-(x**2))

    def a(self, k: ArrayLike) -> jax.Array:
        """`a` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return jnp.zeros_like(k)

    def b(self, k: ArrayLike) -> jax.Array:
        """`b` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return jnp.where(k == 0, jnp.sqrt(jnp.pi), k / 2)

    def _std_norm(self, k: ArrayLike) -> jax.Array:
        return jnp.sqrt(jnp.sqrt(jnp.pi)) * 2 ** (k / 2) * jnp.exp(gammaln(k + 1) / 2)

    def _std_scale(self, k: ArrayLike) -> jax.Array:
        return 2 ** jnp.asarray(k).astype(float)


class HermiteE(ClassicalRecurrenceRelation):
    """Recurrence relation for (probabalists) Hermite polynomials :math:`He_n(x)`

    Hermite polynomials are orthogonal on the interval (-inf, inf)
    with the weight function :math:`w(x) = e^{-x^2/2}`

    Parameters
    ----------
    scale : {"standard", "monic", "normalized"}
        "standard" corresponds to the common scaling found in textbooks such as
        Abramowitz & Stegun. "monic" scales them such that the leading coefficient is 1.
        "normalized" scales them to have a weighted norm of 1.
    """

    def __init__(self, scale: str = "standard"):
        super().__init__(domain=(-jnp.inf, jnp.inf), scale=scale)

    def weight(self, x: ArrayLike) -> jax.Array:
        """Weight function defining inner product."""
        x = jnp.asarray(x)
        return jnp.exp(-0.5 * x**2)

    def a(self, k: ArrayLike) -> jax.Array:
        """`a` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return jnp.zeros_like(k)

    def b(self, k: ArrayLike) -> jax.Array:
        """`b` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return jnp.where(k == 0, jnp.sqrt(2 * jnp.pi), k)

    def _std_norm(self, k: ArrayLike) -> jax.Array:
        return jnp.sqrt(jnp.sqrt(2 * jnp.pi)) * jnp.exp(gammaln(k + 1) / 2)

    def _std_scale(self, k: ArrayLike) -> jax.Array:
        return jnp.ones_like(k)


def _orthonormal_polyval(x, n, a, sb):
    """Evaluate p_n from sb_{k+1} p_{k+1} = (x - a_k) p_k - sb_k p_{k-1}."""
    x = jnp.asarray(x)

    def body(k, state):
        p0, p1 = state
        return p1, ((x - a[k]) * p1 - sb[k] * p0) / sb[k + 1]

    init = (jnp.zeros_like(x), jnp.ones_like(x) / sb[0])
    return jax.lax.fori_loop(0, n, body, init)[1]


def _orthonormal_basis(x, a, sb):
    """Evaluate p_0, ..., p_{n-1} from the same recurrence, for n = len(a)."""
    x = jnp.asarray(x)

    def body(state, k):
        p0, p1 = state
        p2 = ((x - a[k]) * p1 - sb[k] * p0) / sb[k + 1]
        return (p1, p2), p2

    init = (jnp.zeros_like(x), jnp.ones_like(x) / sb[0])
    _, ps = jax.lax.scan(body, init, jnp.arange(a.size - 1))
    return jnp.concatenate([init[1][None], ps])


def generate_recurrence(
    weight: Callable,
    domain: tuple,
    n: int,
    scale: str = "monic",
    quadrule=None,
    quadopts: dict | None = None,
    check: bool = True,
    tol: float | None = None,
    throw: bool = False,
) -> TabulatedRecurrenceRelation:
    r"""Generate recurrence relation coefficients for orthogonal polynomial family.

    Finds coefficients :math:`a_i, b_i, g_i` such that

    .. math::

        \begin{align}
        p_{-1}(x) &= 0 \\
        p_0(x) &= 1 \\
        p_{i+1}(x) &= (x-a_i) p_i(x) - b_i p_{i-1}(x) \\
        \int_D p_i(x) p_j(x) w(x) dx &= g^2_i \delta_{i,j}
        \end{align}


    Parameters
    ----------
    weight : callable
        Weight function.
    domain : tuple of float
        Lower and upper bounds for the domain of the polynomials.
    n : int
        Number of terms to generate, ie, highest order of polynomial desired.
    scale : {"monic", "normalized"}
        How to scale the resulting polynomials.
    quadrule : quadax.AbstractQuadratureRule, optional
        Quadrature rule to use for computing integrals in generating recurrence
        coefficients. Defaults to ``quadax.GaussKronrodRule(order=31)``.
    quadopts : dict, optional
        Additional options passed to ``quadax.adaptive_quadrature``. Default is
        ``max_ninter=500``. The quadrature tolerances are chosen based on ``tol``, and
        overriding them with ``epsabs`` or ``epsrel`` here may prevent reaching it.
    check : bool
        Whether the returned recurrence relation checks that requested indices are
        within the tabulated range. See ``TabulatedRecurrenceRelation``.
    tol : float, optional
        Relative error tolerance for the coefficients. The error in :math:`b_i` and
        :math:`g_i` is measured relative to their values, and the error in :math:`a_i`
        relative to :math:`|a_i| + \sqrt{b_i} + \sqrt{b_{i+1}}`. Default is the square
        root of the machine precision of the dtype of ``domain`` (or the default float
        type if ``domain`` is not a floating point array).
    throw : bool
        Whether to raise an error if the estimated error in the coefficients exceeds
        ``tol``. The estimate is a conservative first order bound based on the error
        estimates from the quadrature, so the actual error is usually much smaller.

    Returns
    -------
    rec : TabulatedRecurrenceRelation
        Recurrence relation coefficients and polynomial norms.

    Notes
    -----
    Requires the ``quadax`` package to be installed.

    """
    assert scale in ["monic", "normalized"]
    try:
        import quadax
    except ImportError as e:
        raise ImportError(
            "quadax must be installed (use ``pip install quadax``) "
            + "to generate custom orthogonal polynomials."
        ) from e

    # Gauss-Kronrod with extrapolation handles the endpoint singularities common in
    # weight functions well, without needing to evaluate the weight close to the
    # endpoint where the distance to it is lost to roundoff.
    rule = quadax.GaussKronrodRule(31) if quadrule is None else quadrule
    interval = jnp.asarray(domain)
    if not jnp.issubdtype(interval.dtype, jnp.inexact):
        interval = interval.astype(jnp.result_type(float))
    dtype = interval.dtype
    if tol is None:
        tol = float(jnp.sqrt(jnp.finfo(dtype).eps))

    def quad(fun, eps):
        opts = {"epsabs": eps, "epsrel": eps, "max_ninter": 500, "interval": interval}
        opts.update(quadopts or {})
        return quadax.adaptive_quadrature(rule, fun, **opts)

    # Stieltjes procedure in orthonormal form, where sb = sqrt(b). Working with
    # orthonormal rather than monic polynomials keeps every integrand O(1), so the
    # quadrature tolerances are effectively relative. The norms of monic polynomials
    # grow or decay geometrically with degree, which lets a fixed absolute tolerance
    # dominate at high degree. The correction below removes the effect of errors made
    # here to first order, so this only needs to be accurate enough to give a well
    # conditioned basis.
    @jax.jit
    def moments(i, a, sb):
        def fun(x):
            p = _orthonormal_polyval(x, i, a, sb)
            return jnp.stack([jnp.ones_like(x), x]) * p**2 * weight(x)

        return quad(fun, jnp.sqrt(tol))[0]

    def body(i, state):
        a, sb = state
        # sb[i] is still 1 here, so p_i is only orthonormal up to a factor of sqrt(b_i)
        # which the zeroth moment gives.
        m0, m1 = moments(i, a, sb)
        return a.at[i].set(m1 / m0), sb.at[i].set(jnp.sqrt(m0))

    init = (jnp.zeros(n, dtype), jnp.ones(n, dtype))
    aa, sb = jax.lax.fori_loop(0, n, body, init)

    # Quadrature errors in early coefficients propagate into all later polynomials. To
    # correct for this, compute the Gram matrix M0 = <p p^T> and M1 = <x p p^T> in the
    # approximate basis, all in a single quadrature so no errors compound. With
    # M0 = L L^T, the polynomials L^-1 p are orthonormal and their Jacobi matrix
    # J = L^-1 M1 L^-T gives corrected coefficients. L is lower triangular with positive
    # diagonal, so this preserves the degree and sign of each polynomial. The result is
    # exact for any basis up to the quadrature error in M0 and M1, which is amplified
    # by roughly 1 + 2|J| in each coefficient, and accumulates over all i in g_i.
    # Using the Jacobi matrix from the loop to estimate that amplification, the
    # quadrature tolerance is chosen so the error bound computed below meets tol.
    offdiag = jnp.concatenate([sb[1:], jnp.zeros(1, dtype)])
    rowsum = jnp.abs(aa) + jnp.concatenate([jnp.zeros(1, dtype), sb[1:]]) + offdiag
    amplification = 1 + 2 * jnp.max(rowsum)
    headroom = jnp.minimum(jnp.min(rowsum), jnp.min(sb[1:], initial=jnp.inf) / n)
    # Asking for accuracy below roundoff gains nothing and can degrade the result, since
    # the adaptive refinement then only accumulates roundoff.
    eps = tol * jnp.minimum(headroom, 1) / (2 * amplification)
    eps = jnp.maximum(eps, 100 * jnp.finfo(dtype).eps)
    iu, ju = jnp.triu_indices(n)

    def gram(x):
        p = _orthonormal_basis(x, aa, sb)
        pp = p[iu] * p[ju] * weight(x)
        return jnp.stack([pp, x * pp])

    def symmetric(m):
        M = jnp.zeros((n, n), dtype).at[iu, ju].set(m)
        return M + jnp.triu(M, 1).T

    (m0, m1), info = jax.jit(lambda: quad(gram, eps))()
    M0, M1 = symmetric(m0), symmetric(m1)
    L = jnp.linalg.cholesky(M0)
    Linv = jax.scipy.linalg.solve_triangular(L, jnp.eye(n, dtype=dtype), lower=True)
    J = Linv @ M1 @ Linv.T

    aa = jnp.diag(J)
    # p_0 = 1/sb_0 so M0[0, 0] = b_0 / sb_0**2, and the remaining sb are off diagonal
    sb = jnp.concatenate([(sb[0] * jnp.sqrt(M0[0, 0]))[None], jnp.diag(J, -1)])
    bb = sb**2
    g = jnp.sqrt(jnp.cumprod(bb))

    if throw:
        # First order bound on the error in J, given an error of at most info.err in
        # each entry of M0 and M1. Perturbing M0 = L L^T gives
        # L^-1 dM0 L^-T = X + X^T for X = L^-1 dL, which is lower triangular, so
        # dJ = L^-1 dM1 L^-T - X J - J X^T.
        r = jnp.abs(Linv).sum(axis=1)
        T = info.err * jnp.outer(r, r)
        X = jnp.tril(T, -1) + jnp.diag(jnp.diag(T)) / 2
        dJ = T + X @ jnp.abs(J) + jnp.abs(J) @ X.T
        da = jnp.diag(dJ) / (jnp.abs(J).sum(axis=1))
        dsb = jnp.concatenate(
            [(info.err / (2 * M0[0, 0]))[None], jnp.diag(dJ, -1) / sb[1:]]
        )
        # relative errors in b and g are 2 dsb/sb and the cumulative sum of dsb/sb
        err = jnp.maximum(jnp.maximum(da, 2 * dsb), jnp.cumsum(dsb))
        g = eqx.error_if(
            g,
            ~jnp.all(err <= tol) | ~jnp.all(jnp.isfinite(g)),
            "Estimated error in recurrence coefficients exceeds tol. Try increasing "
            "tol or max_ninter in quadopts, or using a different quadrature rule.",
        )

    if scale == "monic":
        m = jnp.ones_like(g)
    else:  # normalized
        m = 1 / g

    return TabulatedRecurrenceRelation(weight, domain, aa, bb, g, m, check=check)
