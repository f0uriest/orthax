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

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln


def _asarray(k, kmax=None):
    k = jnp.asarray(k)
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

    Subclasses should declare attributes `_weight` and `_domain` and implement methods
    `a`, `b`, `g`, `m`
    """

    _weight: callable = eqx.field(static=True)
    _domain: tuple[float, float]

    @property
    def weight(self):
        """callable: Weight function defining inner product."""
        return self._weight

    @property
    def domain(self):
        """tuple: Lower and upper bounds for inner product defining orthogonality."""
        return self._domain

    @abc.abstractmethod
    def a(self, k):
        """`a` coefficients of the monic three term recurrence relation."""
        pass

    @abc.abstractmethod
    def b(self, k):
        """`b` coefficients of the monic three term recurrence relation."""
        pass

    @abc.abstractmethod
    def g(self, k):
        """Weighted norm of the kth monic orthogonal polynomial."""
        pass

    @abc.abstractmethod
    def m(self, k):
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

    """

    _ak: jax.Array
    _bk: jax.Array
    _gk: jax.Array
    _mk: jax.Array

    def __init__(self, weight, domain, ak, bk, gk, mk=None):
        if mk is None:
            mk = jnp.ones_like(ak)
        self._ak = ak
        self._bk = bk
        self._gk = gk
        self._mk = mk
        self._weight = weight
        self._domain = domain

    def a(self, k):
        """`a` coefficients of the monic three term recurrence relation."""
        k = _asarray(k, kmax=len(self._ak) - 1)
        return self._ak[k]

    def b(self, k):
        """`b` coefficients of the monic three term recurrence relation."""
        k = _asarray(k, kmax=len(self._bk) - 1)
        return self._bk[k]

    def g(self, k):
        """Weighted norm of the kth monic orthogonal polynomial."""
        k = _asarray(k, kmax=len(self._gk) - 1)
        return self._gk[k]

    def m(self, k):
        """Coefficient of x**k in the kth polynomial in the desired normalization."""
        k = _asarray(k, kmax=len(self._mk) - 1)
        return self._mk[k]


class ClassicalRecurrenceRelation(AbstractRecurrenceRelation, abc.ABC):
    """Base class for recurrence relations for "classical" orthogonal polynomials.

    Parameters
    ----------
    weight : callable
        Weight function.
    domain : tuple
        Lower and upper bounds for inner product defining orthogonality.
    scale : {"standard", "monic", "normalized"}
        Most classical orthogonal polynomials have ad-hoc normalizations (ie,
        the common definitions in textbooks are neither monic nor unit norm). This
        is encompassed in the "standard" scale, which should match that in texts such as
        Abramowitz & Stegun. Alternatively, they can be scaled to be monic or unit norm.
    """

    _scale: str = eqx.field(static=True)

    def __init__(self, weight, domain, scale="standard"):
        assert scale in {"standard", "monic", "normalized"}

        self._weight = weight
        self._domain = domain
        self._scale = scale

    @abc.abstractmethod
    def _std_norm(self, k):
        # norm of the kth polynomial in "standard" scaling (ie, AS, wikipedia, etc)
        pass

    @abc.abstractmethod
    def _std_scale(self, k):
        # coefficient of x**k in "standard" scaling (ie, AS, wikipedia, etc)
        pass

    def g(self, k):
        """Weighted norm of the kth monic orthogonal polynomial."""
        k = _asarray(k)
        return self._std_norm(k) / jnp.abs(self._std_scale(k))

    def m(self, k):
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

    def __init__(self, scale="standard"):
        super().__init__(weight=lambda x: jnp.ones_like(x), domain=(-1, 1), scale=scale)

    def a(self, k):
        """`a` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return jnp.zeros_like(k)

    def b(self, k):
        """`b` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return jnp.where(k == 0, 2, 1 / (4 - 1 / jnp.where(k == 0, 1, k) ** 2))

    def _std_norm(self, k):
        return jnp.sqrt(2 / (2 * k + 1))

    def _std_scale(self, k):
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

    def __init__(self, scale="standard"):
        super().__init__(weight=lambda x: jnp.ones_like(x), domain=(0, 1), scale=scale)

    def a(self, k):
        """`a` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return 0.5 * jnp.ones_like(k)

    def b(self, k):
        """`b` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return jnp.where(k == 0, 1, 0.25 / (4 - 1 / jnp.where(k == 0, 1, k) ** 2))

    def _std_norm(self, k):
        return jnp.sqrt(1 / (2 * k + 1))

    def _std_scale(self, k):
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

    def __init__(self, scale="standard"):
        super().__init__(
            weight=lambda x: 1.0 / jnp.sqrt(1 - x**2), domain=(-1, 1), scale=scale
        )

    def a(self, k):
        """`a` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return jnp.zeros_like(k)

    def b(self, k):
        """`b` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return jnp.where(k == 0, jnp.pi, jnp.where(k == 1, 1 / 2, 1 / 4))

    def _std_norm(self, k):
        return jnp.sqrt(jnp.where(k == 0, jnp.pi, jnp.pi / 2))

    def _std_scale(self, k):
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

    def __init__(self, scale="standard"):
        super().__init__(
            weight=lambda x: jnp.sqrt(1 - x**2), domain=(-1, 1), scale=scale
        )

    def a(self, k):
        """`a` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return jnp.zeros_like(k)

    def b(self, k):
        """`b` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return jnp.where(k == 0, jnp.pi / 2, 1 / 4)

    def _std_norm(self, k):
        return jnp.full(k.shape, jnp.sqrt(jnp.pi / 2))

    def _std_scale(self, k):
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

    def __init__(self, scale="standard"):
        super().__init__(
            weight=lambda x: jnp.sqrt((1 + x) / (1 - x)), domain=(-1, 1), scale=scale
        )

    def a(self, k):
        """`a` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return jnp.where(k == 0, 0.5, 0)

    def b(self, k):
        """`b` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return jnp.where(k == 0, jnp.pi, 1 / 4)

    def _std_norm(self, k):
        return jnp.full(k.shape, jnp.sqrt(jnp.pi))

    def _std_scale(self, k):
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

    def __init__(self, scale="standard"):
        super().__init__(
            weight=lambda x: jnp.sqrt((1 - x) / (1 + x)), domain=(-1, 1), scale=scale
        )

    def a(self, k):
        """`a` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return jnp.where(k == 0, -0.5, 0)

    def b(self, k):
        """`b` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return jnp.where(k == 0, jnp.pi, 1 / 4)

    def _std_norm(self, k):
        return jnp.full(k.shape, jnp.sqrt(jnp.pi))

    def _std_scale(self, k):
        return 2 ** jnp.asarray(k).astype(float)


class Gegenbauer(ClassicalRecurrenceRelation):
    r"""Recurrence relation for Gegenbauer polynomials :math:`C^\lambda_n(x)`

    Also known as Ultraspherical harmonics.

    Gegenbauer polynomials are orthogonal on the interval (-1, 1)
    with the weight function :math:`w(x) = (1-x^2)^{\lambda - 1/2}`

    Parameters
    ----------
    lmbda : float > -1/2
        Hyperparameter λ.
    scale : {"standard", "monic", "normalized"}
        "standard" corresponds to the common scaling found in textbooks such as
        Abramowitz & Stegun. "monic" scales them such that the leading coefficient is 1.
        "normalized" scales them to have a weighted norm of 1.
    """

    lmbda: float

    def __init__(self, lmbda, scale="standard"):
        self.lmbda = eqx.error_if(lmbda, lmbda <= -0.5, "lmbda must be > -1/2")
        super().__init__(
            weight=lambda x: (1 - x**2) ** (self.lmbda - 0.5),
            domain=(-1, 1),
            scale=scale,
        )

    def a(self, k):
        """`a` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return jnp.zeros_like(k)

    def b(self, k):
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

    def _std_norm(self, k):
        lognum = (
            (1 - 2 * self.lmbda) * jnp.log(2)
            + jnp.log(jnp.pi)
            + gammaln(k + 2 * self.lmbda)
        )
        logden = jnp.log(k + self.lmbda) + 2 * gammaln(self.lmbda) + gammaln(k + 1)
        return jnp.exp(0.5 * (lognum - logden))

    def _std_scale(self, k):
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

    alpha: float
    beta: float

    def __init__(self, alpha, beta, scale="standard"):
        self.alpha = eqx.error_if(alpha, alpha <= -1, "alpha must be > -1")
        self.beta = eqx.error_if(beta, beta <= -1, "beta must be > -1")
        super().__init__(
            weight=lambda x: (1 - x) ** self.alpha * (1 + x) ** self.beta,
            domain=(-1, 1),
            scale=scale,
        )

    def a(self, k):
        """`a` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        a, b = self.alpha, self.beta
        num = jnp.where(k == 0, b - a, b**2 - a**2)
        den = jnp.where(k == 0, a + b + 2, (2 * k + a + b) * (2 * k + a + b + 2))
        return num / den

    def b(self, k):
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

    def _std_norm(self, k):
        a, b = self.alpha, self.beta
        lognum = (a + b + 1) * jnp.log(2) + gammaln(k + a + 1) + gammaln(k + b + 1)
        logden = jnp.log(2 * k + a + b + 1) + gammaln(k + a + b + 1) + gammaln(k + 1)
        return jnp.exp(0.5 * (lognum - logden))

    def _std_scale(self, k):
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

    def __init__(self, scale="standard"):
        super().__init__(weight=lambda x: jnp.exp(-x), domain=(0, jnp.inf), scale=scale)

    def a(self, k):
        """`a` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return 2 * k + 1

    def b(self, k):
        """`b` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return jnp.where(k == 0, 1, k**2)

    def _std_norm(self, k):
        return jnp.ones(jnp.asarray(k).shape)

    def _std_scale(self, k):
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

    alpha: float

    def __init__(self, alpha, scale="standard"):
        self.alpha = eqx.error_if(alpha, alpha <= -1, "alpha must be > -1")
        super().__init__(
            weight=lambda x: x**self.alpha * jnp.exp(-x),
            domain=(0, jnp.inf),
            scale=scale,
        )

    def a(self, k):
        """`a` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return 2 * k + self.alpha + 1

    def b(self, k):
        """`b` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return jnp.where(k == 0, jnp.exp(gammaln(self.alpha + 1)), k * (k + self.alpha))

    def _std_norm(self, k):
        return jnp.exp(-0.5 * gammaln(k + 1) + 0.5 * gammaln(k + self.alpha + 1))

    def _std_scale(self, k):
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

    def __init__(self, scale="standard"):
        super().__init__(
            weight=lambda x: jnp.exp(-(x**2)), domain=(-jnp.inf, jnp.inf), scale=scale
        )

    def a(self, k):
        """`a` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return jnp.zeros_like(k)

    def b(self, k):
        """`b` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return jnp.where(k == 0, jnp.sqrt(jnp.pi), k / 2)

    def _std_norm(self, k):
        return jnp.sqrt(jnp.sqrt(jnp.pi)) * 2 ** (k / 2) * jnp.exp(gammaln(k + 1) / 2)

    def _std_scale(self, k):
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

    def __init__(self, scale="standard"):
        super().__init__(
            weight=lambda x: jnp.exp(-0.5 * x**2),
            domain=(-jnp.inf, jnp.inf),
            scale=scale,
        )

    def a(self, k):
        """`a` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return jnp.zeros_like(k)

    def b(self, k):
        """`b` coefficients of the monic three term recurrence relation."""
        k = _asarray(k)
        return jnp.where(k == 0, jnp.sqrt(2 * jnp.pi), k)

    def _std_norm(self, k):
        return jnp.sqrt(jnp.sqrt(2 * jnp.pi)) * jnp.exp(gammaln(k + 1) / 2)

    def _std_scale(self, k):
        return jnp.ones_like(k)


@jax.jit
def _polyval(x, n, a, b):
    a, b = map(lambda x: jnp.atleast_1d(jnp.asarray(x)), (a, b))
    x = jnp.asarray(x)

    p0 = jnp.zeros_like(x)
    p1 = jnp.ones_like(x)
    pn = p1

    def body(i, state):
        p0, p1, pn = state
        pn = (x - a[i]) * p1 - b[i] * p0
        p0 = p1
        p1 = pn
        return p0, p1, pn

    npos = lambda: jax.lax.fori_loop(0, n, body, (p0, p1, pn))[-1]
    nneg = lambda: jnp.zeros_like(x)
    return jax.lax.cond(n >= 0, npos, nneg)


def generate_recurrence(weight, domain, n, scale="monic", quadrule=None, quadopts=None):
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
        coefficients. Defaults to ``quadax.TanhSinhRule(order=129)``.
    quadopts : dict, optional
        Additional options passed to ``quadax.adaptive_quadrature``. Default options
        are ``epsabs=1e-15``, ``epsrel=1e-15``, ``max_ninter=500``.

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

    # p-adaptive might be better here, or bootstrapped gauss
    rule = quadrule or quadax.TanhSinhRule(129)
    opts = quadopts or {}
    opts.setdefault("epsabs", 1e-15)
    opts.setdefault("epsrel", 1e-15)
    opts.setdefault("max_ninter", 500)
    opts.setdefault("interval", domain)
    quad = lambda fun: quadax.adaptive_quadrature(rule, fun, **opts)

    @jax.jit
    def inner(n, a, b):
        fun = lambda x: _polyval(x, n, a, b) ** 2 * weight(x)
        return quad(fun)

    @jax.jit
    def innerx(n, a, b):
        fun = lambda x: x * _polyval(x, n, a, b) ** 2 * weight(x)
        return quad(fun)

    def body(i, state):
        aa, bb, cc, errs, status = state
        m0, out = inner(i, aa, bb)
        errs = errs.at[i, 0].set(out.err)
        status = status.at[i, 0].set(out.status)
        cc = cc.at[i].set(m0)
        ai, out = innerx(i, aa, bb)
        errs = errs.at[i, 1].set(out.err)
        status = status.at[i, 1].set(out.status)
        aa = aa.at[i].set(ai / cc[i])
        bb = bb.at[i].set(jnp.where(i == 0, m0, cc[i] / cc[i - 1]))
        return aa, bb, cc, errs, status

    aa = jnp.zeros(n)
    bb = jnp.zeros(n)
    cc = jnp.zeros(n)
    status = jnp.zeros((n, 2))
    errs = jnp.zeros((n, 2))

    aa, bb, cc, errs, status = jax.lax.fori_loop(0, n, body, (aa, bb, cc, errs, status))
    # TODO: figure out uncertainties better
    g = jnp.sqrt(cc)
    if scale == "monic":
        m = jnp.ones_like(g)
    else:  # normalized
        m = 1 / g

    return TabulatedRecurrenceRelation(weight, domain, aa, bb, g, m)
