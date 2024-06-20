r"""
==================================================================
General orthogonal polynomials via three term recurrence relations
==================================================================

Every family of orthogonal polynomials can be shown to satisfy a three term recurrence
relation of the form

.. math::

    \begin{align}
    p_{-1}(x) &= 0 \\
    p_0(x) &= 1 \\
    p_{i+1}(x) &= (x-a_i) p_i(x) - b_i p_{i-1}(x) \\
    \end{align}

Knowing the coefficients of the recurrence relation :math:`a_i, b_i` along with the
normalization constants :math:`\gamma_i` such that

.. math::

    \int_D p_i(x) p_j(x) w(x) dx = \gamma^2_i \delta_{i,j}


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

   LegendreRecurrenceRelation
   ShiftedLegendreRecurrenceRelation
   ChebyshevTRecurrenceRelation
   ChebyshevURecurrenceRelation
   ChebyshevVRecurrenceRelation
   ChebyshevWRecurrenceRelation
   GegenbauerRecurrenceRelation
   JacobiRecurrenceRelation
   LaguerreRecurrenceRelation
   GeneralizedLaguerreRecurrenceRelation
   HermiteRecurrenceRelation
   HermiteERecurrenceRelation

Recurrence Coefficients
-----------------------
.. autosummary::
   :toctree: generated/

   AbstractRecurrenceCoefficient
   FunctionRecurrenceCoefficient
   TabulatedRecurrenceCoefficient

"""

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln


class AbstractRecurrenceCoefficient(eqx.Module):
    """Abstract base class for recurrence relation coefficients."""

    @abc.abstractmethod
    def __getitem__(self, k):
        pass


class FunctionRecurrenceCoefficient(AbstractRecurrenceCoefficient):
    """Recurrence coefficient with a known functional form.

    Parameters
    ----------
    fun : callable
        Function of the form fun(k:int, *params)-> float to get kth coefficient.
    params : tuple
        Additional parameters passed to fun.
    """

    _fun: callable = eqx.field(static=True)
    _params: any

    def __init__(self, fun, params=()):
        self._fun = fun
        self._params = params

    def __getitem__(self, k):
        return jnp.where(k >= 0, self._fun(k, *self._params), 0)


class TabulatedRecurrenceCoefficient(AbstractRecurrenceCoefficient):
    """Precomputed recurrence coefficients as an array.

    Parameters
    ----------
    arr : jax.Array
        Array of tabulated recurrence coefficients.
    """

    _arr: jax.Array

    def __init__(self, arr):
        self._arr = arr

    def __getitem__(self, k):
        k = jnp.asarray(k)
        k = eqx.error_if(k, (k >= len(self._arr)).any(), "out of bounds!")
        return jnp.where(k >= 0, self._arr[k], 0)


class AbstractRecurrenceRelation(eqx.Module):
    """Base class for three term recurrence relations.

    Parameters
    ----------
    weight : callable
        Weight function.
    domain : tuple
        Lower and upper bounds for inner product defining orthogonality.
    a, b : AbstractRecurrenceCoefficient
        Coefficients of the three term recurrence relation.
    g : AbstractRecurrenceCoefficient
        ``g[k]`` is the weighted norm of the kth orthogonal polynomial.
    m : AbstractRecurrenceCoefficient
        ``m[k]`` is the coefficient of x**k in the kth orthogonal polynomial.
    scale : {"monic", "normalized"}
        Whether the a,b coefficients are for the standard "monic" form of the
        polynomial, or the normalized form.

    """

    weight: callable = eqx.field(static=True)
    domain: tuple[float, float]
    a: AbstractRecurrenceCoefficient
    b: AbstractRecurrenceCoefficient
    g: AbstractRecurrenceCoefficient
    m: AbstractRecurrenceCoefficient
    scale: str = eqx.field(static=True)

    def __init__(self, weight, domain, a, b, g, m=None, scale="monic"):
        assert scale in {"monic", "normalized"}
        if m is None:
            m = FunctionRecurrenceCoefficient(lambda x: jnp.ones(jnp.asarray(x).shape))
        self.a = a
        self.b = b
        self.m = m
        self.g = g
        self.weight = weight
        self.domain = domain
        self.scale = scale


class TabulatedRecurrenceRelation(AbstractRecurrenceRelation):
    """Recurrence relation from tabulated values.

    Parameters
    ----------
    weight : callable
        Weight function.
    domain : tuple
        Lower and upper bounds for inner product defining orthogonality.
    a, b : jax.Array
        Coefficients of the three term recurrence relation.
    g : jax.Array
        ``g[k]`` is the weighted norm of the kth orthogonal polynomial.
    m : jax.Array
        ``m[k]`` is the coefficient of x**k in the kth orthogonal polynomial.
    scale : {"monic", "normalized"}
        Whether the a,b coefficients are for the standard "monic" form of the
        polynomial, or the normalized form.

    """

    ak: jax.Array
    bk: jax.Array
    gk: jax.Array
    mk: jax.Array

    def __init__(self, weight, domain, ak, bk, gk, mk=None, scale="monic"):
        assert scale in {"monic", "normalized"}
        if mk is None:
            mk = jnp.ones_like(ak)
        self.ak = ak
        self.bk = bk
        self.gk = gk
        self.mk = mk
        self.a = TabulatedRecurrenceCoefficient(ak)
        self.b = TabulatedRecurrenceCoefficient(bk)
        self.g = TabulatedRecurrenceCoefficient(gk)
        self.m = TabulatedRecurrenceCoefficient(mk)
        self.weight = weight
        self.domain = domain
        self.scale = scale


class ClassicalRecurrenceRelation(AbstractRecurrenceRelation):
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

    a: FunctionRecurrenceCoefficient = eqx.field(static=True)
    b: FunctionRecurrenceCoefficient = eqx.field(static=True)
    g: FunctionRecurrenceCoefficient = eqx.field(static=True)
    m: FunctionRecurrenceCoefficient = eqx.field(static=True)

    def __init__(self, weight, domain, scale="standard"):
        assert scale in {"standard", "monic", "normalized"}

        self.a = FunctionRecurrenceCoefficient(self._ak)
        self.b = FunctionRecurrenceCoefficient(self._bk)
        self.g = FunctionRecurrenceCoefficient(self._gk)
        self.m = FunctionRecurrenceCoefficient(self._mk)
        self.weight = weight
        self.domain = domain
        self.scale = scale

    @abc.abstractmethod
    def _ak(self, k):
        # alpha coefficients of monic recurrence relation
        pass

    @abc.abstractmethod
    def _bk(self, k):
        # beta coefficients of monic recurrence relation
        pass

    @abc.abstractmethod
    def _std_norm(self, k):
        # norm of the kth polynomial in "standard" scaling (ie, AS, wikipedia, etc)
        pass

    @abc.abstractmethod
    def _std_scale(self, k):
        # leading order coefficient in "standard" scaling (ie, AS, wikipedia, etc)
        pass

    def _gk(self, k):
        # norm of kth polynomial in whatever scale is specified by the user,
        # ie for scale="normalized" this is 1
        # ie, evaluate normalized polynomials then multiply by g to get scaled version
        if self.scale == "monic":
            return self._std_norm(k) / jnp.abs(self._std_scale(k))
        elif self.scale == "standard":
            return self._std_norm(k)
        else:  # normalized
            return jnp.ones_like(k)

    def _mk(self, k):
        # scaling factor. polynomials are evaluated in monic form then multiplied
        # by this scale factor
        # ie, evaluate monic polynomials then multiply by m to get scaled version
        if self.scale == "monic":
            return jnp.ones_like(k)
        elif self.scale == "standard":
            return self._std_scale(k)
        else:  # normalized
            return self._std_scale(k) / self._std_norm(k)


# General notes:
# ak, bk from Gautschi, Orthogonal Polynomials: Computation and Approximation,
# Table 1.1, ak=alpha_k, bk = beta_k
# std_scale, std_norm from NIST Handbook of Mathematical Functions,
# Table 18.3.1, std_scale = k_n, std_norm = sqrt(h_n)


class LegendreRecurrenceRelation(ClassicalRecurrenceRelation):
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

    def _ak(self, k):
        return jnp.zeros_like(k)

    def _bk(self, k):
        k = jnp.asarray(k, dtype=jnp.result_type(k, 1.0))
        return jnp.where(k == 0, 2, 1 / (4 - k ** (-2)))

    def _std_norm(self, k):
        return jnp.sqrt(2 / (2 * k + 1))

    def _std_scale(self, k):
        return jnp.exp(
            k * jnp.log(2) - gammaln(k + 1) + gammaln(k + 0.5) - gammaln(0.5)
        )


class ShiftedLegendreRecurrenceRelation(ClassicalRecurrenceRelation):
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

    def _ak(self, k):
        return 0.5 * jnp.ones_like(k)

    def _bk(self, k):
        k = jnp.asarray(k, dtype=jnp.result_type(k, 1.0))
        return jnp.where(k == 0, 1, 1 / (4 * (4 - k ** (-2))))

    def _std_norm(self, k):
        return jnp.sqrt(1 / (2 * k + 1))

    def _std_scale(self, k):
        return jnp.exp(
            2 * k * jnp.log(2) - gammaln(k + 1) + gammaln(k + 0.5) - gammaln(0.5)
        )


class ChebyshevTRecurrenceRelation(ClassicalRecurrenceRelation):
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

    def _ak(self, k):
        return jnp.zeros_like(k)

    def _bk(self, k):
        return jnp.where(k == 0, jnp.pi, jnp.where(k == 1, 1 / 2, 1 / 4))

    def _std_norm(self, k):
        return jnp.sqrt(jnp.where(k == 0, jnp.pi, jnp.pi / 2))

    def _std_scale(self, k):
        return jnp.where(k == 0, 1, 2 ** jnp.maximum(0.0, k - 1.0))


class ChebyshevURecurrenceRelation(ClassicalRecurrenceRelation):
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

    def _ak(self, k):
        return jnp.zeros_like(k)

    def _bk(self, k):
        return jnp.where(k == 0, jnp.pi / 2, 1 / 4)

    def _std_norm(self, k):
        return jnp.full(k.shape, jnp.sqrt(jnp.pi / 2))

    def _std_scale(self, k):
        return 2 ** jnp.asarray(k).astype(float)


class ChebyshevVRecurrenceRelation(ClassicalRecurrenceRelation):
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

    def _ak(self, k):
        return jnp.where(k == 0, 0.5, 0)

    def _bk(self, k):
        return jnp.where(k == 0, jnp.pi, 1 / 4)

    def _std_norm(self, k):
        return jnp.full(k.shape, jnp.sqrt(jnp.pi))

    def _std_scale(self, k):
        return 2 ** jnp.asarray(k).astype(float)


class ChebyshevWRecurrenceRelation(ClassicalRecurrenceRelation):
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

    def _ak(self, k):
        return jnp.where(k == 0, -0.5, 0)

    def _bk(self, k):
        return jnp.where(k == 0, jnp.pi, 1 / 4)

    def _std_norm(self, k):
        return jnp.full(k.shape, jnp.sqrt(jnp.pi))

    def _std_scale(self, k):
        return 2 ** jnp.asarray(k).astype(float)


class GegenbauerRecurrenceRelation(ClassicalRecurrenceRelation):
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

    def _ak(self, k):
        return jnp.zeros_like(k)

    def _bk(self, k):
        b0 = jnp.sqrt(jnp.pi) * jnp.exp(
            gammaln(self.lmbda + 0.5) - gammaln(self.lmbda + 1)
        )
        bknum = jnp.where(self.lmbda == 0, 1, k * (k + 2 * self.lmbda - 1))
        bkden = jnp.where(
            self.lmbda == 0, 4, (4 * (k + self.lmbda) * (k + self.lmbda - 1))
        )
        bk = bknum / bkden
        return jnp.where(k == 0, b0, bk)

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


class JacobiRecurrenceRelation(ClassicalRecurrenceRelation):
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

    def _ak(self, k):
        a, b = self.alpha, self.beta
        num = jnp.where(k == 0, b - a, b**2 - a**2)
        den = jnp.where(k == 0, a + b + 2, (2 * k + a + b) * (2 * k + a + b + 2))
        return num / den

    def _bk(self, k):
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


class LaguerreRecurrenceRelation(ClassicalRecurrenceRelation):
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

    def _ak(self, k):
        return 2 * k + 1

    def _bk(self, k):
        return jnp.where(k == 0, 1, k**2)

    def _std_norm(self, k):
        return jnp.ones(jnp.asarray(k).shape)

    def _std_scale(self, k):
        return (-1) ** k * jnp.exp(-gammaln(k + 1))


class GeneralizedLaguerreRecurrenceRelation(ClassicalRecurrenceRelation):
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

    def _ak(self, k):
        return 2 * k + self.alpha + 1

    def _bk(self, k):
        return jnp.where(k == 0, jnp.exp(gammaln(self.alpha + 1)), k * (k + self.alpha))

    def _std_norm(self, k):
        return jnp.exp(-0.5 * gammaln(k + 1) + 0.5 * gammaln(k + self.alpha + 1))

    def _std_scale(self, k):
        return (-1) ** k * jnp.exp(-gammaln(k + 1))


class HermiteRecurrenceRelation(ClassicalRecurrenceRelation):
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

    def _ak(self, k):
        return jnp.zeros_like(k)

    def _bk(self, k):
        return jnp.where(k == 0, jnp.sqrt(jnp.pi), k / 2)

    def _std_norm(self, k):
        return jnp.sqrt(jnp.sqrt(jnp.pi)) * 2 ** (k / 2) * jnp.exp(gammaln(k + 1) / 2)

    def _std_scale(self, k):
        return 2 ** jnp.asarray(k).astype(float)


class HermiteERecurrenceRelation(ClassicalRecurrenceRelation):
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

    def _ak(self, k):
        return jnp.zeros_like(k)

    def _bk(self, k):
        return jnp.where(k == 0, jnp.sqrt(2 * jnp.pi), k)

    def _std_norm(self, k):
        return jnp.sqrt(jnp.sqrt(2 * jnp.pi)) * jnp.exp(gammaln(k + 1) / 2)

    def _std_scale(self, k):
        return jnp.ones_like(k)


def generate_recurrence(weight, domain, n, scale="monic"):
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

    Returns
    -------
    rec : TabulatedRecurrenceRelation
        Recurrence relation coefficients and polynomial norms.

    """
    try:
        from quadax import quadts as quad
    except ImportError as e:
        raise ImportError(
            "quadax must be installed (use ``pip install quadax``) "
            + "to generate custom orthogonal polynomials."
        ) from e

    # p-adaptive might be better here, or bootstrapped gauss
    @jax.jit
    def inner(n, a, b):
        fun = lambda x: polyval(x, n, a, b) ** 2 * weight(x)
        return quad(fun, domain, epsabs=1e-14, epsrel=1e-14, order=128)

    @jax.jit
    def innerx(n, a, b):
        fun = lambda x: x * polyval(x, n, a, b) ** 2 * weight(x)
        return quad(fun, domain, epsabs=1e-14, epsrel=1e-14, order=128)

    @jax.jit
    def polyval(x, n, a, b):
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
    cc = jnp.sqrt(cc)
    if scale == "monic":
        g = cc
        m = jnp.ones_like(cc)
    else:  # normalized
        g = jnp.ones_like(cc)
        m = 1 / cc

    return TabulatedRecurrenceRelation(weight, domain, aa, bb, g, m)
