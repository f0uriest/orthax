"""Ensure that the jax specific features work"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

import jax
from jax import numpy as jnp, config

from orthax import (
    Chebyshev,
    Hermite,
    HermiteE,
    Laguerre,
    Legendre,
    Polynomial,
    polybase as pb,
)

config.update("jax_enable_x64", True)


DEFAULT_KINDS = [Chebyshev, Hermite, HermiteE, Laguerre, Legendre, Polynomial]


def random_polynomial(kind, rng: np.random.Generator):
    """create random polynomial of kind supplied"""
    coef = rng.standard_normal(rng.integers(low=1, high=20))
    return kind(coef)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("kind", DEFAULT_KINDS)
class TestArithmetic:
    """Test arithmetic features, especially composition with jit"""

    def test_mul(self, seed, kind):
        """Test multiplication"""
        rng = np.random.default_rng(seed)
        p = random_polynomial(kind, rng)
        q = random_polynomial(kind, rng)

        # jit has no impact
        assert_allclose(jax.jit(lambda: p * q)().coef, (p * q).coef, rtol=1e-11)

        # functional equivalence
        x = rng.uniform(-1, 1, 100)
        assert_allclose(jax.jit(lambda: (p * q)(x))(), p(x) * q(x), rtol=1e-11)

    def test_add(self, seed, kind):
        """Test addition"""
        rng = np.random.default_rng(seed)
        p = random_polynomial(kind, rng)
        q = random_polynomial(kind, rng)

        # jit has no impact
        assert_allclose(jax.jit(lambda: p + q)().coef, (p + q).coef, rtol=1e-11)

        # functional equivalence
        x = rng.uniform(-1, 1, 100)
        assert_allclose(jax.jit(lambda: (p + q)(x))(), p(x) + q(x), rtol=1e-11)

    def test_sub(self, seed, kind):
        """Test subtraction"""
        rng = np.random.default_rng(seed)
        p = random_polynomial(kind, rng)
        q = random_polynomial(kind, rng)

        # jit has no impact
        assert_allclose(jax.jit(lambda: p - q)().coef, (p - q).coef, rtol=1e-11)

        # functional equivalence
        x = rng.uniform(-1, 1, 100)
        assert_allclose(jax.jit(lambda: (p - q)(x))(), p(x) - q(x), rtol=1e-11)

    def test_div(self, seed, kind):
        """Test division"""
        rng = np.random.default_rng(seed)
        p = random_polynomial(kind, rng)
        num = rng.standard_normal()

        # jit has no impact
        assert_allclose(jax.jit(lambda: p / num)().coef, (p / num).coef, rtol=1e-11)

        # functional equivalence
        x = rng.uniform(-1, 1, 100)
        assert_allclose(jax.jit(lambda: (p / num)(x))(), p(x) / num, rtol=1e-11)

    def test_pow(self, seed, kind):
        """Test powers"""
        rng = np.random.default_rng(seed)
        p = random_polynomial(kind, rng)
        num = rng.integers(0, 5)

        # jit has no impact
        assert_allclose(jax.jit(lambda: p**num)().coef, (p**num).coef, rtol=1e-11)

        # functional equivalence
        x = rng.uniform(-1, 1, 100)

        # this seems to be numerically unstable:
        # only require pass if numpy passes
        npp = getattr(np.polynomial, kind.__name__)(p.coef)
        if np.allclose((npp**num)(x), npp(x) ** num):
            # no jit
            assert_allclose(
                (p**num)(x),
                p(x) ** num,
            )
            # with jit
            assert_allclose(
                jax.jit(lambda: (p**num)(x))(),
                p(x) ** num,
            )


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("kind", DEFAULT_KINDS)
class TestDifferentiation:
    """Test AutoDiff"""

    def test_differentiation(self, seed, kind):
        """Compare AutoDiff with manual derivatives"""
        rng = np.random.default_rng(seed)
        coef = rng.standard_normal(rng.integers(low=1, high=20))
        p: pb.ABCPolyBase = kind(jnp.asarray(coef))

        # evaluate derivative at one point
        x = rng.standard_normal()
        assert_allclose(jax.jit(jax.grad(p))(x), p.deriv()(x), rtol=1e-06)

        # vectorize
        x_vec = rng.standard_normal(10)
        assert_allclose(
            jax.jit(jax.vmap(jax.grad(p)))(x_vec),
            jax.vmap(p.deriv())(x_vec),
        )

        # second order derivative
        assert_allclose(jax.jit(jax.grad(jax.grad(p)))(x), p.deriv(2)(x), rtol=1e-05)


class TestMisc:
    """Other tests"""

    @pytest.mark.parametrize("seed", range(3))
    @pytest.mark.parametrize("kind1", DEFAULT_KINDS)
    @pytest.mark.parametrize("kind2", DEFAULT_KINDS)
    def test_conversion(
        self, seed, kind1: type[pb.ABCPolyBase], kind2: type[pb.ABCPolyBase]
    ):
        """Test converting Polynomials from one kind to the other"""
        rng = np.random.default_rng(seed)
        coef = rng.standard_normal(10)
        p = kind1(coef)
        q = p.convert(kind=kind2)

        assert isinstance(p, kind1)
        assert isinstance(q, kind2)

        # roundtrip
        qp_coef = q.convert(kind=kind1).coef
        n = min(len(qp_coef), len(p.coef))
        err_msg = "roundtrip conversion is broken"
        assert_allclose(
            qp_coef[:n],
            p.coef[:n],
            err_msg=err_msg,
            rtol=1e-4 if kind2 == Laguerre else 1e-7,
            atol=1e-4 if kind2 == Laguerre else 0,
        )
        assert_allclose(qp_coef[n:], np.zeros_like(qp_coef[n:]), err_msg=err_msg)
        assert_allclose(p.coef[n:], np.zeros_like(p.coef[n:]), err_msg=err_msg)

        # functional equivalence
        x = rng.uniform(low=p.domain[0], high=p.domain[1], size=100)
        assert_allclose(
            p(x),
            q(x),
            err_msg="conversion does not result in the same polynomial",
            rtol=1e-4 if kind2 == Laguerre else 1e-6,
            atol=1e-4 if kind2 == Laguerre else 0,
        )
