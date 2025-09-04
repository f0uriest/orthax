"""Ensure that the jax specific features work"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

import jax
from jax import numpy as jnp, config

from orthax import (
    Chebyshev,
    Hermite,
    HermiteE,
    Laguerre,
    Legendre,
    OrthPoly,
    Polynomial,
    polybase as pb,
)

config.update("jax_enable_x64", True)


DEFAULT_KINDS = [Chebyshev, Hermite, HermiteE, Laguerre, Legendre, Polynomial]

@pytest.mark.parametrize("kind", DEFAULT_KINDS)
class TestArithmetic:
    def test_mul(self, kind):
        p = kind([1.,2,3])
        q = kind([1,2])
        assert jax.jit(lambda: p*q)() == p*q

    def test_add(self, kind):
        p = kind([1.,2,3])
        q = kind([1,2])
        assert jax.jit(lambda: p+q)() == p+q

    def test_div(self, kind):
        p = kind([1.,2,3])
        assert jax.jit(lambda: p/2)() == p/2

    def test_pow(self, kind):
        p = kind([1.,2,3])
        assert jax.jit(lambda: p ** 5)() == p ** 5


# @pytest.mark.skip()
@pytest.mark.parametrize("seed", range(7))
@pytest.mark.parametrize("kind1", DEFAULT_KINDS)
@pytest.mark.parametrize("kind2", DEFAULT_KINDS)
def test_conversion(seed, kind1: type[pb.ABCPolyBase], kind2: type[pb.ABCPolyBase]):
    rng = np.random.default_rng(seed)
    coef = rng.standard_normal(10)
    p = kind1(coef)
    q = p.convert(kind=kind2)

    assert isinstance(p, kind1)
    assert isinstance(q, kind2)

    # roundtrip
    assert_array_almost_equal(
        q.convert(kind=kind1).coef, p.coef, err_msg="roundtrip conversion is broken"
    )

    # functional equivalence
    x = rng.uniform(low=p.domain[0], high=p.domain[1], size=100)
    assert_array_almost_equal(
        p(x), q(x), err_msg="conversion does not result in the same polynomial"
    )


@pytest.mark.parametrize("kind", DEFAULT_KINDS)
class TestDifferentiation:
    def test_differentiation(self, kind):
        p: pb.ABCPolyBase = kind(jnp.array([0.0, 1.0]))
        assert jax.grad(p)(3.0) == p.deriv()(3.)



def test_vmap():
    p = Polynomial(jnp.array([0.0, 1.0]))

    assert jnp.all(jax.vmap(p)(jnp.array([0.0, 1, 2])) == jnp.array([0.0, 1, 2]))
