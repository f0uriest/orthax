"""Tests for general orthogonal series functions."""

import numpy as np
import pytest
from jax import config
from numpy.testing import assert_allclose

import orthax

config.update("jax_enable_x64", True)

names = [
    "leg",
    "sleg",
    "chebT",
    "chebU",
    "chebV",
    "chebW",
    "geg",
    "jac",
    "lag",
    "glag",
    "herm",
    "herme",
]

recs = {
    "leg": (orthax.recurrence.Legendre, ()),
    "sleg": (orthax.recurrence.ShiftedLegendre, ()),
    "chebT": (orthax.recurrence.ChebyshevT, ()),
    "chebU": (orthax.recurrence.ChebyshevU, ()),
    "chebV": (orthax.recurrence.ChebyshevV, ()),
    "chebW": (orthax.recurrence.ChebyshevW, ()),
    "geg": (orthax.recurrence.Gegenbauer, (0.23,)),
    "jac": (orthax.recurrence.Jacobi, (-0.27, 0.64)),
    "lag": (orthax.recurrence.Laguerre, ()),
    "glag": (orthax.recurrence.GeneralizedLaguerre, (2.5,)),
    "herm": (orthax.recurrence.Hermite, ()),
    "herme": (orthax.recurrence.HermiteE, ()),
}


@pytest.mark.parametrize("scale", ["monic", "normalized"])
@pytest.mark.parametrize("name", names)
def test_generate_recurrence(name, scale):
    rec, args = recs[name]
    rec1 = rec(*args, scale=scale)

    n = 10
    rec2 = orthax.recurrence.generate_recurrence(
        rec1.weight, rec1.domain, n, scale=scale
    )

    nn = np.arange(n)

    assert_allclose(rec1.a(nn), rec2.a(nn), atol=1e-8, rtol=1e-8)
    assert_allclose(rec1.b(nn), rec2.b(nn), atol=1e-8, rtol=1e-8)
    assert_allclose(rec1.g(nn), rec2.g(nn), atol=1e-8, rtol=1e-8)
    assert_allclose(abs(rec1.m(nn)), abs(rec2.m(nn)), atol=1e-8, rtol=1e-8)
