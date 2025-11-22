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
    "leg": (orthax.recurrence.Legendre, (), 1e-12),
    "sleg": (orthax.recurrence.ShiftedLegendre, (), 1e-12),
    "chebT": (orthax.recurrence.ChebyshevT, (), 1e-8),
    "chebU": (orthax.recurrence.ChebyshevU, (), 1e-8),
    "chebV": (orthax.recurrence.ChebyshevV, (), 1e-8),
    "chebW": (orthax.recurrence.ChebyshevW, (), 1e-8),
    "jac_pp": (orthax.recurrence.Jacobi, (0.27, 0.27), 1e-8),
    "jac_pn": (orthax.recurrence.Jacobi, (0.27, -0.27), 1e-8),
    "jac_np": (orthax.recurrence.Jacobi, (-0.27, 0.27), 1e-8),
    "jac_nn": (orthax.recurrence.Jacobi, (-0.27, -0.27), 1e-8),
    "geg_pos": (orthax.recurrence.Gegenbauer, (0.1,), 1e-8),
    "geg_neg": (orthax.recurrence.Gegenbauer, (-0.1,), 1e-6),
    "lag": (orthax.recurrence.Laguerre, (), 1e-12),
    "glag_pos": (orthax.recurrence.GeneralizedLaguerre, (2.5,), 1e-8),
    "glag_neg": (orthax.recurrence.GeneralizedLaguerre, (-0.34,), 1e-8),
    "herm": (orthax.recurrence.Hermite, (), 1e-12),
    "herme": (orthax.recurrence.HermiteE, (), 1e-12),
}


@pytest.mark.parametrize("scale", ["monic", "normalized"])
@pytest.mark.parametrize("name", recs.keys())
def test_generate_recurrence(name, scale):
    rec, args, tol = recs[name]
    rec1 = rec(*args, scale=scale)

    n = 10
    rec2 = orthax.recurrence.generate_recurrence(
        lambda x: rec1.weight(x), rec1.domain, n, scale=scale
    )

    nn = np.arange(n)

    assert_allclose(rec1.a(nn), rec2.a(nn), atol=tol, rtol=tol)
    assert_allclose(rec1.b(nn), rec2.b(nn), atol=tol, rtol=tol)
    assert_allclose(rec1.g(nn), rec2.g(nn), atol=tol, rtol=tol)
    assert_allclose(abs(rec1.m(nn)), abs(rec2.m(nn)), atol=tol, rtol=tol)
