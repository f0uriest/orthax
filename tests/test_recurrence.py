"""Tests for general orthogonal series functions."""

import numpy as np
import pytest
from jax import config
from numpy.testing import assert_allclose

import orthax

config.update("jax_enable_x64", True)


recs = [
    (orthax.recurrence.LegendreRecurrenceRelation, ()),
    (orthax.recurrence.ShiftedLegendreRecurrenceRelation, ()),
    (orthax.recurrence.ChebyshevTRecurrenceRelation, ()),
    (orthax.recurrence.ChebyshevURecurrenceRelation, ()),
    (orthax.recurrence.ChebyshevVRecurrenceRelation, ()),
    (orthax.recurrence.ChebyshevWRecurrenceRelation, ()),
    (orthax.recurrence.GegenbauerRecurrenceRelation, (0.23,)),
    (orthax.recurrence.JacobiRecurrenceRelation, (-0.27, 0.64)),
    (orthax.recurrence.LaguerreRecurrenceRelation, ()),
    (orthax.recurrence.GeneralizedLaguerreRecurrenceRelation, (2.5,)),
    (orthax.recurrence.HermiteRecurrenceRelation, ()),
    (orthax.recurrence.HermiteERecurrenceRelation, ()),
]


@pytest.mark.parametrize("scale", ["monic", "normalized"])
@pytest.mark.parametrize("rec", recs)
def test_generate_recurrence(rec, scale):
    rec, args = rec
    rec1 = rec(*args, scale=scale)

    n = 10
    rec2 = orthax.recurrence.generate_recurrence(
        rec1.weight, rec1.domain, n, scale=scale
    )

    nn = np.arange(n)

    assert_allclose(rec1.a[nn], rec2.a[nn], atol=1e-8, rtol=1e-8)
    assert_allclose(rec1.b[nn], rec2.b[nn], atol=1e-8, rtol=1e-8)
    assert_allclose(rec1.g[nn], rec2.g[nn], atol=1e-8, rtol=1e-8)
    assert_allclose(abs(rec1.m[nn]), abs(rec2.m[nn]), atol=1e-8, rtol=1e-8)
