"""Tests for general orthogonal series functions."""

import numpy as np
import pytest
from jax import config
from numpy.testing import assert_allclose

import orthax

config.update("jax_enable_x64", True)


names = ["cheb", "herm", "herme", "lag", "leg"]

name_map = {
    "cheb": (orthax.recurrence.ChebyshevT(), orthax.chebyshev),
    "herm": (orthax.recurrence.Hermite(), orthax.hermite),
    "herme": (orthax.recurrence.HermiteE(), orthax.hermite_e),
    "lag": (orthax.recurrence.Laguerre(), orthax.laguerre),
    "leg": (orthax.recurrence.Legendre(), orthax.legendre),
}


@pytest.mark.parametrize("name", names)
def test_orth2poly(name):
    rec, mod = name_map[name]
    c = np.random.random(10)

    out1 = orthax.orth2poly(c, rec)
    out2 = getattr(mod, name + "2poly")(c)
    assert_allclose(out1, out2)


@pytest.mark.parametrize("name", names)
def test_poly2orth(name):
    rec, mod = name_map[name]
    c = np.random.random(10)

    out1 = orthax.poly2orth(c, rec)
    out2 = getattr(mod, "poly2" + name)(c)
    assert_allclose(out1, out2)


@pytest.mark.parametrize("name", names)
def test_orthadd(name):
    rec, mod = name_map[name]
    c1 = np.random.random(10)
    c2 = np.random.random(5)

    out1 = orthax.orthadd(c1, c2, rec)
    out2 = getattr(mod, name + "add")(c1, c2)
    assert_allclose(out1, out2)


@pytest.mark.parametrize("name", names)
def test_orthsub(name):
    rec, mod = name_map[name]
    c1 = np.random.random(10)
    c2 = np.random.random(5)

    out1 = orthax.orthsub(c1, c2, rec)
    out2 = getattr(mod, name + "sub")(c1, c2)
    assert_allclose(out1, out2)


@pytest.mark.parametrize("name", names)
def test_orthmul(name):
    rec, mod = name_map[name]
    for n1 in [1, 2, 5, 10, 20]:
        for n2 in [1, 2, 5, 10, 20]:
            c1 = np.random.random(n1)
            c2 = np.random.random(n2)

            out1 = orthax.orthmul(c1, c2, rec)
            out2 = getattr(mod, name + "mul")(c1, c2)
            assert_allclose(out1, out2)


@pytest.mark.parametrize("name", names)
def test_orthdiv(name):
    rec, mod = name_map[name]

    for n1 in [1, 2, 5, 10, 20]:
        for n2 in [1, 2, 5, 10, 20]:
            c1 = np.random.random(n1)
            c2 = np.random.random(n2)

            q1, r1 = orthax.orthdiv(c1, c2, rec)
            q2, r2 = getattr(mod, name + "div")(c1, c2)
            assert_allclose(q1, q2)
            assert_allclose(r1, r2)


@pytest.mark.parametrize("name", names)
def test_orthmulx(name):
    rec, mod = name_map[name]
    c1 = np.random.random(10)

    out1 = orthax.orthmulx(c1, rec)
    out2 = getattr(mod, name + "mulx")(c1)
    assert_allclose(out1, out2)


@pytest.mark.parametrize("pow", np.arange(1, 4))
@pytest.mark.parametrize("name", names)
def test_orthpow(name, pow):
    rec, mod = name_map[name]
    c1 = np.random.random(5)

    out1 = orthax.orthpow(c1, pow, rec)
    out2 = getattr(mod, name + "pow")(c1, pow)
    # loose tols here due to pseudospectral multiplication
    assert_allclose(out1, out2, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("off", np.array([-1, 1]))
@pytest.mark.parametrize("scl", np.array([-1.5, 1.5]))
@pytest.mark.parametrize("name", names)
def test_orthline(name, off, scl):
    rec, mod = name_map[name]

    out1 = orthax.orthline(off, scl, rec)
    out2 = getattr(mod, name + "line")(off, scl)
    assert_allclose(out1, out2)


@pytest.mark.parametrize("name", names)
def test_orthval(name):
    rec, mod = name_map[name]

    c = np.random.random(10)
    x = np.random.random(100)
    out1 = orthax.orthval(x, c, rec)
    out2 = getattr(mod, name + "val")(x, c)
    assert_allclose(out1, out2)


@pytest.mark.parametrize("name", names)
def test_orthval2d(name):
    rec, mod = name_map[name]

    c = np.random.random((10, 10))
    x = np.random.random(100)
    y = np.random.random(100)
    out1 = orthax.orthval2d(x, y, c, rec)
    out2 = getattr(mod, name + "val2d")(x, y, c)
    assert_allclose(out1, out2)


@pytest.mark.parametrize("name", names)
def test_orthval3d(name):
    rec, mod = name_map[name]

    c = np.random.random((10, 10, 10))
    x = np.random.random(100)
    y = np.random.random(100)
    z = np.random.random(100)
    out1 = orthax.orthval3d(x, y, z, c, rec)
    out2 = getattr(mod, name + "val3d")(x, y, z, c)
    assert_allclose(out1, out2)


@pytest.mark.parametrize("name", names)
def test_orthgrid2d(name):
    rec, mod = name_map[name]

    c = np.random.random((10, 10))
    x = np.random.random(9)
    y = np.random.random(12)
    out1 = orthax.orthgrid2d(x, y, c, rec)
    out2 = getattr(mod, name + "grid2d")(x, y, c)
    assert_allclose(out1, out2)


@pytest.mark.parametrize("name", names)
def test_orthgrid3d(name):
    rec, mod = name_map[name]

    c = np.random.random((10, 10, 10))
    x = np.random.random(5)
    y = np.random.random(6)
    z = np.random.random(7)
    out1 = orthax.orthgrid3d(x, y, z, c, rec)
    out2 = getattr(mod, name + "grid3d")(x, y, z, c)
    assert_allclose(out1, out2)


@pytest.mark.parametrize("name", names)
def test_orthvander(name):
    rec, mod = name_map[name]

    x = np.random.random(100)
    deg = 10
    out1 = orthax.orthvander(x, deg, rec)
    out2 = getattr(mod, name + "vander")(x, deg)
    assert_allclose(out1, out2)


@pytest.mark.parametrize("name", names)
def test_orthvander2d(name):
    rec, mod = name_map[name]

    x = np.random.random(100)
    y = np.random.random(100)
    deg = (10, 10)
    out1 = orthax.orthvander2d(x, y, deg, rec)
    out2 = getattr(mod, name + "vander2d")(x, y, deg)
    assert_allclose(out1, out2)


@pytest.mark.parametrize("name", names)
def test_orthvander3d(name):
    rec, mod = name_map[name]

    x = np.random.random(100)
    y = np.random.random(100)
    z = np.random.random(100)
    deg = (5, 5, 5)
    out1 = orthax.orthvander3d(x, y, z, deg, rec)
    out2 = getattr(mod, name + "vander3d")(x, y, z, deg)
    assert_allclose(out1, out2)


@pytest.mark.parametrize("n", [4, 8])
@pytest.mark.parametrize("m", [0, 2])
@pytest.mark.parametrize("name", names)
def test_orthfit(name, n, m):
    rec, mod = name_map[name]

    x = np.random.random(100)
    c = 0.5 - np.random.random(n)
    y = orthax.orthval(x, c, rec)
    deg = n - m
    out1 = orthax.orthfit(x, y, deg, rec)
    out2 = getattr(mod, name + "fit")(x, y, deg)
    assert_allclose(out1, out2, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("name", names)
def test_orthfromroots(name):
    rec, mod = name_map[name]

    r = np.random.random(10)
    out1 = orthax.orthfromroots(r, rec)
    out2 = getattr(mod, name + "fromroots")(r)
    assert_allclose(out1, out2)


@pytest.mark.parametrize("name", names)
def test_orthweight(name):
    rec, mod = name_map[name]

    x = np.random.random(10)
    out1 = orthax.orthweight(x, rec)
    out2 = getattr(mod, name + "weight")(x)
    assert_allclose(out1, out2)


@pytest.mark.parametrize("name", names)
def test_orthnorm(name):
    rec, mod = name_map[name]

    n = np.arange(10)
    out1 = orthax.orthnorm(n, rec)
    out2 = getattr(mod, name + "norm")(n)
    assert_allclose(out1, out2)


@pytest.mark.parametrize("n", np.arange(1, 101, 20))
@pytest.mark.parametrize("name", names)
def test_orthgauss(name, n):
    rec, mod = name_map[name]

    x1, w1 = orthax.orthgauss(n, rec)
    x2, w2 = getattr(mod, name + "gauss")(n)
    assert_allclose(x1, x2, atol=1e-12, rtol=1e-12)
    assert_allclose(w1, w2, atol=1e-12, rtol=1e-12)


def test_orthgauss2():
    """Test for Gauss-Radau and Gauss-Lobatto quadrature."""
    rec = orthax.recurrence.Legendre()

    # gauss-legendre-lobatto, order 5
    xl = np.array([-1, -np.sqrt(3 / 7), 0, np.sqrt(3 / 7), 1])
    wl = np.array([1 / 10, 49 / 90, 32 / 45, 49 / 90, 1 / 10])
    x, w = orthax.orthgauss(5, rec, x0=-1, x1=1)
    assert_allclose(x, xl, atol=1e-14, rtol=1e-14)
    assert_allclose(w, wl, atol=1e-14, rtol=1e-14)

    # gauss-legendre-radau, order 3
    xr = np.array([-1, 1 / 5 * (1 - np.sqrt(6)), 1 / 5 * (1 + np.sqrt(6))])
    wr = np.array([2 / 9, 1 / 18 * (16 + np.sqrt(6)), 1 / 18 * (16 - np.sqrt(6))])
    x, w = orthax.orthgauss(3, rec, x0=-1)
    assert_allclose(x, xr, atol=1e-14, rtol=1e-14)
    assert_allclose(w, wr, atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("name", names)
@pytest.mark.parametrize("m", [0, 1, 2])
@pytest.mark.parametrize("scl", [1.0, 1.2])
def test_orthder(name, m, scl):

    rec, mod = name_map[name]

    c = np.random.random(10)
    p1 = orthax.orthder(c, rec, m, scl)
    p2 = getattr(mod, name + "der")(c, m, scl)
    assert_allclose(p1, p2, atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("name", names)
@pytest.mark.parametrize("m", [0, 1, 2])
@pytest.mark.parametrize("scl", [1.0, 1.2])
@pytest.mark.parametrize("lbnd", [0, 0.2])
@pytest.mark.parametrize("k", [0, 1.3])
def test_orthint(name, m, scl, lbnd, k):

    rec, mod = name_map[name]

    c = np.random.random(10)
    k = [k] * m
    p1 = orthax.orthint(c, rec, m, k, lbnd, scl)
    p2 = getattr(mod, name + "int")(c, m, k, lbnd, scl)
    assert_allclose(p1, p2, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("name", names)
@pytest.mark.parametrize("n", [1, 2, 5, 10])
def test_orthroots(name, n):
    rec, mod = name_map[name]

    c = 0.5 - np.random.random(n)
    r1 = orthax.orthroots(c, rec)
    r2 = getattr(mod, name + "roots")(c)
    assert_allclose(r1, r2, atol=1e-13, rtol=1e-13)
