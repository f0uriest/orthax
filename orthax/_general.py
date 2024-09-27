"""
============================
Orthogonal Polynomial Series
============================

This module provides a number of functions useful for dealing with general orthogonal
polynomial series.

Arithmetic
----------

.. autosummary::
   :toctree: generated/

   orthadd
   orthsub
   orthmulx
   orthmul
   orthpow
   orthval
   orthval2d
   orthval3d
   orthgrid2d
   orthgrid3d

Calculus
--------

.. autosummary::
   :toctree: generated/

   orthder
   orthint

Misc Functions
--------------

.. autosummary::
   :toctree: generated/

   orthfromroots
   orthroots
   orthvander
   orthvander2d
   orthvander3d
   orthgauss
   orthweight
   orthnorm
   orthcompanion
   orthfit
   orthtrim
   orthline
   orth2poly
   poly2orth

"""

import functools

import jax
import jax.numpy as jnp
from jax import jit

from . import polyutils as pu

__all__ = [
    "orthadd",
    "orthsub",
    "orthmulx",
    "orthmul",
    "orthpow",
    "orthval",
    "orthval2d",
    "orthval3d",
    "orthgrid2d",
    "orthgrid3d",
    "orthder",
    "orthint",
    "orthfromroots",
    "orthroots",
    "orthvander",
    "orthvander2d",
    "orthvander3d",
    "orthgauss",
    "orthweight",
    "orthnorm",
    "orthcompanion",
    "orthfit",
    "orthtrim",
    "orthline",
    "orth2poly",
    "poly2orth",
]

orthtrim = pu.trimcoef


def tridiagmv(d, l, u, x):
    """Matvec for tridiagonal matrix."""
    a = u * x[1:]
    b = d * x
    c = l * x[:-1]
    return b.at[:-1].add(a).at[1:].add(c)


def last_nonzero(x):
    """Get index and value of last nonzero element of x"""
    i = len(x) - 1 - jnp.nonzero(x[::-1], size=1, fill_value=len(x))[0][0]
    return i, x[i]


@jit
def poly2orth(pol, rec):
    """Convert a polynomial to an orthogonal series.

    Convert an array representing the coefficients of a polynomial (relative
    to the "standard" basis) ordered from lowest degree to highest, to an
    array of the coefficients of the equivalent Chebyshev series, ordered
    from lowest to highest degree.

    Parameters
    ----------
    pol : array_like
        1-D array containing the polynomial coefficients
    rec : AbstractRecurrenceRelation
        Recurrence relation for the family of orthogonal polynomials.

    Returns
    -------
    c : ndarray
        1-D array containing the coefficients of the equivalent orthogonal
        series.

    See Also
    --------
    orth2poly

    """
    pol = pu.as_series(pol)
    deg = len(pol) - 1

    res = jnp.zeros_like(pol)

    def body(i, res):
        k = deg - i
        res = orthadd(orthmulx(res, rec, mode="same"), pol[k], rec)
        return res

    res = jax.lax.fori_loop(0, deg + 1, body, res)
    return res


@jit
def orth2poly(c, rec):
    """Convert an orthogonal series to a polynomial in standard basis.

    Convert an array representing the coefficients of an orthogonal series,
    ordered from lowest degree to highest, to an array of the coefficients
    of the equivalent polynomial (relative to the "standard" basis) ordered
    from lowest to highest degree.

    Parameters
    ----------
    c : array_like
        1-D array containing the orthogonal series coefficients, ordered
        from lowest order term to highest.
    rec : AbstractRecurrenceRelation
        Recurrence relation for the family of orthogonal polynomials.

    Returns
    -------
    pol : ndarray
        1-D array containing the coefficients of the equivalent polynomial
        (relative to the "standard" basis) ordered from lowest order term
        to highest.

    See Also
    --------
    poly2orth

    """
    A = jax.jacfwd(poly2orth)(c, rec)
    return jnp.linalg.solve(A, c)


@jit
def orthline(off, scl, rec):
    """Orthogonal series whose graph is a straight line.

    Parameters
    ----------
    off, scl : scalars
        The specified line is given by ``off + scl*x``.
    rec : AbstractRecurrenceRelation
        Recurrence relation for the family of orthogonal polynomials.

    Returns
    -------
    y : ndarray
        This module's representation of the orthogonal series for
        ``off + scl*x``.

    """
    return poly2orth(jnp.array([off, scl]), rec)


@jit
def orthfromroots(roots, rec):
    """Generate an orthogonal series with given roots.

    The function returns the coefficients of the polynomial

    .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n),

    in the form of an orthogonal series with recurrence relation ``rec``, where the
    `r_n` are the roots specified in `roots`.
    If a zero has multiplicity n, then it must appear in `roots` n times.
    For instance, if 2 is a root of multiplicity three and 3 is a root of
    multiplicity 2, then `roots` looks something like [2, 2, 2, 3, 3]. The
    roots can appear in any order.

    If the returned coefficients are `c`, then

    .. math:: p(x) = c_0 + c_1 * P_1(x) + ... +  c_n * P_n(x)

    Parameters
    ----------
    roots : array_like
        Sequence containing the roots.
    rec : AbstractRecurrenceRelation
        Recurrence relation for the family of orthogonal polynomials.

    Returns
    -------
    out : ndarray
        1-D array of coefficients.  If all roots are real then `out` is a
        real array, if some of the roots are complex, then `out` is complex
        even if all the coefficients in the result are real (see Examples
        below).

    """
    return pu._fromroots(
        functools.partial(orthline, rec=rec),
        functools.partial(orthmul, rec=rec),
        roots,
    )


@jit
def orthadd(c1, c2, rec):
    """Add one orthogonal series to another.

    Returns the sum of two orthogonal series `c1` + `c2`.  The arguments
    are sequences of coefficients ordered from lowest order term to
    highest, i.e., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of orthogonal series coefficients ordered from low to
        high.
    rec : AbstractRecurrenceRelation
        Recurrence relation for the family of orthogonal polynomials.

    Returns
    -------
    out : ndarray
        Array representing the orthogonal series of their sum.

    See Also
    --------
    orthsub, orthmulx, orthmul, orthdiv, orthpow

    """
    return pu._add(c1, c2)


@jit
def orthsub(c1, c2, rec):
    """Subtract one orthogonal series from another.

    Returns the difference of two orthogonal series `c1` - `c2`.  The
    sequences of coefficients are from lowest order term to highest, i.e.,
    [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of orthogonal series coefficients ordered from low to
        high.
    rec : AbstractRecurrenceRelation
        Recurrence relation for the family of orthogonal polynomials.

    Returns
    -------
    out : ndarray
        Of orthogonal series coefficients representing their difference.

    See Also
    --------
    orthadd, orthmulx, orthmul, orthdiv, orthpow

    """
    return pu._sub(c1, c2)


@functools.partial(jit, static_argnames="mode")
def orthmulx(c, rec, mode="full"):
    """Multiply an orthogonal series by x.

    Multiply the polynomial `c` by x, where x is the independent
    variable.

    Parameters
    ----------
    c : array_like
        1-D array of series coefficients ordered from low to
        high.
    rec : AbstractRecurrenceRelation
        Recurrence relation for the family of orthogonal polynomials.
    mode : {"full", "same"}
        If "full", output has shape (len(c) + 1). If "same", output has shape
        (len(c)), possibly truncating high order modes.

    Returns
    -------
    out : ndarray
        Array representing the result of the multiplication.
    """
    n = jnp.arange(len(c) + 1)
    a = rec.a(n)
    b = rec.b(n)
    m = rec.m(n)

    prd = jnp.pad(c, (0, 1))
    # f = c*(m*p) where p is monic polynomial
    prd *= m  # convert c to monic form

    diagonal = a
    lower_diagonal = b[1:]
    upper_diagonal = jnp.ones_like(b[1:])

    # multiplying by transpose of monic jacobi matrix, so lower and upper are swapped
    prd = tridiagmv(diagonal, upper_diagonal, lower_diagonal, prd)

    # convert back to user scale
    prd /= m

    if mode == "same":
        prd = prd[: len(c)]

    return prd


@functools.partial(jit, static_argnames="mode")
def orthmul(c1, c2, rec, mode="full"):
    """Multiply one orthogonal series by another.

    Returns the product of two series `c1` * `c2`.  The arguments
    are sequences of coefficients, from lowest order "term" to highest,
    e.g., [1,2,3] represents the series ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of orthogonal series coefficients ordered from low to
        high.
    rec : AbstractRecurrenceRelation
        Recurrence relation for the family of orthogonal polynomials.
    mode : {"full", "same"}
        If "full", output has shape (len(c1) + len(c2) - 1). If "same", output has shape
        max(len(c1), len(c2)), possibly truncating high order modes.

    Returns
    -------
    out : ndarray
        Of series coefficients representing their product.

    See Also
    --------
    orthadd, orthsub, orthmulx, orthdiv, orthpow

    """
    # assume c1 is longer, we iterate over c2 so want that shortest
    if len(c2) > len(c1):
        c1, c2 = c2, c1

    if len(c2) == 1:
        return c1 * c2[0]

    # convert coefficients form scaled to monic form
    # ie f * p_scaled(x) = f*m * p_monic(x)
    f = c1 * rec.m(jnp.arange(c1.size))
    g = c2 * rec.m(jnp.arange(c2.size))

    # ensure leading coefficient is 1
    fi, fn = last_nonzero(f)
    gi, gm = last_nonzero(g)
    f = f / fn
    g = g / gm

    # order of polynomials
    n = len(f) - 1
    m = len(g) - 1
    N = n + m

    # elements of jacobi matrix
    nn = jnp.arange(N)
    a = rec.a(nn)  # diagonal
    b = rec.b(nn)[1:]  # lower diagonal
    c = jnp.ones_like(b)  # upper diagonal

    r1 = jnp.pad(f, (0, m - 1))
    r2 = jnp.zeros_like(r1)
    u = r1 * g[0]

    def bodyfun(i, state):
        r1, r2, u = state
        ri = tridiagmv(a - rec.a(i - 2), c, b, r1) - rec.b(i - 2) * r2
        u += g[i - 1] * ri
        r2 = r1
        r1 = ri
        return r1, r2, u

    r1, r2, u = jax.lax.fori_loop(2, m + 1, bodyfun, (r1, r2, u))

    ri = tridiagmv(a - rec.a(gi - 1), c, b, r1) - rec.b(gi - 1) * r2
    u += g[-1] * ri
    u = jnp.append(u, 0.0)
    u = u.at[fi + gi].set(1.0)

    # scale by original leading coefficients
    u *= fn * gm
    # undo rescaling to monic form
    u /= rec.m(jnp.arange(N + 1))

    if mode == "same":
        u = u[: max(len(c1), len(c2))]

    return u


@jit
def orthdiv(c1, c2, rec):
    """Divide one orthogonal series by another.

    Returns the quotient-with-remainder of two orthogonal series
    `c1` / `c2`.  The arguments are sequences of coefficients from lowest
    order "term" to highest, e.g., [1,2,3] represents the series
    ``P_0 + 2*P_1 + 3*P_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of orthogonal series coefficients ordered from low to
        high.
    rec : AbstractRecurrenceRelation
        Recurrence relation for the family of orthogonal polynomials.

    Returns
    -------
    [quo, rem] : ndarrays
        Of orthogonal series coefficients representing the quotient and
        remainder.

    See Also
    --------
    orthadd, orthsub, orthmulx, orthdiv, orthpow

    Notes
    -----
    In general, the (polynomial) division of one orthogonal series by another
    results in quotient and remainder terms that are not in the original
    polynomial basis set.  Thus, to express these results as an orthogonal
    series, it is necessary to "reproject" the results onto the original
    basis set, which may produce "unintuitive" (but correct) results.

    """
    return pu._div(functools.partial(orthmul, rec=rec), c1, c2)


@functools.partial(jit, static_argnames=("pow", "maxpower"))
def orthpow(c, pow, rec, maxpower=16):
    """Raise an orthogonal series to a power.

    Returns the orthogonal series `c` raised to the power `pow`. The
    argument `c` is a sequence of coefficients ordered from low to high.
    i.e., [1,2,3] is the series  ``P_0 + 2*P_1 + 3*P_2.``

    Parameters
    ----------
    c : array_like
        1-D array of orthogonal series coefficients ordered from low to
        high.
    pow : integer
        Power to which the series will be raised
    rec : AbstractRecurrenceRelation
        Recurrence relation for the family of orthogonal polynomials.
    maxpower : integer, optional
        Maximum power allowed. This is mainly to limit growth of the series
        to unmanageable size. Default is 16

    Returns
    -------
    coef : ndarray
        Orthogonal series of power.

    See Also
    --------
    orthadd, orthsub, orthmulx, orthmul, orthdiv

    """
    return pu._pow(functools.partial(orthmul, rec=rec), c, pow, maxpower)


@functools.partial(jit, static_argnames=("tensor",))
def orthval(x, c, rec, tensor=True):
    """Evaluate an orthogonal series.

    Parameters
    ----------
    x : jax.Array
        Evaluation points.
    c : jax.Array
        Coefficients of series, in ascending order
    rec : AbstractRecurrenceRelation
        Recurrence relation for the family of orthogonal polynomials
    tensor : boolean, optional
        If True, the shape of the coefficient array is extended with ones
        on the right, one for each dimension of `x`. Scalars have dimension 0
        for this action. The result is that every column of coefficients in
        `c` is evaluated for every element of `x`. If False, `x` is broadcast
        over the columns of `c` for the evaluation.  This keyword is useful
        when `c` is multidimensional. The default value is True.

    Returns
    -------
    y : jax.Array
        Series evaluated at x
    """
    c = jnp.asarray(c)
    x = jnp.asarray(x)
    if tensor:
        c = c.reshape(c.shape + (1,) * x.ndim)

    c1 = c[-2] * jnp.zeros_like(x)
    c2 = c[-1] * jnp.zeros_like(x)

    def body(i, val):
        k = len(c) - i
        c1, c2 = val
        ck_plus1 = c1
        # on first step c2 is 0 so we don't care about rec.b(k+1), but we don't
        # want to try to get it to avoid out of bounds errors on tabulated coeffs.
        bk1 = rec.b(jnp.minimum(k + 1, len(c) - 1))
        ck = c[k] * rec.m(k) + (x - rec.a(k)) * c1 - bk1 * c2
        return ck, ck_plus1

    c1, c2 = jax.lax.fori_loop(1, len(c), body, (c1, c2))

    p0 = 1
    p1 = (x - rec.a(0)) * p0
    return p0 * c[0] * rec.m(0) + p1 * c1 - rec.b(1) * p0 * c2


@jit
def orthval2d(x, y, c, rec):
    r"""Evaluate a 2-D orthogonal series at points (x, y).

    This function returns the values:

    .. math:: p(x,y) = \sum_{i,j} c_{i,j} * P_i(x) * P_j(y)

    The parameters `x` and `y` are converted to arrays only if they are
    tuples or a lists, otherwise they are treated as a scalars and they
    must have the same shape after conversion. In either case, either `x`
    and `y` or their elements must support multiplication and addition both
    with themselves and with the elements of `c`.

    If `c` is a 1-D array a one is implicitly appended to its shape to make
    it 2-D. The shape of the result will be c.shape[2:] + x.shape.

    Parameters
    ----------
    x, y : array_like, compatible objects
        The two dimensional series is evaluated at the points `(x, y)`,
        where `x` and `y` must have the same shape. If `x` or `y` is a list
        or tuple, it is first converted to an ndarray, otherwise it is left
        unchanged and if it isn't an ndarray it is treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficient of the term
        of multi-degree i,j is contained in ``c[i,j]``. If `c` has
        dimension greater than two the remaining indices enumerate multiple
        sets of coefficients.
    rec : AbstractRecurrenceRelation
        Recurrence relation for the family of orthogonal polynomials

    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional polynomial at points formed with
        pairs of corresponding values from `x` and `y`.

    See Also
    --------
    orthval, orthgrid2d, orthval3d, orthgrid3d

    """
    return pu._valnd(functools.partial(orthval, rec=rec), c, x, y)


@jit
def orthgrid2d(x, y, c, rec):
    r"""Evaluate a 2-D orthogonal series on the Cartesian product of x and y.

    This function returns the values:

    .. math:: p(a,b) = \sum_{i,j} c_{i,j} * P_i(a) * P_j(b)

    where the points `(a, b)` consist of all pairs formed by taking
    `a` from `x` and `b` from `y`. The resulting points form a grid with
    `x` in the first dimension and `y` in the second.

    The parameters `x` and `y` are converted to arrays only if they are
    tuples or a lists, otherwise they are treated as a scalars. In either
    case, either `x` and `y` or their elements must support multiplication
    and addition both with themselves and with the elements of `c`.

    If `c` has fewer than two dimensions, ones are implicitly appended to
    its shape to make it 2-D. The shape of the result will be c.shape[2:] +
    x.shape + y.shape.

    Parameters
    ----------
    x, y : array_like, compatible objects
        The two dimensional series is evaluated at the points in the
        Cartesian product of `x` and `y`.  If `x` or `y` is a list or
        tuple, it is first converted to an ndarray, otherwise it is left
        unchanged and, if it isn't an ndarray, it is treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficient of the term of
        multi-degree i,j is contained in `c[i,j]`. If `c` has dimension
        greater than two the remaining indices enumerate multiple sets of
        coefficients.
    rec : AbstractRecurrenceRelation
        Recurrence relation for the family of orthogonal polynomials

    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional Chebyshev series at points in the
        Cartesian product of `x` and `y`.

    See Also
    --------
    orthval, orthval2d, orthval3d, orthgrid3d

    """
    return pu._gridnd(functools.partial(orthval, rec=rec), c, x, y)


@jit
def orthval3d(x, y, z, c, rec):
    r"""Evaluate a 3-D orthogonal series at points (x, y, z).

    This function returns the values:

    .. math:: p(x,y,z) = \sum_{i,j,k} c_{i,j,k} * P_i(x) * P_j(y) * P_k(z)

    The parameters `x`, `y`, and `z` are converted to arrays only if
    they are tuples or a lists, otherwise they are treated as a scalars and
    they must have the same shape after conversion. In either case, either
    `x`, `y`, and `z` or their elements must support multiplication and
    addition both with themselves and with the elements of `c`.

    If `c` has fewer than 3 dimensions, ones are implicitly appended to its
    shape to make it 3-D. The shape of the result will be c.shape[3:] +
    x.shape.

    Parameters
    ----------
    x, y, z : array_like, compatible object
        The three dimensional series is evaluated at the points
        `(x, y, z)`, where `x`, `y`, and `z` must have the same shape.  If
        any of `x`, `y`, or `z` is a list or tuple, it is first converted
        to an ndarray, otherwise it is left unchanged and if it isn't an
        ndarray it is  treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficient of the term of
        multi-degree i,j,k is contained in ``c[i,j,k]``. If `c` has dimension
        greater than 3 the remaining indices enumerate multiple sets of
        coefficients.
    rec : AbstractRecurrenceRelation
        Recurrence relation for the family of orthogonal polynomials

    Returns
    -------
    values : ndarray, compatible object
        The values of the multidimensional polynomial on points formed with
        triples of corresponding values from `x`, `y`, and `z`.

    See Also
    --------
    orthval, orthval2d, orthgrid2d, orthgrid3d

    """
    return pu._valnd(functools.partial(orthval, rec=rec), c, x, y, z)


@jit
def orthgrid3d(x, y, z, c, rec):
    r"""Evaluate a 3-D orthogonal series on the Cartesian product of x, y, and z.

    This function returns the values:

    .. math:: p(a,b,c) = \sum_{i,j,k} c_{i,j,k} * P_i(a) * P_j(b) * P_k(c)

    where the points `(a, b, c)` consist of all triples formed by taking
    `a` from `x`, `b` from `y`, and `c` from `z`. The resulting points form
    a grid with `x` in the first dimension, `y` in the second, and `z` in
    the third.

    The parameters `x`, `y`, and `z` are converted to arrays only if they
    are tuples or a lists, otherwise they are treated as a scalars. In
    either case, either `x`, `y`, and `z` or their elements must support
    multiplication and addition both with themselves and with the elements
    of `c`.

    If `c` has fewer than three dimensions, ones are implicitly appended to
    its shape to make it 3-D. The shape of the result will be c.shape[3:] +
    x.shape + y.shape + z.shape.

    Parameters
    ----------
    x, y, z : array_like, compatible objects
        The three dimensional series is evaluated at the points in the
        Cartesian product of `x`, `y`, and `z`.  If `x`,`y`, or `z` is a
        list or tuple, it is first converted to an ndarray, otherwise it is
        left unchanged and, if it isn't an ndarray, it is treated as a
        scalar.
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree i,j are contained in ``c[i,j]``. If `c` has dimension
        greater than two the remaining indices enumerate multiple sets of
        coefficients.
    rec : AbstractRecurrenceRelation
        Recurrence relation for the family of orthogonal polynomials

    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional polynomial at points in the Cartesian
        product of `x` and `y`.

    See Also
    --------
    orthval, orthval2d, orthgrid2d, orthval3d

    """
    return pu._gridnd(functools.partial(orthval, rec=rec), c, x, y, z)


@functools.partial(jit, static_argnames=("deg",))
def orthvander(x, deg, rec):
    """Pseudo-Vandermonde matrix of given degree.

    Returns the pseudo-Vandermonde matrix of degree `deg` and sample points
    `x`. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., i] = p_i(x),

    where `0 <= i <= deg`. The leading indices of `V` index the elements of
    `x` and the last index is the degree of the Chebyshev polynomial.

    If `c` is a 1-D array of coefficients of length `n + 1` and `V` is the
    matrix ``V = orthvander(x, n, rec)``, then ``np.dot(V, c)`` and
    ``orthval(x, c, rec)`` are the same up to roundoff.  This equivalence is
    useful both for least squares fitting and for the evaluation of a large
    number of Chebyshev series of the same degree and sample points.

    Parameters
    ----------
    x : jax.Array
        Array of points. The dtype is converted to float64 or complex128
        depending on whether any of the elements are complex. If `x` is
        scalar it is converted to a 1-D array.
    deg : int
        Degree of the resulting matrix.
    rec : AbstractRecurrenceRelation
        Recurrence relation for the family of orthogonal polynomials.

    Returns
    -------
    vander : jax.Array
        The pseudo Vandermonde matrix. The shape of the returned matrix is
        ``x.shape + (deg + 1,)``, where The last index is the degree of the
        corresponding polynomial.  The dtype will be the same as
        the converted `x`.

    """
    if deg < 0:
        raise ValueError("deg must be non-negative")

    x = jnp.array(x, ndmin=1)
    dims = (deg + 1,) + x.shape
    dtyp = jnp.promote_types(x.dtype, jnp.array(0.0).dtype)
    x = x.astype(dtyp)
    v = jnp.empty(dims, dtype=dtyp)
    # easiest to evaluate in monic form then scale at the end
    v = v.at[0].set(jnp.ones_like(x))
    if deg > 0:
        v = v.at[1].set(x - rec.a(0))

        def body(i, v):
            p0 = v[i - 2]
            p1 = v[i - 1]
            pn = (x - rec.a(i - 1)) * p1 - rec.b(i - 1) * p0
            v = v.at[i].set(pn)
            return v

        v = jax.lax.fori_loop(2, deg + 1, body, v)

    return jnp.moveaxis(v, 0, -1) * rec.m(jnp.arange(deg + 1))


@functools.partial(jit, static_argnames=("deg",))
def orthvander2d(x, y, deg, rec):
    r"""Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points `(x, y)`. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (deg[1] + 1)*i + j] = P_i(x) * P_j(y),

    where `0 <= i <= deg[0]` and `0 <= j <= deg[1]`. The leading indices of
    `V` index the points `(x, y)` and the last index encodes the degrees of
    the orthogonal polynomials.

    If ``V = orthvander2d(x, y, [xdeg, ydeg])``, then the columns of `V`
    correspond to the elements of a 2-D coefficient array `c` of shape
    (xdeg + 1, ydeg + 1) in the order

    .. math:: c_{00}, c_{01}, c_{02} ... , c_{10}, c_{11}, c_{12} ...

    and ``np.dot(V, c.flat)`` and ``orthval2d(x, y, c)`` will be the same
    up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 2-D orthogonal
    series of the same degrees and sample points.

    Parameters
    ----------
    x, y : array_like
        Arrays of point coordinates, all of the same shape. The dtypes
        will be converted to either float64 or complex128 depending on
        whether any of the elements are complex. Scalars are converted to
        1-D arrays.
    deg : tuple of ints
        Tuple of maximum degrees of the form (x_deg, y_deg).
    rec : AbstractRecurrenceRelation
        Recurrence relation for the family of orthogonal polynomials.

    Returns
    -------
    vander2d : ndarray
        The shape of the returned matrix is ``x.shape + (order,)``, where
        :math:`order = (deg[0]+1)*(deg[1]+1)`.  The dtype will be the same
        as the converted `x` and `y`.

    See Also
    --------
    orthvander, orthvander3d, orthval2d, orthval3d

    """
    vander = functools.partial(orthvander, rec=rec)
    return pu._vander_nd_flat((vander, vander), (x, y), deg)


@functools.partial(jit, static_argnames=("deg",))
def orthvander3d(x, y, z, deg, rec):
    r"""Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points `(x, y, z)`. If `l, m, n` are the given degrees in `x, y, z`,
    then The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (m+1)(n+1)i + (n+1)j + k] = P_i(x)*P_j(y)*P_k(z),

    where `0 <= i <= l`, `0 <= j <= m`, and `0 <= j <= n`.  The leading
    indices of `V` index the points `(x, y, z)` and the last index encodes
    the degrees of the orthogonal polynomials.

    If ``V = orthvander3d(x, y, z, [xdeg, ydeg, zdeg])``, then the columns
    of `V` correspond to the elements of a 3-D coefficient array `c` of
    shape (xdeg + 1, ydeg + 1, zdeg + 1) in the order

    .. math:: c_{000}, c_{001}, c_{002},... , c_{010}, c_{011}, c_{012},...

    and  ``np.dot(V, c.flat)`` and ``lagval3d(x, y, z, c)`` will be the
    same up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 3-D orthogonal
    series of the same degrees and sample points.

    Parameters
    ----------
    x, y, z : array_like
        Arrays of point coordinates, all of the same shape. The dtypes will
        be converted to either float64 or complex128 depending on whether
        any of the elements are complex. Scalars are converted to 1-D
        arrays.
    deg : tuple of ints
        Tuple of maximum degrees of the form (x_deg, y_deg, z_deg).
    rec : AbstractRecurrenceRelation
        Recurrence relation for the family of orthogonal polynomials.

    Returns
    -------
    vander3d : ndarray
        The shape of the returned matrix is ``x.shape + (order,)``, where
        :math:`order = (deg[0]+1)*(deg[1]+1)*(deg[2]+1)`.  The dtype will
        be the same as the converted `x`, `y`, and `z`.

    See Also
    --------
    orthvander, orthvander3d, orthval2d, orthval3d

    """
    vander = functools.partial(orthvander, rec=rec)
    return pu._vander_nd_flat((vander, vander, vander), (x, y, z), deg)


@functools.partial(jit, static_argnames=("deg", "full"))
def orthfit(x, y, deg, rec, rcond=None, full=False, w=None):
    r"""Least squares fit of orthogonal series to data.

    Return the coefficients of an orthogonal series of degree `deg` that is the
    least squares fit to the data values `y` given at points `x`. If `y` is
    1-D the returned coefficients will also be 1-D. If `y` is 2-D multiple
    fits are done, one for each column of `y`, and the resulting
    coefficients are stored in the corresponding columns of a 2-D return.
    The fitted polynomial(s) are in the form

    .. math::  p(x) = c_0 + c_1 * P_1(x) + ... + c_n * P_n(x),

    where ``n`` is `deg`.

    Parameters
    ----------
    x : array_like, shape (M,)
        x-coordinates of the M sample points ``(x[i], y[i])``.
    y : array_like, shape (M,) or (M, K)
        y-coordinates of the sample points. Several data sets of sample
        points sharing the same x-coordinates can be fitted at once by
        passing in a 2D-array that contains one dataset per column.
    deg : int or 1-D array_like
        Degree(s) of the fitting polynomials. If `deg` is a single integer
        all terms up to and including the `deg`'th term are included in the
        fit. A tuple of integers specifying the degrees of the terms to include
        may be used instead.
    rec : AbstractRecurrenceRelation
        Recurrence relation for the family of orthogonal polynomials.
    rcond : float, optional
        Relative condition number of the fit. Singular values smaller than
        this relative to the largest singular value will be ignored. The
        default value is len(x)*eps, where eps is the relative precision of
        the float type, about 2e-16 in most cases.
    full : bool, optional
        Switch determining nature of return value. When it is False (the
        default) just the coefficients are returned, when True diagnostic
        information from the singular value decomposition is also returned.
    w : array_like, shape (`M`,), optional
        Weights. If not None, the weight ``w[i]`` applies to the unsquared
        residual ``y[i] - y_hat[i]`` at ``x[i]``. Ideally the weights are
        chosen so that the errors of the products ``w[i]*y[i]`` all have the
        same variance.  When using inverse-variance weighting, use
        ``w[i] = 1/sigma(y[i])``.  The default value is None.

    Returns
    -------
    coef : ndarray, shape (M,) or (M, K)
        Orthogonal coefficients ordered from low to high. If `y` was 2-D,
        the coefficients for the data in column *k*  of `y` are in column
        *k*.

    [residuals, rank, singular_values, rcond] : list
        These values are only returned if ``full == True``

        - residuals -- sum of squared residuals of the least squares fit
        - rank -- the numerical rank of the scaled Vandermonde matrix
        - singular_values -- singular values of the scaled Vandermonde matrix
        - rcond -- value of `rcond`.

        For more details, see `jax.numpy.linalg.lstsq`.

    See Also
    --------
    orthax.polynomial.polyfit
    orthax.laguerre.lagfit
    orthax.legendre.legfit
    orthax.chebyshev.chebfit
    orthax.hermite.hermfit
    orthax.hermite_e.hermefit
    orthval : Evaluates an orthogonal series.
    orthvander : pseudo Vandermonde matrix of orthogonal series.
    jax.numpy.linalg.lstsq : Computes a least-squares fit from the matrix.

    Notes
    -----
    The solution is the coefficients of the orthogonal series ``p`` that
    minimizes the sum of the weighted squared errors

    .. math:: E = \sum_j w_j^2 * |y_j - p(x_j)|^2,

    where the :math:`w_j` are the weights. This problem is solved by
    setting up as the (typically) overdetermined matrix equation

    .. math:: V(x) * c = w * y,

    where ``V`` is the weighted pseudo Vandermonde matrix of `x`, ``c`` are the
    coefficients to be solved for, `w` are the weights, and `y` are the
    observed values.  This equation is then solved using the singular value
    decomposition of ``V``.

    If some of the singular values of `V` are so small that they are
    neglected, then a `RankWarning` will be issued. This means that the
    coefficient values may be poorly determined. Using a lower order fit
    will usually get rid of the warning.  The `rcond` parameter can also be
    set to a value smaller than its default, but the resulting fit may be
    spurious and have large contributions from roundoff error.

    Fits using orthogonal series are probably most useful when the data can
    be approximated by ``sqrt(w(x)) * p(x)``, where ``w(x)`` is the series
    weight function. In that case the weight ``sqrt(w(x[i]))`` should be used
    together with data values ``y[i]/sqrt(w(x[i]))``. The weight function is
    available as `rec.weight`.

    """
    vander = functools.partial(orthvander, rec=rec)
    return pu._fit(vander, x, y, deg, rcond, full, w)


@jit
def orthweight(x, rec):
    r"""Weight function of orthogonal polynomials.

    Parameters
    ----------
    x : array_like
       Values at which the weight function will be computed.

    Returns
    -------
    w : ndarray
       The weight function at `x`.

    """
    return rec.weight(x)


@jit
def orthnorm(n, rec):
    r"""Norm of nth orthogonal polynomial.

    The norm :math:`\gamma_n` is defined such that

    :math:`\int_{a}^{b} P_n^2(x) w(x) dx = \gamma_n^2`

    Parameters
    ----------
    n : int
       Order of orthogonal polynomial.

    Returns
    -------
    gamma_n : float
       Norm of the nth orthogonal polynomial.

    """
    return rec.g(n) * jnp.abs(rec.m(n))


@functools.partial(jit, static_argnames=("monic",))
def polyval(x, n, rec, monic=False):
    """Evaluate the nth order polynomial from the family given by rec

    Parameters
    ----------
    x : jax.Array
        Evaluation points.
    n : int
        Order of polynomial
    rec : AbstractRecurrenceRelation
        Recurrence relation for the family of orthogonal polynomials.

    Returns
    -------
    y : jax.Array
        Polynomial evaluated at x
    """
    x = jnp.asarray(x)

    p0 = jnp.zeros_like(x)
    p1 = jnp.ones_like(x)
    pn = p1

    def body(i, state):
        p0, p1, pn = state
        pn = (x - rec.a(i)) * p1 - rec.b(i) * p0
        p0 = p1
        p1 = pn
        return p0, p1, pn

    npos = lambda: jax.lax.fori_loop(0, n, body, (p0, p1, pn))[-1]
    nneg = lambda: jnp.zeros_like(x)
    out = jax.lax.cond(n >= 0, npos, nneg)
    if not monic:
        out *= rec.m(n)
    return out


def differentiation_matrix(rec, n):
    """Differentiation matrix for series up to degree n

    Parameters
    ----------
    rec : AbstractRecurrenceRelation
        Recurrence relation for the family of orthogonal polynomials
    n : int
        Maximum order of polynomial

    Returns
    -------
    D : jax.Array, shape(n+1,n+1)
        Differentiation matrix, such that f' = D@f
    """
    a = rec.a
    b = rec.b
    m = rec.m
    n += 1
    nn = jnp.arange(n)
    Q = jnp.diag(nn.astype(float))

    def iloop(i, Q):
        def jloop(j, Q):
            im1 = jnp.maximum(i - 1, 0)
            jm1 = jnp.maximum(j - 1, 0)
            Q = Q.at[i, j].add(((j > 0) * a(jm1) - (i > 0) * a(im1)) * Q[i - 1, j])
            Q = Q.at[i, j].add((j > 1) * Q[i - 1, j - 1])
            Q = Q.at[i, j].add(b(j) * Q[i - 1, j + 1])
            Q = Q.at[i, j].add(-1 * (i > 1) * b(im1) * Q[i - 2, j])
            return Q

        return jax.lax.fori_loop(0, i, jloop, Q)

    Q = jax.lax.fori_loop(0, n, iloop, Q)
    D = jnp.pad(Q[1:, 1:], ((1, 0), (0, 1))).T
    D = 1 / m(nn)[:, None] * D * m(nn)
    return D


def integration_matrix(rec, n):
    """Integration matrix for series up to degree n

    Parameters
    ----------
    rec : AbstractRecurrenceRelation
        Recurrence relation for the family of orthogonal polynomials
    n : int
        Maximum order of polynomial

    Returns
    -------
    S : jax.Array, shape(n,n)
        Integration matrix, such that f = S@f'
    """
    D = differentiation_matrix(rec, n)
    return jnp.linalg.pinv(D)


def _pad_along_axis(array: jax.Array, pad: tuple = (0, 0), axis: int = 0):
    """Pad with zeros or truncate a given dimension."""
    array = jnp.moveaxis(array, axis, 0)

    if pad[0] < 0:
        array = array[abs(pad[0]) :]
        pad = (0, pad[1])
    if pad[1] < 0:
        array = array[: -abs(pad[1])]
        pad = (pad[0], 0)

    npad = [(0, 0)] * array.ndim
    npad[0] = pad

    array = jnp.pad(array, pad_width=npad, mode="constant", constant_values=0)
    return jnp.moveaxis(array, 0, axis)


@functools.partial(jit, static_argnames=("m", "axis"))
def orthder(c, rec, m=1, scl=1, axis=0):
    """Differentiate an orthogonal series.

    Returns the orthogonal series coefficients `c` differentiated `m` times
    along `axis`.  At each iteration the result is multiplied by `scl` (the
    scaling factor is for use in a linear change of variable). The argument
    `c` is an array of coefficients from low to high degree along each
    axis, e.g., [1,2,3] represents the series ``1*P_0 + 2*P_1 + 3*P_2``
    while [[1,2],[1,2]] represents ``1*P_0(x)*P_0(y) + 1*P_1(x)*P_0(y) +
    2*P_0(x)*P_1(y) + 2*P_1(x)*P_1(y)`` if axis=0 is ``x`` and axis=1 is
    ``y``.

    Parameters
    ----------
    c : array_like
        Array of orthogonal series coefficients. If c is multidimensional the
        different axis correspond to different variables with the degree in
        each axis given by the corresponding index.
    rec : AbstractRecurrenceRelation
        Recurrence relation for the family of orthogonal polynomials
    m : int, optional
        Number of derivatives taken, must be non-negative. (Default: 1)
    scl : scalar, optional
        Each differentiation is multiplied by `scl`.  The end result is
        multiplication by ``scl**m``.  This is for use in a linear change of
        variable. (Default: 1)
    axis : int, optional
        Axis over which the derivative is taken. (Default: 0).

    Returns
    -------
    der : ndarray
        Orthogonal series of the derivative.

    See Also
    --------
    orthint

    """
    if m < 0:
        raise ValueError("The order of derivation must be non-negative")

    c = pu.as_series(c)

    if m == 0:
        return c

    c = jnp.moveaxis(c, axis, 0)
    n = len(c)
    if m >= n:
        c = jnp.zeros_like(c[:1])
    else:
        D = differentiation_matrix(rec, len(c) - 1)
        # TODO: figure out how to get rid of this python loop
        for i in range(m):
            c = D @ c * scl

    c = c[:-m]
    c = jnp.moveaxis(c, 0, axis)
    return c


@functools.partial(jit, static_argnames=("m", "axis"))
def orthint(c, rec, m=1, k=[], lbnd=0, scl=1, axis=0):
    """Integrate an orthogonal series.

    Returns the orthogonal series coefficients `c` integrated `m` times from
    `lbnd` along `axis`. At each iteration the resulting series is
    **multiplied** by `scl` and an integration constant, `k`, is added.
    The scaling factor is for use in a linear change of variable.  ("Buyer
    beware": note that, depending on what one is doing, one may want `scl`
    to be the reciprocal of what one might expect; for more information,
    see the Notes section below.)  The argument `c` is an array of
    coefficients from low to high degree along each axis, e.g., [1,2,3]
    represents the series ``P_0 + 2*P_1 + 3*P_2`` while [[1,2],[1,2]]
    represents ``1*P_0(x)*P_0(y) + 1*P_1(x)*P_0(y) + 2*P_0(x)*P_1(y) +
    2*P_1(x)*P_1(y)`` if axis=0 is ``x`` and axis=1 is ``y``.

    Parameters
    ----------
    c : array_like
        Array of orthogonal series coefficients. If c is multidimensional the
        different axis correspond to different variables with the degree in
        each axis given by the corresponding index.
    rec : AbstractRecurrenceRelation
        Recurrence relation for the family of orthogonal polynomials
    m : int, optional
        Order of integration, must be positive. (Default: 1)
    k : {[], list, scalar}, optional
        Integration constant(s).  The value of the first integral at
        ``lbnd`` is the first value in the list, the value of the second
        integral at ``lbnd`` is the second value, etc.  If ``k == []`` (the
        default), all constants are set to zero.  If ``m == 1``, a single
        scalar can be given instead of a list.
    lbnd : scalar, optional
        The lower bound of the integral. (Default: 0)
    scl : scalar, optional
        Following each integration the result is *multiplied* by `scl`
        before the integration constant is added. (Default: 1)
    axis : int, optional
        Axis over which the integral is taken. (Default: 0).

    Returns
    -------
    S : ndarray
        Orthogonal series coefficient array of the integral.

    Raises
    ------
    ValueError
        If ``m < 0``, ``len(k) > m``, ``np.ndim(lbnd) != 0``, or
        ``np.ndim(scl) != 0``.

    See Also
    --------
    orthder

    Notes
    -----
    Note that the result of each integration is *multiplied* by `scl`.
    Why is this important to note?  Say one is making a linear change of
    variable :math:`u = ax + b` in an integral relative to `x`.  Then
    :math:`dx = du/a`, so one will need to set `scl` equal to
    :math:`1/a` - perhaps not what one would have first thought.

    """
    c = pu.as_series(c)
    lbnd, scl = map(jnp.asarray, (lbnd, scl))

    if not jnp.iterable(k):
        k = [k]
    if len(k) > m:
        raise ValueError("Too many integration constants")
    if jnp.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if jnp.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")

    if m == 0:
        return c

    c = jnp.moveaxis(c, axis, 0)
    k = jnp.array(list(k) + [0] * (m - len(k)), ndmin=1)

    I = integration_matrix(rec, len(c) + m - 1)
    c = _pad_along_axis(c, (0, m), axis=0)
    # TODO: figure out how to get rid of this python loop
    for i in range(m):
        c *= scl
        c = I @ c
        c = c.at[0].add(k[i] - orthval(lbnd, c, rec))
    c = jnp.moveaxis(c, 0, axis)
    return c


def jacobi_matrix(rec, n):
    """Return the Jacobi matrix for a given set of orthogonal polynomials.


    Parameters
    ----------
    rec : AbstractRecurrenceRelation
        Recurrence relation for the family of orthogonal polynomials
    n : int
        Degree of Jacobi matrix

    Returns
    -------
    J : jax.Array, shape(n,n)
        Jacobi matrix.
    """
    nn = jnp.arange(n)
    a = rec.a(nn)
    b = rec.b(nn)
    return (
        jnp.diag(a) + jnp.diag(jnp.sqrt(b[1:]), k=1) + jnp.diag(jnp.sqrt(b[1:]), k=-1)
    )


def orthgauss(deg, rec, x0=None, x1=None):
    """Compute Gaussian quadrature nodes and weights for given orthogonal polynomials.

    Can optionally compute Gauss-Radau or Gauss-Lobatto points if x0 and/or x1 are
    given.

    Parameters
    ----------
    deg : int
        Number of sample points and weights. It must be >= 1.
    rec : AbstractRecurrenceRelation
        Recurrence relation for the family of orthogonal polynomials
    x0, x1 : float, optional
        Optional points to include in the nodes, for Gauss-Radau (if x0 is given) or
        Gauss-Lobatto (if x0 and x1 given). Generally taken to be the start and/or end
        points of the interval of integration.

    Returns
    -------
    x, w : jax.Array, shape(n,)
        Nodes and weights for n point quadrature rule.
    """
    n = deg - (x0 is not None) - (x1 is not None)
    J = jacobi_matrix(rec, n)
    if x0 is not None:
        x0 = jnp.asarray(x0).astype(float)
        J = jnp.pad(J, ((0, 1), (0, 1)))
        J = J.at[n, n - 1].set(jnp.sqrt(rec.b(n)))
        J = J.at[n - 1, n].set(jnp.sqrt(rec.b(n)))
        if x1 is None:
            # gauss radau
            astar = x0 - rec.b(n) * polyval(x0, n - 1, rec, True) / polyval(
                x0, n, rec, True
            )
            J = J.at[n, n].set(astar)
        else:
            # gauss lobatto
            x1 = jnp.asarray(x1).astype(float)
            J = J.at[n, n].set(rec.a(n))
            mat = jnp.array(
                [
                    [polyval(x0, n + 1, rec, True), polyval(x0, n, rec, True)],
                    [polyval(x1, n + 1, rec, True), polyval(x1, n, rec, True)],
                ]
            )
            rhs = jnp.array(
                [x0 * polyval(x0, n + 1, rec, True), x1 * polyval(x1, n + 1, rec, True)]
            )
            astar, bstar = jnp.linalg.solve(mat, rhs)
            J = jnp.pad(J, ((0, 1), (0, 1)))
            J = J.at[n + 1, n + 1].set(astar)
            J = J.at[n, n + 1].set(jnp.sqrt(bstar))
            J = J.at[n + 1, n].set(jnp.sqrt(bstar))
    else:
        assert x1 is None
    x, v = jnp.linalg.eigh(J)
    w = rec.b(0) * v[0] ** 2
    return x, w


@jit
def orthcompanion(c, rec):
    """Return the companion matrix of c.

    Parameters
    ----------
    c : array_like
        1-D array of orthogonal series coefficients ordered from low to high
        degree.
    rec : AbstractRecurrenceRelation
        Recurrence relation for the family of orthogonal polynomials

    Returns
    -------
    mat : jax.Array
        Companion matrix of dimensions (deg, deg).

    """
    c = pu.as_series(c)

    if len(c) < 2:
        raise ValueError("Series must have maximum degree of at least 1.")

    nn = jnp.arange(len(c))
    # convert to monic form
    c = c * rec.m(nn)
    c = c / c[-1]  # divide out leading coeff
    # based on
    # "A companion matrix analogue for orthogonal polynomials", S. Barnett, 1975

    a = rec.a(nn)
    b = rec.b(nn)
    alpha = jnp.ones_like(a)
    beta = -a
    gamma = b
    A = (
        jnp.diag(-beta[:-1] / alpha[:-1])
        + jnp.diag(1 / alpha[:-2], k=1)
        + jnp.diag(gamma[1:-1] / alpha[1:-1], k=-1)
    )
    return A.at[-1, :].add(-c[:-1] / alpha[-1])


@jit
def orthroots(c, rec):
    r"""Compute the roots of an orthogonal series.

    Return the roots (a.k.a. "zeros") of the polynomial

    .. math:: p(x) = \sum_i c[i] * P_i(x).

    Parameters
    ----------
    c : 1-D array_like
        1-D array of coefficients.
    rec : AbstractRecurrenceRelation
        Recurrence relation for the family of orthogonal polynomials

    Returns
    -------
    out : ndarray
        Array of the roots of the series.

    Notes
    -----
    The root estimates are obtained as the eigenvalues of the companion
    matrix, Roots far from the origin of the complex plane may have large
    errors due to the numerical instability of the series for such values.
    Roots with multiplicity greater than 1 will also show larger errors as
    the value of the series near such points is relatively insensitive to
    errors in the roots. Isolated roots near the origin can be improved by
    a few iterations of Newton's method.

    """
    c = pu.as_series(c)
    if len(c) <= 1:
        return jnp.array([], dtype=c.dtype)

    # rotated companion matrix reduces error
    m = orthcompanion(c, rec)[::-1, ::-1]
    r = jnp.linalg.eigvals(m)
    r = jnp.sort(r)
    # TODO: add newton iterations to improve roots
    return r
