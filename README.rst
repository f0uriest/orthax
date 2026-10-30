.. image:: https://raw.githubusercontent.com/f0uriest/orthax/main/docs/_static/images/logo_banner.svg
    :alt: orthax: orthogonal polynomials in JAX
    :width: 600

|License| |DOI| |Issues| |Pypi|

|Docs| |UnitTests| |Codecov|

``orthax`` is a library for working with orthogonal polynomials in `JAX <https://github.com/jax-ml/jax>`__.
It provides a drop-in replacement for ``numpy.polynomial`` that works with ``jit``,
``grad``, ``vmap`` and runs on CPU, GPU and TPU, and goes further with a general
framework built on three term recurrence relations that handles *any* family of
orthogonal polynomials, including ones you define yourself from an arbitrary weight
function.

Highlights
==========

- **Familiar API**: ``orthax.chebyshev``, ``orthax.legendre``, ``orthax.hermite``,
  ``orthax.hermite_e``, ``orthax.laguerre`` and ``orthax.polynomial`` mirror their
  ``numpy.polynomial`` counterparts function for function.
- **Fully JAX native**: written in pure ``jax.numpy``, so functions compose with
  ``jit``, ``grad`` and ``vmap`` over coefficients or evaluation points.
- **Twelve classical families built in**: Legendre, shifted Legendre, Chebyshev of all
  four kinds, Gegenbauer, Jacobi, Laguerre, generalized Laguerre and both Hermite
  conventions, each in standard, monic or orthonormal scaling.
- **Bring your own weight**: ``generate_recurrence`` computes the recurrence relation
  for polynomials orthogonal with respect to any weight on any (possibly infinite)
  interval, with error control.
- **The full toolkit for every family**: arithmetic, evaluation in 1D/2D/3D,
  differentiation and integration, least squares fitting, roots, Vandermonde and
  companion matrices, conversion to and from power series, and Gauss, Gauss-Radau and
  Gauss-Lobatto quadrature.

📖 **Documentation:** `orthax.readthedocs.io <https://orthax.readthedocs.io/en/latest/>`__

.. image:: https://raw.githubusercontent.com/f0uriest/orthax/main/docs/_static/images/families.png
    :alt: The first five orthonormal basis functions for six families of orthogonal polynomials, including one generated from a custom weight function.

Quick start
===========

A drop-in for ``numpy.polynomial``
----------------------------------

.. code-block:: python

    import jax
    import jax.numpy as jnp
    from orthax import chebyshev as C

    x = jnp.linspace(-1, 1, 200)
    y = jnp.exp(x) * jnp.sin(5 * x)

    c = C.chebfit(x, y, deg=20)       # least squares fit
    dc = C.chebder(c)                 # derivative of the series
    ic = C.chebint(c, lbnd=-1)        # antiderivative of the series
    r = C.chebroots(c[:8])            # roots of a truncated series

...that also works with JAX transformations
-------------------------------------------

.. code-block:: python

    # gradient of a loss with respect to the series coefficients
    loss = lambda c: jnp.sum((C.chebval(x, c) - y) ** 2)
    g = jax.jit(jax.grad(loss))(c)

    # evaluate a whole batch of series at once
    batch = jnp.ones((32, 10))
    vals = jax.vmap(C.chebval, in_axes=(None, 0))(x, batch)  # shape (32, 200)

Any classical family
--------------------

The functions in the base ``orthax`` namespace take a recurrence relation describing
the polynomial family, so families without a dedicated ``numpy`` module get the same
toolkit.

.. code-block:: python

    import orthax

    rec = orthax.recurrence.Jacobi(alpha=0.5, beta=-0.5)

    c = jnp.array([0, 1.2, 0, 2])        # 1.2 P_1(x) + 2 P_3(x)
    f = orthax.orthval(x, c, rec)        # evaluate
    df = orthax.orthder(c, rec)          # differentiate
    c2 = orthax.orthmul(c, c, rec)       # multiply two series

    # Gauss-Jacobi quadrature: nodes and weights for the integral of f(x) w(x)
    xq, wq = orthax.orthgauss(16, rec)
    integral = jnp.sum(wq * jnp.cos(xq))

Custom weight functions
-----------------------

Generate polynomials orthogonal with respect to any weight. For example, the
"half-range Hermite" or Maxwell polynomials, orthogonal on [0, ∞) with weight
exp(-x²):

.. code-block:: python

    rec = orthax.recurrence.generate_recurrence(
        lambda x: jnp.exp(-x**2), domain=(0, jnp.inf), n=20
    )

    # 20 point Gauss quadrature for the integral of f(x) exp(-x^2) over [0, inf)
    xq, wq = orthax.orthgauss(20, rec)
    jnp.sum(wq * jnp.cos(xq))  # 0.6901942..., matches scipy.integrate.quad

    # and everything else works as before
    c = orthax.orthfit(xq, jnp.sqrt(xq), deg=10, rec=rec)

Supported families
==================

.. list-table::
    :header-rows: 1

    * - Family
      - Recurrence relation
      - ``numpy.polynomial`` style module
    * - Power series
      -
      - ``orthax.polynomial``
    * - Legendre
      - ``Legendre``, ``ShiftedLegendre``
      - ``orthax.legendre``
    * - Chebyshev
      - ``ChebyshevT``, ``ChebyshevU``, ``ChebyshevV``, ``ChebyshevW``
      - ``orthax.chebyshev`` (first kind)
    * - Gegenbauer / Jacobi
      - ``Gegenbauer``, ``Jacobi``
      -
    * - Laguerre
      - ``Laguerre``, ``GeneralizedLaguerre``
      - ``orthax.laguerre``
    * - Hermite
      - ``Hermite`` (physicists'), ``HermiteE`` (probabilists')
      - ``orthax.hermite``, ``orthax.hermite_e``
    * - Anything else
      - ``generate_recurrence``, ``TabulatedRecurrenceRelation``
      -

Differences from ``numpy.polynomial``
=====================================

JAX requires array shapes to be known at compile time, so trailing zeros are not
automatically trimmed from series as they are in ``numpy``. Use ``orthtrim`` (or the
module specific ``*trim`` functions) outside of jitted code if this becomes a concern.

For full details see the `documentation <https://orthax.readthedocs.io/en/latest/>`__.

Installation
============

orthax is installable with ``pip``:

.. code-block:: console

    pip install orthax

``generate_recurrence`` additionally uses `quadax <https://github.com/f0uriest/quadax>`__
for adaptive quadrature:

.. code-block:: console

    pip install quadax

Citing
======

If you use orthax in your research, please cite it using the information on
`Zenodo <https://zenodo.org/doi/10.5281/zenodo.10774399>`__ or in ``CITATION.cff``.


.. |License| image:: https://img.shields.io/github/license/f0uriest/orthax?color=blue&logo=open-source-initiative&logoColor=white
    :target: https://github.com/f0uriest/orthax/blob/main/LICENSE
    :alt: License

.. |DOI| image:: https://zenodo.org/badge/763821973.svg
    :target: https://zenodo.org/doi/10.5281/zenodo.10774399
    :alt: DOI

.. |Docs| image:: https://img.shields.io/readthedocs/orthax?logo=Read-the-Docs
    :target: https://orthax.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation

.. |UnitTests| image:: https://github.com/f0uriest/orthax/actions/workflows/unittest.yml/badge.svg
    :target: https://github.com/f0uriest/orthax/actions/workflows/unittest.yml
    :alt: UnitTests

.. |Codecov| image:: https://codecov.io/github/f0uriest/orthax/graph/badge.svg?token=MB11I7WE3I
    :target: https://codecov.io/github/f0uriest/orthax
    :alt: Coverage

.. |Issues| image:: https://img.shields.io/github/issues/f0uriest/orthax
    :target: https://github.com/f0uriest/orthax/issues
    :alt: GitHub issues

.. |Pypi| image:: https://img.shields.io/pypi/v/orthax
    :target: https://pypi.org/project/orthax/
    :alt: Pypi
