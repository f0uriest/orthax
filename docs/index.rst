.. include:: ../README.rst


API Documentation
=================


Within the documentation for this package, a "finite power series,"
i.e., a polynomial (also referred to simply as a "series") is represented
by a 1-D JAX array of the polynomial's coefficients, ordered from lowest
order term to highest.  For example, array([1,2,3]) represents
``P_0 + 2*P_1 + 3*P_2``, where P_n is the n-th order basis polynomial
applicable to the specific module in question, e.g., ``polynomial`` (which
"wraps" the "standard" basis) or ``chebyshev``.  For optimal performance,
all operations on polynomials, including evaluation at an argument, are
implemented as operations on the coefficients.  Additional (module-specific)
information can be found in the docstring for the module of interest.

General orthogonal polynomials
------------------------------

The core of ``orthax`` is based around the three term recurrence relation for a general
orthogonal polynomial. The ``orthax.recurrence`` module provides recurrence relations
for many of the "classical" orthogonal polynomials, as well as the ability to generate
recurrence relations for arbitrary weight functions and domains.

Here's an example for evaluating a Chebyshev series of the second kind:

.. code-block:: python

   rec = orthax.recurrence.ChebyshevU() # Chebyshev polynomials of the 2nd kind
   c = jnp.array([0, 1.2, 0, 2]) # 2*U_3(x) + 1.2*U_1(x)
   x = jnp.linspace(-1, 1, 10)
   f = orthax.orthval(x, c, rec)

Or generating non-classical polynomials, such as the "Maxwell polynomials" or one sided
Hermite:

.. code-block:: python

   weight = lambda x: jnp.exp(-x**2)
   domain = (0, jnp.inf)
   rec = orthax.recurrence.generate_recurrence(weight, domain, n=10)
   x, w = orthax.orthgauss(10, rec)

For more information, see the following sections:

.. toctree::
   :maxdepth: 1

   api_recurrence
   api_general

``numpy.polynomial`` interface
------------------------------

``orthax`` also contains submodules for working with many of the "classic" families
of orthogonal polynomials. These submodules are meant to be drop in replacements for
the corresponding modules from the ``numpy.polynomial`` package.

.. toctree::
   :maxdepth: 1

   api_chebyshev
   api_hermite
   api_hermite_e
   api_laguerre
   api_legendre
   api_polynomial
   api_polyutils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
