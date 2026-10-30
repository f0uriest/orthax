#############
API Reference
#############

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
==============================

The core of ``orthax`` is based around the three term recurrence relation for a general
orthogonal polynomial. The ``orthax.recurrence`` module provides recurrence relations
for many of the "classical" orthogonal polynomials, as well as the ability to generate
recurrence relations for arbitrary weight functions and domains. The functions in the
base ``orthax`` namespace accept any of these recurrence relations.

.. toctree::
   :maxdepth: 1

   api_recurrence
   api_general

``numpy.polynomial`` interface
==============================

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

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
