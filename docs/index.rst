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

This package provides functions for operations on each of six different kinds
of polynomials:

.. automodule:: orthax
   :no-members:
   :no-inherited-members:
   :no-special-members:


.. toctree::
   :maxdepth: 1

   api_recurrence
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
