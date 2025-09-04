"""
====================================
General Orthogonal Polynomial Series
====================================

The base ``orthax`` module provides a number of functions useful for dealing with
general orthogonal polynomial series, based around the idea of the three term
recurrence relation.

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

from . import (
    _version,
    chebyshev,
    hermite,
    hermite_e,
    laguerre,
    legendre,
    polynomial,
    polyutils,
    recurrence,
)

from .polynomial import Polynomial
from .chebyshev import Chebyshev
from .legendre import Legendre
from .laguerre import Laguerre
from .hermite import Hermite
from .hermite_e import HermiteE

from ._general import (
    orth2poly,
    orthadd,
    orthcompanion,
    orthder,
    orthdiv,
    orthfit,
    orthfromroots,
    orthgauss,
    orthgrid2d,
    orthgrid3d,
    orthint,
    orthline,
    orthmul,
    orthmulx,
    orthnorm,
    orthpow,
    orthroots,
    orthsub,
    orthtrim,
    orthval,
    orthval2d,
    orthval3d,
    orthvander,
    orthvander2d,
    orthvander3d,
    orthweight,
    poly2orth,
    OrthPoly
)

def set_default_printstyle(style):
    """
    Set the default format for the string representation of polynomials.

    Values for ``style`` must be valid inputs to ``__format__``, i.e. 'ascii'
    or 'unicode'.

    Parameters
    ----------
    style : str
        Format string for default printing style. Must be either 'ascii' or
        'unicode'.

    Notes
    -----
    The default format depends on the platform: 'unicode' is used on
    Unix-based systems and 'ascii' on Windows. This determination is based on
    default font support for the unicode superscript and subscript ranges.

    Examples
    --------
    >>> p = np.polynomial.Polynomial([1, 2, 3])
    >>> c = np.polynomial.Chebyshev([1, 2, 3])
    >>> np.polynomial.set_default_printstyle('unicode')
    >>> print(p)
    1.0 + 2.0·x + 3.0·x²
    >>> print(c)
    1.0 + 2.0·T₁(x) + 3.0·T₂(x)
    >>> np.polynomial.set_default_printstyle('ascii')
    >>> print(p)
    1.0 + 2.0 x + 3.0 x**2
    >>> print(c)
    1.0 + 2.0 T_1(x) + 3.0 T_2(x)
    >>> # Formatting supersedes all class/package-level defaults
    >>> print(f"{p:unicode}")
    1.0 + 2.0·x + 3.0·x²
    """
    if style not in ('unicode', 'ascii'):
        raise ValueError(
            f"Unsupported format string '{style}'. Valid options are 'ascii' "
            f"and 'unicode'"
        )
    _use_unicode = True
    if style == 'ascii':
        _use_unicode = False
    from .polybase import ABCPolyBase
    ABCPolyBase._use_unicode = _use_unicode


__version__ = _version.get_versions()["version"]
