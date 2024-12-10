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
)

__version__ = _version.get_versions()["version"]
