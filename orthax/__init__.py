"""orthax: orthogonal polynomial series with JAX."""

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
