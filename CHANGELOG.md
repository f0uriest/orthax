Changelog
=========

0.2.10
------
- `generate_recurrence` is more accurate and has an error estimate:
  - New `tol` argument sets the relative error tolerance for the coefficients, and
  quadrature tolerances are chosen to meet it. Default is the square root of machine
  precision for the dtype of `domain`.
  - New `throw` argument raises an error if the estimated error exceeds `tol`.
  - Coefficients from the Stieltjes procedure are refined using the Gram matrix of the
  resulting basis, which removes the effect of quadrature errors to first order.
- `TabulatedRecurrenceRelation` and `generate_recurrence` take a `check` argument.
Setting `check=False` skips bounds checking on the requested coefficient indices, which
uses a runtime callback that can be slow.
- Vectorized several loops, reducing compile and run time for `*der`, `legmulx`, and the
differentiation matrix used by `orthder` and `orthint`.
- Type hints throughout, and orthax is now marked as typed (PEP 561). Array arguments
are hinted as `jax.typing.ArrayLike`. Lists and tuples are still accepted, but are not
included in the type hints since converting them can be a silent performance cost.
- Bumps maximum jax version to 0.11 and maximum numpy version to 2.5.
- Bumps minimum jax version to 0.5.0 and minimum numpy version to 1.25.0.

**Full Changelog**: https://github.com/f0uriest/orthax/compare/v0.2.9...main


v0.2.9
------
- Support building from sdists.
- Bumps maximum jax version to 0.10.

**Full Changelog**: https://github.com/f0uriest/orthax/compare/v0.2.8...v0.2.9


v0.2.8
------
- Bumps maximum jax version to 0.9 and maximum numpy version to 2.4.
- Ignore deprecation warnings from third party libraries in tests.

**Full Changelog**: https://github.com/f0uriest/orthax/compare/v0.2.7...v0.2.8


v0.2.7
------
- Fixes to avoid recompilation when using classical recurrence relations. As part of
this, `AbstractRecurrenceRelation.weight` is now an abstract method rather than a
property, and `ClassicalRecurrenceRelation` no longer takes a `weight` argument. Custom
subclasses should implement `weight` directly.

**Full Changelog**: https://github.com/f0uriest/orthax/compare/v0.2.6...v0.2.7


v0.2.6
------
- `Gegenbauer` now raises an error for `lmbda=0`, where the classical normalization is
undefined. `ChebyshevT` has similar orthogonality properties with a well behaved
normalization.
- Bumps maximum jax version to 0.8.

**Full Changelog**: https://github.com/f0uriest/orthax/compare/v0.2.5...v0.2.6


v0.2.5
------
- Bumps maximum jax version to 0.7 and maximum equinox version to 0.13.
- Adds testing on python 3.13.

**Full Changelog**: https://github.com/f0uriest/orthax/compare/v0.2.4...v0.2.5


v0.2.4
------
- Minimum jax version is now 0.4.36, maximum is 0.6.
- Adds testing against all supported jax versions.

**Full Changelog**: https://github.com/f0uriest/orthax/compare/v0.2.3...v0.2.4


v0.2.3
------
- Bumps maximum jax version to 0.5.3.
- Fix installation on windows.

**Full Changelog**: https://github.com/f0uriest/orthax/compare/v0.2.2...v0.2.3


v0.2.2
------
- Maintenance release for compatibility with jax 0.5.0, numpy 2.2.2 and equinox 0.11.11.

**Full Changelog**: https://github.com/f0uriest/orthax/compare/v0.2.1...v0.2.2


v0.2.1
------
- Maintenance release, bumping maximum numpy version.

**Full Changelog**: https://github.com/f0uriest/orthax/compare/v0.2.0...v0.2.1


v0.2.0
------
- Adds three term recurrence relations for many classical orthogonal families, and the
ability to generate relations for non-classical weight functions and domains.
- Adds new functions to the base `orthax` namespace for working with general orthogonal
series by specifying the three term recurrence coefficients.
- Adds `*norm` functions for the norms of the classical polynomial families.
- Removes support for python 3.8.

**Full Changelog**: https://github.com/f0uriest/orthax/compare/v0.1.0...v0.2.0


v0.1.0
------
Initial release
