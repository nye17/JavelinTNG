
## 0.5.3

- Fixed a regression in the optimized demo mock generator introduced in 0.5.1.
  The Pmap photometric mock had been formed with an independently generated
  continuum realization, breaking the covariance assumed by Pmap and causing
  broad/aliased lag posteriors.
- The efficient generator now uses one DRW continuum realization for all
  derived signals, evaluates both line responses on a common epoch grid, and
  constructs YelmBand = Continuum + Yelm and YelmZingBand = Yelm + Zing,
  matching the historical joint-realization construction.
- A local Pmap regression run on a freshly generated mock recovers the injected
  100-day lag (quick 30-walker run: median ~100.65 d).

# JAVELIN modernization

- Python >=3.10, NumPy >=2, SciPy >=1.10.
- `numpy.distutils` and installation-time Fortran/f2py compilation removed.
- Historical BLAS/LAPACK helper wrappers replaced by NumPy/SciPy.
- SPEAR covariance kernel translated from the bundled Fortran formulas to Python/NumPy.
- Bundled historical `emcee_internal` retained and NumPy-2 compatibility fixed.
- External emcee >=3.1.6 supported through `javelin.samplers` compatibility adapter.
- Select with `javelin.set_sampler_backend("internal")`, `"emcee"`, or `"auto"`.

## 0.4.1 compatibility iteration

This iteration was performed against Python 3.13 and NumPy 2.x after an
installed-package import exposed additional legacy assumptions.

Changes:

- Replaced the removed private import `numpy.linalg.linalg.LinAlgError` with
  the public `numpy.linalg.LinAlgError` API.
- Updated the bundled historical emcee autocorrelation code for NumPy 2.x:
  multidimensional indexing now uses tuples of slices rather than lists.
- Added pytest compatibility for the historical nose-style sampler test class.
- Fixed a Python-3 integer-division assumption in an old sampler test helper.
- Modernized SciPy ndimage imports that used deprecated `filters` and
  `morphology` submodule paths.
- Modernized deprecated threading calls (`setDaemon`, `isSet`).
- Hardened light-curve input handling: filename errors are no longer swallowed
  and then misinterpreted as file-like objects; genuine open handles remain
  supported and are not closed by JAVELIN.
- Improved the external-emcee adapter so the old JAVELIN `rstate0` continuation
  contract is preserved through an emcee 3 `State` object when possible.
- Added a small modern pytest smoke suite and configured pytest not to execute
  the long research examples merely because two of them are named `test_*.py`.

Validation performed in this iteration:

- `import javelin` succeeds under Python 3.13 / NumPy 2.3.x.
- Bundled historical emcee suite: 9 passed.
- `examples/demo.py test` executes.
- `examples/plotcov.py` executes with a non-interactive Matplotlib backend.
- `javelin.predict.test_PredictSpear()` executes.
- A short `Cont_Model.do_mcmc()` run with the internal sampler produces a
  finite chain and HPD array.
- The external-emcee compatibility adapter was exercised against an emcee-3
  shaped API contract, including burn-in/reset/continuation and legacy chain
  orientation.
- A pure-Python wheel builds successfully.

The long `thindisk` and `doubledustecho` scripts remain research examples rather
than unit tests; their full production MCMC settings are intentionally not run
as part of `pytest`.


## 0.4.2 prediction compatibility iteration

This iteration fixes NumPy 2.x scalar-assignment failures in the SPEAR prediction
path, exposed by `examples/demo.py show`.

Changes:

- Added a small `_scalarize()` helper in `javelin.predict` that converts a true
  scalar or size-1 ndarray/matrix to a Python float and rejects non-scalar shapes.
- Applied scalar extraction consistently to the predicted mean, prior variance,
  and covariance-correction terms in all five `_fastpredict()` implementations
  (Rmap/Pmap/SPmap/SCmap/DPmap-related paths).
- Preserved `spear()` and `spear_threading()` as matrix-returning covariance APIs;
  this avoids breaking callers that legitimately expect covariance matrices.
- Replaced the remaining removed NumPy alias `np.alltrue(...)` with `np.all(...)`
  throughout `predict.py`.
- Added a regression test for size-1 covariance scalarization.

Validation:

- Automated suite: 13 passed under Python 3.13 / NumPy 2.3.x.
- `examples/demo.py show` proceeds through the Rmap `do_pred()` call that previously
  failed with `ValueError: setting an array element with a sequence`, and continues
  into the subsequent multi-line model. The full interactive plotting demo is long
  and was not run to completion in the sandbox.

## 0.4.4: Python 3.13 LaTeX/string-literal cleanup

- Converted LaTeX-bearing docstrings to raw docstrings where appropriate.
- Converted Matplotlib mathtext labels to raw string literals.
- Normalized LaTeX backslashes so commands such as `\tau`, `\beta`, `\Delta`, `\mathrm`, and `\sum` are not interpreted as Python escape sequences.
- Audited all Python string tokens in `javelin/`, `examples/`, and `tests/`; no math-bearing non-raw string literals remain.
- Verified with `python -Werror::SyntaxWarning -m compileall` and the full pytest suite (14 passed).

Remaining warnings are unrelated to LaTeX: legacy `numpy.matrix` usage and multiprocessing/fork deprecations.


## 0.5.0: full ndarray linear-algebra conversion

This release removes the remaining runtime use of `numpy.matrix` from JAVELIN's
Gaussian-process and covariance implementation.

Changes:

- Replaced all `np.asmatrix(...)` / `asmatrix(...)` construction in the runtime
  package with ordinary NumPy `ndarray` objects.
- Replaced matrix-class `*` multiplication with explicit `@` matrix
  multiplication at every affected linear-algebra site.
- Updated the SPEAR, Brownian, and generic covariance wrappers to return
  `ndarray` covariance matrices.
- Updated full-rank, nearly-full-rank, basis-covariance, observation-update,
  and triangular-solve paths to preserve their historical shapes using
  explicit ndarray operations.
- Removed matrix-only `(N,1)` assignment assumptions in the modernized distance
  routines; distance columns are now assigned as 1-D ndarray slices.
- Fixed a latent `BasisCovariance(x, x)` same-object fast-path bug: it now
  returns the actual covariance `B.T @ B`, rather than the basis factor itself.
- Removed the last deliberate `np.matrix` use from the modern regression tests.

Validation:

- No `np.matrix`, `numpy.matrix`, or `asmatrix(...)` runtime calls remain under
  `javelin/`.
- Full automated suite passes under Python 3.13 / NumPy 2.x.
- `examples/demo.py test` reproduces the 0.4.4 likelihood values to numerical
  precision.
- `examples/plotcov.py` runs without `numpy.matrix` deprecation warnings.
- A basis-covariance regression verifies `U.T @ U == C` and diagonal agreement.
- A short internal-sampler `Cont_Model.do_mcmc()` inference produces a finite
  `(32, 2)` chain and HPD output.

This intentionally does not change the legacy multiprocessing architecture;
that remains a separate future modernization task.

## 0.5.1: fast DRW demo generation and vectorized Rmap covariance

- Replaced the normal SPEAR/Rmap covariance pair loop with a block-vectorized
  NumPy implementation. Randomized regression tests compare it bit-for-bit
  against the historical scalar formulas.
- Fixed `examples/demo.py run` appearing to hang at `generating DRW light curves...`.
  The old `generateTrueLC2()` built and Cholesky-factorized one 6000 x 6000
  joint covariance matrix for three dense 2000-point curves. For the DRW demo,
  the two line curves are deterministic top-hat responses to one continuum
  realization, so the demo now generates the continuum once and constructs the
  line curves by convolution via the existing `generateTrueLC()` path.
- Renamed the startup message to `use MCMC multiprocessing on ... cpus` to make
  clear that the CPU count applies to the later sampler stage, not truth-curve
  generation.

Validation:

- `getTrue(..., mode="run", covfunc="drw")` completes in about 0.6 s in the
  container instead of spending minutes in a dense 6000-point factorization.
- Full automated suite: 16 passed.

## 0.5.2 MCMC performance pass

- Reworked the bundled historical ensemble sampler to use a persistent, spawn-safe worker pool. Each worker receives the wrapped log-posterior and light-curve data once at startup; subsequent MCMC evaluations send only parameter vectors. This avoids repeated serialization and the legacy macOS fork path.
- Worker initialization limits native BLAS/LAPACK threading to one thread via `threadpoolctl`, avoiding process × BLAS oversubscription.
- Added explicit sampler pool shutdown at the end of JAVELIN `do_mcmc()` calls to prevent accumulating workers/semaphores across successive model fits.
- Vectorized the Pmap and DPmap covariance kernels. On a 220-point benchmark, covariance filling was roughly 70x (Pmap) and 80x (DPmap) faster than the scalar Python loops, with agreement to machine precision.
- Vectorized diagonal nugget addition and Cholesky log-determinant bookkeeping.
- Fixed `examples/demo.py::fitCon()` so its `threads` argument is actually passed to the continuum sampler.
- Capped automatic demo workers at 8.
- `python demo.py run` is now an integration smoke run using 24 walkers, 10 burn-in steps, and 10 production steps. Set `JAVELIN_DEMO_FULL=1` to recover the historical 100 walkers / 100 burn-in / 100 production settings. The model API defaults are unchanged.
- End-to-end `python demo.py run` was executed successfully through continuum, Rmap, multi-line Rmap, Pmap, and DPmap fits.
