# {octicon}`megaphone` Releases & Changelog

## Version 0.8
### Version 0.8.16

Release Date: Jun 17, 2023

**What's Changed**
* fix for [\#409](https://github.com/casact/chainladder-python/issues/409)  
* fix for [\#411](https://github.com/casact/chainladder-python/issues/411) 
* fix for [\#438](https://github.com/casact/chainladder-python/issues/438) 

## New Contributors
* [@MatthewCaseres](https://github.com/MatthewCaseres) made their first contribution.
* [@andrejakobsen](https://github.com/andrejakobsen) made their first contribution.





### Version 0.8.15

Release Date: Apr 11, 2023

**What's Changed**
- revamping drop functions by @henrydingliu in [\#385](https://github.com/casact/chainladder-python/issues/385)
- Fixing link_ratio by @henrydingliu in [\#387](https://github.com/casact/chainladder-python/issues/387)
- adding some tests for the revamped drop function by @henrydingliu in [\#388](https://github.com/casact/chainladder-python/issues/388)
- Adding test wth CL and BF by @henrydingliu in [\#391](https://github.com/casact/chainladder-python/issues/391)
- Henrydingliu enhancement 1 2022 11 29 by @henrydingliu in [\#392](https://github.com/casact/chainladder-python/issues/392)
- w_v2_ and pipeline in Development by @henrydingliu in [\#394](https://github.com/casact/chainladder-python/issues/394)
- next step for #376 by @henrydingliu in [\#397](https://github.com/casact/chainladder-python/issues/397)
- consolidating weighted regression factor fit by @henrydingliu in [\#398](https://github.com/casact/chainladder-python/issues/398)
- cum_zeta_ by @henrydingliu in [\#401](https://github.com/casact/chainladder-python/issues/401)
- Expanded mappings to include semester sampling by @A108669 in [\#403](https://github.com/casact/chainladder-python/issues/403)
- fix one more ref to semiannual by @wleescor in [\#410](https://github.com/casact/chainladder-python/issues/410)
- TailCurve argument validation by @genedan in [\#414](https://github.com/casact/chainladder-python/issues/414)
- chore(version): specify version of numba by @pafechet in [\#418](https://github.com/casact/chainladder-python/issues/418)
- Add Friedland Sample - Auto BI Insurer (Expected Loss Method) by @genedan in [\#419](https://github.com/casact/chainladder-python/issues/419)
- Cas webinar by @kennethshsu in [\#424](https://github.com/casact/chainladder-python/issues/424)
- [BUG] Argument validation needed for TailCurve [\#413](https://github.com/casact/chainladder-python/issues/413)
- [BUG] origin_as_datetime does nothing [\#407](https://github.com/casact/chainladder-python/issues/407)
- [BUG] CapeCod predict at different index grain [\#400](https://github.com/casact/chainladder-python/issues/400)
- [BUG] Chainladder doesn't coerce valuation triangle correctly [\#399](https://github.com/casact/chainladder-python/issues/399)

**New Contributors**
[@A108669](https://github.com/A108669) made their first contribution in [\#403](https://github.com/casact/chainladder-python/issues/403)
[@pafechet](https://github.com/pafechet) made their first contribution in [\#418](https://github.com/casact/chainladder-python/issues/418)

### Version 0.8.14

Release Date: Nov 25, 2022



### Version 0.8.13

Release Date: Jun 27, 2022

**Bug Fixes**

- [\#270](https://github.com/casact/chainladder-python/issues/270) Fixed an issue affecting triangle instantiation
- [\#251](https://github.com/casact/chainladder-python/issues/251) Fixed a bug in `predict` method that failed to honor development ages appropriatly
- [\#274](https://github.com/casact/chainladder-python/issues/274) Addressed a pandas deprecation notice in the `heatmap` method.
- [\#258](https://github.com/casact/chainladder-python/issues/258) Addressed silent error when predicting on a Triangle that is larger than the underlying model.
- [\#288](https://github.com/casact/chainladder-python/issues/288) Addressed Bug in `to_frame` method
- [\#283](https://github.com/casact/chainladder-python/issues/283) Addressed Bug in the calculation of LDFs in `CaseOutstanding`

**Enhancements**

- [\#293](https://github.com/casact/chainladder-python/issues/293) Expanded functionality `drop_high` and `drop_low` arguments of `Development` estimator.
- [\#298, #299, #304, #305, #308](https://github.com/casact/chainladder-python/issues/298) Documentation clean-up.

### Version 0.8.12

Release Date: Mar 8, 2022

**Bug Fixes**

- [\#254](https://github.com/casact/chainladder-python/issues/254) Fixed an undesired mutation when using cl.concat
- [\#257](https://github.com/casact/chainladder-python/issues/257) to_frame bug fix on empty triangles
- [\#248](https://github.com/casact/chainladder-python/issues/248) to_frame bug and deprecation notice in origin_as_datetime argument.
- [\#250](https://github.com/casact/chainladder-python/issues/250) Bug fix in triangle initialization
- [\#258](https://github.com/casact/chainladder-python/issues/258) Addressed silent error when predicting on a Triangle that is larger than the underlying model.
- [\#261](https://github.com/casact/chainladder-python/issues/261) Addressed a pandas>1.4.0 bug

**Enhancements**

- Introduced ExpectedLoss method
- [\#242](https://github.com/casact/chainladder-python/issues/242) Added threshold based dropping of link ratios to Development estimator
- [\#158](https://github.com/casact/chainladder-python/issues/158) Triangles can now be instantiated with only one data point.
- [\#250](https://github.com/casact/chainladder-python/issues/250) python 3.10 support
- [\#260](https://github.com/casact/chainladder-python/issues/260) fillna method added to Triangle class that supports filling with constants or other Triangles.

### Version 0.8.10

Release Date: Dec 4, 2021

**Enhancements**

- [\#225](https://github.com/casact/chainladder-python/issues/225) - Added support for mid-year triangles
- Remove `xlcompose` as a dependency
- [\#233](https://github.com/casact/chainladder-python/issues/233) and [\#219](https://github.com/casact/chainladder-python/issues/219) - Added more explicit warning when `is_cumulative` property of triangle is not set.
- [\#192](https://github.com/casact/chainladder-python/issues/192) Expanded `drop_high` and `drop_low` functionality to include integers and variable length lists. PR courtesy of @kennethshsu.

**Bug Fixes**

- [\#234](https://github.com/casact/chainladder-python/issues/234) Coerce `MackChainladder` to zero variance when `inbr_` and `cdf_` are 0.
- Fix bug in `BarnettZehnwirth.transform` that was previously not applying the log-transform correctly

### Version 0.8.9

Release Date: Oct 24, 2021

**Enhancements**

- [\#198](https://github.com/casact/chainladder-python/issues/198) Added `projection_period` to all Tail estimators. This allows for analyzing run-off beyond a one year time horizon.
- [\#214](https://github.com/casact/chainladder-python/issues/214) A refreshed docs theme using jupyter-book!
- [\#200](https://github.com/casact/chainladder-python/issues/200) Added more flexibility to `TailCurve.fit_period` to allow boolean lists - similar to `Development.drop_high`
- [kennethshsu](https://github.com/kennethshsu) further improved tutorials

**Bug Fixes**

- [\#210](https://github.com/casact/chainladder-python/issues/210) Fixed regression in triangle instantiation where `grain=='S'` is being interpreted as seconds and not semesters.
- [\#213](https://github.com/casact/chainladder-python/issues/213) Fixed an unintended Triangle mutation when reassigning columns on a sparse backend.
- [\#221](https://github.com/casact/chainladder-python/issues/221) Fixed `origin`/`development` broadcasting issue that was causing silent bugs in calculations on malformed triangles.

### Version 0.8.8

Release Date: Sep 13, 2021

**Enhancements**

- [\#140](https://github.com/casact/chainladder-python/issues/140) Improved test coverage
- [\#126](https://github.com/casact/chainladder-python/issues/126) Relaxed `BootstrapODPSample` column restriction
- [\#207](https://github.com/casact/chainladder-python/issues/207) Improved tutorials

**Bug Fixes**

- [\#204](https://github.com/casact/chainladder-python/issues/204) Fixed regression in `grain`
- [\#203](https://github.com/casact/chainladder-python/issues/203) Fixed regression in `slice`
- Fixed regression in `Benktander` index broadcasting
- [\#205](https://github.com/casact/chainladder-python/issues/205) Fixed CI/CD build

### Version 0.8.7

Release Date: Aug 29, 2021

Minor release to fix some `chainladder==0.8.6` regressions.

**Bug Fixes**

- Fixed [\#190](https://github.com/casact/chainladder-python/issues/190) - 0 values getting into `heatmap`
- Fixed [\#191](https://github.com/casact/chainladder-python/issues/191) regression that didn't support earlier versions of pandas
- Fixed [\#197](https://github.com/casact/chainladder-python/issues/197) Index broadcasting edge case
- Fixed `valuation_date` regression when instantiating Triangles as vectors

**Enhancements**

- [\#99 ](https://github.com/casact/chainladder-python/issues/99)Added Semester as an additional grain
- Added parallel support to `GridSearch` using `n_jobs` argument
- More tutorial improvements from [kennethshsu](https://github.com/kennethshsu)

### Version 0.8.6

Release Date: Aug 18, 2021

**Ehancements**

- [kennethshsu](https://github.com/kennethshsu) improved heatmap shading
- Support for numpy reductions, e.g. `np.sum(cl.load_sample('raa'))`
- [kennethshsu](https://github.com/kennethshsu) further improved tutorials

**Bug fixes**

- [\#186](https://github.com/casact/chainladder-python/issues/186) Fix bug that disallowed instantiating a full triangles
- [\#180](https://github.com/casact/chainladder-python/issues/180) Fix bug that mutated original DataFrame when instantiating Triangle
- [\#182](https://github.com/casact/chainladder-python/issues/182) Eliminate new deprecations warnings from pandas>=1.3
- [\#183](https://github.com/casact/chainladder-python/issues/183) Better alignment with pandas on index broadcasting
- [\#179](https://github.com/casact/chainladder-python/issues/179) Fixed nan behavior on `val_to_dev`
- implement `TriangleGroupby.__getitem__` to support column selection on groupby operations to align with pandas, e.g. `cl.load_sample('clrd').groupby('LOB')['CumPaidLoss']`

### Version 0.8.5

Release Date: Jul 11, 2021

**Enhancements**

- [\#154](https://github.com/casact/chainladder-python/issues/154) -
  Added groupby hyperparameter to several more estimators including
  the widely used Development estimator. This allows fitting
  development patterns at a higher grain than the Triangle all within
  the estiamtor or Pipeline.
- Improved index broadcasting for sparse arrays. Prior to 0.8.5, this
  code would inappropriately consume too much memory. For example:

> \>\>\> prism = cl.load_sample('prism') \>\>\> prism /
> prism.groupby('Line').sum()

- Arithmetic label matching improved for all axes to align more with
  pandas
- Added `model_diagnostics` utility function to be used on fitted
  estimators.
- Initial support for `dask` arrays. Current support is basic, but
  will eventually allow for distributed computations on massive
  triangles.
- added numpy array protocol support to the Triangle class. Now numpy
  functions can be called on Triangles. For example:

  \>\>\> np.sin(cl.load_sample('raa'))

- [\#169](https://github.com/casact/chainladder-python/issues/169) -
  Made Triangle tutorial more beginner friendly - courtesy of
  [kennethshsu](https://github.com/kennethshsu)

**Bug Fixes**

- Better Estimator pickling support when callables are included in
  estimators.
- Minor bug fix in grain estimator.

### Version 0.8.4

Release Date: May 9, 2021

**Enhancements**

- [\#153](https://github.com/casact/chainladder-python/issues/153) -
  Introduced CapeCod `groupby` parameter that allows for apriori
  computation at user-specified grain for granular triangles
- [\#154](https://github.com/casact/chainladder-python/issues/154) -
  Introduced `groupby` support in key development estimators

**Bug Fixes**

- [\#152](https://github.com/casact/chainladder-python/issues/152) -
  CapeCod apriori is not supporting nans effectively
- [\#157](https://github.com/casact/chainladder-python/issues/157) -
  `cum_to_incr()` not working as expected on `full_triangle_`
- [\#155](https://github.com/casact/chainladder-python/issues/155) -
  `full_triangle_` cdf broadcasting bug
- [\#156](https://github.com/casact/chainladder-python/issues/156) -
  Unable to inspect `sigma_` or `std_err_` properties after tail fit

### Version 0.8.3

Release Date: Apr 25, 2021

**Enhancements**

- [\#135](https://github.com/casact/chainladder-python/issues/135) -
  Added `.at` and `.iat` slicing and value assignment

**Bug Fixes**

- [\#144](https://github.com/casact/chainladder-python/issues/144) -
  Eliminated error when trying to assign a column from a different
  array backend.
- [\#134](https://github.com/casact/chainladder-python/issues/134) -
  Clearer error handling when attempting to instantiate a triangle
  with ages instead of date-likes
- [\#143](https://github.com/casact/chainladder-python/issues/143) -
  Reworked `full_triangle_` run-off for expected loss methods.

### Version 0.8.2

Release Date: Mar 27, 2021

**Enhancements**

- [\#131](https://github.com/casact/chainladder-python/issues/131) -
  Added a BarnettZenwirth development estimator
- [\#117](https://github.com/casact/chainladder-python/issues/117) -
  VotingChainladder enhancements to allow for more flexibility in
  defining weights (courtesy of @cbalona)

**Bug Fixes**

- [\#130](https://github.com/casact/chainladder-python/issues/130) -
  Fixed bug in triangle aggregation
- [\#134](https://github.com/casact/chainladder-python/issues/134) -
  Fixed a bug in triangle broadcasting
- [\#137](https://github.com/casact/chainladder-python/issues/137) -
  Fixed `valuation_date` bug occuring when partial year Triangle is
  instantiated as a vector.
- [\#138](https://github.com/casact/chainladder-python/issues/138) -
  Introduced fix for incompatibility with `sparse>=0.12.0`
- [\#122](https://github.com/casact/chainladder-python/issues/122) -
  Implemented nightly continuous integration to identify new bugs
  associated with verion bumps of dependencies.

### Version 0.8.1

Release Date: Feb 28, 2021

**Enhancements**

- Included a `truncation_age` in the `TailClark` estimator to
  replicate examples from the paper
- [\#129](https://github.com/casact/chainladder-python/issues/129)
  Included new development estimators `TweedieGLM` and `DevelopmentML`
  to take advantage of a broader set of (sklearn-compliant) regression
  frameworks.
- Included a helper Transformer `PatsyFormula` to make working with ML
  algorithms easier.

**Bug Fixes**

- [\#128](https://github.com/casact/chainladder-python/issues/128)
  Made `IncrementalAdditive` method comply with the rest of the
  package API

### Version 0.8.0

Release Date: Feb 15, 2021.

**Enhancements**

- [\#112](https://github.com/casact/chainladder-python/issues/112)
  Advanced `groupby` support.
- [\#113](https://github.com/casact/chainladder-python/issues/113)
  Added `reg_threshold` argument to `TailCurve` for finer control of
  fit. Huge thanks to [Brian-13](https://github.com/Brian-13) for the
  contribution.
- [\#115](https://github.com/casact/chainladder-python/issues/115)
  Introducing `VotingChainladder` workflow estimator to allow for
  averaging multiple IBNR estimators Huge thanks to
  [cbalona](https://github.com/cbalona) for the contribution.
- Introduced `CaseOutstanding` development estimator
- [\#124](https://github.com/casact/chainladder-python/issues/124)
  Standardized `CapCod` functionality to that of `Benktander` and
  `BornhuetterFerguson`

**Bug Fixes**

- [\#83](https://github.com/casact/chainladder-python/issues/83) Fixed
  `grain` issues when using the `trailing` argument
- [\#119](https://github.com/casact/chainladder-python/issues/119)
  Fixed pickle compatibility issue introduced in `v0.7.12`
- [\#120](https://github.com/casact/chainladder-python/issues/120)
  Fixed `Trend` estimator API
- [\#121](https://github.com/casact/chainladder-python/issues/121)
  Fixed numpy==1.20.1 compatibility issue
- [\#123](https://github.com/casact/chainladder-python/issues/123)
  Fixed `BerquistSherman` estimator
- [\#125](https://github.com/casact/chainladder-python/issues/125)
  Fixed various issues with compatibility of estimators in a
  `Pipeline`

**Deprecations**

- [\#118](https://github.com/casact/chainladder-python/issues/118)
  Deprecation warnings on `xlcompose`. Will be removed as a
  depdendency in v0.9.0

## Version 0.7

### Version 0.7.12

Release Date: Jan 19, 2021

No code changes from `0.7.11`. Bump release to fix conda packaging.

### Version 0.7.11

Release Date: Jan 18, 2021

**Enhancements**

- [\#110](https://github.com/casact/chainladder-python/issues/110)
  Added `virtual_column` functionality

**Bug Fixes**

- Minor bug fix on `sort_index` not accepting kwargs
- Minor bug fix in `DevelopmentConstant` when using a callable

**Misc**

- Bringing release up to parity with docs.

### Version 0.7.10

Release Date: Jan 16, 2021

**Bug Fixes**

- [\#108](https://github.com/casact/chainladder-python/issues/108)
  - `sample_weight` error handling on predict - thank you
    [cbalona](https://github.com/cbalona)
- [\#107](https://github.com/casact/chainladder-python/issues/107)
  - Latest diagonal of empty triangle now resolves
- [\#106](https://github.com/casact/chainladder-python/issues/106)
  - Improved `loc` consistency with pandas
- [\#105](https://github.com/casact/chainladder-python/issues/105)
  - Addressed broadcasting error during triangle arithmetic
- [\#103](https://github.com/casact/chainladder-python/issues/103)
  - Fixed Index alignment with triangle arithmetic consistent with
    `pd.Series`

### Version 0.7.9

Release Date: Nov 5, 2020

**Bug Fixes**

- [\#101](https://github.com/casact/chainladder-python/issues/101) Bug
  where LDF labels were not aligned with underlying LDF array

**Enhancements**

- [\#66](https://github.com/casact/chainladder-python/issues/66) Allow
  for onleveling with new `ParallelogramOLF` transformer
- [\#98](https://github.com/casact/chainladder-python/issues/98) Allow
  for more complex trends in estimators with `Trend` transformer Refer
  to this
  [example](https://chainladder-python.readthedocs.io/en/latest/auto_examples/plot_capecod_onlevel.html#sphx-glr-auto-examples-plot-capecod-onlevel-py)
  on how to apply the new estimators.

### Version 0.7.8

Release Date: Oct 22, 2020

**Bug Fixes**

- Resolved
  [\#87](https://github.com/casact/chainladder-python/issues/87)
  val_to_dev with malformed triangle

**Enhancements**

- Major overhaul of Triangle internals for better code clarity and
  more efficiency
- Made sparse operations more efficient for larger triangles
- `to_frame` now works on Triangles that are 3D or 4D. For example
  `clrd.to_frame()`
- Advanced `groupby` operations supported. For (trivial) example:

> \>\>\> clrd = cl.load_sample('clrd') \>\>\> \# Split companies with
> names less than 15 characters vs those above: \>\>\>
> clrd.groupby(clrd.index['GRNAME'].str.len()\<15).sum()

### Version 0.7.7

Release Date: Sep 13, 2020

**Enhancements**

- [\#97](https://github.com/casact/chainladder-python/issues/97), loc
  and iloc now support Ellipsis
- `Development` can now take a float value for averaging. When float
  value is used, it corresponds to weight exponent (delta in
  Barnett/Zenwirth). Only special cases had previously existed -`{"regression": 0.0, "volume": 1.0, "simple": 2.0}`
- Major improvements in slicing performance.

**Bug Fixes**

- [\#96](https://github.com/casact/chainladder-python/issues/96), Fix
  for TailBase transform
- [\#94](https://github.com/casact/chainladder-python/issues/94),
  `n_periods` with asymmetric triangles fixed

### Version 0.7.6

Release Date: Aug 26, 2020

**Enhancements**

- Four Dimensional slicing is now supported.

> \>\>\> clrd = cl.load_sample('clrd') \>\>\> clrd.iloc[[0,10, 3], 1:8,
> :5, :] \>\>\> clrd.loc[:'Aegis Grp', 'CumPaidLoss':, '1990':'1994',
> > :48]

- [\#92](https://github.com/casact/chainladder-python/issues/92)
  to_frame() now takes optional `origin_as_datetime` argument for
  better compatibility with various plotting libraries (Thank you
  [johalnes](https://github.com/johalnes) )

  \>\>\> tri.to_frame(origin_as_datetime=True)

**Bug Fixes**

- Patches to the interaction between `sparse` and `numpy` arrays to
  accomodate more scenarios.
- Patches to multi-index broadcasting
- Improved performance of `latest_diagonal` for sparse backends
- [\#91](https://github.com/casact/chainladder-python/issues/91) Bug
  fix to `MackChainladder` which errored on asymmetric triangles
  (Thank you [johalnes](https://github.com/johalnes) for reporting)

### Version 0.7.5

Release Date: Aug 15, 2020

**Enhancements**

- Enabled multi-index broadcasting.

> \>\>\> clrd = cl.load_sample('clrd') \>\>\> clrd /
> clrd.groupby('LOB').sum() \# LOB alignment works now instead of
> throwing error

- Added sparse representation of triangles which substantially increases
  the size limit of in-memory triangles. Check out the new [Large
  Datasets](https://chainladder-python.readthedocs.io/en/latest/tutorials/large-datasets.html)
  tutorial for details

**Bug Fixes**

- Fixed cupy backend which had previously been neglected
- Fixed xlcompose issue where Period fails when included as column
  header

### Version 0.7.4

Release Date: Jul 26, 2020

**Bug Fixes**

- Fixed a bug where Triangle did not support full accident dates at
  creation
- Fixed an inappropriate index mutation in Triangle index

**Enhancements**

- Added `head` and `tail` methods to Triangle
- Prepped Triangle class to support sparse backend
- Added prism sample dataset for sparse demonstrations and unit tests

### Version 0.7.3

Release Date: Jul 11, 2020

**Enhancements**

- Improved performance of `valuation` axis
- Improved performance of `groupby`
- Added `sort_index` method to `Triangle` consistent with pandas
- Allow for `fit_predict` to be called on a `Pipeline` estimator

**Bug Fixes**

- Fixed issue with Bootstrap process variance where it was being
  applied more than once
- Fixed but where Triangle.index did not ingest numeric columns
  appropriately.

### Version 0.7.2

Release Date: Jul 1, 2020

**Bug Fixes**

- Index slicing not compatible with pandas
  [\#84](https://github.com/casact/chainladder-python/issues/84) fixed
- arithmetic fail
  [\#68](https://github.com/casact/chainladder-python/issues/68)
  - Substantial reworking of how arithmetic works.
- JSON IO on sub-triangles now works
- `predict` and `fit_predict` methods added to all IBNR models and now
  function as expected

**Enhancements**

- Allow `DevelopmentConstant` to take on more than one set of patterns
  by passing in a callable
- `MunichAdjustment`Allow \` does not work when P/I or I/P ratios
  cannot be calculated. You can now optionally back-fill zero values
  with expectaton from simple chainladder so that Munich can be
  performed on sparser triangles.

**Refactors**

- Performance optimized several triangle functions including slicing
  and `val_to_dev`
- Reduced footprint of `ldf_`, `sigma`, and `std_err_` triangles
- Standardized IBNR model methods
- Changed `cdf_`, `full_triangle_`, `full_expectation_`, `ibnr_` to
  function-based properties instead of in-memory objects to reduce
  memory footprint

### Version 0.7.1

Release Date: Jun 22, 2020

**Enhancements**

- Added heatmap method to Triangle - allows for conditionally
  formatting a 2D triangle. Useful for detecting `link_ratio` outliers
- Introduced BerquistSherman estimator
- Better error messaging when triangle columns are non-numeric
- Broadened the functionality of `Triangle.trend`
- Allow for nested estimators in `to_json`. Required addition for the
  new `BerquistSherman` method
- Docs, docs, and more docs.

**Bug Fixes**

- Mixed an inappropriate mutation in
  : `MunichAdjustment.transform`

- Triangle column slicing now supports pd.Index objects
  : instead of just lists

**Misc**

- Moved `BootstrapODPSample` to workflow section as it is not a
  development estimator.

### Version 0.7.0

Release Date: Jun 2, 2020

**Bug Fixes**

- `TailBondy` now works with multiple (4D) triangles
- `TailBondy` computes correctly when `earliest_age` is selected
- Sub-triangles now honor index and column slicing of the parent.
- `fit_transform` for all tail estimators now correctly propagate all
  estimator attributes
- `Bondy` decay now uses the generalized Bondy formula instead of
  exponential decay

**Enhancements**

- Every tail estimator now has a `tail_` attribute representing the
  point estimate of the tail
- Every tail estimator how has an `attachment_age` parameter to allow
  for attachment before the end of the triangle
- `TailCurve` now has `slope_` and `intercept_` attributes for a
  diagnostics of the estimator.
- `TailBondy` now has `earliest_ldf_` attributes to allow for
  diagnostics of the estimator.
- Substantial improvement to the
  [documents](https://chainladder-python.readthedocs.io/en/latest/modules/tails.html#tails)
  on Tails.
- Introduced the deterministic components of
  [ClarkLDF](https://chainladder-python.readthedocs.io/en/latest/modules/generated/chainladder.ClarkLDF.html#chainladder.ClarkLDF)
  and
  [TailClark](https://chainladder-python.readthedocs.io/en/latest/modules/generated/chainladder.TailClark.html#chainladder.TailClark)
  estimators to allow for growth curve selection of development
  patterns.

## Version 0.6

### Version 0.6.3

Release Date: May 21, 2020

**Enhancements (courtesy of gig67)**

- Added `Triangle.calendar_correlation` method and companion class
  `CalendarCorrelation` to support detecting calendar year
  correlations in triangles.
- Added `Triangle.developmen_correlation` method and companion class
  `DevelopmentCorrelation` to support detecting development
  correlations in triangles.

### Version 0.6.2

Release Date: Apr 27, 2020

patch to 0.6.1

### Version 0.6.1

Release Date: Apr 25, 2020

**Bug Fixes**

- Corrected a bug where `TailConstant` couldn't decay when the contant
  is set to 1.0
- [\#71](https://github.com/casact/chainladder-python/issues/71) Fixed
  issue where
  \`Pipeline.predict\\ would not honor the\\ sample_weight\\ argument

**Enhancements**

- [\#72](https://github.com/casact/chainladder-python/issues/72) Added
  `drop` method to `Triangle` similar to `pd.DataFrame.drop` for
  dropping columns
- Added `xlcompose` yaml templating
- [\#74](https://github.com/casact/chainladder-python/issues/74)
  Dropped link ratios now show as ommitted when callinng `link_ratio`
  on a `Development` transformed triangle
- [\#73](https://github.com/casact/chainladder-python/issues/73)
  `Triangle.grain` now has a `trailing` argument that will aggregate
  triangle on a trailing basis

### Version 0.6.0

Release Date: Mar 17, 2020

**Enhancements**

- Added `TailBondy` method
- Propagate `std_err_` and `sigma_` on determinsitic tails in line
  with Mack for better compatibility with `MackChainladder`
- Improved consistency between `to_frame` and `__repr__` for 2D
  triangles.

**Bug Fixes**

- Fixed a bug where the latest origin period was dropped from
  `Triangle` initialization when sure data was present
- resolves
  [\#69](https://github.com/casact/chainladder-python/issues/69) where
  `datetime` was being mishandled when ingested into `Triangle`.
