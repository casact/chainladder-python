# Architecture

## 1. Directory Structure

```
chainladder-python/
├── chainladder/                    # Main package
│   ├── __init__.py                 # Public API, global options, sample data loader
│   │
│   ├── core/                       # Triangle data structure
│   │   ├── triangle.py             # Triangle (the public-facing class)
│   │   ├── base.py                 # TriangleBase (assembles all mixins)
│   │   ├── common.py               # Common (shared helpers used by Triangle and estimators)
│   │   ├── dunders.py              # TriangleDunders (arithmetic, comparison operators)
│   │   ├── pandas.py               # TrianglePandas, TriangleGroupBy (pandas-style API)
│   │   ├── slice.py                # TriangleSlicer, Location, Ilocation, At, Iat, VirtualColumns
│   │   ├── display.py              # TriangleDisplay (__repr__, _repr_html_)
│   │   ├── io.py                   # TriangleIO, EstimatorIO (pickle, JSON, spreadsheet I/O)
│   │   ├── typing.py               # TriangleProtocol, type aliases (BackendArray, etc.)
│   │   └── tests/
│   │
│   ├── development/                # Development pattern estimators
│   │   ├── base.py                 # DevelopmentBase (shared fit/transform logic)
│   │   ├── development.py          # Development (weighted LDF, volume/simple/regression)
│   │   ├── constant.py             # DevelopmentConstant (user-supplied LDFs)
│   │   ├── incremental.py          # IncrementalAdditive
│   │   ├── munich.py               # MunichAdjustment (paid/incurred correlation)
│   │   ├── clark.py                # ClarkLDF (growth-curve LDFs)
│   │   ├── glm.py                  # TweedieGLM
│   │   ├── barnzehn.py             # BarnettZehnwirth (extends TweedieGLM)
│   │   ├── learning.py             # DevelopmentML (sklearn regressor wrapper)
│   │   ├── outstanding.py          # CaseOutstanding
│   │   └── tests/
│   │
│   ├── tails/                      # Tail factor estimators
│   │   ├── base.py                 # TailBase (extends DevelopmentBase)
│   │   ├── constant.py             # TailConstant
│   │   ├── curve.py                # TailCurve (exponential/inverse-power extrapolation)
│   │   ├── bondy.py                # TailBondy
│   │   ├── clark.py                # TailClark
│   │   └── tests/
│   │
│   ├── methods/                    # Reserve estimation methods
│   │   ├── base.py                 # MethodBase (fit/predict contract)
│   │   ├── chainladder.py          # Chainladder
│   │   ├── mack.py                 # MackChainladder (extends Chainladder)
│   │   ├── benktander.py           # Benktander
│   │   ├── bornferg.py             # BornhuetterFerguson (extends Benktander)
│   │   ├── capecod.py              # CapeCod (extends Benktander)
│   │   ├── expectedloss.py         # ExpectedLoss (extends Benktander)
│   │   └── tests/
│   │
│   ├── adjustments/                # Pre-processing transformers
│   │   ├── berqsherm.py            # BerquistSherman
│   │   ├── bootstrap.py            # BootstrapODPSample (extends DevelopmentBase)
│   │   ├── parallelogram.py        # ParallelogramOLF
│   │   ├── trend.py                # Trend, TrendConstant
│   │   └── tests/
│   │
│   ├── workflow/                   # Pipeline and ensemble utilities
│   │   ├── gridsearch.py           # GridSearch, Pipeline
│   │   ├── voting.py               # VotingChainladder
│   │   └── tests/
│   │
│   ├── utils/                      # Internal utilities
│   │   ├── sparse.py               # COO sparse array wrapper
│   │   ├── cupy.py                 # CuPy GPU array shim
│   │   ├── dask.py                 # Dask parallel shim
│   │   ├── utility_functions.py    # num_to_nan, set_common_backend, etc.
│   │   ├── weighted_regression.py  # WeightedRegression helper
│   │   ├── triangle_weight.py      # Triangle weighting helpers
│   │   ├── data/                   # Bundled sample datasets (CSV)
│   │   └── tests/
│   │
│   └── tests/
│       └── test_public_api.py      # Smoke-tests for the public API surface
│
├── docs/                           # JupyterBook/Sphinx documentation source
├── conftest.py                     # pytest fixtures (raa, clrd, qtr, …)
├── pyproject.toml                  # Development environment
└── pyrightconfig.json              # Type checking configuration
```

## 2. Inheritance Diagrams

### 2a. Triangle

`Triangle` is assembled from a stack of single-responsibility mixins. Python resolves methods left-to-right across the MRO (method resolution order), so the order in `TriangleBase` determines which mixin wins on any name collision.

```
object
  └── ABC
  └── Common                 core/common.py     — backend switching, grain helpers, valuation utilities
  └── TrianglePandas         core/pandas.py     — pandas-style API (to_frame, groupby, rename, …)
  └── TriangleDunders        core/dunders.py    — arithmetic & comparison operators
  └── TriangleSlicer         core/slice.py      — .loc / .iloc / .at / .iat / virtual columns
  └── TriangleDisplay        core/display.py    — __repr__ / _repr_html_
  └── TriangleIO             core/io.py         — to_pickle / to_json / to_excel / …
        │
        └── TriangleBase     core/base.py       — __init__, array allocation, grain/dim logic
              │               (inherits all of the above as direct bases)
              └── Triangle   core/triangle.py   — public class; thin layer, exposes full API
```

Supporting classes owned by the slice layer:

```
_LocBase
  ├── Location   (.loc accessor)
  │     └── At  (.at accessor)
  └── Ilocation  (.iloc accessor)
        └── Iat  (.iat accessor)

VirtualColumns     (lazy computed-column registry attached to Triangle)
TriangleGroupBy    (returned by Triangle.groupby(); consumed by arithmetic operators)
```

These classes are related to `Triangle` by **composition**, not inheritance. `Triangle` does not extend `Location` or `Ilocation` — instead, each `Triangle` *instance* holds references to instances of these classes as its own attributes. `TriangleSlicer._set_slicers()` (called during `Triangle.__init__` and whenever the index or column shape changes) creates fresh instances and assigns them:

```python
# core/slice.py
class TriangleSlicer:
    def _set_slicers(self) -> None:
        self.iloc = Ilocation(self)
        self.loc = Location(self)
        self.iat = Iat(self)
        self.at = At(self)
        self.virtual_columns = VirtualColumns(self, self.virtual_columns.columns)
```

Each accessor receives a reference back to the owning `Triangle` (`self`), which is how `triangle.iloc[0]` can read and slice the triangle's underlying arrays. The relationship is:

```
Triangle instance
  ├── .iloc  →  Ilocation instance  (wraps the same Triangle)
  │                 └── [0]  calls back into Triangle to produce a sliced copy
  ├── .loc   →  Location instance
  ├── .iat   →  Iat instance
  ├── .at    →  At instance
  └── .virtual_columns  →  VirtualColumns instance
```

#### Type-hinting the mixin stack

`core/typing.py` defines `TriangleProtocol`, a `typing.Protocol` that declares the interface every mixin assumes will be present on `self` - properties such as `shape`, `index`, `values`, `array_backend`, aggregation methods such as `sum`, and the indexer attributes `iloc`, `loc`, `at`, `iat`.

**The problem with direct Protocol inheritance at runtime**

A mixin that inherits from `TriangleProtocol` at runtime places the Protocol's stub descriptors into the MRO ahead of any concrete implementations provided by other mixins or `Triangle` itself. For example, if `TrianglePandas(TriangleProtocol)` were written literally, then `TriangleProtocol.set_backend` (a stub that returns `...`) would shadow `Common.set_backend` (the real implementation), causing `AttributeError` at runtime. The same applies to any Protocol stub that happens to match a name defined elsewhere in the mixin stack.

**The adopted pattern: `TYPE_CHECKING` conditional base**

Each mixin that needs `TriangleProtocol` for type-checking purposes declares its base class conditionally so that the Protocol is visible to Pyright/Pylance but is replaced by `object` at runtime:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chainladder.core.typing import TriangleProtocol
    _MixinBase = TriangleProtocol
else:
    _MixinBase = object

class TriangleMixin(_MixinBase):
    # Pyright sees TriangleProtocol as the base — self has .shape, .values, .sum, etc.
    # At runtime the base is object — no Protocol stubs in the MRO.
    ...
```

This is the pattern recommended by the Protocols page in the Python documentation.
[Explicitly declaring implementation](https://typing.python.org/en/latest/spec/protocol.html#explicitly-declaring-implementation).

**Type-hinting methods that accept a Triangle-like object and return `Triangle`**

Use `TriangleProtocol` for inputs and `Triangle` for the concrete return type. Import both under `TYPE_CHECKING` to keep the runtime import-free:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chainladder import Triangle
    from chainladder.core.typing import TriangleProtocol

def transform(X: TriangleProtocol) -> Triangle:
    ...
```

- **Input typed as `TriangleProtocol`**: accepts any object that structurally satisfies the protocol (a real `Triangle`, a mock in tests, a future subclass) without requiring a concrete import.
- **Return typed as `Triangle`**: callers know they get the fully-featured concrete class with all mixin methods available, not just the minimal protocol surface.

Both imports live inside `if TYPE_CHECKING:`, so they are never executed at runtime and cannot create circular-import cycles.  The `from __future__ import annotations` at the top of the file makes all annotations lazy strings, which means the names are never resolved at import time even if `TYPE_CHECKING` is `False`.

---

### 2b. Estimators

All estimators follow the scikit-learn `BaseEstimator` / `TransformerMixin` / `fit` / `transform` / `predict` contract.

```
sklearn.BaseEstimator
  │
  ├── EstimatorIO              core/io.py          — to_pickle / to_json for fitted estimators
  │
  └── DevelopmentBase          development/base.py — shared LDF fitting helpers
        │
        ├── Development        development/development.py   — weighted LDF (volume/simple/regression)
        ├── DevelopmentConstant development/constant.py
        ├── IncrementalAdditive development/incremental.py
        ├── MunichAdjustment   development/munich.py
        ├── ClarkLDF           development/clark.py
        ├── DevelopmentML      development/learning.py
        ├── CaseOutstanding    development/outstanding.py
        ├── TweedieGLM         development/glm.py
        │     └── BarnettZehnwirth  development/barnzehn.py
        ├── BootstrapODPSample adjustments/bootstrap.py
        │
        └── TailBase           tails/base.py       — appends tail column, extends DevelopmentBase
              ├── TailConstant tails/constant.py
              ├── TailCurve    tails/curve.py
              ├── TailBondy    tails/bondy.py
              └── TailClark    tails/clark.py


sklearn.BaseEstimator
  └── Common                   core/common.py
  └── EstimatorIO              core/io.py
        └── MethodBase         methods/base.py     — fit/predict contract for reserve methods
              │
              ├── Chainladder  methods/chainladder.py
              │     └── MackChainladder  methods/mack.py
              │
              └── Benktander   methods/benktander.py
                    ├── BornhuetterFerguson  methods/bornferg.py
                    ├── CapeCod              methods/capecod.py
                    └── ExpectedLoss         methods/expectedloss.py


sklearn.BaseEstimator + TransformerMixin + EstimatorIO   (standalone transformers)
  ├── BerquistSherman     adjustments/berqsherm.py
  ├── ParallelogramOLF    adjustments/parallelogram.py
  ├── Trend               adjustments/trend.py
  └── TrendConstant       adjustments/trend.py


sklearn.BaseEstimator
  └── GridSearch          workflow/gridsearch.py
  └── Pipeline            workflow/gridsearch.py   (extends sklearn Pipeline + EstimatorIO)


MethodBase
  └── VotingChainladder   workflow/voting.py       (ensemble of MethodBase estimators)
```
