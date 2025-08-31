# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. This branch (pl_tri) is a refactor away from the existing legacy internals of the Triangle to an arrow-backed internals. These changes will require:
1. Refactoring all models (chainladder/development, chainladder/methods, chainladder/tails, chainladder/utils, chainladder/adjustments, chainladder/workflow) to only use the public API of the triangle class. Any accessing the internals of a triangle should not be permitted
2. Refactoring tests to very clearly have public API tests and rewriting the internal API tests to use the new public API
3. A close review of the /legacy code when ambiguities on Triangle functionality exist. It is permissible to write interim tests that test equality between core and legacy implementations, but these should be removed before merging to master.

## Project Overview

Chainladder is a Python package for Property and Casualty insurance loss reserving. It provides actuarial tools for triangle data manipulation, link ratios calculation, and IBNR (Incurred But Not Reported) estimates using both deterministic and stochastic models. The package mimics pandas for data manipulation and scikit-learn for model construction.

## Architecture

The package is organized into several key modules:

- `chainladder/core/` - Core triangle data structures and base classes
- `chainladder/development/` - Development pattern estimation methods (e.g., chain-ladder, Clark, Munich)
- `chainladder/methods/` - Loss reserving methods (e.g., Mack, Benktander, CapeCod)
- `chainladder/tails/` - Tail factor estimation methods
- `chainladder/adjustments/` - Data adjustments (trend, bootstrap, etc.)
- `chainladder/utils/` - Utilities and sample data sets
- `chainladder/workflow/` - Model composition tools (pipelines, voting, grid search)

## Documentation
A very good look into the public API can be obtained from our docs in `chainladder/docs`. While some of these may be out of date, they provide a good overview of the public API.


### Core Components

- **Triangle**: The fundamental data structure representing claims triangles with origin, development periods, and values
- **Base Classes**: All estimators inherit from base classes that provide common functionality following scikit-learn patterns
- **Array Backend System**: Supports multiple array backends (numpy, sparse, dask, cupy) via the `ARRAY_BACKEND` option. This is currently mostly using numpy and sparse. We want to move away from this to a polars backend.

## Development Commands

### Dependencies & Environment
- Uses `uv` for dependency management (faster alternative to pip/conda)
- Dependencies defined in `pyproject.toml`
- Install development environment: `uv sync --extra dev`
- Install test dependencies: `uv sync --extra test`

### Testing
- Test framework: pytest
- Run all tests: `uv run pytest`
- Run tests with coverage: `uv run pytest --cov=chainladder --cov-report=xml`
- Tests are organized in `tests/` subdirectories within each module
- Global test fixtures defined in `conftest.py` (sample datasets: raa, qtr, clrd, genins, prism, xyz)

### Windows Environment Notes
- This project uses `uv` for dependency management
- On Windows cmd.exe, use: `cmd /c "uv run python script.py"` instead of `cmd /c "python script.py"`
- All Python commands should be prefixed with `uv run` to use the virtual environment

### Building
- Build system: setuptools (configured in `pyproject.toml`)
- Build package: `python -m build`

## Key Patterns

### Array Backend Flexibility
The package supports multiple array backends. Code should be written to work with the active backend:
```python
import chainladder as cl
cl.options.set_option('ARRAY_BACKEND', 'sparse')  # or 'numpy', 'dask', 'cupy'
```

### Scikit-learn API Compatibility  
All estimators follow scikit-learn patterns:
- `fit()` method for training/estimation
- `transform()` or `predict()` methods for applying models
- Parameter validation and consistent interfaces

### Triangle Data Structure
The core `Triangle` class handles actuarial triangle data with:
- Index dimensions (e.g., company, line of business)
- Origin periods (accident years/quarters)
- Development periods  
- Values (loss amounts, counts, etc.)

The Triangle class mimics pandas DataFrame behavior for ease of use.

## Testing Notes

- Tests use parametrized fixtures for different array backends - these will be eliminated in favor of only a polars-based backend
- Some tests prefixed with 'r' (e.g., `rtest_*.py`) are reference tests against R implementations. These can be ignored for now.
- Test data includes various actuarial datasets in `chainladder/utils/data/`

## Refactor Goals

- The package is undergoing modernization with a new polars-based backend (`chainladder/core/base.py`) that should have minimal changes to the public API
- The refactor should first follow:
    1. Test re-write to maximize use of public API. 
    2. Temporary tests between legacy and core implementations to ensure parity
    3. Tests must touch a variety of datasets and edge cases (raa - a simply 2d triangle, qtr - a quarterly triangle, clrd - a 4d triangle with many index and column values,  prism - a large triangle with monthly data that can stress-test performance)

Refactor should initially not modify anything outside of the /core path. However, we can leverage /legacy for parity checks and /docs for better context.