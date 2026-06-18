"""Tests that guard the explicit public API of ``chainladder``.

The top-level package exposes a curated set of names via explicit imports
(see ``chainladder/__init__.py`` and each submodule's ``__all__``). These
tests pin that public surface so the refactor away from wildcard imports
cannot silently add or drop a public name, and so submodule names no longer
leak into the package namespace.
"""

from __future__ import annotations

import types

import chainladder as cl


# The intended public API of the top-level ``chainladder`` package. This is the
# union of every submodule ``__all__`` plus the package-level objects defined
# directly in ``chainladder/__init__.py``.
EXPECTED_PUBLIC_API = {
    # core
    "Triangle",
    "DevelopmentCorrelation",
    "ValuationCorrelation",
    # utils
    "WeightedRegression",
    "TriangleWeight",
    "parallelogram_olf",
    "read_csv",
    "read_pickle",
    "read_json",
    "concat",
    "load_sample",
    "list_samples",
    "minimum",
    "maximum",
    "PatsyFormula",
    "model_diagnostics",
    "cp",
    "sp",
    "dp",
    # development
    "DevelopmentBase",
    "Development",
    "MunichAdjustment",
    "IncrementalAdditive",
    "DevelopmentConstant",
    "ClarkLDF",
    "CaseOutstanding",
    "DevelopmentML",
    "TweedieGLM",
    "BarnettZehnwirth",
    # adjustments
    "BootstrapODPSample",
    "BerquistSherman",
    "ParallelogramOLF",
    "Trend",
    "TrendConstant",
    # tails
    "TailBase",
    "TailConstant",
    "TailCurve",
    "TailBondy",
    "TailClark",
    # methods
    "MethodBase",
    "Chainladder",
    "MackChainladder",
    "Benktander",
    "BornhuetterFerguson",
    "CapeCod",
    "ExpectedLoss",
    # workflow
    "GridSearch",
    "Pipeline",
    "VotingChainladder",
    # package-level objects
    "Options",
    "options",
    "version",
}


def test_public_api_present() -> None:
    """Every expected public name is importable from ``chainladder``."""
    missing = {name for name in EXPECTED_PUBLIC_API if not hasattr(cl, name)}
    assert not missing, f"Missing public names: {sorted(missing)}"


def test_no_submodule_leakage() -> None:
    """Implementation submodules no longer leak into the public namespace.

    Wildcard imports previously exposed leaf module names (e.g. ``clark``,
    ``bondy``, ``mack``, ``glm``) on the top-level package. After switching to
    explicit imports, only the subpackages themselves (bound by importing them)
    and a small set of intentionally retained helpers remain as modules.
    """
    # Leaf implementation modules that used to leak via ``import *``.
    leaked_leaf_modules = {
        "barnzehn",
        "base",
        "benktander",
        "berqsherm",
        "bondy",
        "bootstrap",
        "bornferg",
        "capecod",
        "chainladder",
        "clark",
        "common",
        "constant",
        "correlation",
        "curve",
        "display",
        "dunders",
        "expectedloss",
        "glm",
        "gridsearch",
        "incremental",
        "io",
        "learning",
        "mack",
        "munich",
        "outstanding",
        "pandas",
        "parallelogram",
        "slice",
        "triangle",
        "trend",
        "utility_functions",
        "voting",
        "weighted_regression",
    }
    still_leaked = {name for name in leaked_leaf_modules if hasattr(cl, name)}
    assert not still_leaked, (
        f"Implementation submodules leaked into the public namespace: "
        f"{sorted(still_leaked)}"
    )


def test_estimators_match_expected() -> None:
    """The public estimator/util names exactly match the expected set.

    Allows the package to also expose a few standard helper imports
    (``numpy``, ``pandas``, ``copy``) without failing, while ensuring the
    curated API itself does not drift.
    """
    public_non_modules = {
        name
        for name in dir(cl)
        if not name.startswith("_")
        and not isinstance(getattr(cl, name), types.ModuleType)
    }
    # Standard third-party/stdlib helpers that may remain visible.
    allowed_extra = {"np", "pd", "copy"}
    unexpected = public_non_modules - EXPECTED_PUBLIC_API - allowed_extra
    assert not unexpected, f"Unexpected public names: {sorted(unexpected)}"
