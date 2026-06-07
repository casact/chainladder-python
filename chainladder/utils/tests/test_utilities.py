from __future__ import annotations
import pytest

import chainladder as cl
import copy
import numpy as np
import pandas as pd

from chainladder import (
    __dt64_unit__
)
from chainladder.utils.utility_functions import date_delta_adjustment
from chainladder.utils.data._manifest import SAMPLES
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytest import CaptureFixture
    from pytest import MonkeyPatch



def test_triangle_json_io(clrd):
    xp = clrd.get_array_module()
    clrd2 = cl.read_json(clrd.to_json(), array_backend=clrd.array_backend)
    xp.testing.assert_array_equal(clrd.values, clrd2.values)
    xp.testing.assert_array_equal(clrd.kdims, clrd2.kdims)
    xp.testing.assert_array_equal(clrd.vdims, clrd2.vdims)
    xp.testing.assert_array_equal(clrd.odims, clrd2.odims)
    xp.testing.assert_array_equal(clrd.ddims, clrd2.ddims)
    assert np.all(clrd.valuation == clrd2.valuation)


def test_json_for_val(raa):
    x = raa.dev_to_val().to_json()
    assert cl.read_json(x) == raa.dev_to_val()


def test_estimator_json_io():
    assert (
        cl.read_json(cl.Development().to_json()).get_params()
        == cl.Development().get_params()
    )


def test_pipeline_json_io():
    pipe = cl.Pipeline(
        steps=[("dev", cl.Development()), ("model", cl.BornhuetterFerguson())]
    )
    pipe2 = cl.read_json(pipe.to_json())
    assert {item[0]: item[1].get_params() for item in pipe.get_params()["steps"]} == {
        item[0]: item[1].get_params() for item in pipe2.get_params()["steps"]
    }


def test_json_subtri(raa):
    a = cl.read_json(cl.Chainladder().fit_predict(raa).to_json()).full_triangle_
    b = cl.Chainladder().fit_predict(raa).full_triangle_
    assert abs(a - b).max().max() < 1e-4


def test_json_df():
    x = cl.MunichAdjustment(paid_to_incurred=("paid", "incurred")).fit_transform(
        cl.load_sample("mcl")
    )
    assert abs(cl.read_json(x.to_json()).lambda_ - x.lambda_).sum() < 1e-5


def test_read_csv_single(raa):
    # Test the read_csv function for a single dimensional input.

    # Read in the csv file.

    raa_csv_path = Path(__file__).parent.parent / "data" / "raa.csv"

    assert raa == cl.read_csv(
        filepath_or_buffer=raa_csv_path,
        origin="origin",
        development="development",
        columns=["values"],
        index=None,
        cumulative=True,
    )


def test_read_csv_multi(clrd):
    # Test the read_csv function for multidimensional input.

    # Read in the csv file.

    clrd_csv_path = Path(__file__).parent.parent / "data" / "clrd.csv"

    assert clrd == cl.read_csv(
        filepath_or_buffer=clrd_csv_path,
        origin="AccidentYear",
        development="DevelopmentYear",
        columns=[
            "IncurLoss",
            "CumPaidLoss",
            "BulkLoss",
            "EarnedPremDIR",
            "EarnedPremCeded",
            "EarnedPremNet",
        ],
        index=["GRNAME", "LOB"],
        cumulative=True,
    )


def test_concat(clrd):
    tri = clrd.groupby("LOB").sum()
    assert (
        cl.concat([tri.loc["wkcomp"], tri.loc["comauto"]], axis=0)
        == tri.loc[["wkcomp", "comauto"]]
    )


def test_model_diagnostics(qtr):
    cl.model_diagnostics(cl.Chainladder().fit(qtr))


def test_concat_immutability(raa):
    u = cl.Chainladder().fit(raa).ultimate_
    l = raa.latest_diagonal
    u.columns = l.columns
    u_new = copy.deepcopy(u)
    cl.concat((l, u), axis=3)
    assert u == u_new


def test_invalid_sample() -> None:
    """
    Test that an invalid sample name provided to cl.load_sample() raises an error.
    """
    with pytest.raises(ValueError):
        cl.load_sample(key="not_a_real_sample_38473743")

def test_load_sample() -> None:
    """
    Tests whether every sample data set declared in the manifest loads.

    Iterating over the manifest (rather than globbing the data directory)
    means adding a new sample is a one-entry change in
    ``chainladder/utils/data/_manifest.py`` and this test picks it up
    automatically, while non-data files in the folder (``__init__.py``,
    ``_manifest.py``) are never mistaken for datasets.
    """
    # Every manifest entry must load and have a matching CSV on disk.
    data_dir: Path = Path(__file__).parent.parent / "data"
    for dataset in SAMPLES:
        assert (data_dir / f"{dataset}.csv").is_file(), (
            f"manifest lists '{dataset}' but {dataset}.csv is missing"
        )
        cl.load_sample(dataset)

    # Conversely, every CSV on disk must be declared in the manifest, so a
    # newly added data file can't silently go unregistered.
    csv_stems = {f.stem for f in data_dir.glob("*.csv")}
    assert csv_stems == set(SAMPLES), (
        "manifest and data directory are out of sync: "
        f"only in dir={csv_stems - set(SAMPLES)}, "
        f"only in manifest={set(SAMPLES) - csv_stems}"
    )


def test_list_samples() -> None:
    """
    Tests cl.list_samples(): the manifest-driven catalog of bundled datasets.
    """
    df = cl.list_samples()
    # One row per manifest entry, indexed by sample name.
    assert df.index.name == "name"
    assert set(df.index) == set(SAMPLES)
    assert {"index", "columns", "cumulative", "origin_grain", "development_grain"} <= set(df.columns)

    # The fast path skips loading data and therefore omits the grain columns.
    fast = cl.list_samples(include_grain=False)
    assert set(fast.index) == set(SAMPLES)
    assert "origin_grain" not in fast.columns
    assert "development_grain" not in fast.columns

    # Metadata matches the manifest source of truth.
    assert df.loc["clrd2025", "columns"] == SAMPLES["clrd2025"]["columns"]
    assert df.loc["prism", "cumulative"] == SAMPLES["prism"]["cumulative"]


def test_sdist_ships_all_samples(tmp_path) -> None:
    """
    Build a source distribution and assert it contains every sample CSV.

    This is the guard against MANIFEST.in drifting out of sync with the data
    folder again (the bug behind #774: the old per-file include list shipped
    only 22 of the bundled CSVs). It is deliberately self-skipping rather than
    a hard requirement of the fast suite: it needs the ``build`` package and a
    source checkout (a pyproject.toml at the repo root), and it shells out to a
    full sdist build, so it no-ops in environments that lack either.
    """
    import subprocess
    import sys
    import tarfile

    pytest.importorskip("build", reason="requires the build package")

    # Locate the repo root (the directory containing pyproject.toml). When
    # running from an installed wheel there is no source tree, so skip.
    repo_root: Path = Path(__file__).resolve().parents[3]
    if not (repo_root / "pyproject.toml").is_file():
        pytest.skip("not running from a source checkout")

    data_dir: Path = Path(__file__).parent.parent / "data"
    expected_csvs = {f.name for f in data_dir.glob("*.csv")}

    subprocess.run(
        [sys.executable, "-m", "build", "--sdist", "--outdir", str(tmp_path)],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )

    sdists = list(tmp_path.glob("*.tar.gz"))
    assert len(sdists) == 1, f"expected one sdist, found {sdists}"

    with tarfile.open(sdists[0]) as tar:
        shipped = {
            Path(name).name
            for name in tar.getnames()
            if "/utils/data/" in name and name.endswith(".csv")
        }

    missing = expected_csvs - shipped
    assert not missing, f"sdist is missing sample CSVs: {sorted(missing)}"


def test_load_sample_clrd2025() -> None:
    """
    Tests the clrd2025 sample (CAS Schedule P 1998-2007 refresh).
    """
    tri = cl.load_sample("clrd2025")

    # Six LOBs in the CAS Schedule P refresh.
    expected_lobs = {
        "comauto", "medmal", "othliab", "ppauto", "prodliab", "wkcomp"
    }
    assert set(tri.index["LOB"].unique()) == expected_lobs

    # Modern column names (IncurredLosses rather than IncurLoss).
    expected_columns = {
        "IncurredLosses", "CumPaidLoss", "BulkLoss",
        "EarnedPremDIR", "EarnedPremCeded", "EarnedPremNet",
    }
    assert set(str(c) for c in tri.vdims) == expected_columns

    # Accident years span 1998-2007.
    assert str(tri.origin.min()) == "1998"
    assert "2007" in [str(o) for o in tri.origin]

def test_date_delta_adjustment() -> None:
    """
    Tests the date adjustment depending on Pandas default precision, nanosecond for Pandas 2, microsecond for Pandas 3.
    """
    result = date_delta_adjustment("2025-11-01")

    expected = (
        "2025-10-31 23:59:59.999999999"
        if __dt64_unit__ == "ns"
        else "2025-10-31 23:59:59.999999"
    )
    assert result == expected

def test_reset_option() -> None:
    """
    Change some of the options and then reset them. Values after reset should match the original values.

    Returns
    -------
    None

    """

    original_backend = cl.options.ARRAY_BACKEND
    original_auto_sparse = cl.options.AUTO_SPARSE
    original_array_priority = cl.options.ARRAY_PRIORITY

    try:

        cl.options.set_option('ARRAY_BACKEND', 'sparse')
        cl.options.set_option('AUTO_SPARSE', False)
        cl.options.set_option('ARRAY_PRIORITY', ['sparse', 'dask', 'numpy', 'cupy'])

        cl.options.reset_option()

        assert cl.options.ARRAY_BACKEND == original_backend
        assert cl.options.AUTO_SPARSE == original_auto_sparse
        assert cl.options.ARRAY_PRIORITY == original_array_priority

    finally:
    # Manual reset in case of test failure.
        cl.options.set_option('ARRAY_BACKEND', original_backend)
        cl.options.set_option('AUTO_SPARSE', original_auto_sparse)
        cl.options.set_option('ARRAY_PRIORITY', original_array_priority)


def test_options_defaults() -> None:
    """
    When initialized, default options should be correct and accessible from the options variable.

    Returns
    -------
    None

    """
    options = cl.Options()
    assert options.ARRAY_BACKEND == "numpy"
    assert options.AUTO_SPARSE == True
    assert options.ARRAY_PRIORITY == ["dask", "sparse", "cupy", "numpy"]
    assert isinstance(options.ULT_VAL, str)


def test_get_option() -> None:
    """
    get_option should return the appropriate attribute value.

    Returns
    -------
    None

    """
    assert cl.options.get_option('ARRAY_BACKEND') == cl.options.ARRAY_BACKEND
    assert cl.options.get_option('AUTO_SPARSE') == cl.options.AUTO_SPARSE
    assert cl.options.get_option('ARRAY_PRIORITY') == cl.options.ARRAY_PRIORITY
    assert cl.options.get_option('ULT_VAL') == cl.options.ULT_VAL


def test_set_option_consistency() -> None:
    """
    When set_option changes an option value, get_option should return the new option value.

    Returns
    -------
    None

    """
    try:
        cl.options.set_option('ARRAY_BACKEND', 'sparse')
        assert cl.options.ARRAY_BACKEND == 'sparse'
        assert cl.options.get_option('ARRAY_BACKEND') == 'sparse'
    finally:
        # Reset the options to default if the test fails.
        cl.options.reset_option('ARRAY_BACKEND')

def test_reset_single_option() -> None:
    """
    Set an option and check its value, then reset it and check its value.

    Returns
    -------
    None

    """
    cl.options.set_option('ARRAY_BACKEND', 'sparse')
    assert cl.options.ARRAY_BACKEND == 'sparse'
    # Return backend to original state.
    cl.options.reset_option('ARRAY_BACKEND')
    assert cl.options.ARRAY_BACKEND == 'numpy'


def test_reset_option_invalid() -> None:
    """
    Supply in invalid option to cl.options.reset_option() and raise an error.

    Returns
    -------
    None
    """
    with pytest.raises(ValueError):
        cl.options.reset_option('NOT_A_REAL_OPTION')


def test_set_option_cupy_backend_deprecated() -> None:
    """
    Setting ARRAY_BACKEND to 'cupy' should emit a DeprecationWarning. See issue #843.

    Returns
    -------
    None
    """
    try:
        with pytest.warns(DeprecationWarning, match="cupy"):
            cl.options.set_option('ARRAY_BACKEND', 'cupy')
    finally:
        cl.options.reset_option('ARRAY_BACKEND')


def test_set_option_dask_backend_deprecated() -> None:
    """
    Setting ARRAY_BACKEND to 'dask' should emit a DeprecationWarning. See issue #842.

    Returns
    -------
    None
    """
    try:
        with pytest.warns(DeprecationWarning, match="dask"):
            cl.options.set_option('ARRAY_BACKEND', 'dask')
    finally:
        cl.options.reset_option('ARRAY_BACKEND')


def test_set_option_cupy_priority_deprecated() -> None:
    """
    Setting ARRAY_PRIORITY with 'cupy' ahead of a non-deprecated backend
    ('numpy' or 'sparse') should emit a DeprecationWarning. See issue #843.

    Returns
    -------
    None
    """
    try:
        with pytest.warns(DeprecationWarning, match="cupy"):
            cl.options.set_option('ARRAY_PRIORITY', ['cupy', 'numpy', 'sparse', 'dask'])
    finally:
        cl.options.reset_option('ARRAY_PRIORITY')


def test_set_option_dask_priority_deprecated() -> None:
    """
    Setting ARRAY_PRIORITY with 'dask' ahead of a non-deprecated backend
    ('numpy' or 'sparse') should emit a DeprecationWarning. See issue #842.

    Returns
    -------
    None
    """
    try:
        with pytest.warns(DeprecationWarning, match="dask"):
            cl.options.set_option('ARRAY_PRIORITY', ['dask', 'numpy', 'sparse', 'cupy'])
    finally:
        cl.options.reset_option('ARRAY_PRIORITY')


def test_set_option_deprecated_priority_last_no_warning(recwarn) -> None:
    """
    Setting ARRAY_PRIORITY with the deprecated backends ('cupy' and 'dask')
    ranked below every non-deprecated backend should not warn, since neither
    would ever be selected over a supported backend. See issues #842 and #843.

    Returns
    -------
    None
    """
    try:
        cl.options.set_option('ARRAY_PRIORITY', ['numpy', 'sparse', 'dask', 'cupy'])
        assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]
    finally:
        cl.options.reset_option('ARRAY_PRIORITY')


def test_set_option_supported_backend_no_warning(recwarn) -> None:
    """
    Setting a non-deprecated backend ('sparse'), and a priority list where no
    deprecated backend precedes a supported one, should not emit a
    DeprecationWarning.

    Returns
    -------
    None
    """
    try:
        cl.options.set_option('ARRAY_BACKEND', 'sparse')
        cl.options.set_option('ARRAY_PRIORITY', ['sparse', 'numpy'])
        assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]
    finally:
        cl.options.reset_option('ARRAY_BACKEND')
        cl.options.reset_option('ARRAY_PRIORITY')


def test_set_backend_cupy_deprecated(clrd) -> None:
    """
    Triangle.set_backend('cupy') should emit exactly one DeprecationWarning,
    pointing at the caller. See issue #843.

    Returns
    -------
    None
    """
    with pytest.warns(DeprecationWarning, match="cupy") as record:
        clrd.set_backend('cupy', deep=True)
    cupy_warnings = [
        w for w in record
        if issubclass(w.category, DeprecationWarning) and "cupy" in str(w.message)
    ]
    # A single warning should fire at the user's call site, not once per
    # internal recursive / deep child conversion.
    assert len(cupy_warnings) == 1
    assert cupy_warnings[0].filename == __file__


def test_set_backend_dask_deprecated(clrd) -> None:
    """
    Triangle.set_backend('dask') should emit exactly one DeprecationWarning,
    pointing at the caller. See issue #842.

    Returns
    -------
    None
    """
    with pytest.warns(DeprecationWarning, match="dask") as record:
        try:
            clrd.set_backend('dask', deep=True)
        except Exception:
            # The actual conversion can fail when the optional 'dask'
            # dependency is not installed; we only care that the deprecation
            # warning fired at the public entry point.
            pass
    dask_warnings = [
        w for w in record
        if issubclass(w.category, DeprecationWarning) and "dask" in str(w.message)
    ]
    assert len(dask_warnings) == 1
    assert dask_warnings[0].filename == __file__


def test_triangle_dask_input_deprecated() -> None:
    """
    Passing a Dask dataframe to the Triangle constructor should emit a
    DeprecationWarning. The 'dask' dependency is optional and not installed in
    the test environment, so a pandas subclass whose module is spoofed to
    'dask' is used to take the same non-pandas code path in ``_aggregate_data``
    while still supporting the pandas operations performed there. See issue
    #842.

    Returns
    -------
    None
    """
    class _FakeDaskFrame(pd.DataFrame):
        @property
        def _constructor(self):
            return _FakeDaskFrame

    # The warning is gated on the data's top-level module being 'dask', so the
    # detection mirrors a real dask dataframe (e.g. dask.dataframe.core).
    _FakeDaskFrame.__module__ = "dask.dataframe.core"

    data = _FakeDaskFrame({
        "origin": [2020, 2020, 2021],
        "development": [2020, 2021, 2021],
        "values": [100.0, 150.0, 200.0],
    })
    with pytest.warns(DeprecationWarning, match="dask") as record:
        cl.Triangle(
            data,
            origin="origin",
            development="development",
            columns="values",
        )
    dask_warnings = [
        w for w in record
        if issubclass(w.category, DeprecationWarning) and "dask" in str(w.message)
    ]
    assert len(dask_warnings) == 1
    assert dask_warnings[0].filename == __file__


def test_triangle_pandas_subclass_no_dask_warning(recwarn) -> None:
    """
    Passing a pandas subclass (not a Dask dataframe) to the Triangle
    constructor should not emit the dask DeprecationWarning, even though such
    inputs take the same non-pandas code path in ``_aggregate_data``. See
    issue #842.

    Returns
    -------
    None
    """
    class _PandasSubclass(pd.DataFrame):
        @property
        def _constructor(self):
            return _PandasSubclass

    data = _PandasSubclass({
        "origin": [2020, 2020, 2021],
        "development": [2020, 2021, 2021],
        "values": [100.0, 150.0, 200.0],
    })
    cl.Triangle(
        data,
        origin="origin",
        development="development",
        columns="values",
    )
    dask_warnings = [
        w for w in recwarn
        if issubclass(w.category, DeprecationWarning) and "dask" in str(w.message)
    ]
    assert dask_warnings == []


def test_describe_option(capsys: CaptureFixture[str]) -> None:
    """
    Supply an option to cl.options.describe_option(). Attribute name, type, default/current
    settings should be captured in the output.

    Parameters
    ----------
    capsys: CaptureFixture[str]
        pytest built-in fixture to capture stdout

    Returns
    -------
    None

    """
    cl.options.describe_option('ARRAY_BACKEND')
    captured = capsys.readouterr()
    assert 'ARRAY_BACKEND : str' in captured.out
    assert '[default: numpy]' in captured.out
    assert '[currently: numpy]' in captured.out

def test_describe_option_multi(capsys) -> None:
    """
    Supply two options to cl.options.describe_option(). Attribute names, types, default/current
    settings should be captured in the output.

    Parameters
    ----------
    capsys: CaptureFixture[str]
        pytest built-in fixture to capture stdout

    Returns
    -------
    None

    """
    cl.options.describe_option('ARRAY_BACKEND|AUTO_SPARSE')
    captured = capsys.readouterr()
    assert 'ARRAY_BACKEND : str' in captured.out
    assert '[default: numpy]' in captured.out
    assert '[currently: numpy]' in captured.out
    assert 'AUTO_SPARSE : bool' in captured.out
    assert '[default: True]' in captured.out
    assert '[currently: True]' in captured.out
    assert 'ARRAY_PRIORITY' not in captured.out


def test_describe_option_all(capsys) -> None:
    """
    Execute cl.options.describe_option() with default arguments. All attributes
    should be captured.

    Parameters
    ----------
    capsys: CaptureFixture[str]
        pytest built-in fixture to capture stdout

    Returns
    -------
    None

    """
    cl.options.describe_option()
    captured = capsys.readouterr()
    for key in cl.Options()._defaults:
        assert key in captured.out


def test_describe_option_return_string() -> None:
    """
    Execute cl.options.desribe_option() with _print_desc=False. Should return a string. Check
    if attribute info is in the string.

    Returns
    -------
    None

    """
    result = cl.options.describe_option('ARRAY_BACKEND', _print_desc=False)
    assert isinstance(result, str)
    assert 'ARRAY_BACKEND : str' in result
    assert '[default: numpy]' in result
    assert '[currently: numpy]' in result


def test_deprecated_option_kwarg_warns() -> None:
    """
    Passing option= to get_option or set_option should emit a FutureWarning.
    """
    with pytest.warns(FutureWarning, match="'option'"):
        cl.options.get_option(option='ARRAY_BACKEND')

    try:
        with pytest.warns(FutureWarning, match="'option'"):
            cl.options.set_option(option='ARRAY_BACKEND', value='numpy')
    finally:
        cl.options.reset_option('ARRAY_BACKEND')


def test_deprecated_option_kwarg_reset_option_warns() -> None:
    """
    Passing option= to reset_option should emit a FutureWarning.
    """
    try:
        cl.options.set_option('ARRAY_BACKEND', 'sparse')
        with pytest.warns(FutureWarning, match="'option'"):
            cl.options.reset_option(option='ARRAY_BACKEND')
        assert cl.options.ARRAY_BACKEND == 'numpy'
    finally:
        cl.options.reset_option('ARRAY_BACKEND')


def test_get_option_missing_pat_raises() -> None:
    """
    Calling get_option() with neither pat nor option should raise TypeError.
    """
    with pytest.raises(TypeError, match="missing required argument"):
        cl.options.get_option()


def test_describe_option_no_docstring_match(monkeypatch: MonkeyPatch) -> None:
    """
    When the class docstring has no entry for an option, describe_option should fall back
    to 'No description available.' rather than raising an error.

    Parameters
    ----------
    monkeypatch: MonkeyPatch
        The pytest built-in monkeypatch fixture.

    Returns
    -------
    None
    """
    monkeypatch.setattr(cl.Options, '__doc__', '')
    result = cl.options.describe_option('ARRAY_BACKEND', _print_desc=False)
    assert 'No description available.' in result


def test_describe_option_invalid() -> None:
    """
    Execute cl.options.desribe_option() with an invalid argument. Should raise a ValueError.

    Returns
    -------
    None

    """
    with pytest.raises(ValueError):
        cl.options.describe_option('NOT_A_REAL_OPTION')


def test_both_pat_and_option_raises() -> None:
    """
    Passing both pat and option to get_option, set_option, or reset_option should raise TypeError.
    """
    with pytest.raises(TypeError, match="Cannot specify both"):
        cl.options.get_option(pat='ARRAY_BACKEND', option='ARRAY_BACKEND')


def test_set_option_missing_value_raises() -> None:
    """
    Calling set_option with pat but no value should raise TypeError.
    """
    with pytest.raises(TypeError, match="missing required argument"):
        cl.options.set_option('ARRAY_BACKEND')


def test_describe_option_invalid_regex() -> None:
    """
    Passing a malformed regular expression to describe_option should raise ValueError.
    """
    with pytest.raises(ValueError, match="not a valid regular expression"):
        cl.options.describe_option('[')