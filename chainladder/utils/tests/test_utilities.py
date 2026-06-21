from __future__ import annotations

import pytest

import chainladder as cl
import copy
import dill
import warnings
import numpy as np
import pandas as pd

from chainladder import (
    __dt64_unit__
)
from chainladder.utils.utility_functions import date_delta_adjustment
from chainladder.utils.data._manifest import SAMPLES
from chainladder.utils.utility_functions import (
    date_delta_adjustment,
    maximum,
    minimum
)

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytest import CaptureFixture
    from pytest import MonkeyPatch
    from chainladder import Triangle

class _FakeBag:
    """
    Minimal stand-in for a dask bag that runs the mapped function eagerly.

    Lets the dask-accelerated parallel-compute paths be exercised in tests even
    though the optional 'dask' dependency is not installed. See issue #842.
    """

    def __init__(self, seq):
        self._seq = list(seq)
        self._func = None
        self._args = ()

    def map(self, func, *args):
        self._func = func
        self._args = args
        return self

    def compute(self, scheduler=None):
        return [self._func(item, *self._args) for item in self._seq]


class _FakeDaskBag:
    """Stands in for the ``dask.bag`` module in the parallel-compute paths."""

    @staticmethod
    def from_sequence(seq):
        return _FakeBag(seq)


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


def test_model_diagnostics_erorr(raa,atol):
    with pytest.raises(ValueError):
        cl.model_diagnostics(raa)
    dev = cl.Development().fit_transform(raa)
    est = cl.Chainladder().fit(dev)
    emerg = est.full_expectation_.cum_to_incr()
    md = cl.model_diagnostics(est)
    assert np.allclose(
        md['Run Off 1'].values,
        emerg[emerg.valuation.year==1991].latest_diagonal.values,
        atol=atol,
        equal_nan=True
    )
    assert np.allclose(
        md['Year Incremental'].values,
        raa.cum_to_incr().latest_diagonal.values,
        atol=atol,
        equal_nan=True
    )
    assert np.allclose(
        md['LDF'].values.flatten()[:0:-1],
        dev.ldf_.values.flatten(),
        atol=atol,
        equal_nan=True
    )
    assert np.allclose(
        md['CDF'].values.flatten()[:0:-1],
        dev.cdf_.values.flatten(),
        atol=atol,
        equal_nan=True
    )


def test_model_diagnostics_groupby(prism,atol):
    dev = cl.Development().fit(prism["Incurred"].sum())
    est = cl.Chainladder().fit(dev.transform(prism["Incurred"]))
    lhs = cl.model_diagnostics(est,groupby=['Line'])
    rhs = cl.model_diagnostics(cl.Chainladder().fit(dev.transform(prism["Incurred"].groupby('Line').sum())))
    assert np.allclose(lhs['Ultimate'].values,rhs['Ultimate'].values,atol=atol,equal_nan=True)
    assert np.allclose(np.nan_to_num(lhs['IBNR'].values),np.nan_to_num(rhs['IBNR'].values),atol=atol,equal_nan=True)


def test_concat_immutability(raa):
    u = cl.Chainladder().fit(raa).ultimate_
    l = raa.latest_diagonal
    u.columns = l.columns
    u_new = copy.deepcopy(u)
    cl.concat((l, u), axis=3)
    assert u == u_new


def test_to_pickle_read_pickle(raa):
    import tempfile
    import os
    dev = cl.Development(average="simple", n_periods=4).fit(raa)
    fd, path = tempfile.mkstemp(suffix=".pkl")
    os.close(fd)
    try:
        dev.to_pickle(path)
        restored = cl.read_pickle(path)
        assert restored.average == dev.average
        assert restored.n_periods == dev.n_periods
        np.testing.assert_array_almost_equal(
            restored.ldf_.values, dev.ldf_.values
        )
    finally:
        os.remove(path)


def test_maximum(raa):
    ult_vol = cl.Chainladder().fit(
        cl.Development(average="volume").fit_transform(raa)
    ).ultimate_
    ult_sim = cl.Chainladder().fit(
        cl.Development(average="simple").fit_transform(raa)
    ).ultimate_
    high_side = cl.maximum(ult_vol, ult_sim)
    np.testing.assert_array_almost_equal(
        high_side.values, np.maximum(ult_vol.values, ult_sim.values)
    )


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


def test_load_sample_uspp() -> None:
    """Pin the manifest column schema for the uspp Friedland family.

    Loadability of every sample is already covered by ``test_load_sample``,
    but no other test asserts the columns a sample is configured with. This
    guards the manifest entries for the uspp datasets against the regression
    where ``Earned Premium`` was dropped from their column config, in the
    same dataset-specific style as ``test_load_sample_clrd2025`` below.
    """
    for key in [
        "friedland_uspp_auto_increasing_case",
        "friedland_uspp_auto_increasing_claim",
        "friedland_uspp_auto_steady_state",
        "friedland_uspp_increasing_claim_case",
    ]:
        tri = cl.load_sample(key)
        assert set(str(c) for c in tri.vdims) == {
            "Reported Claims",
            "Paid Claims",
            "Earned Premium",
        }


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

def test_read_pickle_triangle(raa: Triangle, tmp_path: Path) -> None:
    """
    Create a triangle, dump a pickle of it, and then read it back in. The ingested pickle should result
    in an equal copy of the triangle that was dumped.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.
    tmp_path: Path
        The builtin pytest tmp_path fixture, provides a temporary path to dump the pickle to.

    Returns
    -------
    None

    """
    pkl_path = tmp_path / "triangle.pkl"
    with open(pkl_path, "wb") as f:
        dill.dump(raa, f)
    assert cl.read_pickle(str(pkl_path)) == raa


def test_triangle_to_pickle(
        raa: Triangle,
        clrd: Triangle,
        tmp_path: Path
) -> None:
    """
    Dump a pickle of a triangle and read it back in. The read-in triangle should
    equal the one that was dumped.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set Triangle.
    clrd: Triangle
        The clrd sample data set Triangle.
    tmp_path: Path
        The builtin pytest tmp_path fixture, provides a temporary path to dump the pickle to.

    Returns
    -------
    None

    """
    # Single-dimension case.
    raa_path = tmp_path / "raa.pkl"
    raa.to_pickle(str(raa_path))
    assert raa_path.is_file()
    assert cl.read_pickle(str(raa_path)) == raa

    # Multidimensional case.
    clrd_path = tmp_path / "clrd.pkl"
    clrd.to_pickle(str(clrd_path))
    assert clrd_path.is_file()
    assert cl.read_pickle(str(clrd_path)) == clrd


def test_read_pickle_estimator(raa: Triangle, tmp_path: Path) -> None:
    """
    Create an estimator, dump a pickle of it, and then read it back in. The ingested pickle should result
    produce the same LDFs that the original estimator does.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.
    tmp_path: Path
        The builtin pytest tmp_path fixture, provides a temporary path to dump the pickle to.

    Returns
    -------
    None

    """

    pkl_path = tmp_path / "estimator.pkl"
    dev = cl.Development().fit(raa)
    with open(pkl_path, "wb") as f:
        dill.dump(dev, f)
    assert dev.ldf_ == cl.read_pickle(str(pkl_path)).ldf_


def test_concat_type_error() -> None:
    """
    Supply a string to cl.concat() and raise a TypeError. Concat expects a list or tuple, so supplying a string
    will raise the error.

    Returns
    -------
    None

    """
    with pytest.raises(TypeError):
        cl.concat("not_a_list", axis=0)


def test_concat_empty_error() -> None:
    """
    Supply an empty list to cl.concat(). Trigger a ValueError since the length of the list must be greater than 0.
    Returns
    -------

    """
    with pytest.raises(ValueError):
        cl.concat([], axis=0)


def test_concat_mismatched_columns(clrd: Triangle) -> None:
    """
    Concatenate two triangles where each has a column that the other does not have. This creates new columns
    for each triangle which then get filled with xp.nan.

    Parameters
    ----------
    clrd: Triangle
        The clrd sample data set.

    Returns
    -------
    None

    """
    tri = clrd.groupby("LOB").sum()
    t1 = tri.loc["wkcomp"][["IncurLoss"]].rename("columns", ["A"])
    t2 = tri.loc["comauto"][["CumPaidLoss"]].rename("columns", ["B"])
    result = cl.concat([t1, t2], axis=0)

    # Check that new triangle has both columns.
    assert set(result.columns) == {"A", "B"}
    xp = result.get_array_module()

    # Check that each new column is filled with xp.nan corresponding
    # to the index that did not previously have the column.
    assert xp.all(xp.isnan(result.loc["wkcomp"]["B"].values))
    assert xp.all(xp.isnan(result.loc["comauto"]["A"].values))


def test_concat_sort(clrd: Triangle) -> None:
    """
    Concat two triangles with indexes in reverse alphabetical order then sort the index. Check to see
    that the index gets sorted alphabetically.

    Parameters
    ----------
    clrd: Triangle
        The clrd sample data set.

    Returns
    -------
    None

    """
    tri = clrd.groupby("LOB").sum()
    lobs_expected = ["comauto", "wkcomp"]
    result = cl.concat([tri.loc["wkcomp"], tri.loc["comauto"]], axis=0, sort=True)
    lobs = list(result.index["LOB"])
    assert lobs == lobs_expected


def test_concat_axis1(raa: Triangle) -> None:
    """
    Concat two triangles along the column axis. Check if columns are unique.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.
    Returns
    -------
    None

    """
    t1 = copy.deepcopy(raa).rename("columns", ["A"])
    t2 = copy.deepcopy(raa).rename("columns", ["B"])
    result = cl.concat([t1, t2], axis=1)
    assert set(result.columns) == {"A", "B"}


def test_concat_axis1_duplicate_columns(raa: Triangle) -> None:
    """
    Concat two triangles along the column axis with the same column names. Raise an assertion error since
    columns must be unique.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None

    """
    with pytest.raises(AssertionError):
        cl.concat([raa, raa], axis=1)


def test_maximum(raa: Triangle) -> None:
    """
    Run cl.maximum(raa, 5000) and check if each element in the resulting triangle is at least 5000.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None

    """
    result = maximum(raa, 5000)
    xp = result.get_array_module()
    assert xp.all(xp.nan_to_num(result.values, nan=5000) >= 5000)


def test_minimum(raa):
    """
    Run cl.minimum(raa, 5000) and check if each element in the resulting triangle is at most 5000.

    Parameters
    ----------
    raa: Triangle
        The raa sample data set.

    Returns
    -------
    None

    """
    result = minimum(raa, 5000)
    xp = result.get_array_module()
    assert xp.all(xp.nan_to_num(result.values, nan=5000) <= 5000)


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


def test_dask_parallel_deprecated_warns_once() -> None:
    """
    The dask-accelerated parallel-compute paths share a one-time
    DeprecationWarning helper. Calling it repeatedly in the same process should
    warn at most once, so the automatic dask paths don't flood output. See
    issue #842.

    Returns
    -------
    None
    """
    cl._dask_parallel_warned = False
    try:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            cl._warn_dask_parallel_deprecated()
            cl._warn_dask_parallel_deprecated()
        dask_warnings = [
            w for w in record
            if issubclass(w.category, DeprecationWarning) and "dask" in str(w.message)
        ]
        assert len(dask_warnings) == 1
    finally:
        cl._dask_parallel_warned = False


def test_dask_parallel_groupby_deprecated(monkeypatch: MonkeyPatch) -> None:
    """
    A groupby aggregation on a sparse-backed triangle uses the dask 'bag'
    parallel-compute path when dask is available, which should emit the dask
    DeprecationWarning. dask is optional and not installed in the test
    environment, so a fake bag stands in for it. See issue #842.

    Returns
    -------
    None
    """
    cl._dask_parallel_warned = False
    monkeypatch.setattr("chainladder.core.pandas.db", _FakeDaskBag)
    sparse_clrd = cl.load_sample("clrd").set_backend("sparse")
    try:
        with pytest.warns(DeprecationWarning, match="dask") as record:
            sparse_clrd.groupby("LOB").sum()
        dask_warnings = [
            w for w in record
            if issubclass(w.category, DeprecationWarning) and "dask" in str(w.message)
        ]
        assert len(dask_warnings) == 1
    finally:
        cl._dask_parallel_warned = False


def test_dask_parallel_incr_to_cum_deprecated(monkeypatch: MonkeyPatch) -> None:
    """
    Converting an incremental sparse-backed triangle to cumulative uses the
    dask 'bag' parallel-compute path when dask is available, which should emit
    the dask DeprecationWarning. A fake bag stands in for the optional 'dask'
    dependency. See issue #842.

    Returns
    -------
    None
    """
    cl._dask_parallel_warned = False
    monkeypatch.setattr("chainladder.core.triangle.db", _FakeDaskBag)
    incremental_sparse = cl.load_sample("raa").cum_to_incr().set_backend("sparse")
    try:
        with pytest.warns(DeprecationWarning, match="dask") as record:
            incremental_sparse.incr_to_cum()
        dask_warnings = [
            w for w in record
            if issubclass(w.category, DeprecationWarning) and "dask" in str(w.message)
        ]
        assert len(dask_warnings) == 1
    finally:
        cl._dask_parallel_warned = False


def test_dask_parallel_numpy_groupby_no_warning(
        monkeypatch: MonkeyPatch,
        recwarn,
) -> None:
    """
    The dask 'bag' parallel-compute path is gated on the sparse backend, so a
    groupby aggregation on a numpy-backed triangle should not warn even when a
    dask bag is available. See issue #842.

    Returns
    -------
    None
    """
    cl._dask_parallel_warned = False
    monkeypatch.setattr("chainladder.core.pandas.db", _FakeDaskBag)
    numpy_clrd = cl.load_sample("clrd").set_backend("numpy")
    try:
        numpy_clrd.groupby("LOB").sum()
        dask_warnings = [
            w for w in recwarn
            if issubclass(w.category, DeprecationWarning) and "dask" in str(w.message)
        ]
        assert dask_warnings == []
    finally:
        cl._dask_parallel_warned = False


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
