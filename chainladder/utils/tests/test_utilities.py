import pytest

import chainladder as cl
import copy
import numpy as np

from chainladder import (
    __dt64_unit__
)
from chainladder.utils.utility_functions import date_delta_adjustment
from pathlib import Path




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
    Tests whether the supported sample data sets load.
    """

    # Get the folder containing the datasets.
    data_dir: Path = Path(__file__).parent.parent / 'data'

    # Files to exclude from cl.load_sample().
    files_to_excl: list = [
        '__init__'
    ]

    # Gather list of files to test.
    datasets = [f.stem for f in data_dir.iterdir() if f.is_file() and f.stem not in files_to_excl]

    # Load each file.
    for dataset in datasets:
        cl.load_sample(dataset)


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