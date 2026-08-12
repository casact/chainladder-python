
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_friedland_reference(path="friedland_ch7_part2_reference.json"):
    """Load benchmark values transcribed from the published Friedland exhibits."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _numeric_frame(obj):
    """Convert a chainladder Triangle-like object or DataFrame to a numeric DataFrame."""
    if hasattr(obj, "to_frame"):
        try:
            df = obj.to_frame(origin_as_datetime=False)
        except TypeError:
            df = obj.to_frame()
    else:
        df = pd.DataFrame(obj)
    return df.apply(pd.to_numeric, errors="coerce")


def _assert_rows(actual, expected_rows, atol, label):
    """Compare the non-missing cells in each Friedland accident-year row."""
    actual = _numeric_frame(actual)
    actual.index = actual.index.map(str)

    for year, expected in expected_rows.items():
        if year not in actual.index:
            raise AssertionError(f"{label}: accident year {year} not found.")
        observed = actual.loc[year].dropna().to_numpy(dtype=float)
        expected = np.asarray(expected, dtype=float)

        if observed.size != expected.size:
            raise AssertionError(
                f"{label}, AY {year}: expected {expected.size} displayed values "
                f"but found {observed.size}."
            )

        np.testing.assert_allclose(
            observed, expected, rtol=1e-5, atol=atol,
            err_msg=f"{label}, AY {year}"
        )


def assert_dev_exhibit(
    tri,
    devs,
    reference,
    *,
    selected_key="Selected",
    average_key="volume_5",
    dollar_atol=1.0,
    factor_atol=0.0015,
):
    """
    Validate one of Friedland's four-part development exhibits.

    Parameters
    ----------
    tri:
        The input triangle displayed in Part 1.
    devs:
        Dictionary returned by dev_exhibit().
    reference:
        One sheet from friedland_ch7_part2_reference.json.
    """
    # Part 1: cumulative claims triangle
    _assert_rows(
        tri, reference["triangle"], dollar_atol,
        "Part 1 - Data Triangle"
    )

    # Part 2: age-to-age factors
    _assert_rows(
        tri.age_to_age, reference["age_to_age"], factor_atol,
        "Part 2 - Age-to-Age Factors"
    )

    # Part 3: latest-5 volume-weighted average
    avg = _numeric_frame(devs[average_key].ldf_).to_numpy(dtype=float)
    avg = avg[np.isfinite(avg)]
    expected_avg = np.asarray(reference["latest_5_volume_weighted"], dtype=float)
    # Friedland does not print the tail factor in Part 3.
    avg = avg[: expected_avg.size]
    np.testing.assert_allclose(
        avg, expected_avg, rtol=0, atol=factor_atol,
        err_msg="Part 3 - Latest 5 volume-weighted factors"
    )

    # Part 4: selected LDF, CDF and percent developed
    selected = devs[selected_key].ldf_
    ldf = _numeric_frame(selected.round(3)).to_numpy(dtype=float)
    ldf = ldf[np.isfinite(ldf)]

    np.testing.assert_allclose(
    ldf[:len(reference["selected_ldf"])],
    reference["selected_ldf"],
    rtol=0,
    atol=factor_atol,
    err_msg="Part 4 - Selected LDF",
)

    cdf_obj = selected.incr_to_cum().round(3)

    cdf = _numeric_frame(cdf_obj).to_numpy(dtype=float)
    cdf = cdf[np.isfinite(cdf)]

    np.testing.assert_allclose(
        cdf[:len(reference["cdf_to_ultimate"])],
        reference["cdf_to_ultimate"],
        rtol=0,
        atol=factor_atol,
        err_msg="Part 4 - CDF to Ultimate",
    )

    pct = 1 / cdf
    np.testing.assert_allclose(
        pct[:len(reference["percent_developed"])],
        reference["percent_developed"],
        rtol=0, atol=factor_atol,
        err_msg="Part 4 - Percent Reported/Paid"
    )


def assert_summary_rows(
    actual,
    expected_rows,
    *,
    ratio_positions=(),
    difference_positions=(),
    dollar_atol=2.0,
    dollar_rtol=1e-5,
    difference_atol=5.0,
    factor_atol=0.0005,
    label="summary exhibit",
):
    """
    Compare a summary DataFrame to Friedland row-by-row.

    `ratio_positions` are zero-based positions containing ratios/factors
    rather than dollar values.
    """
    if "Total" in actual.index:
        actual = actual.drop(index="Total")

    actual = actual.copy()
    actual.index = actual.index.map(str)
    arr = actual.apply(pd.to_numeric, errors="coerce")

    for year, expected in expected_rows.items():
        observed = arr.loc[year].to_numpy(dtype=float)
        expected = np.asarray(expected, dtype=float)

        if observed.size != expected.size:
            raise AssertionError(
                f"{label}, AY {year}: expected {expected.size} columns "
                f"but found {observed.size}."
            )

        for j, (obs, exp) in enumerate(zip(observed, expected)):
            if j in ratio_positions:
                ok = np.isclose(
                    obs,
                    exp,
                    rtol=0,
                    atol=factor_atol,
                )
                tolerance = f"atol={factor_atol}"

            elif j in difference_positions:
                ok = np.isclose(
                    obs,
                    exp,
                    rtol=0,
                    atol=difference_atol,
                )
                tolerance = f"atol={difference_atol}"

            else:
                ok = np.isclose(
                    obs,
                    exp,
                    rtol=dollar_rtol,
                    atol=dollar_atol,
                )
                tolerance = (
                    f"atol={dollar_atol}, "
                    f"rtol={dollar_rtol}"
                )

            if not ok:
                raise AssertionError(
                    f"{label}, AY {year}, column {j}: "
                    f"actual={obs}, Friedland={exp}, "
                    f"tolerance={tolerance}"
                )

def assert_ex3_sheet1(results_upper, results_lower, ref, **kwargs):
    # Published row order:
    # EP, ratio, ultimate, reported, IBNR, ratio, ultimate, reported, IBNR
    assert_summary_rows(
        results_upper, ref["upper_rows"],
        ratio_positions=(1, 5), label="Exhibit III Sheet 1 (upper)", **kwargs
    )
    assert_summary_rows(
        results_lower, ref["lower_rows"],
        ratio_positions=(1, 5), label="Exhibit III Sheet 1 (lower)", **kwargs
    )


def assert_ex3_sheet10_or_11(results, expected_rows, **kwargs):
    assert_summary_rows(
        results,
        expected_rows,
        ratio_positions=(4, 5),
        difference_positions=(11, 12),
        label="Exhibit III Sheet 10/11",
        **kwargs,
    )


def assert_ex4_sheet1(results, expected_rows, **kwargs):
    # PP premium, Comm premium, Total premium,
    # PP ratio, Comm ratio, Total ratio,
    # PP ultimate, Comm ultimate, Total ultimate, reported, actual IBNR
    assert_summary_rows(
        results, expected_rows,
        ratio_positions=(3, 4, 5), label="Exhibit IV Sheet 1", **kwargs
    )


def assert_ex4_sheet6(results, expected_rows, **kwargs):
    assert_summary_rows(
        results, expected_rows,
        ratio_positions=(4, 5), label="Exhibit IV Sheet 6", **kwargs
    )
