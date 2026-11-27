"""
Tests for TailCurve.

Sources are cited by their keys in docs/library/references.bib.

What the test suite checks for each fitted attribute, by curve:

====================  ===========  =============  =======
Attribute             exponential  inverse_power  weibull
====================  ===========  =============  =======
slope_, intercept_    published    published      no test
tail_                 published    published      no test
ldf_ before the tail  unchanged    no test        no test
====================  ===========  =============  =======

published
    Matches fits printed in CAS_TFWP2013 (Appendix B.4.1) and boor2006
    (sections 5.3 and 5.4) to their printed decimals, except two CAS_TFWP2013
    values checked to .001 for the reason given in their test.
unchanged
    Fitting the tail leaves ldf_, sigma_ and std_err_ as they were up to the
    tail.

The tail's sigma_ and std_err_ come from TailBase._get_tail_stats whatever the
curve. test_tail_stats_fall_between_mack_neighbors bounds them against
mack1999 through TailConstant; no test checks their values on a TailCurve fit.

Outside this module, the methods and workflow tests run the default
exponential curve without checking its values against a source, and five docs
notebooks run inverse_power the same way. Nothing runs weibull.
"""

from __future__ import annotations

import warnings

import numpy as np

import chainladder as cl
import pytest

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chainladder import Triangle


def test_fit_period(tail_sample: Triangle) -> None:
    dev = cl.Development(average="simple").fit_transform(tail_sample)
    assert (
        round(
            cl
            .TailCurve(fit_period=(tail_sample.ddims[-7], None), extrap_periods=10)
            .fit(dev)
            .cdf_["paid"]
            .set_backend("numpy", inplace=True)
            .values[0, 0, 0, -2],
            3,
        )
        == 1.044
    )


def test_exponential_fit_matches_tail_factor_working_party(
    tail_sample: Triangle,
) -> None:
    """
    Exponential fits to tail_sample match CAS_TFWP2013, Appendix B.4.1.

    The paid triangle is Appendix B.1.1 (p. 98). Appendix B.4.1 (p. 103) fits
    the straight-average factors using all nine periods and the last six, and
    multiplies the fitted factors for periods 10 to 20 to get the tail. It
    prints exp(slope), exp(intercept) and the tail to three decimals. Its link
    ratios were not computed from the amounts it prints, and the last-six
    exp(slope) and exp(intercept) differ from the printed ones by up to .0008,
    so those two are checked to .001.
    """
    dev = cl.Development(average="simple").fit_transform(tail_sample["paid"])

    all_periods = cl.TailCurve(extrap_periods=11).fit(dev)
    assert np.exp(all_periods.slope_.iloc[0, 0]) == pytest.approx(0.623, abs=5e-4)
    assert np.exp(all_periods.intercept_.iloc[0, 0]) == pytest.approx(1.372, abs=5e-4)
    assert all_periods.tail_.iloc[0, 0] == pytest.approx(1.032, abs=5e-4)

    last_six = cl.TailCurve(fit_period=(48, None), extrap_periods=11).fit(dev)
    assert np.exp(last_six.slope_.iloc[0, 0]) == pytest.approx(0.666, abs=1e-3)
    assert np.exp(last_six.intercept_.iloc[0, 0]) == pytest.approx(0.863, abs=1e-3)
    assert last_six.tail_.iloc[0, 0] == pytest.approx(1.044, abs=5e-4)


# boor2006, sections 5.3.1 and 5.4.1: link ratios 1 + 0.5**k. The tables show
# the seventh as 1.007813 or 1.007812, but its log development portion as
# -4.85203, which is ln(0.5**7).
BOOR_CONTRIVED = [1 + 0.5**k for k in range(1, 8)]
# boor2006, sections 5.3.2 and 5.4.2, "a more realistic example".
BOOR_REALISTIC = [2.000, 1.250, 1.090, 1.050, 1.040, 1.030, 1.028, 1.020, 1.018]


def _boor_development(link_ratios: list[float], backend: str) -> Triangle:
    """
    Return a development whose link ratios at 12, 24, ... months are Boor's.

    raa supplies only the shape. It is cut one age past the last link ratio, so
    the tail starts after Boor's last stage.
    """
    ages = [12 * k for k in range(1, len(link_ratios) + 2)]
    triangle = cl.load_sample("raa").set_backend(backend)
    triangle = triangle[triangle.development <= ages[-1]]
    patterns = dict(zip(ages, link_ratios))
    return cl.DevelopmentConstant(patterns=patterns, style="ldf").fit_transform(
        triangle
    )


def _printed(value: str) -> object:
    """Approximate a number to half a unit in its last printed decimal."""
    decimals = len(value.partition(".")[2])
    return pytest.approx(float(value), abs=0.5 * 10**-decimals)


@pytest.mark.parametrize("backend", ["numpy", "sparse"])
@pytest.mark.parametrize(
    ("curve", "link_ratios", "fit_period", "slope", "intercept"),
    [
        ("exponential", BOOR_CONTRIVED, (None, None), "-0.6931", "0.0000"),
        ("exponential", BOOR_REALISTIC, (None, None), "-0.4415", "-0.5723"),
        ("exponential", BOOR_REALISTIC, (48, None), "-0.2073", "-2.1900"),
        ("inverse_power", BOOR_CONTRIVED, (None, None), "-2.10512", "-0.20881"),
        ("inverse_power", BOOR_CONTRIVED, (48, None), "-3.69867", "2.413854"),
        ("inverse_power", BOOR_REALISTIC, (None, None), "-1.82492", "-0.18424"),
        ("inverse_power", BOOR_REALISTIC, (48, None), "-1.28108", "-1.18688"),
    ],
    ids=[
        "5.3.1",
        "5.3.2-all-stages",
        "5.3.2-stages-4-to-9",
        "5.4.1-all-stages",
        "5.4.1-stages-4-to-7",
        "5.4.2-all-stages",
        "5.4.2-stages-4-to-9",
    ],
)
def test_fit_matches_boor(
    curve: str,
    link_ratios: list[float],
    fit_period: tuple[int | None, int | None],
    slope: str,
    intercept: str,
    backend: str,
) -> None:
    """
    Slopes and intercepts match boor2006, sections 5.3 and 5.4.

    Boor regresses the log of each link ratio's development portion on its
    stage for the exponential curve (section 5.3, pp. 366-371) and on the log
    of its stage for the inverse power curve (section 5.4, pp. 373-380).
    """
    development = _boor_development(link_ratios, backend)
    model = cl.TailCurve(curve=curve, fit_period=fit_period).fit(development)
    assert model.slope_.iloc[0, 0] == _printed(slope)
    assert model.intercept_.iloc[0, 0] == _printed(intercept)


@pytest.mark.parametrize("backend", ["numpy", "sparse"])
@pytest.mark.parametrize(
    ("curve", "link_ratios", "fit_period", "extrap_periods", "tail"),
    [
        ("exponential", BOOR_CONTRIVED, (None, None), 15, "1.00783"),
        ("exponential", BOOR_REALISTIC, (None, None), 15, "1.019226"),
        ("inverse_power", BOOR_CONTRIVED, (None, None), 15, "1.056977"),
        ("inverse_power", BOOR_CONTRIVED, (48, None), 15, "1.017077"),
        ("inverse_power", BOOR_REALISTIC, (None, None), 30, "1.114487"),
        ("inverse_power", BOOR_REALISTIC, (48, None), 30, "1.208566"),
    ],
    ids=[
        "5.3.1",
        "5.3.2",
        "5.4.1-all-stages",
        "5.4.1-stages-4-to-7",
        "5.4.2-all-stages",
        "5.4.2-stages-4-to-9",
    ],
)
def test_tail_matches_boor(
    curve: str,
    link_ratios: list[float],
    fit_period: tuple[int | None, int | None],
    extrap_periods: int,
    tail: str,
    backend: str,
) -> None:
    """
    Tails match boor2006, sections 5.3 and 5.4.

    Boor multiplies the fitted link ratios of the next fifteen stages, or of
    the next thirty in section 5.4.2 (p. 377). Section 5.3.2 labels its tail
    "8-22", but the printed 1.019226 is the product for stages 10 to 24. Its
    tail for the fit to stages 4 to 9, 1.075813, is the product for stages 10
    to 27, though labelled 10-24, so it is left out. Section 5.4.2's table for
    the fit to stages 4 to 9 stops at stage 13; thirty stages give its printed
    tail.
    """
    development = _boor_development(link_ratios, backend)
    model = cl.TailCurve(
        curve=curve, fit_period=fit_period, extrap_periods=extrap_periods
    ).fit(development)
    assert model.tail_.iloc[0, 0] == _printed(tail)


@pytest.mark.parametrize("backend", ["numpy", "sparse"])
@pytest.mark.parametrize("data", ["RAA", "ABC", "GenIns", "MW2008", "MW2014"])
def test_tail_doesnt_mutate_development(data: str, backend: str) -> None:
    """Fitting a tail leaves ldf_, sigma_ and std_err_ unchanged up to the tail."""

    def develop() -> Triangle:
        triangle = cl.load_sample(data).set_backend(backend)
        development = cl.Development(average="simple", sigma_interpolation="mack")
        return development.fit_transform(triangle)

    no_tail = develop()
    tail = cl.TailCurve(curve="exponential").fit_transform(develop())
    for attribute in ["ldf_", "sigma_", "std_err_"]:
        expected = getattr(no_tail, attribute).set_backend("numpy").values
        actual = getattr(tail, attribute).set_backend("numpy").values
        np.testing.assert_array_equal(actual[..., : expected.shape[-1]], expected)


@pytest.mark.parametrize("backend", ["numpy", "sparse"])
def test_tail_stats_fall_between_mack_neighbors(backend: str) -> None:
    """
    The tail's sigma_ and std_err_ fall between the neighbours mack1999 names.

    As a plausibility check, mack1999 (p. 364) finds k with f_(k-1) > f_ult >
    f_k and asks whether s.e.(f_(k-1)) > s.e.(f_ult) > s.e.(f_k) holds too. In
    its example (p. 365, Table 1) the tail of 1.05 lies between f_6 = 1.069 and
    f_7 = 1.026, where s.e.(f_k) is 0.023 and 0.015 and sigma_k is 74.60 and
    35.49. TailConstant takes the paper's tail; the tail's sigma_ and std_err_
    come from TailBase._get_tail_stats, which TailCurve also uses.
    """
    triangle = cl.load_sample("mortgage").set_backend(backend)
    development = cl.Development(sigma_interpolation="mack").fit_transform(triangle)
    model = cl.TailConstant(1.05).fit(development)
    ldf = development.ldf_.set_backend("numpy").values.flatten()
    sigma = model.sigma_.set_backend("numpy").values.flatten()
    std_err = model.std_err_.set_backend("numpy").values.flatten()
    assert ldf[5] > 1.05 > ldf[6]
    assert 0.015 < std_err[-1] < 0.023
    assert 35.49 < sigma[-1] < 74.60


def test_curve_validation(tail_sample: Triangle) -> None:
    """
    Test validation of the curve parameter. Should raise a value error if an incorrect argument is supplied.
    """

    with pytest.raises(ValueError):
        cl.TailCurve(curve="Exponential").fit_transform(tail_sample)


def test_errors_validation(tail_sample: Triangle) -> None:
    """
    Test validation of the errors parameter. Should raise a value error if an incorrect argument is supplied.
    """
    with pytest.raises(ValueError):
        cl.TailCurve(errors="Ignore").fit_transform(tail_sample)


def test_no_log_warning_when_only_some_tails_exceed_one(clrd: Triangle) -> None:
    """
    A tail at or below 1.0 must not reach ``log(tail - 1)``.

    The nominal 1.001 used to be substituted only when the *largest* tail was
    at or below 1, so an estimator producing a mix of tails skipped the
    substitution entirely and evaluated the logarithm of a negative number.
    ``TailBondy`` on the grouped ``clrd`` sample gives 6 tails below 1 out of
    12, with a maximum of 1.018. See #1414.

    Parameters
    ----------
    clrd: Triangle
        The clrd sample data set.

    Returns
    -------
    None
    """
    triangle = clrd.groupby("LOB").sum()[["CumPaidLoss", "IncurLoss"]]
    triangle["CaseIncurredLoss"] = triangle["IncurLoss"] - triangle["CumPaidLoss"]
    development = cl.Development().fit_transform(
        triangle[["CumPaidLoss", "CaseIncurredLoss"]]
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cl.TailBondy().fit(development)

    offending = [
        w
        for w in caught
        if issubclass(w.category, RuntimeWarning)
        and "log" in str(w.message)
        and "tails/base.py" in str(w.filename).replace("\\", "/")
    ]
    assert not offending, [str(w.message) for w in offending]


def test_nominal_tail_option_is_honoured(clrd: Triangle) -> None:
    """
    ``NOMINAL_TAIL`` sets the tail substituted for a tail at or below 1.0.

    A theoretical Triangle is constructed from ``clrd`` total, such that the
    fitted tail is negative while its regression coefficients are finite, so
    the substituted value reaches ``sigma_``. See #1414.

    Parameters
    ----------
    clrd: Triangle
        The clrd sample data set.

    Returns
    -------
    None
    """
    # construct a Triangle with small positive development at earlier ages
    # and small negative development at later ages
    # clrd total incurred exhibits consistent small negative development
    tri = clrd.sum()["IncurLoss"]
    tri = tri.cum_to_incr()
    tri.iloc[..., 1:5] = -tri.iloc[..., 1:5]
    tri = tri.incr_to_cum()
    development = cl.Development().fit_transform(tri)

    assert cl.options.NOMINAL_TAIL == 1.001
    baseline = cl.TailBondy().fit(development).sigma_.values.copy()

    try:
        cl.options.set_option("NOMINAL_TAIL", 1.5)
        widened = cl.TailBondy().fit(development).sigma_.values
    finally:
        cl.options.set_option("NOMINAL_TAIL", 1.001)

    assert np.allclose(
        np.nan_to_num(baseline)[..., :-1],
        np.nan_to_num(widened)[..., :-1],
    )
    assert not np.allclose(
        np.nan_to_num(baseline)[..., -1:],
        np.nan_to_num(widened)[..., -1:],
        atol=1,
    )
    assert cl.options.NOMINAL_TAIL == 1.001
