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
    # This sample's paid triangle is Appendix B.1.1 (p. 98) of the CAS Tail
    # Factor Working Party's "The Estimation of Loss Development Tail Factors:
    # A Summary Report", CAS E-Forum, Fall 2013. Appendix B.4.1 (p. 103) fits
    # exponential decay to the selected factors, which are the straight
    # averages, using all nine periods and the last six, and multiplies the
    # fitted factors for periods 10 to 20 to get the tail:
    # https://www.casact.org/sites/default/files/database/forum_13fforum_02-tail-factors-working-party.pdf
    # It prints the decay rate exp(slope), the coefficient exp(intercept) and
    # the tail to three decimals. Its link ratios were not computed from the
    # whole amounts it prints (2.092 for 2001 at 12-24, where 2,712 / 1,297 =
    # 2.091), and the decay rate and coefficient of the last-six fit differ
    # from the printed ones by up to .0008, so those two are checked to .001.
    dev = cl.Development(average="simple").fit_transform(tail_sample["paid"])

    all_periods = cl.TailCurve(extrap_periods=11).fit(dev)
    assert np.exp(all_periods.slope_.iloc[0, 0]) == pytest.approx(0.623, abs=5e-4)
    assert np.exp(all_periods.intercept_.iloc[0, 0]) == pytest.approx(1.372, abs=5e-4)
    assert all_periods.tail_.iloc[0, 0] == pytest.approx(1.032, abs=5e-4)

    last_six = cl.TailCurve(fit_period=(48, None), extrap_periods=11).fit(dev)
    assert np.exp(last_six.slope_.iloc[0, 0]) == pytest.approx(0.666, abs=1e-3)
    assert np.exp(last_six.intercept_.iloc[0, 0]) == pytest.approx(0.863, abs=1e-3)
    assert last_six.tail_.iloc[0, 0] == pytest.approx(1.044, abs=5e-4)


# Section 5.3.1 of Boor's paper (cited below) uses the link ratios 1 + 0.5**k.
# Its tables show the seventh as 1.007813 or 1.007812, but its development
# portion as 0.0078125 and the log of that as -4.85203, which is ln(0.5**7).
BOOR_CONTRIVED = [1 + 0.5**k for k in range(1, 8)]
# Section 5.3.2, "a more realistic example".
BOOR_REALISTIC = [2.000, 1.250, 1.090, 1.050, 1.040, 1.030, 1.028, 1.020, 1.018]


def _triangle_from_link_ratios(link_ratios: list[float], backend: str) -> Triangle:
    """Build a paid triangle in which every origin has the given link ratios."""
    origin, valuation, paid = [], [], []
    for i in range(len(link_ratios) + 1):
        amount = 1.0
        for k, ratio in enumerate([1.0, *link_ratios[: len(link_ratios) - i]]):
            amount *= ratio
            origin.append(2000 + i)
            valuation.append(2000 + i + k)
            paid.append(amount)
    triangle = cl.Triangle(
        data={"origin": origin, "valuation": valuation, "paid": paid},
        origin="origin",
        valuation="valuation",
        columns=["paid"],
        cumulative=True,
    )
    return triangle.set_backend(backend)


@pytest.mark.parametrize("backend", ["numpy", "sparse"])
@pytest.mark.parametrize(
    ("link_ratios", "fit_period", "slope", "intercept"),
    [
        (BOOR_CONTRIVED, (None, None), -0.6931, 0.0000),
        (BOOR_REALISTIC, (None, None), -0.4415, -0.5723),
        (BOOR_REALISTIC, (48, None), -0.2073, -2.1900),
    ],
    ids=["5.3.1", "5.3.2-all-stages", "5.3.2-stages-4-to-9"],
)
def test_exponential_fit_matches_boor(
    link_ratios: list[float],
    fit_period: tuple[int | None, int | None],
    slope: float,
    intercept: float,
    backend: str,
) -> None:
    # Boor, "Estimating Tail Development Factors: What to Do When the Triangle
    # Runs Out", CAS Forum, Winter 2006, section 5.3 (pp. 366-371), regresses
    # the log of each link ratio's development portion on its stage. He prints
    # the slope and intercept to four decimals for section 5.3.1 (p. 368) and
    # for section 5.3.2, using all stages and stages 4 to 9 (pp. 370-371):
    # https://www.casact.org/sites/default/files/database/forum_06wforum_06w348.pdf
    triangle = _triangle_from_link_ratios(link_ratios, backend)
    dev = cl.Development().fit_transform(triangle)
    model = cl.TailCurve(fit_period=fit_period).fit(dev)
    assert model.slope_.iloc[0, 0] == pytest.approx(slope, abs=5e-5)
    assert model.intercept_.iloc[0, 0] == pytest.approx(intercept, abs=5e-5)


@pytest.mark.parametrize("backend", ["numpy", "sparse"])
@pytest.mark.parametrize(
    ("link_ratios", "tail", "tolerance"),
    [(BOOR_CONTRIVED, 1.00783, 5e-6), (BOOR_REALISTIC, 1.019226, 5e-7)],
    ids=["5.3.1", "5.3.2"],
)
def test_exponential_tail_matches_boor(
    link_ratios: list[float], tail: float, tolerance: float, backend: str
) -> None:
    # Boor's tails in sections 5.3.1 (p. 370) and 5.3.2 (p. 371) multiply the
    # fitted link ratios of the next fifteen stages. In section 5.3.2 the label
    # reads "8-22", but the printed 1.019226 is the product for stages 10 to 24.
    # His tail for the fit to stages 4 to 9, 1.075813, is the product for
    # stages 10 to 27, though labelled 10-24, so it is left out.
    triangle = _triangle_from_link_ratios(link_ratios, backend)
    dev = cl.Development().fit_transform(triangle)
    model = cl.TailCurve(extrap_periods=15).fit(dev)
    assert model.tail_.iloc[0, 0] == pytest.approx(tail, abs=tolerance)


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


# Mack, "The Standard Error of Chain Ladder Reserve Estimates: Recursive
# Calculation and Inclusion of a Tail Factor", ASTIN Bulletin 29(2), 1999. As a
# plausibility check, p. 364 finds k with f_(k-1) > f_ult > f_k and asks whether
# s.e.(f_(k-1)) > s.e.(f_ult) > s.e.(f_k) holds too. In the p. 365 example the
# actuary's tail of 1.05 lies between f_6 = 1.069 and f_7 = 1.026 (Table 1), and
# his tail estimates "fit well between the columns k = 6 and k = 7", where
# s.e.(f_k) is 0.023 and 0.015 and sigma_k is 74.60 and 35.49. TailConstant
# takes the paper's tail; the tail's sigma_ and std_err_ come from the
# log-linear rule in TailBase._get_tail_stats, which TailCurve also uses.
@pytest.mark.parametrize("backend", ["numpy", "sparse"])
def test_tail_stats_fall_between_mack_neighbors(backend: str) -> None:
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

    The full ``clrd`` sample has one index entry whose tail is exactly 1.0 while
    its regression coefficients are finite, so the substituted value reaches
    ``sigma_``. See #1414.

    Parameters
    ----------
    clrd: Triangle
        The clrd sample data set.

    Returns
    -------
    None
    """
    development = cl.Development().fit_transform(clrd["CumPaidLoss"])

    assert cl.options.NOMINAL_TAIL == 1.001
    baseline = cl.TailCurve().fit(development).sigma_.values.copy()

    try:
        cl.options.set_option("NOMINAL_TAIL", 1.5)
        widened = cl.TailCurve().fit(development).sigma_.values
    finally:
        cl.options.set_option("NOMINAL_TAIL", 1.001)

    assert not np.allclose(np.nan_to_num(baseline), np.nan_to_num(widened))
    assert cl.options.NOMINAL_TAIL == 1.001
