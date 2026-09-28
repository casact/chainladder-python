import chainladder as cl
import numpy as np


def test_basic_case_outstanding():
    tri = cl.load_sample("usauto")
    m = cl.CaseOutstanding(paid_to_incurred=("paid", "incurred")).fit(tri)
    out = cl.Chainladder().fit(m.fit_transform(tri))
    a = (out.full_triangle_["incurred"] - out.full_triangle_["paid"]).iloc[
        ..., -1, :9
    ] * m.paid_ldf_.values
    b = (out.full_triangle_["paid"].cum_to_incr().iloc[..., -1, 1:10]).values
    assert (a - b).max() < 1e-6


def test_outstanding_friedland_example():
    usauto = cl.load_sample("usauto")
    model = cl.CaseOutstanding(
        paid_to_incurred=("paid", "incurred"), paid_n_periods=3, case_n_periods=3
    ).fit(usauto)

    expected_paid_ldf = np.array([
        [
            0.833,
            0.701,
            0.714,
            0.714,
            0.653,
            0.631,
            0.553,
            0.437,
            0.524,
        ]
    ])
    assert (
        model.paid_ldf_.to_frame(origin_as_datetime=False).values - expected_paid_ldf
        < 0.001
    ).all()

    expected_case_ldf = np.array([
        [
            0.526,
            0.566,
            0.528,
            0.486,
            0.511,
            0.555,
            0.652,
            0.674,
            0.580,
        ]
    ])
    assert (
        model.case_ldf_.to_frame(origin_as_datetime=False).values - expected_case_ldf
        < 0.001
    ).all()


def test_approach_2_friedland_exhibit_iii():
    import pandas as pd
    import pytest

    case_data = pd.DataFrame({
        "origin": [1998, 1999, 2000, 2001, 2002, 2003],
        "development": [132, 120, 108, 96, 84, 72],
        "case": [500000, 650000, 800000, 850000, 975000, 1000000],
    })
    case_tri = cl.Triangle(
        case_data,
        origin="origin",
        development="development",
        columns="case",
        cumulative=True,
    ).val_to_dev()

    rep_cdfs = {132: 1.015, 120: 1.020, 108: 1.030, 96: 1.051, 84: 1.077, 72: 1.131}
    paid_cdfs = {132: 1.046, 120: 1.067, 108: 1.109, 96: 1.187, 84: 1.306, 72: 1.489}

    model = cl.CaseOutstanding(
        reported_pattern=rep_cdfs,
        paid_pattern=paid_cdfs,
        style="cdf",
    ).fit(case_tri)

    assert model.approach_ == 2
    # Verify individual case CDF factors match Friedland Exhibit III
    assert np.isclose(
        model.case_cdf_.to_frame().loc["(All)", "132-Ult"], 1.506, atol=1e-3
    )
    assert np.isclose(
        model.case_cdf_.to_frame().loc["(All)", "120-Ult"], 1.454, atol=1e-3
    )
    assert np.isclose(
        model.case_cdf_.to_frame().loc["(All)", "108-Ult"], 1.421, atol=1e-3
    )
    assert np.isclose(
        model.case_cdf_.to_frame().loc["(All)", "96-Ult"], 1.445, atol=1e-3
    )
    assert np.isclose(
        model.case_cdf_.to_frame().loc["(All)", "84-Ult"], 1.439, atol=1e-3
    )
    assert np.isclose(
        model.case_cdf_.to_frame().loc["(All)", "72-Ult"], 1.545, atol=1e-3
    )

    # Chainladder projection
    cl_model = cl.Chainladder().fit(model.transform(case_tri))
    assert np.isclose(cl_model.ultimate_.sum().sum(), 7011175, rtol=1e-3)

    # Approach 1 properties should raise AttributeError in Approach 2
    with pytest.raises(AttributeError):
        _ = model.case_to_prior_case_
    with pytest.raises(AttributeError):
        _ = model.paid_to_prior_case_


def test_approach_2_with_ldf_style():
    import pandas as pd

    case_data = pd.DataFrame({
        "origin": [2001, 2002, 2003],
        "development": [36, 24, 12],
        "case": [1000, 2000, 3000],
    })
    case_tri = cl.Triangle(
        case_data,
        origin="origin",
        development="development",
        columns="case",
        cumulative=True,
    ).val_to_dev()

    # Link ratios
    rep_ldfs = {12: 1.20, 24: 1.10, 36: 1.00}
    paid_ldfs = {12: 1.50, 24: 1.25, 36: 1.05}

    model = cl.CaseOutstanding(
        reported_pattern=rep_ldfs,
        paid_pattern=paid_ldfs,
        style="ldf",
        approach=2,
    ).fit(case_tri)

    assert model.approach_ == 2
    assert hasattr(model, "case_cdf_")
    assert hasattr(model, "ldf_")


def test_approach_2_with_development_estimator():
    tri = cl.load_sample("usauto")
    case = tri["incurred"] - tri["paid"]

    rep_dev = cl.Development().fit(tri["incurred"])
    paid_dev = cl.Development().fit(tri["paid"])

    model = cl.CaseOutstanding(
        reported_pattern=rep_dev,
        paid_pattern=paid_dev,
    ).fit(case)

    assert model.approach_ == 2
    cl_model = cl.Chainladder().fit(model.transform(case))
    assert cl_model.ultimate_ is not None


def test_approach_2_validation_errors():
    import pytest

    tri = cl.load_sample("usauto")
    case = tri["incurred"] - tri["paid"]

    # Missing reported_pattern
    with pytest.raises(
        ValueError, match="requires both reported_pattern and paid_pattern"
    ):
        cl.CaseOutstanding(approach=2, paid_pattern={12: 1.5}).fit(case)

    # No common development ages
    with pytest.raises(ValueError, match="share no common development ages"):
        cl.CaseOutstanding(
            reported_pattern={12: 1.2},
            paid_pattern={24: 1.5},
            approach=2,
        ).fit(case)


def test_implied_case_development_alias():
    from chainladder.development.outstanding import ImpliedCaseDevelopment

    assert ImpliedCaseDevelopment is cl.CaseOutstanding
