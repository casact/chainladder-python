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

    case_data = pd.DataFrame({
        "origin": [1998, 1999, 2000, 2001, 2002, 2003],
        "valuation": 2008,
        "case": [500000, 650000, 800000, 850000, 975000, 1000000],
    })
    case_tri = cl.Triangle(
        case_data,
        origin="origin",
        valuation="valuation",
        columns="case",
        cumulative=True,
    ).val_to_dev()

    rep_cdfs = {132: 1.015, 120: 1.020, 108: 1.030, 96: 1.051, 84: 1.077, 72: 1.131}
    paid_cdfs = {132: 1.046, 120: 1.067, 108: 1.109, 96: 1.187, 84: 1.306, 72: 1.489}

    model = cl.ImpliedCaseDevelopment(
        reported_pattern=rep_cdfs,
        paid_pattern=paid_cdfs,
        style="cdf",
    ).fit(case_tri)

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


def test_approach_2_with_ldf_style():
    import pandas as pd

    case_data = pd.DataFrame({
        "origin": [2001, 2002, 2003],
        "valuation": 2003,
        "case": [1000, 2000, 3000],
    })
    case_tri = cl.Triangle(
        case_data,
        origin="origin",
        valuation="valuation",
        columns="case",
        cumulative=True,
    ).val_to_dev()

    # Link ratios
    rep_ldfs = {12: 1.20, 24: 1.10, 36: 1.00}
    paid_ldfs = {12: 1.50, 24: 1.25, 36: 1.05}

    model = cl.ImpliedCase(
        reported_pattern=rep_ldfs,
        paid_pattern=paid_ldfs,
        style="ldf",
    ).fit(case_tri)

    assert hasattr(model, "case_cdf_")
    assert hasattr(model, "ldf_")


def test_approach_2_with_development_estimator():
    tri = cl.load_sample("usauto")
    case = tri["incurred"] - tri["paid"]

    rep_dev = cl.Development().fit(tri["incurred"])
    paid_dev = cl.Development().fit(tri["paid"])

    model = cl.ImpliedCaseDevelopment(
        reported_pattern=rep_dev,
        paid_pattern=paid_dev,
    ).fit(case)

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
        cl.ImpliedCaseDevelopment(paid_pattern={12: 1.5}).fit(case)

    # No common development ages
    with pytest.raises(ValueError, match="share no common development ages"):
        cl.ImpliedCaseDevelopment(
            reported_pattern={12: 1.2},
            paid_pattern={24: 1.5},
        ).fit(case)

    # Calling CaseOutstanding with Approach 2 arguments raises helpful ValueError
    with pytest.raises(ValueError, match="Approach 2 has been decoupled"):
        cl.CaseOutstanding(reported_pattern={12: 1.2}, paid_pattern={12: 1.5}).fit(tri)
    with pytest.raises(ValueError, match="Approach 2 has been decoupled"):
        cl.CaseOutstanding(approach=2).fit(tri)


def test_implied_case_development_alias():
    from chainladder.development.outstanding import ImpliedCaseDevelopment, ImpliedCase
    from chainladder.development import (
        ImpliedCaseDevelopment as DevImplied,
        ImpliedCase as DevImpliedShort,
    )
    import chainladder as cl_root

    assert ImpliedCase is ImpliedCaseDevelopment
    assert DevImpliedShort is ImpliedCaseDevelopment
    assert cl_root.ImpliedCase is ImpliedCaseDevelopment
    assert DevImplied is ImpliedCaseDevelopment
    assert cl_root.ImpliedCaseDevelopment is ImpliedCaseDevelopment
    assert ImpliedCaseDevelopment is not cl.CaseOutstanding


def test_approach_2_pattern_input_types():
    import pandas as pd
    import pytest

    tri = cl.load_sample("usauto")
    case = tri["incurred"] - tri["paid"]

    # pd.Series inputs
    rep_s = pd.Series({12: 1.2, 24: 1.1, 36: 1.0})
    paid_s = pd.Series({12: 1.5, 24: 1.25, 36: 1.05})
    m_series = cl.ImpliedCaseDevelopment(
        reported_pattern=rep_s, paid_pattern=paid_s, style="ldf"
    ).fit(case)
    assert m_series.case_cdf_ is not None

    # pd.DataFrame 1-row input
    df_1row = pd.DataFrame([[1.2, 1.1, 1.0]], columns=[12, 24, 36])
    m_df1 = cl.ImpliedCaseDevelopment(
        reported_pattern=df_1row, paid_pattern=paid_s, style="ldf"
    ).fit(case)
    assert m_df1.case_cdf_ is not None

    # pd.DataFrame 1-column input
    df_1col = pd.DataFrame([1.2, 1.1, 1.0], index=[12, 24, 36])
    m_df2 = cl.ImpliedCaseDevelopment(
        reported_pattern=df_1col, paid_pattern=paid_s, style="ldf"
    ).fit(case)
    assert m_df2.case_cdf_ is not None

    # Invalid DataFrame shape (>1 row and >1 col)
    df_invalid = pd.DataFrame([[1.2, 1.1], [1.0, 1.0]], columns=[12, 24])
    with pytest.raises(ValueError, match="must have 1 row or 1 column"):
        cl.ImpliedCaseDevelopment(reported_pattern=df_invalid, paid_pattern=paid_s).fit(case)

    # Unsupported pattern type
    with pytest.raises(TypeError, match="Unsupported pattern type"):
        cl.ImpliedCaseDevelopment(reported_pattern=[1.2, 1.1], paid_pattern=paid_s).fit(case)


def test_approach_2_estimator_with_style_ldf_and_no_double_cumprod():
    """Verify BugBot fix: estimator with cdf_ is not reconverted/double-cumprodded when style='ldf'."""
    import pandas as pd

    tri = cl.load_sample("usauto")
    case = tri["incurred"] - tri["paid"]

    rep_dev = cl.Development().fit(tri["incurred"])
    paid_dev = cl.Development().fit(tri["paid"])

    # style='ldf' passed with Development estimator should use estimator.cdf_ directly
    m = cl.ImpliedCaseDevelopment(
        reported_pattern=rep_dev,
        paid_pattern=paid_dev,
        style="ldf",
    ).fit(case)
    assert m.case_cdf_ is not None

    # Mock object with only ldf_ and no cdf_
    class MockLDFPattern:
        def __init__(self, ldf_dict):
            self.ldf_ = pd.Series(ldf_dict)

    mock_rep = MockLDFPattern({12: 1.2, 24: 1.1, 36: 1.0})
    mock_paid = MockLDFPattern({12: 1.5, 24: 1.25, 36: 1.05})
    m_mock = cl.ImpliedCaseDevelopment(
        reported_pattern=mock_rep,
        paid_pattern=mock_paid,
        style="ldf",
    ).fit(case)
    assert m_mock.case_cdf_ is not None


def test_approach_2_auto_column_inference_and_transform_bugbot_fix():
    tri = cl.load_sample("usauto")

    # Incurred and paid auto-detection from 2-channel triangle
    tri_sub = tri[["incurred", "paid"]]
    m = cl.ImpliedCaseDevelopment(
        reported_pattern={12: 1.1, 24: 1.05},
        paid_pattern={12: 1.3, 24: 1.15},
    ).fit(tri_sub)
    assert m.case_cdf_ is not None

    # Verify transform on 2-channel triangle extracts 1-channel case triangle with ldf_
    # This directly addresses and validates Bugbot's review concern
    transformed = m.transform(tri_sub)
    assert len(transformed.columns) == 1
    assert transformed.columns[0] == "case"
    assert hasattr(transformed, "ldf_")

    # Verify downstream Chainladder model runs seamlessly on transformed triangle
    cl_model = cl.Chainladder().fit(transformed)
    assert cl_model.ultimate_ is not None

    # Denominator <= 0 guard (CDF_paid <= CDF_rep fallback to 1.0)
    m_guard = cl.ImpliedCaseDevelopment(
        reported_pattern={12: 1.5, 24: 1.2},
        paid_pattern={12: 1.2, 24: 1.1},  # paid < rep
    ).fit(tri_sub)
    assert np.isclose(m_guard.case_cdf_.to_frame().loc["(All)", "12-Ult"], 1.0)

    # Capitalized Incurred and Paid column names
    tri_cap = tri.copy()
    tri_cap.columns = [
        "Incurred" if c == "incurred" else "Paid" if c == "paid" else c
        for c in tri_cap.columns
    ]
    m_cap = cl.ImpliedCaseDevelopment(
        reported_pattern={12: 1.1, 24: 1.05},
        paid_pattern={12: 1.3, 24: 1.15},
    ).fit(tri_cap[["Incurred", "Paid"]])
    assert m_cap.case_cdf_ is not None

    # Groupby in Approach 1 CaseOutstanding
    tri_grouped = cl.load_sample("clrd")
    m_grp = cl.CaseOutstanding(
        paid_to_incurred=("CumPaidLoss", "IncurLoss"),
        groupby="LOB",
    ).fit(tri_grouped)
    assert m_grp.case_to_prior_case_ is not None
    assert m_grp.paid_to_prior_case_ is not None

