import numpy as np
import pandas as pd
import chainladder as cl


def test_off_grid_incremental_change_is_not_dropped():
    """A rate change dated mid-month must not vanish in monthly mode."""
    kw = dict(start_date="2000-01-01", end_date="2003-12-31", grain="Y")
    off = cl.parallelogram_olf(values=[0.20], dates=["2001-07-15"], **kw)
    # In monthly mode the change takes effect at the next grid point.
    on = cl.parallelogram_olf(values=[0.20], dates=["2001-08-01"], **kw)
    assert np.isclose(off.loc["2000", "OLF"], 1.2)
    assert np.allclose(off["OLF"].values, on["OLF"].values)


def test_off_grid_incremental_matches_cumulative():
    """Incremental and cumulative inputs agree for off-grid effective dates."""
    data = pd.DataFrame({"Year": list(range(2006, 2016)), "EarnedPremium": [1e4] * 10})
    prem_tri = cl.Triangle(
        data, origin="Year", columns="EarnedPremium", cumulative=True
    )
    dates = ["2006-01-01", "2010-07-15", "2011-02-14", "2012-06-30", "2013-04-10"]
    changes = [0.0, 0.035, 0.05, 0.10, -0.01]
    levels = np.cumprod(np.array(changes) + 1)
    factors = levels[-1] / levels
    for grain in ["M", "D"]:
        for vertical_line in [True, False]:
            kw = dict(
                date_col="EffDate",
                approximation_grain=grain,
                vertical_line=vertical_line,
            )
            incremental = cl.ParallelogramOLF(
                pd.DataFrame({"EffDate": dates, "RateChange": changes}),
                change_col="RateChange",
                **kw,
            ).fit_transform(prem_tri)
            cumulative = cl.ParallelogramOLF(
                pd.DataFrame({"EffDate": dates, "Factor": factors}),
                change_col="Factor",
                cumulative=True,
                **kw,
            ).fit_transform(prem_tri)
            assert np.allclose(incremental.olf_.values, cumulative.olf_.values), (
                grain,
                vertical_line,
            )
