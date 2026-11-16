import chainladder as cl
from sklearn.linear_model import LinearRegression, TweedieRegressor
from sklearn.pipeline import Pipeline
from chainladder.utils.utility_functions import PatsyFormula
import pandas as pd
import numpy as np


def test_incremental(genins):
    response = [genins.columns[0]]
    model = cl.DevelopmentML(
        Pipeline(
            steps=[
                ("design_matrix", PatsyFormula("C(development)")),
                ("model", LinearRegression(fit_intercept=False)),
            ]
        ),
        y_ml=response,
        fit_incrementals=False,
    ).fit(genins)
    assert abs(model.triangle_ml_.loc[:, :, "2010", :] - genins.mean()).max() < 1e2


def test_misc(genins):
    model = cl.DevelopmentML(
        Pipeline(
            steps=[
                ("design_matrix", PatsyFormula("C(development)")),
                ("model", LinearRegression(fit_intercept=False)),
            ]
        ),
        weighted_step=["model"],
        fit_incrementals=False,
    ).fit(genins, sample_weight=genins / genins)
    assert abs(model.triangle_ml_.loc[:, :, "2010", :] - genins.mean()).max() < 1e2


def test_grain():
    grains = {"Y": 2, "2Q": 4, "Q": 8, "M": 24}
    for period in grains.keys():
        tframe = pd.DataFrame()
        devdates = pd.date_range(
            start="2000-01-01", end="2002-01-01", freq=period + "S", inclusive="left"
        )
        for ind, sdate in enumerate(devdates):
            tframe = pd.concat([
                tframe,
                pd.DataFrame({
                    "origin": [sdate] * (len(devdates) - ind),
                    "development": pd.date_range(
                        start=sdate,
                        end="2002-01-01",
                        freq=period + "S",
                        inclusive="left",
                    ),
                    "loss": [100] * (len(devdates) - ind),
                }),
            ])
        tri = cl.Triangle(
            tframe,
            origin="origin",
            valuation="development",
            columns="loss",
            cumulative=False,
        )
        model = cl.DevelopmentML(
            Pipeline(
                steps=[
                    ("design_matrix", PatsyFormula("C(development)")),
                    ("model", LinearRegression(fit_intercept=False)),
                ]
            ),
            weighted_step=["model"],
            fit_incrementals=False,
        ).fit(tri)
        assert grains[period] == len(model.triangle_ml_.development)
        assert grains[period] == len(model.triangle_ml_.origin)


def test_monthly_poisson_glm_matches_chainladder():
    # A cross-classified Poisson GLM reproduces chain ladder (Renshaw and
    # Verrall, 1998), so on a monthly triangle with an exact multiplicative
    # pattern the projected square must be complete and the ultimates equal.
    n = 8
    origins = pd.date_range("2020-01-01", periods=n, freq="MS")
    pattern = 1 - 0.5 ** np.arange(1, n + 1)
    rows = [
        {
            "origin": o,
            "valuation": o + pd.DateOffset(months=k + 1) - pd.Timedelta(days=1),
            "values": 1000.0 * (1 + 0.1 * a) * pattern[k],
        }
        for a, o in enumerate(origins)
        for k in range(n - a)
    ]
    tri = cl.Triangle(
        pd.DataFrame(rows),
        origin="origin",
        valuation="valuation",
        columns="values",
        cumulative=True,
    )
    model = cl.DevelopmentML(
        Pipeline(
            steps=[
                ("design_matrix", PatsyFormula("C(development) + C(origin)")),
                (
                    "model",
                    TweedieRegressor(
                        power=1, link="log", fit_intercept=False, alpha=0, max_iter=1000
                    ),
                ),
            ]
        ),
        y_ml="values",
    ).fit(tri)
    assert not np.isnan(model.triangle_ml_.values).any()
    expected = cl.Chainladder().fit(tri).ultimate_.values
    actual = cl.Chainladder().fit(model.transform(tri)).ultimate_.values
    assert np.allclose(actual, expected, rtol=1e-4)
