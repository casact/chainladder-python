import chainladder as cl
import pytest
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
        assert not np.isnan(model.triangle_ml_.values).any()


GRAINS = [
    "OMDM",
    "OQDM",
    "OSDM",
    "OYDM",
    "OQDQ",
    "OSDQ",
    "OYDQ",
    "OSDS",
    "OYDS",
    "OYDY",
]


@pytest.fixture
def exact_monthly():
    # Two years of monthly origins whose cumulative values follow one
    # multiplicative development pattern, so every grain has at least two
    # origins and a full upper triangle.
    n = 24
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
    return cl.Triangle(
        pd.DataFrame(rows),
        origin="origin",
        valuation="valuation",
        columns="values",
        cumulative=True,
    )


# newton-cholesky converges in about ten iterations here, where lbfgs needs
# thousands. Its gradient step emits a matmul RuntimeWarning, and the fit
# still matches chain ladder at the tolerance below.
@pytest.mark.filterwarnings("ignore:invalid value encountered in matmul:RuntimeWarning")
@pytest.mark.parametrize("grain", GRAINS)
def test_poisson_glm_matches_chainladder(exact_monthly, grain):
    # A cross-classified Poisson GLM reproduces chain ladder (Renshaw and
    # Verrall, 1998, renshaw1998 in docs/library/references.bib), so at every
    # grain the projected square must be complete and its chain ladder
    # ultimates must equal the ultimates of chain ladder on the triangle.
    tri = exact_monthly.grain(grain)
    model = cl.DevelopmentML(
        Pipeline(
            steps=[
                ("design_matrix", PatsyFormula("C(development) + C(origin)")),
                (
                    "model",
                    TweedieRegressor(
                        power=1,
                        link="log",
                        fit_intercept=False,
                        alpha=0,
                        solver="newton-cholesky",
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


@pytest.mark.parametrize("grain", GRAINS)
def test_projection_fills_future_cells(monthly, grain):
    # DevelopmentML projects each origin forward from the latest diagonal, so
    # every future cell of triangle_ml_ must be filled at every grain,
    # including grains whose origin and development periods differ.
    tri = monthly["reportedCount"].grain(grain)
    ml = cl.TweedieGLM(response="reportedCount").fit(tri).triangle_ml_
    values = ml.set_backend("numpy").values[0, 0]
    future = np.isnan(tri.set_backend("numpy").nan_triangle)
    assert values.shape == future.shape
    assert not np.isnan(values[future]).any()
