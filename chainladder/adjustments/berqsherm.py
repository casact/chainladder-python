# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from chainladder.methods.chainladder import Chainladder
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import copy
import warnings
from chainladder.core.io import EstimatorIO


class BerquistSherman(BaseEstimator, TransformerMixin, EstimatorIO):
    """
    Class to alter the inner diagonals of a Triangle using the methods described
    by Berquist and Sherman.

    Parameters
    ----------
    paid_amount: str
        The triangle column associated with paid amounts
    incurred_amount: str
        The triangle column associated with incurred amounts
    reported_count: str
        The triangle column associated with reported claim count
    closed_count: str
        The triangle column associated with closed claim counts
    trend: float (default=0.0)
        Trend rate underlying average open case reserves.
    reported_count_estimator: Estimator
        An IBNR estimator for reported_count used to calculate closed_count
        disposal rates.  Estimator can be a Pipeline.  If None selected then
        basic Chainladder model is used.

    Attributes
    ----------
    adjusted_triangle_: Triangle
        A set of triangles represented by each simulation
    disposal_rate_: Triangle
        The disposal rates of closed claims based on the reported_count_estimator
    a_: Triangle
        Two-period Exponential intercept parameters
    b_: Triangle
        Two-period Exponential slope parameters

    Examples
    --------
    ``trend`` tilts the case-adequacy adjustment before ``Incurred`` is rebuilt;
    on the ``MedMal`` slice the inner diagonals of the adjusted ``Incurred``
    triangle restate materially between ``0%`` and ``15%`` annual drift, while
    the latest diagonal is preserved.

    .. testsetup::

        import chainladder as cl
        import numpy as np

    .. testcode::

        tri = cl.load_sample("berqsherm").loc["MedMal"]
        base = cl.BerquistSherman(
            paid_amount="Paid",
            incurred_amount="Incurred",
            reported_count="Reported",
            closed_count="Closed",
            trend=0.0,
        ).fit(tri)
        tilted = cl.BerquistSherman(
            paid_amount="Paid",
            incurred_amount="Incurred",
            reported_count="Reported",
            closed_count="Closed",
            trend=0.15,
        ).fit(tri)
        print(np.round(base.adjusted_triangle_["Incurred"], 0))

    .. testoutput::
        :options: +NORMALIZE_WHITESPACE

                      12          24          36          48          60          72          84          96
        1969   9883293.0  27420103.0  35879085.0  43105257.0  33438702.0  30397324.0  25723694.0  23506000.0
        1970   8641763.0  31305782.0  41543535.0  48550616.0  38203864.0  36222888.0  32216000.0         NaN
        1971  11733960.0  43887171.0  61649896.0  64917222.0  51410209.0  48377000.0         NaN         NaN
        1972  13638651.0  50987209.0  66696278.0  72777529.0  61163000.0         NaN         NaN         NaN
        1973  14387930.0  45470590.0  56577593.0  73733000.0         NaN         NaN         NaN         NaN
        1974  13630366.0  47189379.0  63477000.0         NaN         NaN         NaN         NaN         NaN
        1975  15036351.0  48904000.0         NaN         NaN         NaN         NaN         NaN         NaN
        1976  15791000.0         NaN         NaN         NaN         NaN         NaN         NaN         NaN

    .. testcode::

        print(np.round(tilted.adjusted_triangle_["Incurred"], 0))

    .. testoutput::
        :options: +NORMALIZE_WHITESPACE

                      12          24          36          48          60          72          84          96
        1969   3793504.0  12084942.0  18563821.0  25924316.0  23516364.0  24979245.0  24016864.0  23506000.0
        1970   3760482.0  15830500.0  24615996.0  33169802.0  30722141.0  33362729.0  32216000.0         NaN
        1971   5982185.0  25583831.0  41384825.0  50323342.0  46191356.0  48377000.0         NaN         NaN
        1972   7819355.0  33794110.0  51361061.0  64559286.0  61163000.0         NaN         NaN         NaN
        1973   9533246.0  34585431.0  49667342.0  73733000.0         NaN         NaN         NaN         NaN
        1974  10348458.0  41241243.0  63477000.0         NaN         NaN         NaN         NaN         NaN
        1975  13102479.0  48904000.0         NaN         NaN         NaN         NaN         NaN         NaN
        1976  15791000.0         NaN         NaN         NaN         NaN         NaN         NaN         NaN

    """

    def __init__(
        self,
        paid_amount=None,
        incurred_amount=None,
        reported_count=None,
        closed_count=None,
        trend=0.0,
        reported_count_estimator=None,
    ):
        self.paid_amount = paid_amount
        self.incurred_amount = incurred_amount
        self.reported_count = reported_count
        self.closed_count = closed_count
        self.trend = trend
        self.reported_count_estimator = reported_count_estimator

    def fit(self, X, y=None, sample_weight=None):
        backend = X.array_backend
        if backend == "sparse":
            obj = X.set_backend("numpy")
        else:
            obj = X.copy()
        xp = obj.get_array_module()
        if not (
            self.paid_amount in X.columns
            and self.incurred_amount in X.columns
            and self.reported_count in X.columns
            and self.closed_count in X.columns
        ):
            raise ValueError(
                "Must enter values valid columns for paid_amount, incurred_amount, reported_count and closed_count"
            )
        paid_amount = self.paid_amount
        incurred_amount = self.incurred_amount
        reported_count = self.reported_count
        closed_count = self.closed_count
        reported_count_estimator = self.reported_count_estimator
        # Case reserve adequacy adjustment
        open_count = obj[reported_count] - obj[closed_count]
        avg_case = (obj[incurred_amount] - obj[paid_amount]) / open_count
        adj_avg_case = (
            avg_case.trend(1 / (1 + self.trend) - 1, axis="valuation")
            / avg_case
            * avg_case[avg_case.valuation == avg_case.valuation_date].sum("origin")
        )
        adj_incurred_amount = adj_avg_case * open_count + obj[paid_amount]

        # Paid and closed claim adjustments
        if reported_count_estimator is None:
            reported_count_estimator = Chainladder()
        reported_count_estimator.fit(obj[reported_count])
        if reported_count_estimator.__class__.__name__ == "Pipeline":
            rep_cnt_ult = reported_count_estimator.named_steps[
                reported_count_estimator.steps[-1][0]
            ].ultimate_
        else:
            rep_cnt_ult = reported_count_estimator.ultimate_
        disposal_rate = obj[closed_count] / rep_cnt_ult
        adj_closed_clm = (
            (disposal_rate * 0 + 1)
            * disposal_rate[disposal_rate.valuation == X.valuation_date].mean("origin")
            * rep_cnt_ult
        )
        adj_closed_clm.valuation_date = disposal_rate.valuation_date = X.valuation_date

        # Two-period exponential regression
        y = obj[paid_amount]
        x = obj[closed_count]
        x0 = x.values[..., :-1]
        x1 = x.values[..., 1:]
        y0 = xp.log(y.values[..., :-1])
        y1 = xp.log(y.values[..., 1:])
        b = ((x0 * y0 + x1 * y1) / 2 - (x0 + x1) / 2 * (y0 + y1) / 2) / (
            (x0 ** 2 + x1 ** 2) / 2 - ((x0 + x1) / 2) ** 2
        )
        a = np.exp((y0 + y1) / 2 - b * (x0 + x1) / 2)

        # Need to Ignore warnings for NaN cells
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lookup = (
                np.maximum(
                    xp.concatenate(
                        [
                            (
                                adj_closed_clm.values[..., i : i + 1]
                                > obj[closed_count].values
                            ).sum(axis=-1, keepdims=True)
                            for i in range(adj_closed_clm.shape[-1])
                        ],
                        axis=-1,
                    ),
                    1,
                )
                - 1
            )

        # Don't allow lookup values beyond the final value.
        n = min(lookup.shape[-1], lookup.shape[-2])
        for j in range(n - 1):
            lookup[:, :, j, :] = np.clip(lookup[:, :, j, :], 0,  n - j - 2)

        a = (
            xp.concatenate([
                xp.concatenate(
                    [a[j:j+1, ..., i, lookup[j, 0, i:i+1, :]] for i in range(lookup.shape[-2])],
                    axis=-2
                )
                for j in range(a.shape[0]) # Process each batch independently
            ], axis=0)
        * adj_closed_clm.nan_triangle[None, None, ...]
        )
        b = (
            xp.concatenate([
                xp.concatenate(
                    [b[j:j+1, ..., i, lookup[j, 0, i:i+1, :]] for i in range(lookup.shape[-2])],
                    axis=-2
                )
                for j in range(b.shape[0]) # Process each batch independently
            ], axis=0)
        * adj_closed_clm.nan_triangle[None, None, ...]
        )
        # Adjust paids
        adj_paid_claims = adj_closed_clm * 0 + xp.exp(adj_closed_clm.values * b) * a
        adj_paid_claims = (
            y[y.valuation == y.valuation_date]
            + adj_paid_claims[
                adj_paid_claims.valuation < adj_paid_claims.valuation_date
            ]
        )

        adjusted_triangle_ = copy.deepcopy(obj)
        adjusted_triangle_[paid_amount] = adj_paid_claims
        adjusted_triangle_[incurred_amount] = adj_incurred_amount
        adjusted_triangle_[closed_count] = adj_closed_clm
        adjusted_triangle_ = adjusted_triangle_[
            adjusted_triangle_.valuation <= obj.valuation_date
        ]
        self.adjusted_triangle_ = adjusted_triangle_.set_backend(backend)
        self.disposal_rate_ = disposal_rate.set_backend(backend)
        self.a_ = a
        self.b_ = b
        return self

    def transform(self, X):
        """ If X and self are of different shapes, align self to X, else
        return self.

        Parameters
        ----------
        X: Triangle
            The triangle to be transformed

        Returns
        -------
            X_new: New triangle with transformed attributes.
        """
        X_new = copy.deepcopy(X)
        X_new[self.paid_amount] = self.adjusted_triangle_[self.paid_amount]
        X_new[self.incurred_amount] = self.adjusted_triangle_[self.incurred_amount]
        X_new[self.reported_count] = self.adjusted_triangle_[self.reported_count]
        X_new[self.closed_count] = self.adjusted_triangle_[self.closed_count]
        X_new.a_ = self.a_
        X_new.b_ = self.b_
        return X_new

    def set_params(self, **params):
        from chainladder.utils.utility_functions import read_json

        if type(params["reported_count_estimator"]) is str:
            params["reported_count_estimator"] = read_json(
                params["reported_count_estimator"]
            )
        return super().set_params(**params)
