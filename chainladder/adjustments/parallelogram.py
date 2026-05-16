# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from sklearn.base import BaseEstimator, TransformerMixin
from chainladder.core.io import EstimatorIO


class ParallelogramOLF(BaseEstimator, TransformerMixin, EstimatorIO):
    """
    Estimator to create and apply on-level factors to a Triangle object. This
    is commonly used for premium vectors expressed as a Triangle object.

    Parameters
    ----------

    rate_history: pd.DataFrame
        A DataFrame with two columns: one containing the effective dates of the rate
        changes and the other containing the rate changes expressed as a decimal.
        For example, 5% decrease should be stated as -0.05.
    change_col: str
        The column containing the rate changes expressed as a decimal. For example,
        5% decrease should be stated as -0.05.
    date_col: str
        A list-like set of effective dates corresponding to each of the changes.
    approximation_grain: str {"M", "D"} (default="M")
        The resolution of the internal calendar spacing used to calculate on-level
        factors can be set to monthly (`'M'`) or daily (`'D'`). Under each
        `approximation_grain`, periods are treated as discrete intervals and a
        weighted current rate level is estimated. In monthly mode, each month is
        treated as an equal-length period, consistent with the methodology presented
        in the Friedland text, although this assumes that all months within a year
        contain the same number of days. In daily mode, each calendar day is treated
        as a full period, providing finer granularity and more accurately accounting
        for differences in month length and leap years when assigning factors to
        origin periods.
    policy_length: int (default=12)
        The length of the policy in months.
    vertical_line: bool (default=False)
        Rates are typically stated on an effective date basis and premiums on
        and earned basis.  By default, this argument is False and produces
        parallelogram OLFs. If True, Parallelograms become squares.  This is
        commonly seen in Workers Compensation with benefit on-leveling or if
        the premium origin is also stated on an effective date basis.

    Attributes
    ----------

    olf_:
        A triangle representation of the on-level factors

    Examples
    --------

    Premium vectors are expressed as a Triangle object. This example shows how to create and apply on-level factors to a Triangle object with one rate change.

    ..  testsetup::

        import chainladder as cl

    ..  testcode::

        import pandas as pd
        import numpy as np

        xyz = cl.load_sample("xyz")
        olf = (
            cl.ParallelogramOLF(
                rate_history=pd.DataFrame(
                    {
                        "EffDate": ["2001-07-01"],
                        "RateChange": [0.20],
                    }
                ),
                change_col="RateChange",
                date_col="EffDate",
            )
            .fit_transform(xyz["Premium"])
            .olf_
        )
        xyz["Leveled Premium"] = xyz["Premium"] * olf
        print(np.round(xyz["Leveled Premium"], 0))

    ..  testoutput::

                   12        24        36        48       60       72       84       96       108      120      132
        1998       NaN       NaN   24000.0   24000.0  24000.0  24000.0  24000.0  24000.0  24000.0  24000.0  24000.0
        1999       NaN   37800.0   37800.0   37800.0  37800.0  37800.0  37800.0  37800.0  37800.0  37800.0      NaN
        2000   54000.0   54000.0   54000.0   54000.0  54000.0  54000.0  54000.0  54000.0  54000.0      NaN      NaN
        2001   58537.0   58537.0   58537.0   58537.0  58537.0  58537.0  58537.0  58537.0      NaN      NaN      NaN
        2002   62485.0   62485.0   62485.0   62485.0  62485.0  62485.0  62485.0      NaN      NaN      NaN      NaN
        2003   69175.0   69175.0   69175.0   69175.0  69175.0  69175.0      NaN      NaN      NaN      NaN      NaN
        2004   99322.0   99322.0   99322.0   99322.0  99322.0      NaN      NaN      NaN      NaN      NaN      NaN
        2005  138151.0  138151.0  138151.0  138151.0      NaN      NaN      NaN      NaN      NaN      NaN      NaN
        2006  107578.0  107578.0  107578.0       NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN
        2007   62438.0   62438.0       NaN       NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN
        2008   47797.0       NaN       NaN       NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN

    Of course, we can have multiple rate changes, or assuems that policies are 24 months
    long with `policy_length`.
    We can also get more accurate OLFs by using the `approximation_grain`
    argument to set the resolution of the internal calendar spacing used to
    calculate on-level factors.

    ..  testcode::

        xyz = cl.load_sample("xyz")
        olf = (
            cl.ParallelogramOLF(
                rate_history=pd.DataFrame(
                    {
                        "EffDate": ["2001-07-01", "2023-10-01"],
                        "RateChange": [0.20, -0.05],
                    }
                ),
                change_col="RateChange",
                date_col="EffDate",
                policy_length=24,
                approximation_grain="D",
            )
            .fit_transform(xyz["Premium"])
            .olf_
        )
        xyz["Leveled Premium"] = xyz["Premium"] * olf
        print(np.round(xyz["Leveled Premium"], 0))

    ..  testoutput::

                   12        24        36        48       60       72       84       96       108      120      132
        1998       NaN       NaN   24000.0   24000.0  24000.0  24000.0  24000.0  24000.0  24000.0  24000.0  24000.0
        1999       NaN   37800.0   37800.0   37800.0  37800.0  37800.0  37800.0  37800.0  37800.0  37800.0      NaN
        2000   54000.0   54000.0   54000.0   54000.0  54000.0  54000.0  54000.0  54000.0  54000.0      NaN      NaN
        2001   59247.0   59247.0   59247.0   59247.0  59247.0  59247.0  59247.0  59247.0      NaN      NaN      NaN
        2002   66720.0   66720.0   66720.0   66720.0  66720.0  66720.0  66720.0      NaN      NaN      NaN      NaN
        2003   69891.0   69891.0   69891.0   69891.0  69891.0  69891.0      NaN      NaN      NaN      NaN      NaN
        2004   99322.0   99322.0   99322.0   99322.0  99322.0      NaN      NaN      NaN      NaN      NaN      NaN
        2005  138151.0  138151.0  138151.0  138151.0      NaN      NaN      NaN      NaN      NaN      NaN      NaN
        2006  107578.0  107578.0  107578.0       NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN
        2007   62438.0   62438.0       NaN       NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN
        2008   47797.0       NaN       NaN       NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN

    """

    def __init__(
        self,
        rate_history=None,
        change_col="",
        date_col="",
        approximation_grain="M",
        policy_length=12,
        vertical_line=False,
    ):
        self.rate_history = rate_history
        self.change_col = change_col
        self.date_col = date_col
        self.approximation_grain = approximation_grain
        self.policy_length = policy_length
        self.vertical_line = vertical_line

    def fit(self, X, y=None, sample_weight=None):
        """Fit the model with X.

        Parameters
        ----------
        X: Triangle-like
            Data to which the model will be applied.
        y: Ignored
        sample_weight: Ignored

        Returns
        -------
        self: object
            Returns the instance itself.
        """
        from chainladder.utils.utility_functions import parallelogram_olf, concat

        if X.array_backend == "sparse":
            obj = X.set_backend("numpy")
        else:
            obj = X.copy()

        groups = list(set(X.key_labels).intersection(self.rate_history.columns))

        if len(groups) == 0:
            idx = obj
        else:
            idx = obj.groupby(groups).sum()

        kw = dict(
            start_date=X.origin[0].to_timestamp(how="s"),
            end_date=X.origin[-1].to_timestamp(how="e"),
            grain=X.origin_grain,
            policy_length=self.policy_length,
            vertical_line=self.vertical_line,
            approximation_grain=self.approximation_grain,
        )

        if len(groups) > 0:
            tris = []
            for item in idx.index.set_index(groups).iterrows():
                r = self.rate_history.set_index(groups).loc[item[0]].copy()
                r[self.change_col] = r[self.change_col] + 1
                r = (r.groupby(self.date_col)[self.change_col].prod() - 1).reset_index()
                date = r[self.date_col]
                values = r[self.change_col]
                olf = parallelogram_olf(values=values, dates=date, **kw).values[
                    None, None
                ]
                if X.array_backend == "cupy":
                    olf = X.get_array_module().array(olf)
                tris.append((idx.loc[item[0]] * 0 + 1) * olf)
            self.olf_ = concat(tris, 0).latest_diagonal
        else:
            r = self.rate_history.copy()
            r[self.change_col] = r[self.change_col] + 1
            r = (r.groupby(self.date_col)[self.change_col].prod() - 1).reset_index()
            date = r[self.date_col]
            values = r[self.change_col]
            olf = parallelogram_olf(values=values, dates=date, **kw)
            self.olf_ = ((idx * 0 + 1) * olf.values[None, None]).latest_diagonal
        return self

    def transform(self, X, y=None, sample_weight=None):
        """If X and self are of different shapes, align self to X, else
        return self.

        Parameters
        ----------
        X: Triangle
            The triangle to be transformed

        Returns
        -------
            X_new: New triangle with transformed attributes.
        """
        X_new = X.copy()
        triangles = ["olf_"]
        for item in triangles:
            setattr(X_new, item, getattr(self, item))
        X_new._set_slicers()
        return X_new
