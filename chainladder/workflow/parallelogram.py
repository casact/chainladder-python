# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from sklearn.base import BaseEstimator, TransformerMixin
from chainladder.core.io import EstimatorIO


class ParallelogramOLF(BaseEstimator, TransformerMixin, EstimatorIO):
    """
    Estimator to create and apply on-level factors to a Triangle object.  This
    is commonly used for premium vectors expressed as a Triangle object.

    Parameters
    ----------

    rate_history : pd.DataFrame
        A DataFrame with
    change_col : str
        The column containing the rate changes expressed as a decimal. For example,
        5% decrease should be stated as -0.05
    date_col : str
        A list-like set of effective dates corresponding to each of the changes
    vertical_line :
        Rates are typically stated on an effective date basis and premiums on
        and earned basis.  By default, this argument is False and produces
        parallelogram OLFs. If True, Parallelograms become squares.  This is
        commonly seen in Workers Compensation with benefit on-leveling or if
        the premium origin is also stated on an effective date basis.

    Attributes
    ----------

    olf_ :
        A triangle representation of the on-level factors
    """

    def __init__(
        self, rate_history=None, change_col="", date_col="", vertical_line=False
    ):
        self.rate_history = rate_history
        self.change_col = change_col
        self.date_col = date_col
        self.vertical_line = vertical_line

    def fit(self, X, y=None, sample_weight=None):
        """Fit the model with X.

        Parameters
        ----------
        X : Triangle-like
            Data to which the model will be applied.
        y : Ignored
        sample_weight : Ignored

        Returns
        -------
        self : object
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
            vertical_line=self.vertical_line,
        )

        if len(groups) > 0:
            tris = []
            for item in idx.index.set_index(groups).iterrows():
                r = self.rate_history.set_index(groups).loc[item[0]].copy()
                r[self.change_col] = r[self.change_col] + 1
                r = (r.groupby(self.date_col)[self.change_col].prod() - 1).reset_index()
                date = r[self.date_col]
                values = r[self.change_col]
                olf = parallelogram_olf(values=values, date=date, **kw).values[
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
            olf = parallelogram_olf(values=values, date=date, **kw)
            self.olf_ = ((idx * 0 + 1) * olf.values[None, None]).latest_diagonal
        return self

    def transform(self, X, y=None, sample_weight=None):
        """ If X and self are of different shapes, align self to X, else
        return self.

        Parameters
        ----------
        X : Triangle
            The triangle to be transformed

        Returns
        -------
            X_new : New triangle with transformed attributes.
        """
        X_new = X.copy()
        triangles = ["olf_"]
        for item in triangles:
            setattr(X_new, item, getattr(self, item))
        X_new._set_slicers()
        return X_new
