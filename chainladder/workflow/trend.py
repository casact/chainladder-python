# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from sklearn.base import BaseEstimator, TransformerMixin
from chainladder.core.io import EstimatorIO
import pandas as pd


class Trend(BaseEstimator, TransformerMixin, EstimatorIO):
    """
    Estimator to create and apply trend factors to a Triangle object.  This
    is commonly used for estimators like `CapeCod`.

    Parameters
    ----------

    trend : list-like
        The list containing the annual trends expressed as a decimal. For example,
        5% decrease should be stated as -0.05
    date : str
        A list-like set of dates at which trends start. If None then the valuation
        date of the triangle is assumed.
    axis : str (options: [‘origin’, ‘valuation’])
        The axis on which to apply the trend

    Attributes
    ----------

    trend_ :
        A triangle representation of the trend factors
    """

    def __init__(self, trend=0.0, end_date=None,  axis='origin'):
        self.trend = trend if type(trend) is list else [trend]
        self.end_date = end_date if type(end_date) is list else [end_date]
        self.axis = axis

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
        date = [item if item else X.valuation_date.strftime('%Y-%m-%d') for item in self.end_date]
        date = pd.to_datetime(date).tolist()
        self.trend_ = X.copy()
        for i, trend in enumerate(self.trend):
            self.trend_ = self.trend_.trend(trend, self.axis, date[i])
        self.trend_  = self.trend_ / X
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
        triangles = ["trend_"]
        for item in triangles:
            setattr(X_new, item, getattr(self, item))
        X_new._set_slicers()
        return X_new
