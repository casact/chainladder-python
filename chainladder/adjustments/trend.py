# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from sklearn.base import BaseEstimator, TransformerMixin
from chainladder.core.io import EstimatorIO
import pandas as pd


class Trend(BaseEstimator, TransformerMixin, EstimatorIO):
    """
    Estimator to create and apply trend factors to a Triangle object.  Allows
    for compound trends as well as storage of the trend matrix to be used in
    other estimators, such as `CapeCod`.

    Parameters
    ----------

    trends : list-like
        The list containing the annual trends expressed as a decimal. For example,
        5% decrease should be stated as -0.05
    dates : list of date-likes
        A list-like of (start, end) dates to correspond to the `trend` list.
    axis : str (options: [‘origin’, ‘valuation’])
        The axis on which to apply the trend

    Attributes
    ----------

    trend_ :
        A triangle representation of the trend factors

    """

    def __init__(self, trends=0.0, dates=None, axis="origin"):
        self.trends = trends
        self.dates = dates
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
        trends = self.trends if type(self.trends) is list else [self.trends]
        dates = [(None, None)] if self.dates is None else self.dates
        dates = [dates] if type(dates) is not list else dates
        if type(dates[0]) is not tuple:
            raise AttributeError(
                'Dates must be specified as a tuple of start and end dates')
        self.trend_ = X.copy()
        for i, trend in enumerate(trends):
            self.trend_ = self.trend_.trend(
                trend, self.axis,
                start=dates[i][0], end=dates[i][1])
        self.trend_ = self.trend_ / X
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
