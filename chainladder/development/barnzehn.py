# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
from chainladder.development.learning import DevelopmentML
from chainladder.development.glm import TweedieGLM
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from chainladder.utils.utility_functions import PatsyFormula


class BarnettZehnwirth(TweedieGLM):
    """ This estimator enables modeling from the Probabilistic Trend Family as
    described by Barnett and Zehnwirth.

    .. versionadded:: 0.8.2

    Parameters
    -----------
    formula : formula-like
        A patsy formula describing the independent variables, X of the GLM
    response :  str
        Column name for the reponse variable of the GLM.  If ommitted, then the
        first column of the Triangle will be used.


    """

    def __init__(self, formula='origin + development + valuation', response=None):
        self.formula = formula
        self.response = response

    def fit(self, X, y=None, sample_weight=None):
        if max(X.shape[:2]) > 1:
            raise ValueError("Only single index/column triangles are supported")
        tri = X.cum_to_incr().log()
        response = X.columns[0] if not self.response else self.response
        self.model_ = DevelopmentML(Pipeline(steps=[
            ('design_matrix', PatsyFormula(self.formula)),
            ('model', LinearRegression(fit_intercept=False))]),
                    y_ml=response, fit_incrementals=False).fit(tri)
        resid = tri - self.model_.triangle_ml_[
            self.model_.triangle_ml_.valuation <= tri.valuation_date]
        self.mse_resid_ = (resid**2).sum(0).sum(1).sum(2).sum() / (
            np.nansum(tri.nan_triangle) -
            len(self.model_.estimator_ml.named_steps.model.coef_))
        self.std_residuals_ = (resid / np.sqrt(self.mse_resid_))
        self.model_.triangle_ml_ = self.model_.triangle_ml_.exp()
        self.model_.triangle_ml_.is_cumulative = False
        return self
