# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import pandas as pd
from chainladder import options
from chainladder.development.learning import DevelopmentML
from chainladder.development.glm import TweedieGLM
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from chainladder.utils.utility_functions import PatsyFormula, PTF_formula

class BarnettZehnwirth(TweedieGLM):
    """ This estimator enables modeling from the Probabilistic Trend Family as
    described by Barnett and Zehnwirth.

    .. versionadded:: 0.8.2

    Parameters
    ----------
    drop: tuple or list of tuples
        Drops specific origin/development combination(s)
    drop_valuation: str or list of str (default = None)
        Drops specific valuation periods. str must be date convertible.
    formula: formula-like
        A patsy formula describing the independent variables, X of the GLM
    response:  str
        Column name for the reponse variable of the GLM.  If ommitted, then the
        first column of the Triangle will be used.
    alpha: list of int
        List of origin periods denoting the first indices of each group 
    gamma: list of int
    iota: list of int

    """

    def __init__(self, drop=None,drop_valuation=None,formula=None, response=None, alpha=None, gamma=None, iota=None):
        self.drop = drop
        self.drop_valuation = drop_valuation

        self.response = response
        if formula and (alpha or gamma or iota):
            raise ValueError("Model can only be specified by either a formula or some combination of alpha, gamma and iota.")
        if not (formula or alpha or gamma or iota):
            raise ValueError("Model must be specified, either a formula or some combination of alpha, gamma and iota.")
        for Greek in [alpha,gamma,iota]:
            if Greek:
                if not ( (type(Greek) is list) and all(type(bound) is int for bound in Greek) ):
                    raise ValueError("Alpha, gamma and iota must be given as lists of integers, specifying periods.")
        self.formula = formula
        self.alpha = alpha
        self.gamma = gamma
        self.iota = iota
        
    def fit(self, X, y=None, sample_weight=None):
        if max(X.shape[:2]) > 1:
            raise ValueError("Only single index/column triangles are supported")
        tri = X.cum_to_incr().log()
        response = X.columns[0] if not self.response else self.response
        if(not self.formula):
            self.formula = PTF_formula(self.alpha,self.gamma,self.iota,dgrain=min(tri.development))
        self.model_ = DevelopmentML(Pipeline(steps=[
            ('design_matrix', PatsyFormula(self.formula)),
            ('model', LinearRegression(fit_intercept=False))]),
                    y_ml=response, fit_incrementals=True, drop=self.drop, drop_valuation = self.drop_valuation, weighted_step = 'model').fit(X = tri, sample_weight = sample_weight)
        resid = tri - self.model_.triangle_ml_[
            self.model_.triangle_ml_.valuation <= tri.valuation_date]
        self.mse_resid_ = (resid**2).sum(0).sum(1).sum(2).sum() / (
            np.nansum(tri.nan_triangle) -
            len(self.model_.estimator_ml.named_steps.model.coef_))
        self.std_residuals_ = (resid / np.sqrt(self.mse_resid_))
        self.model_.triangle_ml_ = self.model_.triangle_ml_.exp()
        self.model_.triangle_ml_.is_cumulative = False
        return self

    def transform(self, X):
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
        X_ml = self.model_._prep_X_ml(X.cum_to_incr().log())
        y_ml = self.model_.estimator_ml.predict(X_ml)
        triangle_ml, predicted_data = self.model_._get_triangle_ml(X_ml, y_ml)
        backend = "cupy" if X.array_backend == "cupy" else "numpy"
        triangle_ml.is_cumulative = False
        X_new.ldf_ = triangle_ml.exp().incr_to_cum().link_ratio.set_backend(backend)
        X_new.ldf_.valuation_date = pd.to_datetime(options.ULT_VAL)
        X_new._set_slicers()
        X_new.predicted_data_ = predicted_data
        return X_new
