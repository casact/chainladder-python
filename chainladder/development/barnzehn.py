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
import warnings
from chainladder.utils.utility_functions import PatsyFormula
from patsy import ModelDesc


class BarnettZehnwirth(TweedieGLM):
    """ This estimator enables modeling from the Probabilistic Trend Family as
    described by Barnett and Zehnwirth.

    .. versionadded:: 0.8.2

    Parameters
    ----------
    formula: formula-like
        A patsy formula describing the independent variables, X of the GLM
    feat_eng: dict
        A dictionary with feature names as keys and a dictionary of function (with a key of 'func') and keyword arguments (with a key of 'kwargs') 
        (e.g. {
            'feature_1':{
                'func': function_name for feature 1,
                'kwargs': keyword arguments for the function
                },
            'feature_2':{
                'func': function_name for feature 2,
                'kwargs': keyword arguments for the function
                }
        );  
        functions should be written with a input Dataframe named df; this is the DataFrame containing origin, development, and valuation that will passed into the function at run time
        (e.g. this function adds 1 to every origin 
        def test_func(df)
            return df['origin'] + 1
        )
    response:  str
        Column name for the reponse variable of the GLM.  If ommitted, then the
        first column of the Triangle will be used.


    """

    def __init__(self, formula='C(origin) + development', feat_eng=None, response=None):
        self.formula = formula
        self.response = response
        self.feat_eng = feat_eng

    def fit(self, X, y=None, sample_weight=None):
        if max(X.shape[:2]) > 1:
            raise ValueError("Only single index/column triangles are supported")
        tri = X.cum_to_incr().log()
        response = X.columns[0] if not self.response else self.response
        # Check for more than one linear predictor 
        linearpredictors = 0
        for term in ModelDesc.from_formula(self.formula).rhs_termlist[1:]:
            if 'C(' not in term.factors[0].code:
                linearpredictors += 1
        if linearpredictors > 1:
            warnings.warn("Using more than one linear predictor with BarnettZehnwirth may lead to issues with multicollinearity.")
        self.model_ = DevelopmentML(Pipeline(steps=[
            ('design_matrix', PatsyFormula(self.formula)),
            ('model', LinearRegression(fit_intercept=False))]),
                    y_ml=response, fit_incrementals=False, feat_eng = self.feat_eng).fit(tri)
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
        triangle_ml = self.model_._get_triangle_ml(X_ml, y_ml)
        backend = "cupy" if X.array_backend == "cupy" else "numpy"
        triangle_ml.is_cumulative = False
        X_new.ldf_ = triangle_ml.exp().incr_to_cum().link_ratio.set_backend(backend)
        X_new.ldf_.valuation_date = pd.to_datetime(options.ULT_VAL)
        X_new._set_slicers()
        return X_new
