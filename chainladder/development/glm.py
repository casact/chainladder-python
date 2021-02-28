# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pandas as pd
import numpy as np
from patsy import dmatrix
from sklearn.base import BaseEstimator, TransformerMixin
from chainladder.development.base import DevelopmentBase
from chainladder.development.learning import DevelopmentML
from sklearn.linear_model import TweedieRegressor
from sklearn.pipeline import Pipeline


class PatsyFormula(BaseEstimator, TransformerMixin):
    """ A sklearn-style wrapper for patsy formulas """
    def __init__(self, formula=None):
        self.formula = formula

    def fit(self, X, y=None, sample_weight=None):
        self.design_info_ = dmatrix(self.formula, X).design_info
        return self

    def transform(self, X):
        return dmatrix(self.design_info_, X)


class TweedieGLM(DevelopmentBase):
    """ This estimator creates development patterns with a GLM using a Tweedie distribution.


    The Tweedie family includes several of the more popular distributions including
    the normal, ODP poisson, and gamma distributions.  This class is a special case
    of `DevleopmentML`.  It restricts to just GLM using a TweedieRegressor and
    provides an R-like formulation of the design matrix.

    Parameters
    -----------
    design_matrix : formula-like
        A patsy formula describing the independent variables, X of the GLM
    response : str, default None
        Name of the response column.
    power : float, default=0
            The power determines the underlying target distribution according
            to the following table:
            +-------+------------------------+
            | Power | Distribution           |
            +=======+========================+
            | 0     | Normal                 |
            +-------+------------------------+
            | 1     | Poisson                |
            +-------+------------------------+
            | (1,2) | Compound Poisson Gamma |
            +-------+------------------------+
            | 2     | Gamma                  |
            +-------+------------------------+
            | 3     | Inverse Gaussian       |
            +-------+------------------------+
            For ``0 < power < 1``, no distribution exists.
    alpha : float, default=1
        Constant that multiplies the penalty term and thus determines the
        regularization strength. ``alpha = 0`` is equivalent to unpenalized
        GLMs. In this case, the design matrix `X` must have full column rank
        (no collinearities).
    link : {'auto', 'identity', 'log'}, default='auto'
        The link function of the GLM, i.e. mapping from linear predictor
        `X @ coeff + intercept` to prediction `y_pred`. Option 'auto' sets
        the link depending on the chosen family as follows:
        - 'identity' for Normal distribution
        - 'log' for Poisson,  Gamma and Inverse Gaussian distributions
    max_iter : int, default=100
        The maximal number of iterations for the solver.
    tol : float, default=1e-4
        Stopping criterion. For the lbfgs solver,
        the iteration will stop when ``max{|g_j|, j = 1, ..., d} <= tol``
        where ``g_j`` is the j-th component of the gradient (derivative) of
        the objective function.
    warm_start : bool, default=False
        If set to ``True``, reuse the solution of the previous call to ``fit``
        as initialization for ``coef_`` and ``intercept_`` .
    verbose : int, default=0
        For the lbfgs solver set verbose to any positive number for verbosity.

    """

    def __init__(self, design_matrix='C(development) + C(origin)',
                 response=None, power=1.0, alpha=1.0, link='log',
                 max_iter=100, tol=0.0001, warm_start=False, verbose=0):
        self.response=response
        self.design_matrix = design_matrix
        self.power=power
        self.alpha=alpha
        self.link=link
        self.max_iter=max_iter
        self.tol=tol
        self.warm_start=warm_start
        self.verbose=verbose

    def fit(self, X, y=None, sample_weight=None):
        response = X.columns[0] if not self.response else self.response
        self.model = DevelopmentML(Pipeline(steps=[
            ('design_matrix', PatsyFormula(self.design_matrix)),
            ('model', TweedieRegressor(
                    link=self.link, power=self.power, max_iter=self.max_iter,
                    tol=self.tol, warm_start=self.warm_start,
                    verbose=self.verbose, fit_intercept=False))]),
                    y_ml=response).fit(X)
        return self

    @property
    def ldf_(self):
        return self.model.ldf_

    @property
    def triangle_glm_(self):
        return self.model.triangle_ml_

    @property
    def coef_(self):
        return pd.Series(
            self.model.estimator_ml.named_steps.model.coef_, name='coef_',
            index=list(self.model.estimator_ml.named_steps.design_matrix.design_info_.column_name_indexes.keys())
        ).to_frame()

    def transform(self, X):
        return self.model.transform(X)
