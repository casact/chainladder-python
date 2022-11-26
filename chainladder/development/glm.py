# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pandas as pd
import numpy as np
from chainladder.development.base import DevelopmentBase
from chainladder.development.learning import DevelopmentML
from sklearn.linear_model import TweedieRegressor
from sklearn.pipeline import Pipeline
from chainladder.utils.utility_functions import PatsyFormula
import warnings

class TweedieGLM(DevelopmentBase):
    """ This estimator creates development patterns with a GLM using a Tweedie distribution.

    The Tweedie family includes several of the more popular distributions including
    the normal, ODP poisson, and gamma distributions.  This class is a special case
    of `DevleopmentML`.  It restricts to just GLM using a TweedieRegressor and
    provides an R-like formulation of the design matrix.

    .. versionadded:: 0.8.1

    Parameters
    ----------
    design_matrix: formula-like
        A patsy formula describing the independent variables, X of the GLM
    response:  str
        Column name for the reponse variable of the GLM.  If ommitted, then the
        first column of the Triangle will be used.
    weight: str
        Column name of any weight to use in the GLM. If none specified, then an
        unweighted regression will be performed.
    power: float, default=0
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
    alpha: float, default=1
        Constant that multiplies the penalty term and thus determines the
        regularization strength. ``alpha = 0`` is equivalent to unpenalized
        GLMs. In this case, the design matrix `X` must have full column rank
        (no collinearities).
    link: {'auto', 'identity', 'log'}, default='auto'
        The link function of the GLM, i.e. mapping from linear predictor
        `X @ coeff + intercept` to prediction `y_pred`. Option 'auto' sets
        the link depending on the chosen family as follows:
        - 'identity' for Normal distribution
        - 'log' for Poisson,  Gamma and Inverse Gaussian distributions
    max_iter: int, default=100
        The maximal number of iterations for the solver.
    tol: float, default=1e-4
        Stopping criterion. For the lbfgs solver,
        the iteration will stop when ``max{|g_j|, j = 1, ..., d} <= tol``
        where ``g_j`` is the j-th component of the gradient (derivative) of
        the objective function.
    warm_start: bool, default=False
        If set to ``True``, reuse the solution of the previous call to ``fit``
        as initialization for ``coef_`` and ``intercept_``.
    verbose: int, default=0
        For the lbfgs solver set verbose to any positive number for verbosity.
    resid_adj: str 
        Scale Pearson residuals. 'hat' uses the hat matrix, 'dof' uses the
        degree of freedom adjustment, None uses no adjustment. If 'hat' is used
        but cannot be calculated, then the 'dof' adjustment is invoked.
    hetero_groups: list, dict or callable
        Development age groupings to adjust for heteroscedasticity in residuals
    hetero_adj: str(default='std')
        Can be either 'std' or 'scale' depending on whether to group based on
        standard deviation or scale factor

    Attributes
    ----------
    ldf_: Triangle
        The estimated loss development patterns
    cdf_: Triangle
        The estimated cumulative development patterns
    model_: sklearn.Pipeline
        A scikit-learn Pipeline of the GLM
    design_matrix_ : pd.DataFrame
        A DataFrame representation of the design matrix used to solve the GLM.
    N_: int
        The number of observations used in fitting the GLM.
    n_params_: int
        The number of parametrs fit in the GLM.
    degrees_of_freedom_: int
        The degrees of freedom of the fitted GLM.
    """

    def __init__(self, design_matrix='C(origin)+C(development)-1',
                 response=None, weight=None, power=1.0, alpha=1.0, link='log',
                 max_iter=100, tol=0.0001, warm_start=False, verbose=0,
                 resid_adj='hat', hetero_groups=None, hetero_adj='std',):
        self.response=response
        self.weight=weight
        self.design_matrix = design_matrix
        self.power=power
        self.alpha=alpha
        self.link=link
        self.max_iter=max_iter
        self.tol=tol
        self.warm_start=warm_start
        self.verbose=verbose
        self.resid_adj = resid_adj 
        self.hetero_groups = hetero_groups
        self.hetero_adj = hetero_adj
        

    def fit(self, X, y=None, sample_weight=None):
        response = X.columns[0] if not self.response else self.response
        self.model_ = DevelopmentML(Pipeline(steps=[
            ('design_matrix', PatsyFormula(self.design_matrix)),
            ('model', TweedieRegressor(
                    link=self.link, power=self.power, max_iter=self.max_iter,
                    tol=self.tol, warm_start=self.warm_start,
                    verbose=self.verbose, fit_intercept=False))]),
                    y_ml=response, weight_ml=self.weight).fit(X)
        self.predicted_incrementals_, self.unscaled_pearson_residuals_ = self._get_unscaled_pearson_residuals(X)
        self.pearson_residuals_ = self._get_pearson_residuals(X)
        #self.hetero_adj_pearson_residuals_ = self._get_std_hetero()
        return self

    @property 
    def response_(self):
        return self.data_[self.model_.y_ml]
    
    @property 
    def data_(self):
        return self.model_.df_

    @property
    def ldf_(self):
        return self.model_.ldf_

    @property
    def coef_(self):
        return pd.Series(
            self.model_.estimator_ml.named_steps.model.coef_, name='coef_',
            index=list(self.model_.estimator_ml.named_steps.design_matrix.
                            design_info_.column_name_indexes.keys())
        ).to_frame()

    def transform(self, X):
        return self.model_.transform(X)

    @property
    def design_matrix_(self):
        return pd.DataFrame(np.asarray(
            self.model_.estimator_ml.named_steps.design_matrix.transform(self.model_.df_)), 
            columns=self.model_.estimator_ml.named_steps.design_matrix.design_info_.column_name_indexes.keys()
            ).astype(int)

    def _get_unscaled_pearson_residuals(self, X):
        xp = X.get_array_module()
        from chainladder import Chainladder
        predicted_incremental = Chainladder().fit(self.transform(X)).full_expectation_.cum_to_incr()
        predicted_incremental = predicted_incremental[predicted_incremental.development<=X.development.max()]
        limited_incremental = predicted_incremental[predicted_incremental.valuation<=X.valuation_date]
        unscaled_residuals = (
            (X.cum_to_incr() - limited_incremental)
            / xp.sqrt(xp.abs(limited_incremental ** self.power))
        ).iloc[0, 0, ...]
        w_ = X.nan_triangle[:, :-1]
        w_[:, 1:] * w_[:, 1:]
        w_ = xp.concatenate((w_[:, 0:1], w_), axis=1)
        unscaled_residuals = unscaled_residuals * w_
        unscaled_residuals.values = np.where(
            np.isnan(unscaled_residuals.values), 
            1e-10, unscaled_residuals.values
        ) * w_ * unscaled_residuals.nan_triangle
        unscaled_residuals.iat[0, 0, 0, -1] = 0 # Cheating the numerical approximation
        return predicted_incremental, unscaled_residuals
    
    @property
    def n_params_(self):
        return self.design_matrix_.shape[1]
    
    @property
    def N_(self):
        return self.design_matrix_.shape[0]
    
    @property
    def degrees_of_freedom_(self):
        return self.N_ - self.n_params_ # - self.hetero_params_

    @property
    def dof_adjustment_factor_(self):
        return (self.N_ / self.degrees_of_freedom_)**0.5

    @property
    def scale_(self):
        return ((self.unscaled_pearson_residuals_**2) / 
                (self.degrees_of_freedom_)).sum().sum()

    def _get_hat(self, X):
        """ The hat matrix adjustment (Shapland eq3.23)"""
        xp = X.get_array_module()
        exp_incr_triangle = xp.nan_to_num(
            self.predicted_incrementals_.values[0, 0]
            ) * X.nan_triangle
        weight_matrix = xp.diag(
            pd.DataFrame(exp_incr_triangle).unstack().dropna().values
        )
        hat = xp.matmul(
            xp.matmul(
                xp.matmul(
                    self.design_matrix_,
                    xp.linalg.inv(
                        xp.matmul(
                            self.design_matrix_.T, 
                            xp.matmul(weight_matrix, self.design_matrix_.values)
                        )
                    ),
                ),
                self.design_matrix_.values.T,
            ),
            weight_matrix,
        )
        hat = xp.diagonal(xp.sqrt(xp.divide(1, abs(1 - hat), where=(1 - hat) != 0)))
        total_length = X.nan_triangle.shape[0]
        reshaped_hat = xp.reshape(hat[:total_length], (1, total_length))
        indices = xp.nansum(X.nan_triangle, axis=0).cumsum().astype(int)
        for num, item in enumerate(indices[:-1]):
            col_length = int(indices[num + 1] - indices[num])
            col = xp.reshape(
                hat[int(indices[num]) : int(indices[num + 1])], (1, col_length)
            )
            nans = xp.repeat(
                xp.array([xp.nan])[None, :], total_length - col_length, axis=1
            )
            col = xp.concatenate((col, nans), axis=1)
            reshaped_hat = xp.concatenate((reshaped_hat, col), axis=0)
        return reshaped_hat.T
    
    def _conform_hetero_groups(self, group):
        if type(group) is list:
            group = dict(zip(self.unscaled_pearson_residuals_.development, group))
        elif callable(group):
            group = {age: group(age) for age in self.unscaled_pearson_residuals_.development}
        gs = pd.Series(group)
        self.hetero_params_ = len(set(group.values())) - 1
        group = {g: list(gs[gs==g].index) for g in set(group.values())}
        return group

    def _get_std_hetero(self):
        pearson_residuals = self.pearson_residuals_
        resid = pearson_residuals.to_frame(origin_as_datetime=True).unstack().dropna()
        std = resid.std()
        groups = self._conform_hetero_groups(self.hetero_groups)
        group_std = {k: resid.loc[v].std() for k, v in groups.items()}
        group_std = pd.Series(
            {d: group_std[k] 
             for k, v in groups.items() for d in v}
             ).sort_index()
        hetero_adj_pearson_residuals_ = pearson_residuals * (
            (std / group_std).values)
        return hetero_adj_pearson_residuals_

    def _get_range_hetero(self):
        pass

    def _get_pearson_residuals(self, X):
        if self.resid_adj == 'hat':
            try:
                hat_ = self._get_hat(X)
                dof_ = 1.0
            except:
                warnings.warn('Cannot compute Hat Matrix. Using the Degree of Freedom Adjustment instead')
                hat_ = 1.0
                dof_ = self.dof_adjustment_factor_
        return self.unscaled_pearson_residuals_ * hat_ * dof_