# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from sklearn.utils import check_random_state

from chainladder.methods.chainladder import Chainladder
from chainladder.development.base import DevelopmentBase, Development
import numpy as np
import pandas as pd
import copy
from warnings import warn


class BootstrapODPSample(DevelopmentBase):
    """
    Class to generate bootstrap samples of triangles.  Currently this Only
    supports 'single' triangles (single index and single column).

    Parameters
    ----------
    n_sims : int (default=1000)
        Number of simulations to generate
    n_periods : integer, optional (default=-1)
        number of origin periods to be used in the ldf average calculation. For
        all origin periods, set n_periods=-1
    hat_adj : bool (default=TRUE)
        Adjust standardized Pearson residuals with the hat matrix adjustment
        factor.
    drop : tuple or list of tuples
        Drops specific origin/development combination(s) from residual sample
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.

    Attributes
    ----------
    resampled_triangles_ : Triangle
        A set of triangles represented by each simulation
    scale_ :
        The scale parameter to be used in generating process risk
    """
    def __init__(self, n_sims=1000, n_periods=-1,
                 hat_adj=True, drop=None, random_state=None):
        self.n_sims = n_sims
        self.n_periods = n_periods
        self.hat_adj = hat_adj
        self.drop = drop
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        if (type(X.ddims) != np.ndarray):
            raise ValueError('Triangle must be expressed with development lags')
        obj = copy.copy(X)
        self.w_ = X._nan_triangle() if not self.drop else self._drop(X)
        lag = {'M': 1, 'Q': 3, 'Y': 12}[X.development_grain]
        if type(self.drop) is not list and self.drop is not None:
            self.drop = [self.drop]
            drop = [(item[0], item[1]-lag) for item in self.drop]
        else:
            drop = self.drop
        obj = Development(n_periods=self.n_periods, drop=drop).fit_transform(obj)
        obj = Chainladder().fit(obj)
        # Works for only a single triangle - can we generalize this
        exp_incr_triangle = obj.full_expectation_.cum_to_incr() \
                               .values[0, 0, :, :X.shape[-1]]
        exp_incr_triangle = np.nan_to_num(exp_incr_triangle) * \
            obj.X_._nan_triangle()
        self.design_matrix_ = self._get_design_matrix(X)
        if self.hat_adj:
            try:
                self.hat_ = self._get_hat(X, exp_incr_triangle)
            except:
                warn('Could not compute hat matrix.  Setting hat_adj to False')
                self.had_adj = False
                self.hat_ = None
        else:
            self.hat_ = None
        self.resampled_triangles_, self.scale_ = \
            self._get_simulation(X, exp_incr_triangle)
        n_obs = np.nansum(self.w_)
        n_origin_params = X.shape[2]
        n_dev_params = X.shape[3] - 1
        deg_free = n_obs - n_origin_params - n_dev_params
        deg_free_adj_fctr = np.sqrt(n_obs/deg_free)
        return self

    def _get_simulation(self, X, exp_incr_triangle):
        k_value = 1  # for ODP Poisson
        if X.shape[:2] != (1, 1):
            raise ValueError('Only single index/column triangles are ',
                             'supported')
        unscaled_residuals = \
            ((X.cum_to_incr().values - exp_incr_triangle) /
             np.sqrt(np.abs(exp_incr_triangle**k_value)))[0, 0, ...]
        if self.hat_ is None:
            standardized_residuals = unscaled_residuals
        else:
            standardized_residuals = self.hat_ * unscaled_residuals
        pearson_chi_sq = sum(sum(np.nan_to_num(unscaled_residuals)**2))
        n_params = self.design_matrix_.shape[1]
        degree_freedom = np.nansum(X._nan_triangle()) - n_params
        # Shapland has a hetero adjustment to degree_freedom here
        # He also adjusts the residuals for the etero adjustment
        scale_phi = pearson_chi_sq / degree_freedom
        k, v, o, d = X.shape
        resids = np.reshape(standardized_residuals, (k, v, o*d))

        adj_resid_dist = resids[np.isfinite(resids)]  # Missing k,v dimensions
        # Suggestions from Using the ODP Bootstrap Model: A Practitioners Guide
        adj_resid_dist = adj_resid_dist[adj_resid_dist != 0]
        adj_resid_dist = adj_resid_dist - np.mean(adj_resid_dist)

        random_state = check_random_state(self.random_state)
        resampled_residual = [random_state.choice(adj_resid_dist,
                              size=exp_incr_triangle.shape,
                              replace=True)*(exp_incr_triangle*0+1)
                              for item in range(self.n_sims)]
        resampled_residual = np.array(resampled_residual) \
                               .reshape(self.n_sims, exp_incr_triangle.shape[0],
                                        exp_incr_triangle.shape[1])
        resampled_residual = resampled_residual
        b = np.repeat(np.expand_dims(exp_incr_triangle, 0), self.n_sims, 0)
        resampled_triangles = (resampled_residual*np.sqrt(abs(b))+b).cumsum(2)
        resampled_triangles = np.swapaxes(
            np.expand_dims(resampled_triangles, 0), 0, 1)
        obj = copy.copy(X)
        obj.kdims = np.arange(self.n_sims)
        obj.values = resampled_triangles
        obj._set_slicers()
        return obj, scale_phi

        # Shapland cites Verral and England 2002 in using gamma as a proxy for
        # poisson because of computational efficiency even though poisson is
        # the more theoretically correct choice.  Does this belong in the fit?

        # Process variance adjustment
        #lower_diagonal = (obj._nan_triangle()*0)
        #lower_diagonal[np.isnan(lower_diagonal)]=1
        #obj.values = Chainladder().fit(Development().fit_transform(obj)).full_expectation_.values[...,:-1]*lower_diagonal
        #obj.nan_override = True
        #if self.process_dist == 'gamma':
        #    process_triangle = np.nan_to_num(np.array([random_state.gamma(shape=abs(item/scale_phi),scale=scale_phi)*np.sign(np.nan_to_num(item)) for item in sim_exp_incr_triangle]))
        #if self.process_dist == 'od poisson':
        #    process_triangle = np.nan_to_num(np.array([random_state.poisson(lam=abs(item))*np.sign(np.nan_to_num(item))for item in sim_exp_incr_triangle]))
        #IBNR = process_triangle.cumsum(axis=2)[:,:,-1]

    def _get_design_matrix(self, X):
        """ The design matrix used in hat matrix adjustment (Shapland eq3.12)
        """
        w = X._nan_triangle()
        arr = np.diag(w[:, 0])
        intra_beta = np.zeros((w.shape[0], w.shape[1]-1))
        arr = np.concatenate((arr, intra_beta), axis=1)
        for i in range(w.shape[1]-1):
            len_alpha = len(w[:, i+1][~np.isnan(w[:, i+1])])
            intra_alpha = np.diag(w[:, i+1])[:len_alpha, :]
            intra_beta[:, i] = 1
            intra_beta = intra_beta[:len_alpha, :]
            intra_arr = np.concatenate((intra_alpha, intra_beta), axis=1)
            arr = np.concatenate((arr, intra_arr), axis=0)
        return arr

    def _get_hat(self, X, exp_incr_triangle):
        """ The hat matrix adjustment (Shapland eq3.23)"""
        weight_matrix = np.diag(pd.DataFrame(exp_incr_triangle).unstack().dropna().values)
        design_matrix = self.design_matrix_
        hat = np.matmul(np.matmul(np.matmul(design_matrix,np.linalg.inv(np.matmul(design_matrix.T, np.matmul(weight_matrix, design_matrix)))), design_matrix.T), weight_matrix)
        hat = np.diagonal(np.sqrt(np.divide(1, abs(1-hat), where=(1-hat)!=0)))
        total_length = X._nan_triangle().shape[0]
        reshaped_hat = np.reshape(hat[:total_length],(1, total_length))
        indices = np.nansum(X._nan_triangle(), axis=0).cumsum().astype(int)
        for num, item in enumerate(indices[:-1]):
            col_length = int(indices[num+1]-indices[num])
            col = np.reshape(hat[int(indices[num]):int(indices[num+1])],(1,col_length))
            nans = np.repeat(np.expand_dims(np.array([np.nan]),0),total_length-col_length, axis=1)
            col = np.concatenate((col, nans), axis=1)
            reshaped_hat = np.concatenate((reshaped_hat,col),axis=0)
        return reshaped_hat.T

    def _get_hetero_adjustment(self):
        pass

    def _drop(self, X):
        drop = [self.drop] if type(self.drop) is not list else self.drop
        arr = X._nan_triangle()
        for item in drop:
            arr[np.where(X.origin == item[0])[0][0],
                np.where(X.development == item[1])[0][0]] = 0
        return arr

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
        X_new = copy.copy(X)
        X_new = self.resampled_triangles_
        X_new.scale_ = self.scale_
        return X_new
