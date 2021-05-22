# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from chainladder.methods.chainladder import Chainladder
from chainladder.development import DevelopmentBase, Development
import numpy as np
import pandas as pd
from warnings import warn
import types


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
    hat_adj : bool (default=False)
        Adjust standardized Pearson residuals with the hat matrix adjustment
        factor.  If false, Degree of Freedom adjustment is used.
    drop : tuple or list of tuples
        Drops specific origin/development combination(s) from residual sample
    drop_high : bool or list of bool (default=None)
        Drops highest link ratio(s) from residual sample
    drop_low : bool or list of bool (default=None)
        Drops lowest link ratio(s) from residual sample
    drop_valuation : str or list of str (default=None)
        Drops specific valuation periods from residual sample. str must be date
        convertible.
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

    def __init__(
        self,
        n_sims=1000,
        n_periods=-1,
        hat_adj=True,
        drop=None,
        drop_high=None,
        drop_low=None,
        drop_valuation=None,
        random_state=None,
    ):
        self.n_sims = n_sims
        self.n_periods = n_periods
        self.hat_adj = hat_adj
        self.drop = drop
        self.drop_high = drop_high
        self.drop_low = drop_low
        self.drop_valuation = drop_valuation
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        backend = X.array_backend
        if backend == "sparse":
            X = X.set_backend("numpy")
        else:
            X = X.copy()
        xp = X.get_array_module()
        if X.shape[:2] != (1, 1):
            raise ValueError("Only single index/column triangles are supported")
        if type(X.ddims) != np.ndarray:
            raise ValueError("Triangle must be expressed with development lags")
        lag = {"M": 1, "Q": 3, "Y": 12}[X.development_grain]
        obj = Development(
            n_periods=self.n_periods,
            drop=self.drop,
            drop_high=self.drop_high,
            drop_low=self.drop_low,
            drop_valuation=self.drop_valuation,
        ).fit_transform(X)
        self.w_ = obj.w_
        obj = Chainladder().fit(obj)
        # Works for only a single triangle - can we generalize this
        exp_incr_triangle = obj.full_expectation_.cum_to_incr().values[
            0, 0, :, : X.shape[-1]
        ]
        exp_incr_triangle = xp.nan_to_num(exp_incr_triangle) * obj.X_.nan_triangle
        self.design_matrix_ = self._get_design_matrix(X)
        if self.hat_adj:
            try:
                self.hat_ = self._get_hat(X, exp_incr_triangle)
            except:
                warn("Could not compute hat matrix.  Setting hat_adj to False")
                self.had_adj = False
                self.hat_ = None
        else:
            self.hat_ = None
        self.resampled_triangles_, self.scale_ = self._get_simulation(
            X, exp_incr_triangle
        )
        n_obs = xp.nansum(self.w_)
        n_origin_params = X.shape[2]
        n_dev_params = X.shape[3] - 1
        deg_free = n_obs - n_origin_params - n_dev_params
        deg_free_adj_fctr = xp.sqrt(n_obs / deg_free)
        return self

    def _get_simulation(self, X, exp_incr_triangle):
        xp = X.get_array_module()
        k_value = 1  # for ODP Poisson
        unscaled_residuals = (
            (X.cum_to_incr().values - exp_incr_triangle)
            / xp.sqrt(xp.abs(exp_incr_triangle ** k_value))
        )[0, 0, ...]
        w_ = self.w_[0, 0]
        w_[:, 1:] * w_[:, 1:]
        w_ = xp.concatenate((w_[:, 0:1], w_), axis=1)
        unscaled_residuals = unscaled_residuals * w_
        pearson_chi_sq = sum(sum(xp.nan_to_num(unscaled_residuals) ** 2))
        if self.hat_ is None:
            standardized_residuals = unscaled_residuals
        else:
            standardized_residuals = self.hat_ * unscaled_residuals
        n_params = self.design_matrix_.shape[1]
        degree_freedom = xp.nansum(X.nan_triangle) - n_params
        # Shapland has a hetero adjustment to degree_freedom here
        # He also adjusts the residuals for the hetero adjustment
        scale_phi = pearson_chi_sq / degree_freedom
        k, v, o, d = X.shape
        resids = xp.reshape(standardized_residuals, (k, v, o * d))

        adj_resid_dist = resids[xp.isfinite(resids)]  # Missing k,v dimensions
        # Suggestions from Using the ODP Bootstrap Model: A Practitioners Guide
        adj_resid_dist = adj_resid_dist[adj_resid_dist != 0]
        adj_resid_dist = adj_resid_dist - xp.mean(adj_resid_dist)
        random_state = xp.random.RandomState(self.random_state)
        resampled_residual = [
            (
                random_state.choice(
                    adj_resid_dist, size=exp_incr_triangle.shape, replace=True
                )
                * (exp_incr_triangle * 0 + 1)
            )[None, ...]
            for item in range(self.n_sims)
        ]
        resampled_residual = xp.concatenate(tuple(resampled_residual), 0).reshape(
            self.n_sims, exp_incr_triangle.shape[0], exp_incr_triangle.shape[1]
        )
        resampled_residual = resampled_residual
        b = xp.repeat(exp_incr_triangle[None, ...], self.n_sims, 0)
        resampled_triangles = (resampled_residual * xp.sqrt(abs(b)) + b).cumsum(2)
        resampled_triangles = xp.swapaxes(resampled_triangles[None, ...], 0, 1)
        obj = X.copy()
        obj.kdims = np.arange(self.n_sims)
        obj.values = resampled_triangles
        obj._set_slicers()
        return obj, scale_phi

    def _get_design_matrix(self, X):
        """ The design matrix used in hat matrix adjustment (Shapland eq3.12)
        """
        xp = X.get_array_module()
        w = X.nan_triangle
        arr = xp.diag(w[:, 0])
        intra_beta = xp.zeros((w.shape[0], w.shape[1] - 1))
        arr = xp.concatenate((arr, intra_beta), axis=1)
        for i in range(w.shape[1] - 1):
            len_alpha = len(w[:, i + 1][~xp.isnan(w[:, i + 1])])
            intra_alpha = xp.diag(w[:, i + 1])[:len_alpha, :]
            intra_beta[:, i] = 1
            intra_beta = intra_beta[:len_alpha, :]
            intra_arr = xp.concatenate((intra_alpha, intra_beta), axis=1)
            arr = xp.concatenate((arr, intra_arr), axis=0)
        return arr

    def _get_hat(self, X, exp_incr_triangle):
        """ The hat matrix adjustment (Shapland eq3.23)"""
        xp = X.get_array_module()
        weight_matrix = xp.diag(
            pd.DataFrame(exp_incr_triangle).unstack().dropna().values
        )
        design_matrix = self.design_matrix_
        hat = xp.matmul(
            xp.matmul(
                xp.matmul(
                    design_matrix,
                    xp.linalg.inv(
                        xp.matmul(
                            design_matrix.T, xp.matmul(weight_matrix, design_matrix)
                        )
                    ),
                ),
                design_matrix.T,
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

    def _get_hetero_adjustment(self):
        pass

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
        X_new = self.resampled_triangles_
        X_new.scale_ = self.scale_
        X_new.random_state = self.random_state
        X_new._get_process_variance = types.MethodType(_get_process_variance, X_new)
        return X_new


def _get_process_variance(self, full_triangle):
    """ Inserts random noise into the full triangles and full expectations """
    # if self.process_dist == 'od poisson':
    #    process_triangle = np.nan_to_num(np.array([random_state.poisson(lam=abs(item))*np.sign(np.nan_to_num(item))for item in sim_exp_incr_triangle]))
    xp = full_triangle.get_array_module()
    lower_tri = full_triangle.cum_to_incr() - self.cum_to_incr()
    random_state = xp.random.RandomState(
        None if not self.random_state else self.random_state + 1
    )
    lower_tri.values = random_state.gamma(
        shape=abs(lower_tri.values) / self.scale_, scale=self.scale_
    ) * xp.sign(xp.nan_to_num(lower_tri.values))
    return (lower_tri + self.cum_to_incr()).incr_to_cum()
