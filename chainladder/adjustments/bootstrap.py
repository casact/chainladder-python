# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from chainladder.methods.chainladder import Chainladder
from chainladder.development import DevelopmentBase, Development, TweedieGLM
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
    n_sims: int (default=1000)
        Number of simulations to generate
    model: TweedieGLM (default=None)
        A TweedieGLM model. If None, then the ODP GLM will be used.
    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.
    hetero_groups: list, dict or callable
        Development age groupings to adjust for heteroskedacitity in residuals
    hetero_adj: str(default='std')
        Can be either 'std' or 'scale' depending on whether to group based on
        standard deviation or scale factor


    Attributes
    ----------
    glm_: TweedieGLM
        The fitted TweedieGLM model used for Bootstrap simulations.
    resampled_triangles_: Triangle
        A set of triangles represented by each simulation
    unscaled_pearson_residuals_:
        The unscaled Pearson residuals
    pearson_residuals_:
        The scaled Pearson residuals
    scale_:
        The scale parameter to be used in generating process risk
    """

    def __init__(
        self,
        n_sims=1000,
        model=None,
        random_state=None,
    ):
        self.n_sims = n_sims
        self.model = model
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        if X.shape[1] > 1:
            # Generalize recursively to support axis 1 > 1
            from chainladder.utils.utility_functions import concat
            out = [BootstrapODPSample(**self.get_params()).fit(X.iloc[:, i])
                   for i in range(X.shape[1])]
            xp = X.get_array_module(out[0].design_matrix_)
            self.resampled_triangles_ = concat([i.resampled_triangles_ for i in out], axis=1)
            self.scale_ = xp.array([i.scale_ for i in out])
        else:
            # Save current backend, but convert to numpy
            backend = X.array_backend
            if backend == "sparse":
                X = X.set_backend("numpy").val_to_dev()
            else:
                X = X.val_to_dev()
            xp = X.get_array_module()
            if len(X) != 1:
                raise ValueError("Only single index triangles are supported")
            if self.model is None:
                self.glm_ = TweedieGLM().fit(X)
            else:
                self.glm_ = self.model.fit(X)
            self.resampled_triangles_ = self._get_simulation(X)
            #deg_free = n_obs - n_origin_params - n_dev_params
            #deg_free_adj_fctr = xp.sqrt(n_obs / deg_free)
        return self

    def _get_simulation(self, X):
        xp = X.get_array_module()
        exp_incr_triangle = xp.nan_to_num(
            self.glm_.predicted_incrementals_.values[0, 0]
            ) * X.nan_triangle
        k, v, o, d = X.shape
        resids = xp.reshape(self.glm_.pearson_residuals_.values, (k, v, o * d))
        adj_resid_dist = resids[xp.isfinite(resids)]  # Missing k, v dimensions
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
        # Equation 3.18
        resampled_triangles = (resampled_residual * xp.sqrt(abs(b)) + b).cumsum(2)
        resampled_triangles = xp.swapaxes(resampled_triangles[None, ...], 0, 1)

        obj = X.copy()
        obj.kdims = np.arange(self.n_sims)
        obj.values = resampled_triangles
        obj._set_slicers()
        return obj

    def transform(self, X):
        """ If X and self are of different shapes, align self to X, else
        return self.

        Parameters
        ----------
        X: Triangle
            The triangle to be transformed

        Returns
        -------
            X_new: New triangle with transformed attributes.
        """
        X_new = X.copy()
        X_new = self.resampled_triangles_
        X_new.glm_ = self.glm_
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
        shape=abs(lower_tri.values) / self.glm_.scale_, scale=self.glm_.scale_
    ) * xp.sign(xp.nan_to_num(lower_tri.values))
    return (lower_tri + self.cum_to_incr()).incr_to_cum()

