# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from sklearn.base import BaseEstimator, TransformerMixin
from chainladder.utils.weighted_regression import WeightedRegression
from chainladder.development import Development
from chainladder.core import EstimatorIO
import numpy as np
from chainladder.utils.cupy import cp
import copy
import warnings


class MunichAdjustment(BaseEstimator, TransformerMixin, EstimatorIO):
    """Applies the Munich Chainladder adjustment to a set of paid/incurred
       ldfs.  The Munich method heavily relies on the ratio of paid/incurred
       and its inverse.

    Parameters
    ----------
    paid_to_incurred : tuple or list of tuples
        A dictionary representing the ``values`` of paid and incurred triangles
        where ``values`` are an appropriate selection from :class:`Triangle`
        ``.values``, such as ``('paid', 'incurred')``

    Attributes
    ----------
    basic_cdf_ : Triangle
        The univariate cumulative development patterns
    basic_sigma_ : Triangle
        Sigma of the univariate ldf regression
    resids_ : Triangle
        Residuals of the univariate ldf regression
    q_ : Triangle
        chainladder age-to-age factors of the paid/incurred triangle and its
        inverse.  For paid measures it is (P/I) and for incurred measures it is
        (I/P).
    q_resids_ : Triangle
        Residuals of q regression.
    rho_ : Triangle
        Estimated conditional deviation around ``q_``
    lambda_ : Series or DataFrame
        Dependency coefficient between univariate chainladder link ratios and
        `q_resids_`
    ldf_ : Triangle
        The estimated bivariate loss development patterns
    cdf_ : Triangle
        The estimated bivariate cumulative development patterns

    """
    def __init__(self, paid_to_incurred):
        if type(paid_to_incurred) is dict:
            warnings.warn("paid_to_incurred dict argument is deprecated, use tuple instead")
            paid_to_incurred = [(k, v) for k, v in paid_to_incurred.items()]
        self.paid_to_incurred = paid_to_incurred

    def fit(self, X, y=None, sample_weight=None):
        """Fit the model with X.

        Parameters
        ----------
        X : Triangle-like
            Set of LDFs to which the munich adjustment will be applied.
        y : Ignored
        sample_weight : Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """


        obj = copy.deepcopy(X)
        xp = cp.get_array_module(obj.values)
        if 'ldf_' not in obj:
            obj = Development().fit_transform(obj)
        self.p_to_i_X_ = self._get_p_to_i_object(obj)
        self.p_to_i_ldf_ = self._get_p_to_i_object(obj.ldf_)
        self.p_to_i_sigma_ = self._get_p_to_i_object(obj.sigma_)
        self.q_f_, self.rho_sigma_ = self._get_MCL_model(obj)
        self.residual_, self.q_resid_ = self._get_MCL_resids(obj)
        self.lambda_coef_ = self._get_MCL_lambda()
        self.cdf_ = self._get_cdf(obj)
        self.ldf_ = self._set_ldf(X)
        self._map = {
            (list(X.columns).index(x)): (num%2, num//2)
            for num, x in enumerate(np.array(self.paid_to_incurred).flatten())}
        self.rho_ = copy.deepcopy(X)
        self.rho_.odims = ['(All)']
        self.rho_.values = self._reshape('rho_sigma_')

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
        X.cdf_ = self.cdf_
        X.ldf_ = self.ldf_
        return X

    def _get_p_to_i_object(self, obj):
        if type(self.paid_to_incurred) is tuple:
            p_to_i = [self.paid_to_incurred]
        else:
            p_to_i = self.paid_to_incurred
        xp = cp.get_array_module(obj.values)
        paid = obj[[item[0] for item in p_to_i][0]]
        for item in [item[0] for item in p_to_i][1:]:
            paid[item] = obj[item]
        incurred = obj[[item[1] for item in p_to_i][0]]
        for item in [item[1] for item in p_to_i][1:]:
            incurred[item] = obj[item]
        paid = paid.values[xp.newaxis]
        incurred = incurred.values[xp.newaxis]
        return xp.concatenate((paid, incurred), axis=0)

    def _p_to_i_concate(self, obj_p, obj_i):
        xp = cp.get_array_module(obj_p)
        return xp.concatenate((obj_p[xp.newaxis], obj_i[xp.newaxis]), 0)

    def _get_MCL_model(self, X):
        xp = cp.get_array_module(X.values)
        p, i = self.p_to_i_X_[0], self.p_to_i_X_[1]
        modelsP = WeightedRegression(axis=2, thru_orig=True)
        modelsP = modelsP.fit(p, i, 1/p).sigma_fill(X.sigma_interpolation)
        modelsI = WeightedRegression(axis=2, thru_orig=True)
        modelsI = modelsI.fit(i, p, 1/i).sigma_fill(X.sigma_interpolation)
        q_f = self._p_to_i_concate(modelsP.slope_, modelsI.slope_)
        rho_sigma = self._p_to_i_concate(modelsP.sigma_, modelsI.sigma_)
        return xp.swapaxes(q_f, -1, -2), xp.swapaxes(rho_sigma, -1, -2)

    def _get_MCL_resids(self, X):
        xp = cp.get_array_module(X.values)
        p_to_i_ata = self._get_p_to_i_object(X.link_ratio)
        p_to_i_ldf = self.p_to_i_ldf_
        p_to_i_sigma = self.p_to_i_sigma_
        paid, incurred = self.p_to_i_X_[0], self.p_to_i_X_[1]
        p_to_i_ldf = xp.unique(p_to_i_ldf, axis=-2)  # May cause issues later
        p_to_i_sigma = xp.unique(p_to_i_sigma, axis=-2)  # May cause issues
        residP = (p_to_i_ata[0]-p_to_i_ldf[0]) / \
            p_to_i_sigma[0]*xp.sqrt(paid[..., :-1, :-1])
        residI = (p_to_i_ata[1]-p_to_i_ldf[1]) / \
            p_to_i_sigma[1]*xp.sqrt(incurred[..., :-1, :-1])
        nans = (X-X._get_latest_diagonal(compress=False)).values[0, 0]*0+1
        q_resid = (paid/incurred - self.q_f_[1]) / \
            self.rho_sigma_[1]*xp.sqrt(incurred)*nans
        q_inv_resid = (incurred/paid - 1/self.q_f_[1]) / \
            self.rho_sigma_[0]*xp.sqrt(paid)*nans
        resid = self._p_to_i_concate(residP, residI)
        q_resid = self._p_to_i_concate(q_inv_resid, q_resid)
        return resid, q_resid

    def _get_MCL_lambda(self):
        xp = cp.get_array_module(self.residual_[1])
        k, v, o, d = self.residual_[1].shape
        w = xp.reshape(self.residual_[1], (k, v, o*d))
        w[w == 0] = xp.nan
        w = w*0+1
        lambdaI = WeightedRegression(thru_orig=True, axis=-1).fit(
            xp.reshape(self.q_resid_[1][..., :-1, :-1], (k, v, o*d)),
            xp.reshape(self.residual_[1], (k, v, o*d)), w).slope_
        lambdaP = WeightedRegression(thru_orig=True, axis=-1).fit(
            xp.reshape(self.q_resid_[0][..., :-1, :-1], (k, v, o*d)),
            xp.reshape(self.residual_[0], (k, v, o*d)), w).slope_
        return self._p_to_i_concate(lambdaP, lambdaI)[..., xp.newaxis]

    @property
    def munich_full_triangle_(self):
        full_paid = self.p_to_i_X_[0][..., 0:1]
        xp = cp.get_array_module(full_paid)
        full_incurred = self.p_to_i_X_[1][..., 0:1]
        for i in range(self.p_to_i_X_[0].shape[-1]-1):
            paid = (self.p_to_i_ldf_[0][..., i:i+1] +
                    self.lambda_coef_[0] *
                    self.p_to_i_sigma_[0][..., i:i+1] /
                    self.rho_sigma_[0][..., i:i+1] *
                    (full_incurred[..., -1:]/full_paid[..., -1:] -
                     self.q_f_[0][..., i:i+1]))*full_paid[..., -1:]
            inc = (self.p_to_i_ldf_[1][..., i:i+1] + self.lambda_coef_[1] *
                   self.p_to_i_sigma_[1][..., i:i+1] /
                   self.rho_sigma_[1][..., i:i+1] *
                   (full_paid[..., -1:]/full_incurred[..., -1:] -
                   self.q_f_[1][..., i:i+1]))*full_incurred[..., -1:]
            full_incurred = xp.concatenate(
                (full_incurred,
                 xp.nan_to_num(self.p_to_i_X_[1][..., i+1:i+2]) +
                 (1-xp.nan_to_num(self.p_to_i_X_[1][..., i+1:i+2]*0+1)) *
                 inc), axis=3)
            full_paid = xp.concatenate(
                (full_paid,
                 xp.nan_to_num(self.p_to_i_X_[0][..., i+1:i+2]) +
                 (1-xp.nan_to_num(self.p_to_i_X_[0][..., i+1:i+2]*0+1)) *
                 paid), axis=3)
        return self._p_to_i_concate(full_paid, full_incurred)

    def _get_cdf(self, X):
        ''' needs to be an attribute that gets assigned.  requires we overwrite
            the cdf and ldf methods with
        '''
        obj = copy.copy(X.cdf_)
        if type(self.paid_to_incurred) is tuple:
            p_to_i = [self.paid_to_incurred]
        else:
            p_to_i = self.paid_to_incurred
        xp = cp.get_array_module(obj.values)
        cdf_triangle = self.munich_full_triangle_
        cdf_triangle = cdf_triangle[..., -1:]/cdf_triangle[..., :-1]
        paid = [item[0] for item in p_to_i]
        for n, item in enumerate(paid):
            idx = np.where(X.cdf_.vdims == item)[0][0]
            obj.values[:, idx:idx+1, ...] = cdf_triangle[0, :, n:n+1, ...]
        incurred = [item[1] for item in p_to_i]
        for n, item in enumerate(incurred):
            idx = np.where(X.cdf_.vdims == item)[0][0]
            obj.values[:, idx:idx+1, ...] = cdf_triangle[1, :, n:n+1, ...]
        obj.nan_override = True
        obj._set_slicers()
        return obj

    def _set_ldf(self, X):
        ldf_tri = self.cdf_.values.copy()
        xp = cp.get_array_module(ldf_tri)
        ldf_tri = xp.concatenate((ldf_tri, xp.ones(ldf_tri.shape)[..., -1:]), -1)
        ldf_tri = ldf_tri[..., :-1]/ldf_tri[..., 1:]
        obj = copy.copy(self.cdf_)
        obj.values = ldf_tri
        obj.ddims = X.link_ratio.ddims
        obj._set_slicers
        return obj

    def _reshape(self, measure):
        map = self._map
        return np.concatenate(
            [getattr(self, measure)[map[k][0],:,map[k][1]:map[k][1]+1,...]
             for k in range(len(map))], axis=1)

    @property
    def lambda_(self):
        obj = copy.deepcopy(self.cdf_)
        obj.odims = obj.ddims = ['(All)']
        obj.values = self._reshape('lambda_coef_')
        return obj.to_frame()

    @property
    def basic_cdf_(self):
        obj = copy.deepcopy(self.ldf_)
        obj.values = self._reshape('p_to_i_ldf_')
        return obj

    @property
    def basic_sigma_(self):
        obj = copy.deepcopy(self.ldf_)
        obj.values = self._reshape('p_to_i_sigma_')
        return obj

    @property
    def resids_(self):
        obj = copy.deepcopy(self.ldf_)
        obj.values = self._reshape('residual_')
        obj.odims = self.cdf_.odims[:obj.values.shape[2]]
        return obj

    @property
    def q_(self):
        obj = copy.deepcopy(self.rho_)
        obj.odims = self.cdf_.odims
        obj.values = self._reshape('q_f_')
        return obj

    @property
    def q_resids_(self):
        obj = copy.deepcopy(self.ldf_)
        obj.values = self._reshape('q_resid_')[
            ..., :self.residual_.shape[-2], :self.residual_.shape[-1]]
        obj.odims = obj.odims[:obj.values.shape[2]]
        obj.ddims = obj.ddims[:obj.values.shape[3]]
        return obj
