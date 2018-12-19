from sklearn.base import BaseEstimator
from chainladder.utils.weighted_regression import WeightedRegression
import numpy as np


class MunichDevelopment(BaseEstimator):
    """ Munich Chainladder
        TODO:
            1. Create 'square' LDF as a triangle obj
            2. Let it take both Development and Tail objects

    """
    def __init__(self, paid_to_incurred={}):
        self.paid_to_incurred = paid_to_incurred

    def fit(self, X, y=None, sample_weight=None):
        if X.__class__.__dict__.get('ldf_', None) is None:
            raise ValueError('Triangle must have LDFs.')
        self.p_to_i_X_ = self._get_p_to_i_object(X)
        self.p_to_i_ldf_ = self._get_p_to_i_object(X.ldf_)
        self.p_to_i_sigma_ = self._get_p_to_i_object(X.sigma_)
        self.q_f_, self.rho_sigma_ = self._get_MCL_model(X)
        self.residual_, self.q_resid_ = self._get_MCL_residuals(X)
        self.lambda_coef_ = self._get_MCL_lambda()
        return self

    def _get_p_to_i_object(self, obj):
        paid = obj[list(self.paid_to_incurred.keys())[0]]
        for item in list(self.paid_to_incurred.keys())[1:]:
            paid[item] = obj[item]
        incurred = obj[list(self.paid_to_incurred.values())[0]]
        for item in list(self.paid_to_incurred.values())[1:]:
            incurred[item] = obj[item]
        paid = np.expand_dims(paid.triangle, 0)
        incurred = np.expand_dims(incurred.triangle, 0)
        return np.concatenate((paid, incurred), axis=0)

    def _p_to_i_concate(self, obj_p, obj_i):
        obj = np.concatenate((np.expand_dims(obj_p, 0),
                              np.expand_dims(obj_i, 0)), 0)
        return obj

    def _get_MCL_model(self, X):
        p, i = self.p_to_i_X_[0], self.p_to_i_X_[1]
        modelsP = WeightedRegression(w=1/p, x=p, y=i, axis=2, thru_orig=True)
        modelsP = modelsP.fit().sigma_fill(X.sigma_interpolation)
        modelsI = WeightedRegression(w=1/i, x=i, y=p, axis=2, thru_orig=True)
        modelsI = modelsI.fit().sigma_fill(X.sigma_interpolation)
        q_f = self._p_to_i_concate(modelsP.slope_, modelsI.slope_)
        rho_sigma = self._p_to_i_concate(modelsP.sigma_, modelsI.sigma_)
        return np.swapaxes(q_f, -1, -2), np.swapaxes(rho_sigma, -1, -2)

    def _get_MCL_residuals(self, X):
        p_to_i_ata = self._get_p_to_i_object(X.link_ratio)
        p_to_i_ldf = self.p_to_i_ldf_
        p_to_i_sigma = self.p_to_i_sigma_
        paid, incurred = self.p_to_i_X_[0], self.p_to_i_X_[1]
        residualP = (p_to_i_ata[0]-p_to_i_ldf[0]) / \
            p_to_i_sigma[0]*np.sqrt(paid[:, :, :-1, :-1])
        residualI = (p_to_i_ata[1]-p_to_i_ldf[1]) / \
            p_to_i_sigma[1]*np.sqrt(incurred[:, :, :-1, :-1])
        nans = X.nan_triangle_x_latest()
        q_resid = (paid/incurred - self.q_f_[1]) / \
            self.rho_sigma_[1]*np.sqrt(incurred)*nans
        q_inv_resid = (incurred/paid - 1/self.q_f_[1]) / \
            self.rho_sigma_[0]*np.sqrt(paid)*nans
        residual = self._p_to_i_concate(residualP, residualI)
        q_resid = self._p_to_i_concate(q_inv_resid, q_resid)
        return residual, q_resid

    def _get_MCL_lambda(self):
        k, v, o, d = self.residual_[1].shape
        w = np.reshape(self.residual_[1], (k, v, o*d))
        w[w == 0] = np.nan
        w = w*0+1
        lambdaI = WeightedRegression(
            w, x=np.reshape(self.q_resid_[1][:, :, :-1, :-1], (k, v, o*d)),
            y=np.reshape(self.residual_[1], (k, v, o*d)),
            thru_orig=True, axis=-1).fit().slope_
        lambdaP = WeightedRegression(
            w, x=np.reshape(self.q_resid_[0][:, :, :-1, :-1], (k, v, o*d)),
            y=np.reshape(self.residual_[0], (k, v, o*d)),
            thru_orig=True, axis=-1).fit().slope_
        return np.expand_dims(self._p_to_i_concate(lambdaP, lambdaI), -1)

    def get_MCL_full_triangle(self):
        full_paid = self.p_to_i_X_[0][:, :, :, 0:1]
        full_incurred = self.p_to_i_X_[1][:, :, :, 0:1]
        for i in range(self.p_to_i_X_[0].shape[-1]-1):
            paid = (self.p_to_i_ldf_[0][:, :, :, i:i+1] +
                    self.lambda_coef_[0] *
                    self.p_to_i_sigma_[0][:, :, :, i:i+1] /
                    self.rho_sigma_[0][:, :, :, i:i+1] *
                    (full_incurred[:, :, :, -1:]/full_paid[:, :, :, -1:] -
                     self.q_f_[0][:, :, :, i:i+1]))*full_paid[:, :, :, -1:]
            inc = (self.p_to_i_ldf_[1][:, :, :, i:i+1] +
                   self.lambda_coef_[1] *
                   self.p_to_i_sigma_[1][:, :, :, i:i+1] /
                   self.rho_sigma_[1][:, :, :, i:i+1] *
                   (full_paid[:, :, :, -1:]/full_incurred[:, :, :, -1:] -
                   self.q_f_[1][:, :, :, i:i+1]))*full_incurred[:, :, :, -1:]
            full_incurred = np.concatenate(
                (full_incurred,
                 np.nan_to_num(self.p_to_i_X_[1][:, :, :, i+1:i+2]) +
                 (1-np.nan_to_num(self.p_to_i_X_[1][:, :, :, i+1:i+2]*0+1)) *
                 inc), axis=3)
            full_paid = np.concatenate(
                (full_paid,
                 np.nan_to_num(self.p_to_i_X_[0][:, :, :, i+1:i+2]) +
                 (1-np.nan_to_num(self.p_to_i_X_[0][:, :, :, i+1:i+2]*0+1)) *
                 paid), axis=3)
        return self._p_to_i_concate(full_paid, full_incurred)
