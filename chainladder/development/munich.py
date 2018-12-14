from chainladder.development.base import DevelopmentBase
from chainladder.utils.weighted_regression import WeightedRegression
import numpy as np


class MunichDevelopment(DevelopmentBase):
    """ Munich Chainladder
    """
    def __init__(self, n_per=-1, average='volume',
                 sigma_interpolation='log-linear', paid_to_incurred={}):
        self.n_per = n_per
        self.average = average
        self.sigma_interpolation = sigma_interpolation
        self.paid_to_incurred = paid_to_incurred

    def fit(self, X, y=None, sample_weight=None):
        super().fit(X, y, sample_weight)
        rhoI_sigma, rhoP_sigma, q_f, qinverse_f = self._get_MCL_model()
        paid_residual, incurred_residual, Q_inverse_residual, Q_residual = self._get_MCL_residuals()

    def _get_MCL_model(self):
        modelsP, modelsI = [], []
        for k, v in self.paid_to_incurred.items():
            p, i = self.X_[k].triangle, self.X_[k].triangle
            modelsP.append(
                WeightedRegression(w=1/p, x=p, y=i, axis=2, thru_orig=True)
                .fit().sigma_fill(self.sigma_interpolation))
            modelsI.append(
                WeightedRegression(w=1/i, x=i, y=p, axis=2, thru_orig=True)
                .fit().sigma_fill(self.sigma_interpolation))
        q_f = np.array([item.slope_ for item in modelsI])
        qinverse_f = np.array([item.slope_ for item in modelsP])
        rhoI_sigma = np.array([item.sigma_ for item in modelsI])
        rhoP_sigma = np.array([item.sigma_ for item in modelsP])
        return rhoI_sigma, rhoP_sigma, q_f, qinverse_f

    def _get_MCL_residuals(self):
        return 1, 2, 3, 4
        ## Estimate the residuals
        #residualP, residualI = [], []
        #for k, v in self.paid_to_incurred.items():
        #    # age_to_age needs to contemplate n_per?
        #    residualP.append(((self.X_[k].age_to_age-self.ldf_[k]) / \
        #                       self.sigma_[k]*np.sqrt(self.X_))
        #    residualP.append(((self.X_[v].age_to_age-self.ldf_[v]) / \
        #                       self.sigma_[v]*np.sqrt(self.X_))
        #Q_ratios = self.X_[k]/self.X_[v]
        #Q_f = np.array(([self.q_f[:-1]]*(len(self.Paid.data))))
        #Q_residual = ((Q_ratios - Q_f) \
        #              / np.array(([self.rhoI_sigma[:-1]]*(len(self.Incurred.data))))
        #              * np.array(np.sqrt(self.Incurred.data.iloc[:,:-1])))
        #Q_inverse_residual = ((1/Q_ratios - 1/Q_f) \
        #                        / np.array(([self.rhoP_sigma[:-1]]*(len(self.Paid.data))))
        #                        * np.array(np.sqrt(self.Paid.data.iloc[:,:-1])))

        #paid_residual = self.__MCL_vector(paid_residual.iloc[:,:-1],len(self.Paid.data.columns))
        #incurred_residual = self.__MCL_vector(incurred_residual.iloc[:,:-1],len(self.Paid.data.columns))
        #Q_inverse_residual = self.__MCL_vector(Q_inverse_residual.iloc[:,:-1],len(self.Paid.data.columns))
        #Q_residual = self.__MCL_vector(Q_residual.iloc[:,:-1],len(self.Paid.data.columns))
        #return paid_residual, incurred_residual, Q_inverse_residual, Q_residual
