''' Munich Chainladder
Use constant.py as a template
reference:
    https://www.casact.org/pubs/forum/13fforum/02-Tail-Factors-Working-Party.pdf
'''
from chainladder.development.base import DevelopmentBase
from chainladder.utils.weighted_regression import WeightedRegression
import numpy as np

class MunichDevelopment(DevelopmentBase):
    def __init__(self, n_per=-1, average='volume',
                 sigma_interpolation='log-linear', paid_to_incurred={}):
        self.n_per = n_per
        self.average = average
        self.sigma_interpolation = sigma_interpolation
        self.paid_to_incurred = paid_to_incurred

    def get_MCL_model(self):
        modelsI = []
        modelsP = []
        for k, v in self.paid_to_incurred.items():
            modelsP.append(
                WeightedRegression(w=1/self.X_[k].triangle,
                                   x=self.X_[k].triangle,
                                   y=self.X_[v].triangle,
                                   axis=2, thru_orig=True) \
                .fit().sigma_fill(self.sigma_interpolation))
            modelsI.append(
                WeightedRegression(w=1/self.X_[v].triangle,
                                   x=self.X_[v].triangle,
                                   y=self.X_[k].triangle,
                                   axis=2, thru_orig=True) \
                .fit().sigma_fill(self.sigma_interpolation))
        q_f = np.array([item.slope_ for item in modelsI])
        qinverse_f = np.array([item.slope_ for item in modelsP])
        rhoI_sigma = np.array([item.sigma_ for item in modelsI])
        rhoP_sigma = np.array([item.sigma_ for item in modelsP])
        return rhoI_sigma, rhoP_sigma, q_f, qinverse_f
