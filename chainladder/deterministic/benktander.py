import numpy as np
import copy
from sklearn.base import BaseEstimator

class Benktander(BaseEstimator):
    def __init__(self, apriori=.5, n_iters=1):
        self.apriori = apriori
        self.n_iters = n_iters

    def fit(self, X, y=None, sample_weight=None):
        '''This method generalizes both the DFM and Bornheuetter-Ferguson (BF)
           methods.  It serves as the computational base for these other two methods
           by setting the n_iters parameter to 1 in the case of the BF method and
           a sufficiently high n_iters for the chainladder.
           L\sum_{k=0}^{n-1}(1-\frac{1}{CDF}) + Apriori\times (1-\frac{1}{CDF})^{n}
        '''

        latest, cdf = np.expand_dims(X.triangle[:, :, :, 0],-1), np.expand_dims(X.triangle[:, :, :, 1],-1)
        apriori = sample_weight.triangle * self.apriori
        cdf = np.expand_dims(np.array(cdf), 4)
        cdf = 1-1/np.repeat(cdf, self.n_iters + 1, axis=4)
        exponents = np.expand_dims(np.expand_dims(np.arange(self.n_iters+1), 0),0)
        exponents = np.repeat(exponents, cdf.shape[2], axis=0)
        exponents = X.expand_dims(exponents)
        cdf = cdf**exponents
        ultimates = np.sum(cdf[:,:,:,:,:-1],axis=4)*latest+cdf[:,:,:,:,-1]*apriori
        obj = copy.deepcopy(X)
        obj.triangle = ultimates
        obj.ddims = ['Ultimate']
        return obj

class BornFerg(BaseEstimator):
    def __init__(self, apriori=.5):
        self.apriori = apriori

    def fit(self, X, y=None, sample_weight=None):
        return Benktander(apriori=self.apriori, n_iters=1).fit(X=X, sample_weight=sample_weight)
