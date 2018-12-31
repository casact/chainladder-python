import numpy as np
import copy
from chainladder.methods import MethodBase

class Benktander(MethodBase):
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
        super().fit(X, y, sample_weight)
        self.sample_weight_ = sample_weight
        latest = self.X_.latest_diagonal.triangle
        apriori = sample_weight.triangle * self.apriori
        obj = copy.deepcopy(self.X_)
        obj.triangle = self.X_.cdf_.triangle * (obj.triangle*0+1)
        cdf = obj.latest_diagonal.triangle
        cdf = np.expand_dims(1-1/cdf, 0)
        exponents = np.arange(self.n_iters+1)
        exponents = np.reshape(exponents, tuple([len(exponents)]+[1]*4))
        cdf = cdf**exponents
        obj.triangle = np.sum(cdf[:-1, ...], 0)*latest+cdf[-1, ...]*apriori
        obj.ddims = ['Ultimate']
        self.ultimate_ = obj
        return self


class BornhuetterFerguson(MethodBase):
    def __init__(self, apriori=.5):
        self.apriori = apriori

    def fit(self, X, y=None, sample_weight=None):
        super().fit(X, y, sample_weight)
        self.sample_weight_ = sample_weight
        obj = Benktander(apriori=self.apriori, n_iters=1) \
            .fit(X=X, sample_weight=sample_weight)
        self.ultimate_ = obj.ultimate_
        return self
