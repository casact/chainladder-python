import numpy as np
import copy
from sklearn.base import BaseEstimator


class Benktander(BaseEstimator):

    def __init__(self, apriori=.5, n_iters=1, development=None):
        self.apriori = apriori
        self.n_iters = n_iters
        self.development = development

    def fit(self, X, y=None, sample_weight=None):
        '''This method generalizes both the DFM and Bornheuetter-Ferguson (BF)
           methods.  It serves as the computational base for these other two methods
           by setting the n_iters parameter to 1 in the case of the BF method and
           a sufficiently high n_iters for the chainladder.
           L\sum_{k=0}^{n-1}(1-\frac{1}{CDF}) + Apriori\times (1-\frac{1}{CDF})^{n}
        '''

        latest, cdf = X[0].latest_diagonal, X[1].latest_diagonal
        exposure = sample_weight
        apriori = exposure.triangle * self.apriori
        cdf = np.expand_dims(np.array(cdf), 0)
        cdf = 1-1/np.repeat(cdf, self.n_iters + 1, axis=0)
        return cdf
        #exponents = np.expand_dims(list(range(n_iters+1)),0)
        #exponents = np.repeat(exponents,CDF.shape[0], axis=0)
        #CDF = CDF**exponents
        #self.ultimates = Series(np.sum(CDF[:,:-1],axis=1)*latest+CDF[:,-1]*my_apriori,
        #                        index=triangle.data.index)
        #self.method = 'benktander'
        #self.apriori = apriori
        #self.exposure = exposure
        #self.n_iters = n_iters
        #return self
