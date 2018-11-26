import numpy as np
import copy
from sklearn.base import BaseEstimator

class CapeCod(BaseEstimator):
    def __init__(trend = 0, decay = 1):
        self.trend = trend
        self.decay = decay

    def fit(self, X, y=None, sample_weight=None):
        ''' capecod
        '''

        latest, cdf = np.expand_dims(X.triangle[:, :, :, 0],-1), np.expand_dims(X.triangle[:, :, :, 1],-1)
        exposure = sample_weight.triangle
        latest = latest[exposure>0]
        reported_exposure = exposure[exposure>0]/ cdf
        trend_array = np.array([(1+trend)**(sum(exposure>0) - (i+1)) for i in range(sum(exposure>0))])
        decay_matrix = np.array([[decay**abs(i-j) for i in range(sum(exposure>0) )] for j in range(sum(exposure>0))])
        weighted_exposure = reported_exposure * decay_matrix
        trended_ultimate = np.repeat(np.array([(latest * trend_array) /(reported_exposure)]),sum(exposure>0),axis=0)
        apriori = np.sum(weighted_exposure*trended_ultimate,axis=1)/np.sum(weighted_exposure,axis=1)
        detrended_ultimate = apriori/trend_array
        IBNR = detrended_ultimate * (1-1/CDF) * np.array(exposure[exposure>0])
        self.ultimates = Series(latest + IBNR, index=exposure[exposure>0].index)
        obj = copy.deepcopy(X)
        obj.triangle = ultimates
        obj.ddims = ['Ultimate']
        return obj
