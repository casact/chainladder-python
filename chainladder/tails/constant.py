from chainladder.tails import TailBase


class TailConstant(TailBase):
    ''' This class must take an object of type Development
        Development will already have LDFs fit.

        TODO: Expand constant to be an array that varies by kdim and vdim
    '''
    def __init__(self, tail=1.0):
        self.tail = tail

    def fit(self, X, y=None, sample_weight=None):
        super().fit(X, y, sample_weight)
        self.ldf_.triangle[..., -1] = self.ldf_.triangle[..., -1]*self.tail
        return self

    def predict(self, X):
        X.std_err_ = self.std_err_
        X.cdf_ = self.cdf_
        X.ldf_ = self.ldf_
        X.sigma_ = self.sigma_
        return X
