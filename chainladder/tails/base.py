import copy
import numpy as np
from chainladder.development import Development


class TailBase(Development):
    def fit(self, X, y=None, sample_weight=None):
        self.X_ = X.X_
        self._params = copy.deepcopy(X._params)
        self._params.ddims = \
            np.append(self._params.ddims,
                      [str(int(len(self._params.ddims) + 1)) + '-Ult'])
        self.y_ = y
        self.sample_weight_ = sample_weight

    def transform(self, X):
        return self

    def fit_transform(self, X, y=None, sample_weight=None):
        self.fit(X, y, sample_weight)
        return self

    def _get_tail_sigma(self):
        """ Method to produce the sigma of the tail factor
        """
        sigma_ = self._params.triangle[:,:,1:2,:]
        # Mack Method
        if self.sigma_est == 'mack':
            y = sigma_
            return np.sqrt(abs(min((y[:, : , :, -1]**4 / y[:, : , :, -2]**2),
                                    min(y[:, : , :, -2]**2, y[:, : , :, -1]**2))))
