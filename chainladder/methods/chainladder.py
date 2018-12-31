import numpy as np
import copy
from chainladder.methods import MethodBase


class Chainladder(MethodBase):
    @property
    def ultimate_(self):
        obj = copy.deepcopy(self.X_)
        obj.triangle = np.repeat(self.X_.latest_diagonal.triangle,
                                 self.cdf_.shape[3], 3)
        obj.triangle = (self.cdf_.triangle*obj.triangle)*self.X_.nan_triangle()
        obj = obj.latest_diagonal
        obj.ddims = ['Ultimate']
        return obj
