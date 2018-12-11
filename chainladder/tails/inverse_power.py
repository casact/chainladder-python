import numpy as np
from chainladder.tails import CurveFit
from chainladder import WeightedRegression

class InversePower(CurveFit):
    def get_x(self, w, y):
        ''' For InversePower fit, we must transform x -> ln(x) '''
        reg = WeightedRegression(w, None, y, 3, False).infer_x_w()
        return np.log(reg.x)

    def predict_tail(self, slope, intercept, extrapolate):
        return np.expand_dims(np.product(1 + np.exp(intercept)*(extrapolate**slope), -1), -1)
