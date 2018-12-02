import numpy as np
from chainladder.tails import CurveFit


class Exponential(CurveFit):
    """Exponential Tail Fit
    Parameters
    ----------
    n_per : slice, default: slice(None, None, None)
    The LDF slice to be used in the computation of the tail factor. The
    default value indicates LDFs from all ages will be used.
    extrap_per : int, default: 100
    The number of periods over which the LDFs will be extrapolated. The
    product of LDFs from the extrapolations will be used as the tail factor.
    errors : {'ignore', 'raise'}, default: 'ignore'
    Exponentail decay requires LDFs strictly larger than 1.0.  If LDFs are
    less than 1.0, they will be removed from the calculation ('ignore') or
    they will raise an error ('raise').

    Attributes
    ----------
    ldf_ : Triangle

    cdf_ : Triangle
    Labels of each point

    Examples
    --------
    >>> from chainladder.tails import Exponential

    See also
    --------
    InversePower
    Another curve fit approach
    Notes
    ------
    None
    """
    def get_x(self, w, y):
        ''' For Exponential decay, no transformation on x is needed '''
        return None

    def predict_tail(self, slope, intercept, extrapolate):
        tail = np.exp(slope*extrapolate + intercept)
        return np.expand_dims(np.product(1 + tail, -1), -1)
