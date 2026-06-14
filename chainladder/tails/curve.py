# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from chainladder.tails import TailBase
from chainladder.utils import WeightedRegression
from chainladder.development import Development
import pandas as pd
import warnings


class TailCurve(TailBase):
    """Allows for extraploation of LDFs to form a tail factor.

    Parameters
    ----------
    curve : str ('exponential', 'inverse_power')
        The type of curve extrapolation you'd like to use
    fit_period : tuple (start, stop) or list(bool)
        A tuple representing the range of ldfs to use in the curve fit.
        The use of ``None`` will use the edge of the triangle.  For example,
        (48, None) will use development factors for age 48 and beyond.
        Alternatively, passing a list of booleans [True, False, ...] will
        allow for excluding (False) any development patterns from fitting.
    extrap_periods : int
        Then number of development periods from attachment point to extrapolate
        the fit.
    errors : str ('raise' or 'ignore')
        Whether to raise an error or ignore observations that violate the
        distribution being fit.  The most common is ldfs < 1.0 will not work
        in either the ``exponential`` or ``inverse_power`` fits.
    attachment_age: int (default=None)
        The age at which to attach the fitted curve.  If None, then the latest
        age is used. Measures of variability from original ``ldf_`` are retained
        when being used in conjunction with the MackChainladder method.
    reg_threshold : tuple (lower, upper)
        A tuple representing the lower and upper thresholds for the ldfs to be
        considered in the log regression of the tail fitting. Default lower
        threshold set to 1.00001 to avoid distortion caused by ldfs close to 1.
        Upper threshold can be used as an alternative to the fit_period start,
        to make the selection value based rather then period based.
    projection_period : int
        The number of months beyond the latest available development age the
        `ldf_` and `cdf_` vectors should extend.


    Attributes
    ----------
    ldf_ : Triangle
        ldf with tail applied.
    cdf_ : Triangle
        cdf with tail applied.
    tail_ : DataFrame
        Point estimate of tail at latest maturity available in the Triangle.
    slope_ : DataFrame
        Slope parameter of the curve fit.
    intercept : DataFrame
        Intercept parameter of the curve fit.

    Examples
    --------
    Suppose a reserving actuary has credible development factors through age
    120, but the line is clearly still developing at the oldest observed
    ages, so a tail factor must be appended. ``TailCurve`` extrapolates one
    by regressing the observed development factors against age and extending
    the fitted curve beyond the edge of the triangle. The two main curve
    families encode very different assumptions about how quickly the
    remaining development decays:

    - ``exponential`` regresses ``ln(ldf - 1)`` against age, so the
      development portion of each factor decays geometrically. This yields a
      light tail that converges quickly, appropriate when development is
      expected to run off within a few periods of the triangle edge.
    - ``inverse_power`` regresses ``ln(ldf - 1)`` against the log of age,
      giving a power-law decay. Development persists much longer, producing a
      heavier and more conservative tail suited to long-tailed lines. Because
      it approaches its asymptote slowly, it is also far more sensitive to
      the ``extrap_periods`` parameter than the exponential curve.

    .. testsetup::

        import chainladder as cl

    .. testcode::

        dev = cl.Development().fit_transform(cl.load_sample("tail_sample")["paid"])
        exp = cl.TailCurve(curve="exponential").fit(dev)
        inv = cl.TailCurve(curve="inverse_power").fit(dev)
        print(exp.ldf_)
        print(inv.ldf_)

    .. testoutput::

                  12-24     24-36     36-48     48-60     60-72     72-84     84-96    96-108   108-120   120-132   132-144
        (All)  2.026309  1.559087  1.320123  1.184491  1.107264  1.074001  1.046207  1.032158  1.023925  1.012067  1.020099
                  12-24     24-36     36-48     48-60     60-72     72-84     84-96    96-108   108-120   120-132   132-144
        (All)  2.026309  1.559087  1.320123  1.184491  1.107264  1.074001  1.046207  1.032158  1.023925  1.027083  1.325559

    The same nine observed factors support either a 3.2% tail under
    exponential decay (``1.012067 * 1.020099``) or a 36.1% tail under the
    inverse power curve (``1.027083 * 1.325559``). To judge which family the
    data actually supports, attach each fitted curve at an early age and
    compare the smoothed factors against the observed ones over the ages
    where data exists:

    .. testcode::

        print(dev.ldf_)
        print(cl.TailCurve(curve="exponential", attachment_age=24).fit(dev).ldf_)
        print(cl.TailCurve(curve="inverse_power", attachment_age=24).fit(dev).ldf_)

    .. testoutput::

                  12-24     24-36     36-48     48-60     60-72     72-84     84-96    96-108   108-120
        (All)  2.026309  1.559087  1.320123  1.184491  1.107264  1.074001  1.046207  1.032158  1.023925
                  12-24     24-36     36-48     48-60     60-72     72-84    84-96    96-108   108-120   120-132   132-144
        (All)  2.026309  1.531333  1.331052  1.206265  1.128515  1.080073  1.04989  1.031084  1.019367  1.012067  1.020099
                  12-24     24-36     36-48     48-60     60-72     72-84     84-96    96-108   108-120   120-132   132-144
        (All)  2.026309  1.466969  1.227905  1.136998  1.092314  1.066862  1.050903  1.040192  1.032632  1.027083  1.325559

    The exponential curve tracks the observed decay closely at every age.
    The inverse power curve understates development at the middle ages and
    overstates it at the oldest ages, and that overstatement at the oldest
    ages is exactly what compounds into its 36.1% tail. On this data the
    lighter exponential tail is the better supported selection; booking the
    inverse power tail instead would require external support, such as
    industry benchmarks for an unusually long-tailed line.

    """

    def __init__(
            self,
            curve="exponential",
            fit_period=(None, None),
            extrap_periods=100,
            errors="ignore",
            attachment_age=None,
            reg_threshold=(1.00001, None),
            projection_period=12
    ):
        # validate arguments

        if curve not in [
            'exponential',
            'inverse_power',
            'weibull'
        ]:
            raise ValueError("Invalid curve type specified. Accepted values are 'exponential', 'inverse_power' and 'weibull'.")

        if errors not in [
            'ignore',
            'raise'
        ]:
            raise ValueError("Invalid value argument supplied to the errors parameter. Accepted values are 'raise' "
                             "and 'ignore'.")
        self.curve = curve
        self.fit_period = fit_period
        self.extrap_periods = extrap_periods
        self.errors = errors
        self.attachment_age = attachment_age
        self.reg_threshold = reg_threshold
        self.projection_period = projection_period

    def fit(self, X, y=None, sample_weight=None):
        """Fit the model with X.

        Parameters
        ----------
        X : Triangle-like
            Set of LDFs to which the tail will be applied.
        y : Ignored
        sample_weight : Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        from chainladder.utils.utility_functions import num_to_nan

        X = X.copy()
        xp = X.get_array_module()
        if type(self.fit_period) == slice:
            warnings.warn(
                "Slicing for fit_period is deprecated and will be removed. Please use a tuple (start_age, end_age)."
            )
            fit_period = self.fit_period
        elif type(self.fit_period) is list:
            fit_period = xp.array(self.fit_period)[None, None, None, :]
        else:
            grain = {"Y": 12, "S": 6, "Q": 3, "M": 1}[X.development_grain]
            start = (
                None
                if self.fit_period[0] is None
                else int(self.fit_period[0] / grain - 1)
            )
            end = (
                None
                if self.fit_period[1] is None
                else int(self.fit_period[1] / grain - 1)
            )
            fit_period = slice(start, end, None)
        super().fit(X, y, sample_weight)
        xp = self.ldf_.get_array_module()
        _y = self.ldf_.values[..., : X.shape[-1] - 1].copy()
        _w = xp.zeros(_y.shape)
        if type(fit_period) is slice:
            _w[..., fit_period] = 1.0
        else:
            _w = (_w + 1) * fit_period
        if self.reg_threshold[0] is None:
            warnings.warn(
                "Lower threshold for ldfs not set. Lower threshold will be set to 1.0 to ensure"
                "valid inputs for regression.")
            lower_threshold = 1
        elif self.reg_threshold[0] < 1:
            warnings.warn(
                "Lower threshold for ldfs set too low (<1). Lower threshold will be set to 1.0 to ensure "
                "valid inputs for regression.")
            lower_threshold = 1
        else:
            lower_threshold = self.reg_threshold[0]
        if self.reg_threshold[1] is not None:
            if self.reg_threshold[1] <= lower_threshold:
                warnings.warn(
                    "Can't set upper threshold for ldfs below lower threshold. Upper threshold will be set to 'None'.")
                upper_threshold = None
            else:
                upper_threshold = self.reg_threshold[1]
        else:
            upper_threshold = self.reg_threshold[1]
        if self.errors == "ignore":
            if upper_threshold is None:
                _w[_y <= lower_threshold] = 0
                _y[_y <= lower_threshold] = 1.01
            else:
                _w[(_y <= lower_threshold) | (_y > upper_threshold)] = 0
                _y[(_y <= lower_threshold) | (_y > upper_threshold)] = 1.01
        elif self.errors == "raise" and xp.any(_y < 1.0):
            raise ZeroDivisionError("Tail fit requires all LDFs to be greater than 1.0")
        if self.curve == "weibull":
            _y = xp.log(xp.log(_y / (_y - 1)))
        else:
            _y = xp.log(_y - 1)
        n_obs = X.shape[-1] - 1
        k, v = X.shape[:2]
        _x = self._get_x(_w, _y)
        # Get LDFs
        coefs = WeightedRegression(axis=3, xp=xp).fit(_x, _y, _w)
        self._slope_, self._intercept_ = coefs.slope_, coefs.intercept_
        extrapolate = xp.cumsum(
            xp.ones(tuple(list(_y.shape)[:-1] + [self.extrap_periods + n_obs])), -1
        )
        tail = self._predict_tail(extrapolate)
        if self.attachment_age:
            attach_idx = xp.min(xp.where(X.ddims >= self.attachment_age))
        else:
            attach_idx = len(X.ddims) - 1
        self.ldf_.values = xp.concatenate(
            (self.ldf_.values[..., :attach_idx], tail[..., attach_idx:]), -1
        )
        obj = Development().fit_transform(X) if "ldf_" not in X else X
        self._get_tail_stats(obj)
        return self

    def _get_x(self, w, y):
        # For Exponential decay, no transformation on x is needed
        if self.curve == "exponential":
            return None
        if self.curve in ("inverse_power", "weibull"):
            xp = self.ldf_.get_array_module()
            reg = WeightedRegression(3, False, xp=xp).fit(None, y, w).infer_x_w()
            return xp.log(reg.x)

    def _predict_tail(self, extrapolate):
        xp = self.ldf_.get_array_module()
        if self.curve == "exponential":
            tail_ldf = xp.exp(self._slope_ * extrapolate + self._intercept_)
        if self.curve == "inverse_power":
            tail_ldf = xp.exp(self._intercept_) * (extrapolate ** self._slope_)
        if self.curve == "weibull":
            tail_ldf = 1/(1-xp.exp(-xp.exp(self._intercept_)
                          * extrapolate**self._slope_))-1
        return self._get_tail_prediction(tail_ldf)

    @property
    def slope_(self):
        """ Does not work with munich """
        rows = self.ldf_.index.set_index(self.ldf_.key_labels).index
        return pd.DataFrame(
            self._slope_[..., 0, 0], index=rows, columns=self.ldf_.vdims
        )

    @property
    def intercept_(self):
        """ Does not work with munich """
        rows = self.ldf_.index.set_index(self.ldf_.key_labels).index
        return pd.DataFrame(
            self._intercept_[..., 0, 0], index=rows, columns=self.ldf_.vdims
        )
