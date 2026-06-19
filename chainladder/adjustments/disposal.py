# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from chainladder.methods import Chainladder
from chainladder.development import DevelopmentBase
import numpy as np
import copy
import warnings
from chainladder.utils import TriangleWeight, concat
from chainladder import Triangle

class DisposalRate(DevelopmentBase):
    """
    Calculates the bottom of a fitted full_triangle_ using the Disposal Rate method described
    by Friedland.

    Parameters
    ----------
    n_periods: integer, optional (default = -1)
        number of origin periods to be used in the ldf average calculation. For
        all origin periods, set n_periods = -1
    drop: tuple or list of tuples
        Drops specific origin/development combination(s). See order of operations
        below when combined with multiple drop parameters.
    drop_high: bool, int, list of bools, or list of ints (default = None)
        Drops highest (by rank) link ratio(s) from LDF calculation
        If a boolean variable is passed, drop_high is set to 1, dropping only the
        highest value. Protected by ``preserve``.
        See order of operations below when combined with multiple drop parameters.
    drop_low: bool, int, list of bools, or list of ints (default = None)
        Drops lowest (by rank) link ratio(s) from LDF calculation
        If a boolean variable is passed, drop_low is set to 1, dropping only the
        lowest value. Protected by ``preserve``.
        See order of operations below when combined with multiple drop parameters.
    drop_above: float or list of floats (default = numpy.inf)
        Drops all link ratio(s) above the given parameter from the LDF calculation.
        Protected by ``preserve``.
        See order of operations below when combined with multiple drop parameters.
    drop_below: float or list of floats (default = 0.00)
        Drops all link ratio(s) below the given parameter from the LDF calculation.
        Protected by ``preserve``.
        See order of operations below when combined with multiple drop parameters.
    preserve: int (default = 1)
        The minimum number of link ratio(s) required for LDF calculation.
        See order of operations below when combined with multiple drop parameters.
    drop_valuation: str or list of str (default = None)
        Drops specific valuation periods. str must be date convertible.
        See order of operations below when combined with multiple drop parameters.

        .. note ::
    
            (Order of Drop Operations)
            
            When multiple drop parameters are used together, the weights are built in this order (steps 4 and 5 are reversed from `Development`):
        
            1. ``n_periods`` — limit to the most recent origin periods.
            2. ``drop`` — remove specific origin/development cells.
            3. ``drop_valuation`` — remove entire valuation diagonal in the triangle.
            4. ``drop_above`` / ``drop_below`` — remove link ratios outside a range
               (Protected by``preserve``, which may relax exclusions from this step if too few ratios would remain
               then this step is skipped).
            5. ``drop_high`` / ``drop_low`` — remove highest/lowest link ratios by rank
               (eligible factors from ``n_periods`` are used; protected by ``preserve``,
               which may relax exclusions from this step if too few ratios would remain then this step is skipped).
            6. Calculate the loss development factors using ``average`` method.

    Attributes
    ----------
    disposal_rate_tri: Triangle
        actual disposal rates by origin and development

    disposal_: Triangle
        fitted disposal rates

    incr_disposal_: Triangle
        incremental of disposal_

    Examples
    --------
    ``trend`` tilts the case-adequacy adjustment before ``Incurred`` is rebuilt;
    on the ``MedMal`` slice the inner diagonals of the adjusted ``Incurred``
    triangle restate materially between ``0%`` and ``15%`` annual drift, while
    the latest diagonal is preserved.

    .. testsetup::

        import chainladder as cl
        import numpy as np

    .. testcode::

        tri = cl.load_sample("berqsherm").loc["MedMal"]
        base = cl.BerquistSherman(
            paid_amount="Paid",
            incurred_amount="Incurred",
            reported_count="Reported",
            closed_count="Closed",
            trend=0.0,
        ).fit(tri)
        tilted = cl.BerquistSherman(
            paid_amount="Paid",
            incurred_amount="Incurred",
            reported_count="Reported",
            closed_count="Closed",
            trend=0.15,
        ).fit(tri)
        print(np.round(base.adjusted_triangle_["Incurred"], 0))

    .. testoutput::
        :options: +NORMALIZE_WHITESPACE

                      12          24          36          48          60          72          84          96
        1969   9883293.0  27420103.0  35879085.0  43105257.0  33438702.0  30397324.0  25723694.0  23506000.0
        1970   8641763.0  31305782.0  41543535.0  48550616.0  38203864.0  36222888.0  32216000.0         NaN
        1971  11733960.0  43887171.0  61649896.0  64917222.0  51410209.0  48377000.0         NaN         NaN
        1972  13638651.0  50987209.0  66696278.0  72777529.0  61163000.0         NaN         NaN         NaN
        1973  14387930.0  45470590.0  56577593.0  73733000.0         NaN         NaN         NaN         NaN
        1974  13630366.0  47189379.0  63477000.0         NaN         NaN         NaN         NaN         NaN
        1975  15036351.0  48904000.0         NaN         NaN         NaN         NaN         NaN         NaN
        1976  15791000.0         NaN         NaN         NaN         NaN         NaN         NaN         NaN

    .. testcode::

        print(np.round(tilted.adjusted_triangle_["Incurred"], 0))

    .. testoutput::
        :options: +NORMALIZE_WHITESPACE

                      12          24          36          48          60          72          84          96
        1969   3793504.0  12084942.0  18563821.0  25924316.0  23516364.0  24979245.0  24016864.0  23506000.0
        1970   3760482.0  15830500.0  24615996.0  33169802.0  30722141.0  33362729.0  32216000.0         NaN
        1971   5982185.0  25583831.0  41384825.0  50323342.0  46191356.0  48377000.0         NaN         NaN
        1972   7819355.0  33794110.0  51361061.0  64559286.0  61163000.0         NaN         NaN         NaN
        1973   9533246.0  34585431.0  49667342.0  73733000.0         NaN         NaN         NaN         NaN
        1974  10348458.0  41241243.0  63477000.0         NaN         NaN         NaN         NaN         NaN
        1975  13102479.0  48904000.0         NaN         NaN         NaN         NaN         NaN         NaN
        1976  15791000.0         NaN         NaN         NaN         NaN         NaN         NaN         NaN

    """

    def __init__(
        self,
        n_periods: int = -1,
        average: str | list[str] = 'volume',
        drop: tuple | list[tuple] | None = None,
        drop_high: bool | int | list[bool] | list[int] | None = None,
        drop_low: bool | int | list[bool] | list[int] | None = None,
        preserve: int = 1,
        drop_valuation: str | list[str] | None = None,
        drop_above: float = np.inf,
        drop_below: float = 0.00,
    ):
        self.n_periods = n_periods
        self.average = average
        self.drop_high = drop_high
        self.drop_low = drop_low
        self.preserve = preserve
        self.drop_valuation = drop_valuation
        self.drop_above = drop_above
        self.drop_below = drop_below
        self.drop = drop

    def fit(
            self, 
            X:Triangle, 
            y:None=None, 
            sample_weight:Triangle|None=None
    ):
        """
        Estimate disposal rate for a given Triangle and ultimate

        Parameters
        ----------
        X : Triangle
            Triangle to which the Disposal Rate method is applied
        y : None
            Ignored
        sample_weight : Triangle
            Ultimate

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        if sample_weight is None:
            raise ValueError("sample_weight is required.")
        #convert to numpy
        if X.array_backend == "sparse":
            X = X.set_backend("numpy").incr_to_cum()
        else:
            X = X.copy().incr_to_cum()
        if sample_weight.array_backend == "sparse":
            ult = sample_weight.set_backend("numpy")
        else:
            ult = sample_weight.copy()
        #get backend
        self.xp = X.get_array_module()
        self.disposal_rate_tri = X / ult.values
        tw = TriangleWeight(
            n_periods = self.n_periods,
            drop_high = self.drop_high,
            drop_low = self.drop_low,
            drop_above = self.drop_above,
            drop_below = self.drop_below,
            drop_valuation = self.drop_valuation,
            preserve = self.preserve,
            drop = self.drop
        )
        if hasattr(X, "w_"):
            self.w_ = tw.fit(X=self.disposal_rate_tri * X.w_).w_.values
        else:
            self.w_ = tw.fit(X=self.disposal_rate_tri).w_.values
        #calculate factors
        super().fit(ult.values,X.values,self.w_)
        #keep attributes
        self.disposal_ = self._param_property(self.disposal_rate_tri,self.params_.slope_[...,0][..., None, :])
        self.disposal_ = concat((self.disposal_,(ult/ult).iloc[:,:,0,:].rename("development", [9999])),axis=3)
        self.disposal_.is_cumulative = True
        self.disposal_.is_pattern = False
        self.incr_disposal_ = self.disposal_.cum_to_incr()
        self.incr_disposal_.is_pattern = True
        self.disposal_.is_pattern = True
        return self

    def transform(
            self, 
            X: Triangle, 
            sample_weight: Triangle | None = None
    ) -> Triangle:
        """ If X and self are of different shapes, align self to X, else
        return self.

        Parameters
        ----------
        X: Triangle
            The triangle to be transformed

        sample_weight: Triangle
            Ultimate

        Returns
        -------
            X_new: New triangle with transformed attributes.
        """
        if sample_weight is None:
            raise ValueError("sample_weight is required.")
        X_new = copy.deepcopy(X)
        X_new.disposal_rate_tri = self.disposal_rate_tri
        X_new.disposal_ = self.disposal_
        X_new.incr_disposal_ = self.incr_disposal_
        X_new.ultimate_ = sample_weight.latest_diagonal
        ibnr_pct = 1 - X_new.disposal_.align_pattern(X_new.disposal_rate_tri)
        run_off = X_new.incr_disposal_ / ibnr_pct * X_new.ibnr_
        run_off = run_off[run_off.valuation > X_new.valuation_date]
        X_new.ldf_ = (X_new.cum_to_incr() + run_off).incr_to_cum().age_to_age
        return X_new
    
    def fit_transform(self, X, y=None, sample_weight=None):
        """Fit and return predictions for VotingChainladder

        Parameters
        ----------
        X : Triangle
            Loss data to which the model will be applied.

        y : None
            Ignored

        sample_weight : Triangle, default=None
            Ultimate

        Returns
        -------
        X_new: Triangle
            Loss data with VotingChainladder ultimate applied
        """
        return self.fit(X, y, sample_weight).transform(X, sample_weight=sample_weight)
    def _test(self, X, ult):
        return 'test'