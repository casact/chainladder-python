# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from chainladder.methods import Chainladder, MethodBase
from chainladder.development import DevelopmentBase
import numpy as np
import copy
from chainladder.utils import TriangleWeight, concat

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chainladder.core import Triangle

class DisposalMixin:
    '''
    This class provides attributes for the DisposalRate adjustment method and transformed `Triangle`    
    '''

    @property
    def disposal_rate_(self) -> Triangle:
        '''
        Gets the estimated disposal rate
        '''
        if not hasattr(self, "_disposal_rate_"):
            x = self.__class__.__name__
            raise AttributeError("'" + x + "' object has no attribute 'disposal_rate_'")
        return self._disposal_rate_
    
    @disposal_rate_.setter
    def disposal_rate_(self,value) -> None:
        '''
        Sets disposal_rate_
        '''
        obj = copy.deepcopy(value)
        obj.is_pattern = True
        obj.is_disposal_rate = True
        obj.is_cumulative = True
        self._disposal_rate_ = obj

    @property
    def incr_disposal_rate_(self) -> Triangle:
        '''
        Gets the incremental of the estimated disposal rate
        '''
        return self.disposal_rate_.cum_to_incr()

    @incr_disposal_rate_.setter
    def incr_disposal_rate_(self,value) -> None:
        '''
        Sets incr_disposal_rate_
        '''
        obj = copy.deepcopy(value)
        obj.is_pattern = True
        obj.is_disposal_rate = True
        obj.is_cumulative = False
        self._disposal_rate_ = obj.incr_to_cum()

class DisposalRate(DevelopmentBase, DisposalMixin):
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
    disposal_rate_: Triangle
        fitted disposal rates

    incr_disposal_rate_: Triangle
        incremental of disposal_

    Examples
    --------
    This adjustment method re-apportions future loss emergence based on a '% of ultimate' emergence pattern. 
    The ultimate can come from another triangle. A common use case is to forecast payment pattern based on incurred ultimate.
    
    .. testsetup::

        import chainladder as cl
        import numpy as np

    .. testcode::

        clrd = cl.load_sample('clrd').sum()
        ult = cl.Chainladder().fit(clrd['IncurLoss']).ultimate_
        dr = cl.DisposalRate().fit_transform(clrd['CumPaidLoss'],sample_weight = ult)

    Once we apply this adjustment method via a `fit_transform`, we can examine the emergence pattern via `disposal_rate_tri`. 

    .. testcode::
    
        dr.disposal_rate_tri

    .. testoutput::
        
                12        24        36        48        60        72        84        96        108       120
        1988  0.313923  0.619459  0.774429  0.865377  0.919077  0.948898  0.964643  0.973184  0.980224  0.983063
        1989  0.321526  0.626023  0.781086  0.872345  0.924842  0.952533  0.967690  0.977373  0.981938       NaN
        1990  0.329567  0.634056  0.790752  0.880273  0.927029  0.952951  0.968379  0.976049       NaN       NaN
        1991  0.330035  0.636233  0.791888  0.881010  0.929460  0.954694  0.968533       NaN       NaN       NaN
        1992  0.342613  0.650521  0.801875  0.885976  0.932865  0.956495       NaN       NaN       NaN       NaN
        1993  0.353784  0.663303  0.810639  0.894414  0.939009       NaN       NaN       NaN       NaN       NaN
        1994  0.367530  0.670460  0.814661  0.897244       NaN       NaN       NaN       NaN       NaN       NaN
        1995  0.379650  0.680979  0.821603       NaN       NaN       NaN       NaN       NaN       NaN       NaN
        1996  0.395603  0.688621       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN
        1997  0.393820       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN

    The estimated pattern is stored in `disposal_rate_`. 

    .. testcode::

        dr.disposal_rate_
        
    .. testoutput::

                12-Ult    24-Ult    36-Ult    48-Ult    60-Ult    72-Ult    84-Ult    96-Ult   108-Ult  120-Ult  132-Ult
        (All)  0.112105  0.336242  0.545897  0.693774  0.812877  0.905045  0.942998  0.974365  0.990868      1.0      1.0

    `full_triangle_` now reflects the disposal-rate-based forecast. 

    .. testcode::

        dr.full_triangle_
        
    .. testoutput::

                12            24            36            48            60            72            84            96            108           120           9999
        1988  3577780.0  7.059966e+06  8.826151e+06  9.862687e+06  1.047470e+07  1.081458e+07  1.099401e+07  1.109136e+07  1.117159e+07  1.120395e+07  1.139698e+07
        1989  4090680.0  7.964702e+06  9.937520e+06  1.109859e+07  1.176649e+07  1.211879e+07  1.231163e+07  1.243483e+07  1.249290e+07  1.251646e+07  1.272270e+07
        1990  4578442.0  8.808486e+06  1.098535e+07  1.222900e+07  1.287854e+07  1.323867e+07  1.345299e+07  1.355956e+07  1.363458e+07  1.366101e+07  1.389229e+07
        1991  4648756.0  8.961755e+06  1.115424e+07  1.240959e+07  1.309204e+07  1.344748e+07  1.364241e+07  1.375400e+07  1.382878e+07  1.385512e+07  1.408564e+07
        1992  5139142.0  9.757699e+06  1.202798e+07  1.328948e+07  1.399282e+07  1.434727e+07  1.454438e+07  1.465904e+07  1.473589e+07  1.476295e+07  1.499983e+07
        1993  5653379.0  1.059942e+07  1.295381e+07  1.429252e+07  1.500514e+07  1.533589e+07  1.553037e+07  1.564351e+07  1.571933e+07  1.574603e+07  1.597976e+07
        1994  6246447.0  1.139496e+07  1.384576e+07  1.524933e+07  1.593547e+07  1.629529e+07  1.650686e+07  1.662994e+07  1.671242e+07  1.674147e+07  1.699574e+07
        1995  6473843.0  1.161215e+07  1.401010e+07  1.527961e+07  1.597603e+07  1.634123e+07  1.655597e+07  1.668088e+07  1.676460e+07  1.679409e+07  1.705215e+07
        1996  6591599.0  1.147391e+07  1.365956e+07  1.491261e+07  1.559998e+07  1.596045e+07  1.617240e+07  1.629570e+07  1.637833e+07  1.640743e+07  1.666215e+07
        1997  6451896.0  1.106345e+07  1.330436e+07  1.458909e+07  1.529384e+07  1.566342e+07  1.588073e+07  1.600714e+07  1.609186e+07  1.612170e+07  1.638286e+07

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
        #validate dimensions of sample weight
        MethodBase().validate_weight(X, sample_weight)
        #set backeneds to numpy
        if X.array_backend == "sparse":
            X = X.set_backend("numpy")
        else:
            X = X.copy()
        if sample_weight.array_backend == "sparse":
            ult = sample_weight.set_backend("numpy")
        else:
            ult = sample_weight.copy()
        #calculate disposal rate triangle
        self.xp = X.get_array_module()
        self.X_ = X.incr_to_cum().sort_index()
        self.X_.ultimate_ = ult
        #get weights for estimation
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
        if hasattr(self.X_, "disposal_w_"):
            self.disposal_w_ = tw.fit(X=self.X_.disposal_rate_tri * self.X_.disposal_w_).w_.values
        else:
            self.disposal_w_ = tw.fit(X=self.X_.disposal_rate_tri).w_.values
        #calculate factors
        super().fit(ult.values,self.X_.values,self.disposal_w_)
        #keep attributes
        disposal = self._param_property(self.X_.disposal_rate_tri,self.params_.slope_[...,0][..., None, :])
        self.disposal_rate_ = concat((disposal,(self.X_.latest_diagonal*0 + 1).iloc[:,:,0,:].rename("development", [9999])),axis=3)
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
        #validate dimensions of sample weight
        MethodBase().validate_weight(X, sample_weight)
        #align backeneds
        X_new.disposal_w_ = self.disposal_w_
        X_new.ultimate_ = sample_weight.set_backend(self.X_.array_backend).latest_diagonal
        X_new.disposal_rate_ = self.disposal_rate_
        ibnr_pct = 1 - X_new.disposal_rate_.align_pattern(X_new.disposal_rate_tri)
        run_off = X_new.incr_disposal_rate_ / ibnr_pct * X_new.ibnr_
        run_off = run_off[run_off.valuation > X_new.valuation_date]
        X_new.ldf_ = (X_new.cum_to_incr() + run_off).incr_to_cum().age_to_age
        return X_new
    
    def fit_transform(self, X, y=None, sample_weight=None):
        """Fit and return transformed full_triangle_ based on the Disposal Rate

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
            Triangle with new full_triangle_
        """
        return self.fit(X, y, sample_weight).transform(X, sample_weight=sample_weight)