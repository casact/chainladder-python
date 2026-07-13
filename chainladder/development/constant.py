# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from chainladder.development.base import DevelopmentBase
import numpy as np
import pandas as pd


class DevelopmentConstant(DevelopmentBase):
    """An Estimator that allows for including of external patterns into a
        Development style model. When this estimator is fit against a triangle,
        only the grain of the existing triangle is retained.

    Parameters
    ----------
    patterns: dict or callable
        A dictionary key:value representation of age(in months):value. If callable
        is supplied, callable must return a dict for each element of the callable axis
    style: Literal['cdf', 'ldf'], optional (default='ldf')
        Type of pattern given to the Estimator. `cdf` for cumulative development factors
        and `ldf` for (non-cumulative) loss development factors.
    callable_axis: 0 or 1
        If a callable is supplied, the axis, index (0) or column (1) along which to apply
        the callable. If patterns is not a callable, then this parameter is ignored.
    groupby:
        option to group levels of the triangle index together for the purposes
        estimating patterns.  If omitted, each level of the triangle
        index will receive its own patterns.

    Attributes
    ----------
    ldf_: Triangle
        The estimated loss development patterns
    cdf_: Triangle
        The estimated cumulative development patterns

    Examples
    --------
    Commonly, we will have to use a pattern from another source, such as a pattern populated
    by the regulator to develop our triangle.

    Once we have the pattern, we can use the DevelopmentConstant estimator to develop our
    triangle. Further IBNR models will use these patterns to develop triangles to ultimate.

    .. testsetup::

        import chainladder as cl
        import numpy as np
        import pandas as pd

    .. testcode::

        xyz = cl.load_sample("xyz")
        reported_patterns = {
            12: 4.0,
            24: 2.9,
            36: 1.8,
            48: 1.4,
            60: 1.2,
            72: 1.1,
            84: 1.03,
            96: 1.02,
            108: 1.005,
        }
        print(np.round(cl.DevelopmentConstant(patterns=reported_patterns, style="cdf").fit_transform(
            xyz["Incurred"]
        ).ldf_,4))
        print(np.round(cl.DevelopmentConstant(patterns=reported_patterns, style="cdf").fit_transform(
            xyz["Incurred"]
        ).cdf_,4))

    .. testoutput::

                12-24   24-36   36-48   48-60   60-72  72-84   84-96  96-108  108-120  120-132
        (All)  1.3793  1.6111  1.2857  1.1667  1.0909  1.068  1.0098  1.0149    1.005      1.0
               12-Ult  24-Ult  36-Ult  48-Ult  60-Ult  72-Ult  84-Ult  96-Ult  108-Ult  120-Ult
        (All)     4.0     2.9     1.8     1.4     1.2     1.1    1.03    1.02    1.005      1.0

    When patterns vary by triangle index (such as LOBs), pass a callable to ``patterns`` and set
    ``callable_axis=0`` to apply it along the index.

    .. testcode::

        clrd = cl.load_sample("clrd")
        agway = clrd.loc["Agway Ins Co", "CumPaidLoss"]
        cdfs = {
            "comauto": [3.832, 1.874, 1.386, 1.181, 1.085, 1.043, 1.022, 1.013, 1.007, 1],
            "medmal": [24.168, 4.127, 2.103, 1.528, 1.275, 1.161, 1.088, 1.047, 1.018, 1],
            "othliab": [10.887, 3.416, 1.957, 1.433, 1.231, 1.119, 1.06, 1.031, 1.011, 1],
            "ppauto": [2.559, 1.417, 1.181, 1.084, 1.04, 1.019, 1.009, 1.004, 1.001, 1],
            "prodliab": [13.703, 5.613, 2.92, 1.765, 1.385, 1.177, 1.072, 1.034, 1.008, 1],
            "wkcomp": [4.106, 1.865, 1.418, 1.234, 1.141, 1.09, 1.056, 1.03, 1.01, 1],
        }
        patterns = pd.DataFrame(cdfs, index=range(12, 132, 12)).T
        model = cl.DevelopmentConstant(
            patterns=lambda x: patterns.loc[x.loc["LOB"]].to_dict(),
            callable_axis=0,
            style="cdf",
        )
        print(model.fit_transform(agway).cdf_.loc["comauto"])

    .. testoutput::

               12-Ult  24-Ult  36-Ult  48-Ult  60-Ult  72-Ult  84-Ult  96-Ult  108-Ult  120-Ult
        (All)   3.832   1.874   1.386   1.181   1.085   1.043   1.022   1.013    1.007      1.0

    """

    def __init__(self, patterns=None, style="ldf", callable_axis=0, groupby=None):
        self.patterns = patterns
        self.style = style
        self.callable_axis = callable_axis
        self.groupby = groupby

    def fit(self, X, y=None, sample_weight=None):
        """Fit the model with X.
        
        Parameters
        ----------
        X : Triangle-like
            Set of LDFs to which the munich adjustment will be applied.
        y : Ignored
        sample_weight : Ignored
        
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        from chainladder import options

        if X.is_cumulative == False:
            obj = self._set_fit_groups(X).incr_to_cum().val_to_dev().copy()
        else:
            obj = self._set_fit_groups(X).val_to_dev().copy()

        xp = obj.get_array_module()

        if callable(self.patterns):
            if self.callable_axis == 0:
                sample_pattern = obj.index.apply(self.patterns, axis=1).iloc[0]
            elif self.callable_axis == 1:
                sample_pattern = (
                    obj.columns.to_frame(index=False)
                    .apply(self.patterns, axis=1)
                    .iloc[0]
                )
            else:
                raise ValueError("callable axis needs to be 0 or 1")
            pattern_length = len(sample_pattern)
            pattern_ddims = sorted(sample_pattern.keys())
        else:
            pattern_length = len(self.patterns)
            pattern_ddims = sorted(self.patterns.keys())

        # pattern supplied is much shorter than the triangle
        if pattern_length < len(obj.ddims) - 1:
            obj = obj.iloc[..., 0, :-1] * 0 + 1
        # pattern supplied is exactly one short of the triangle
        elif pattern_length == len(obj.ddims) - 1:
            obj = obj.iloc[..., 0, :-1] * 0 + 1
        # pattern supplied is exactly the same length as the triangle
        elif pattern_length == len(obj.ddims):
            obj = obj.iloc[..., 0, :] * 0 + 1
        # pattern supplied is longer than the triangle
        else:
            obj = obj.iloc[..., 0, :] * 0 + 1
            extra = len(pattern_ddims) - len(obj.ddims)
            if extra > 0:
                tail = xp.ones(obj.shape)[..., -1:]
                tail = xp.repeat(tail, extra, -1)
                obj.values = xp.concatenate((obj.values, tail), -1)
                obj.ddims = np.array(pattern_ddims)
                obj._set_slicers()

        if callable(self.patterns):
            if self.callable_axis == 0:
                ldf = obj.index.apply(self.patterns, axis=1)
                ldf = (
                    pd.concat(ldf.apply(pd.DataFrame, index=[0]).values, axis=0)
                    .fillna(1)[obj.ddims]
                    .values
                )
                ldf = xp.array(ldf[:, None, None, :])
            elif self.callable_axis == 1:
                ldf = obj.columns.to_frame(index=False).apply(self.patterns, axis=1)
                ldf = (
                    pd.concat(ldf.apply(pd.DataFrame, index=[0]).values, axis=0)
                    .fillna(1)[obj.ddims]
                    .values
                )
                ldf = xp.array(ldf[None, :, None, :])
            else:
                raise ValueError("callable axis needs to be 0 or 1")
        else:
            ldf = xp.array([self.patterns.get(item, 1.0) for item in obj.ddims])
            ldf = ldf[None, None, None, :]

        if self.style == "cdf":
            ldf = xp.concatenate((ldf[..., :-1] / ldf[..., 1:], ldf[..., -1:]), -1)

        obj = obj * ldf
        obj._set_slicers()

        self.ldf_ = obj
        self.ldf_.is_pattern = True
        self.ldf_.is_cumulative = False
        self.ldf_.valuation_date = pd.to_datetime(options.ULT_VAL)
        return self

    def transform(self, X):
        """If X and self are of different shapes, align self to X, else
        return self.

        Parameters
        ----------
        X : Triangle
            The triangle to be transformed

        Returns
        -------
            X_new : New triangle with transformed attributes.
        """
        X_new = X.copy()
        X_new.group_index = self._set_transform_groups(X_new)
        triangles = ["ldf_"]
        for item in triangles:
            setattr(X_new, item, getattr(self, item))
        X_new._set_slicers()
        return X_new
