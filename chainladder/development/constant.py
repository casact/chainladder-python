# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from chainladder.development.base import DevelopmentBase
import pandas as pd
import numpy as np


class DevelopmentConstant(DevelopmentBase):
    """A Estimator that allows for including of external patterns into a
        Development style model. When this estimator is fit against a triangle,
        only the grain of the existing triangle is retained.

    Parameters
    ----------
    patterns: dict or callable
        A dictionary key:value representation of age(in months):value. If callable
        is supplied, callable must return a dict for each element of the callable axis
    style: string, optional (default='ldf')
        Type of pattern given to the Estimator. Options include 'cdf' or 'ldf'.
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

        print("In DevelopmentConstant fit")

        # convert to cumulative triangle
        if X.is_cumulative == False:
            obj = self._set_fit_groups(X).incr_to_cum().val_to_dev().copy()
        else:
            obj = self._set_fit_groups(X).val_to_dev().copy()

        xp = obj.get_array_module()

        if callable(self.patterns):
            if self.callable_axis == 0:  # varying patterns by index
                pattern = self.patterns(obj.index.iloc[0])
            elif self.callable_axis == 1:  # varying patterns by column
                pattern = self.patterns(obj.columns.to_frame(index=False).iloc[0])
            else:
                raise ValueError("callable axis needs to be 0 or 1")

        else:  # static patterns
            pattern = self.patterns

        print("pattern\n", pattern)
        # print("len(pattern)\n", len(pattern))
        # print("len(obj.ddims)\n", len(obj.ddims))

        # the pattern provided is longer than the development triangle
        if len(pattern) > len(obj.ddims) - 1:
            print("len(pattern) > len(obj.ddims) - 1")

            from chainladder.tails import TailConstant

            sorted_keys = sorted(pattern.keys())
            print("sorted_keys\n", sorted_keys)
            tail_cdf = pattern[sorted_keys[len(obj.ddims) - 1]]
            print("tail_cdf\n", tail_cdf)

            normalized_pattern_values = np.array(list(pattern.values())) / tail_cdf

            # Zip the original keys back with the new vectorized array
            pattern = dict(zip(pattern.keys(), normalized_pattern_values))

            print("pattern\n", pattern)

            tail = TailConstant(tail=tail_cdf, projection_period=0).fit(obj)
            print("tail.cdf_\n", tail.cdf_)

            if tail_cdf == 1:
                obj = tail.ldf_.iloc[..., :1, :-1] * 0 + 1
            else:
                obj = tail.ldf_.iloc[..., :1, :] * 0 + 1

        else:
            print("len(pattern) < len(obj.ddims)")
            obj = obj.iloc[..., :1, :-1] * 0 + 1

        print("obj\n", obj)

        if callable(self.patterns):
            if self.callable_axis == 0:
                ldf = obj.index.apply(self.patterns, axis=1)
            elif self.callable_axis == 1:
                ldf = obj.columns.to_frame(index=False).apply(self.patterns, axis=1)
            else:
                raise ValueError("callable axis needs to be 0 or 1")
            ldf = (
                pd.concat(ldf.apply(pd.DataFrame, index=[0]).values, axis=0)
                .fillna(1)[obj.ddims]
                .values
            )
            if self.callable_axis == 0:
                ldf = xp.array(ldf[:, None, None, :])
            else:
                ldf = xp.array(ldf[None, :, None, :])
            print("ldf\n", ldf)

        else:
            print("self.patterns\n", self.patterns)
            print("obj.ddims\n", obj.ddims)
            ldf = xp.array([float(self.patterns[item]) for item in obj.ddims])
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
