# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from chainladder.development.base import DevelopmentBase
import pandas as pd
import numpy as np
import warnings


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

        # print("In DevelopmentConstant fit")

        # convert to cumulative triangle
        if X.is_cumulative == False:
            obj = self._set_fit_groups(X).incr_to_cum().val_to_dev().copy()
        else:
            obj = self._set_fit_groups(X).val_to_dev().copy()

        n_dev_periods = len(obj.ddims)
        xp = obj.get_array_module()

        if callable(self.patterns):
            if self.callable_axis == 0:  # varying patterns by index
                patterns = self.patterns(obj.index.iloc[0])
            elif self.callable_axis == 1:  # varying patterns by column
                patterns = self.patterns(obj.columns.to_frame(index=False).iloc[0])
            else:
                raise ValueError("callable axis needs to be 0 or 1")

        else:  # static patterns
            patterns = self.patterns

        # convert patterns to CDFs so it's easier to work with
        sorted_keys = sorted(patterns.keys())
        pattern_values = np.array([float(patterns[k]) for k in sorted_keys])
        if self.style == "ldf":
            cdf_patterns = dict(
                zip(sorted_keys, np.cumprod(pattern_values[::-1])[::-1])
            )
        else:
            cdf_patterns = patterns

        print("CDF patterns\n", cdf_patterns)

        # patterns provided is longer than the triangle development periods,
        # this step resizes and gets tail_cdf to apply to the tail of the triangle later
        if len(cdf_patterns) > len(obj.ddims) - 1:
            # print("patterns is longer than the triangle development periods")

            tail_key = sorted_keys[len(obj.ddims) - 1]
            tail_cdf = cdf_patterns[tail_key]

            if tail_cdf == 1:
                obj = obj.iloc[..., :1, :-1] * 0 + 1
            else:
                obj = obj.iloc[..., :1, :] * 0 + 1
        else:
            obj = obj.iloc[..., :1, :-1] * 0 + 1
            tail_key = None
            tail_cdf = 1

        # warn if the patterns are shorter than the triangle development periods
        if len(cdf_patterns) < n_dev_periods:
            warnings.warn(
                "Supplied patterns are shorter than the triangle development "
                "periods. Missing ages will be filled with a factor of 1.0.",
                UserWarning,
                stacklevel=2,
            )

        # fill the cdf_patterns dictionary with 1.0 for any development periods
        for ddim in obj.ddims:
            if not any(ddim == k or int(ddim) == int(k) for k in cdf_patterns):
                cdf_patterns[int(ddim)] = 1.0
            if self.style == "ldf" and not any(
                ddim == k or int(ddim) == int(k) for k in patterns
            ):
                patterns[int(ddim)] = 1.0

        print("obj to fill\n", obj)
        print("tail_cdf", tail_key, tail_cdf)

        # from chainladder.tails import TailConstant

        # tail = TailConstant(tail=tail_cdf, projection_period=0).fit(obj)
        # print("tail.cdf_\n", tail.cdf_)

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

        else:
            fit_patterns = patterns if self.style == "ldf" else cdf_patterns
            ldf = xp.array([float(fit_patterns[int(item)]) for item in obj.ddims])
            ldf = ldf[None, None, None, :]

        if self.style == "cdf":
            ldf = xp.concatenate((ldf[..., :-1] / ldf[..., 1:], ldf[..., -1:]), -1)

        print("final ldf\n", ldf)
        obj = obj * ldf
        # tail = TailConstant(tail=tail_cdf, projection_period=0).fit(obj)
        # print("tail.cdf_\n", tail.cdf_)
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
