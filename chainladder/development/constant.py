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

    def _prepare_cdf_patterns(self, patterns, n_dev_periods):

        patterns = dict(patterns)

        sorted_keys = sorted(patterns.keys())
        pattern_values = np.array([float(patterns[k]) for k in sorted_keys])

        # convert ldfs to cdfs; cdf patterns are used as-is
        if self.style == "ldf":
            cdf_values = np.cumprod(pattern_values[::-1])[::-1]
        else:
            cdf_values = pattern_values

        cdf_patterns = {int(k): float(v) for k, v in zip(sorted_keys, cdf_values)}

        # patterns that fit within the triangle have no tail
        if len(cdf_patterns) <= n_dev_periods:
            return cdf_patterns, 1.0

        # separate the tail factor and rebase the remaining cdfs onto it
        tail_cdf = cdf_patterns[int(sorted_keys[n_dev_periods])]
        for k in sorted_keys[:n_dev_periods]:
            cdf_patterns[int(k)] /= tail_cdf

        return cdf_patterns, tail_cdf

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

        # convert to cumulative triangle
        if not X.is_cumulative:
            obj = self._set_fit_groups(X).incr_to_cum().val_to_dev().copy()
        else:
            obj = self._set_fit_groups(X).val_to_dev().copy()

        xp = obj.get_array_module()
        tri_dev_periods = len(obj.ddims)

        if callable(self.patterns):
            # on index
            if self.callable_axis == 0:
                rows = obj.index
            # on columns
            elif self.callable_axis == 1:
                rows = obj.columns.to_frame(index=False)
            else:
                raise ValueError("callable axis needs to be 0 or 1")

            patterns = self.patterns(rows.iloc[0])
        else:
            # force the patterns to a dictionary
            patterns = dict(self.patterns)

        # separate the cdf patterns from the tail; _prepare_cdf_patterns already
        # returns tail_cdf=1 when the patterns do not extend past the triangle.
        cdf_patterns, tail_cdf = self._prepare_cdf_patterns(patterns, tri_dev_periods)

        # drops everything in the tail that is 1, recursively
        while (
            cdf_patterns[max(cdf_patterns)] == 1 and len(cdf_patterns) > tri_dev_periods
        ):
            del cdf_patterns[max(cdf_patterns)]

        pattern_dev_periods = len(cdf_patterns)

        # determine whether to include the last development period in the patterns
        if pattern_dev_periods < tri_dev_periods:
            warnings.warn(
                "Supplied patterns are shorter than the triangle development "
                "periods. Missing ages will be filled with a factor of 1.0.",
                UserWarning,
                stacklevel=2,
            )
            include_last = False

        elif pattern_dev_periods == tri_dev_periods:
            include_last = True

        else:
            include_last = tail_cdf != 1

        dev_slice = slice(None) if include_last else slice(None, -1)

        # this is the object to fill out the patterns, skeleton frame
        obj = obj.iloc[..., :1, dev_slice] * 0 + 1

        if callable(self.patterns):

            def _callable_row(row):
                raw_patterns = self.patterns(row)
                cdf_row, row_tail_cdf = self._prepare_cdf_patterns(
                    raw_patterns, tri_dev_periods
                )
                fit_row = raw_patterns if self.style == "ldf" else cdf_row
                return dict(fit_row), row_tail_cdf

            prepared = rows.apply(_callable_row, axis=1)
            ldf = (
                pd.concat(
                    [pd.DataFrame(item[0], index=[0]) for item in prepared],
                    axis=0,
                )
                .fillna(1)[obj.ddims]
                .values
            )
            tail_cdfs = xp.array([item[1] for item in prepared])

            if self.callable_axis == 0:
                ldf = xp.array(ldf[:, None, None, :])
                tail_cdfs = tail_cdfs[:, None, None]
            else:
                ldf = xp.array(ldf[None, :, None, :])
                tail_cdfs = tail_cdfs[None, :, None]

        else:
            fit_patterns = patterns if self.style == "ldf" else cdf_patterns

            # fill any triangle ages missing from the patterns with a factor of 1.0
            for ddim in obj.ddims:
                if not any(ddim == k or int(ddim) == int(k) for k in fit_patterns):
                    fit_patterns[int(ddim)] = 1.0

            ldf = xp.array([float(fit_patterns[int(item)]) for item in obj.ddims])
            ldf = ldf[None, None, None, :]
            tail_cdfs = tail_cdf

        if self.style == "cdf":
            ldf = xp.concatenate((ldf[..., :-1] / ldf[..., 1:], ldf[..., -1:]), -1)

        # apply tail_cdf to the last ldfs of the triangle
        ldf[..., -1] = ldf[..., -1] * tail_cdfs

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
