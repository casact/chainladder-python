# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from typing import Any, Literal

from chainladder.development.base import DevelopmentBase
import numpy as np
import pandas as pd
import warnings


class DevelopmentConstant(DevelopmentBase):
    """
    An Estimator that allows for including of external patterns into a
    Development style model. When this estimator is fit against a triangle,
    only the grain of the existing triangle is retained.

    Parameters
    ----------
    patterns: dict, Series, DataFrame, Triangle, estimator, or callable
        A representation of development factors indexed by age (in months). Can be
        a dictionary of age:value, a pandas Series, a 1-row or 1-column DataFrame,
        a 1D Triangle pattern (such as another estimator's ``ldf_`` or ``cdf_``),
        or a fitted estimator. When passing a fitted estimator, the pattern matching
        ``style`` (``ldf_`` or ``cdf_``) is used. If the estimator only provides one
        pattern attribute (e.g. only ``cdf_`` or only ``ldf_``), it is used as a
        convenience fallback and a UserWarning is issued if ``style`` does not match.
        If callable is supplied, callable must return a dict for each element of the
        callable axis.
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

    def __init__(
        self,
        patterns: Any = None,
        style: Literal["cdf", "ldf"] = "ldf",
        callable_axis: int = 0,
        groupby: Any = None,
    ):
        self.patterns = patterns
        self.style = style
        self.callable_axis = callable_axis
        self.groupby = groupby

    def fit(self, X, y=None, sample_weight=None):
        """
        Fit the model with X.

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

        if not X.is_cumulative:
            obj = self._set_fit_groups(X).incr_to_cum().val_to_dev().copy()
        else:
            obj = self._set_fit_groups(X).val_to_dev().copy()

        xp = obj.get_array_module()

        if callable(self.patterns):
            if self.callable_axis == 0:
                sample_pattern = obj.index.apply(self.patterns, axis=1).iloc[0]
            elif self.callable_axis == 1:
                sample_pattern = (
                    obj.columns
                    .to_frame(index=False)
                    .apply(self.patterns, axis=1)
                    .iloc[0]
                )
            else:
                raise ValueError("callable axis needs to be 0 or 1")
            pattern_length = len(sample_pattern)
            pattern_dev = sorted(sample_pattern.keys())
            patterns_dict = None
        else:
            patterns_dict = self._normalize_patterns(self.patterns, self.style)
            pattern_length = len(patterns_dict)
            pattern_dev = sorted(patterns_dict.keys())

        # pattern supplied is much shorter than the triangle
        if pattern_length < len(obj.development) - 1:
            obj = obj.iloc[..., 0, :-1] * 0 + 1
        # pattern supplied is exactly one short of the triangle
        elif pattern_length == len(obj.development) - 1:
            obj = obj.iloc[..., 0, :-1] * 0 + 1
        # pattern supplied is exactly the same length as the triangle
        elif pattern_length == len(obj.development):
            obj = obj.iloc[..., 0, :] * 0 + 1
        # pattern supplied is longer than the triangle
        else:
            obj = obj.iloc[..., 0, :] * 0 + 1
            extra = len(pattern_dev) - len(obj.development)
            if extra > 0:
                tail = xp.ones(obj.shape)[..., -1:]
                tail = xp.repeat(tail, extra, -1)
                obj.values = xp.concatenate((obj.values, tail), -1)
                obj.development = np.array(pattern_dev)
                obj._set_slicers()

        if callable(self.patterns):
            if self.callable_axis == 0:
                ldf = obj.index.apply(self.patterns, axis=1)
                ldf = (
                    pd
                    .concat(ldf.apply(pd.DataFrame, index=[0]).values, axis=0)
                    .fillna(1)[obj.development]
                    .values
                )
                ldf = xp.array(ldf[:, None, None, :])
            elif self.callable_axis == 1:
                ldf = obj.columns.to_frame(index=False).apply(self.patterns, axis=1)
                ldf = (
                    pd
                    .concat(ldf.apply(pd.DataFrame, index=[0]).values, axis=0)
                    .fillna(1)[obj.development]
                    .values
                )
                ldf = xp.array(ldf[None, :, None, :])
            else:
                raise ValueError("callable axis needs to be 0 or 1")
        else:
            ldf = xp.array([patterns_dict.get(item, 1.0) for item in obj.development])
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

    @staticmethod
    def _normalize_patterns(patterns, style="ldf"):
        """Normalize pattern input (dict, Series, DataFrame, Triangle, estimator) into a dict."""
        is_triangle = (
            not isinstance(patterns, (pd.DataFrame, pd.Series, dict))
            and hasattr(patterns, "shape")
            and len(patterns.shape) == 4
            and hasattr(patterns, "development")
        )
        if not is_triangle:
            if hasattr(patterns, "cdf_") and style == "cdf":
                patterns = patterns.cdf_
            elif hasattr(patterns, "ldf_") and style == "ldf":
                patterns = patterns.ldf_
            elif hasattr(patterns, "cdf_") and not hasattr(patterns, "ldf_"):
                warnings.warn(
                    f"Estimator provides only 'cdf_', which does not match style='{style}'. "
                    f"Using 'cdf_' as fallback pattern.",
                    UserWarning,
                    stacklevel=2,
                )
                patterns = patterns.cdf_
            elif hasattr(patterns, "ldf_") and not hasattr(patterns, "cdf_"):
                warnings.warn(
                    f"Estimator provides only 'ldf_', which does not match style='{style}'. "
                    f"Using 'ldf_' as fallback pattern.",
                    UserWarning,
                    stacklevel=2,
                )
                patterns = patterns.ldf_

        if (
            not isinstance(patterns, (pd.DataFrame, pd.Series, dict))
            and hasattr(patterns, "shape")
            and len(patterns.shape) == 4
            and hasattr(patterns, "development")
        ):
            if patterns.shape[2] > 1:
                raise ValueError(
                    "Triangle pattern must have a single origin row (e.g. model.ldf_ or model.cdf_)."
                )
            vals = patterns.values[0, 0, 0, :]
            vals = vals.todense() if hasattr(vals, "todense") else np.asarray(vals)
            dev = patterns.development
            s = pd.Series(vals, index=dev, dtype=float)
        elif isinstance(patterns, pd.DataFrame):
            if patterns.shape[0] == 1:
                s = patterns.iloc[0].astype(float)
            elif patterns.shape[1] == 1:
                s = patterns.iloc[:, 0].astype(float)
            else:
                raise ValueError("DataFrame pattern must have 1 row or 1 column.")
        elif isinstance(patterns, pd.Series):
            s = patterns.astype(float).copy()
        elif isinstance(patterns, dict):
            s = pd.Series(patterns, dtype=float)
        else:
            raise TypeError(
                f"Unsupported pattern type: {type(patterns)}. "
                f"Expected dict, Series, DataFrame, Triangle, estimator, or callable."
            )

        parsed_index = []
        for x in s.index:
            try:
                parsed_index.append(int(str(x).split("-")[0]))
            except (ValueError, TypeError):
                parsed_index.append(x)
        s.index = parsed_index
        return s.to_dict()

    def transform(self, X):
        """
        If X and self are of different shapes, align self to X, else
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
