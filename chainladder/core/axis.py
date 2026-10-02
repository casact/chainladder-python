"""Axis descriptor for Triangle dimensions, analogous to Pandas AxisProperty."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable
import numpy as np
import pandas as pd

from chainladder.utils.utility_functions import _get_axis_name, _get_axis_number

if TYPE_CHECKING:
    from chainladder.core.triangle import Triangle


class TriangleAxis:
    """
    Generalized descriptor for representing a Triangle dimension,
    analogous to Pandas AxisProperty.

    Parameters
    ----------
    axis : int or str
        Axis number (0..3) or canonical name ('index', 'columns', 'origin',
        'development') used to store and access raw axis data in ``obj._axes``.
    fget : callable, optional
        Transformation function taking ``(obj, raw)`` and returning the
        public axis representation. If None, the raw value is returned.
    fset : callable, optional
        Transformation function taking ``(obj, value)`` and returning the
        raw axis representation to be stored in ``obj._axes``. If None,
        the value is stored as-is.
    doc : str, optional
        Docstring for the descriptor attribute on the class.
    """

    def __init__(
        self,
        axis: int | str,
        *,
        fget: Callable[[Triangle, Any], Any] | None = None,
        fset: Callable[[Triangle, Any], Any] | None = None,
        doc: str | None = None,
    ) -> None:
        """
        Initialize a TriangleAxis descriptor.

        Parameters
        ----------
        axis : int or str
            Axis number or name mapped to an integer in ``obj._axes``.
        fget : callable, optional
            Getter callable mapping ``(obj, raw)`` to public representation.
        fset : callable, optional
            Setter callable mapping ``(obj, value)`` to internal representation.
        doc : str, optional
            Docstring for the attribute.
        """
        self.axis: int = _get_axis_number(axis)
        self.key: int = self.axis
        self.fget = fget  # raw -> public; identity if None
        self.fset = fset  # (obj, public) -> raw; identity if None
        self.__doc__ = doc

    def __get__(self, obj: Triangle | None, objtype: type | None = None) -> Any:
        """
        Access the axis value on a Triangle instance or return descriptor on class.

        Parameters
        ----------
        obj : Triangle or None
            Instance of Triangle or None if accessed via class.
        objtype : type, optional
            Type of the owning class.

        Returns
        -------
        Any
            The descriptor itself if accessed on class, or the transformed
            axis value if accessed on an instance.

        Raises
        ------
        AttributeError
            If the axis key has not been initialized in ``obj._axes``.
        """
        if obj is None:
            return self
        if not hasattr(obj, "_axes") or self.axis not in obj._axes:
            axis_name = _get_axis_name(self.axis)
            raise AttributeError(
                f"'{type(obj).__name__}' object has no attribute '{axis_name}'"
            )
        raw = obj._axes[self.axis]
        return self.fget(obj, raw) if self.fget else raw

    def __set__(self, obj: Triangle, value: Any) -> None:
        """
        Set the axis value on a Triangle instance.

        Transforms ``value`` using ``fset`` if defined, stores the raw value
        in ``obj._axes[self.axis]``, and updates slicers if necessary.

        Parameters
        ----------
        obj : Triangle
            Instance of Triangle.
        value : Any
            New axis value provided by the user.
        """
        raw = self.fset(obj, value) if self.fset else value
        if not hasattr(obj, "_axes"):
            obj._axes = {}
        obj._axes[self.axis] = raw
        if hasattr(obj, "virtual_columns"):
            obj._set_slicers()


def _set_columns(obj: Triangle, value: Any) -> pd.Index:
    """
    Validate and transform assigned columns into a pandas Index.

    Parameters
    ----------
    obj : Triangle
        The Triangle instance being updated.
    value : Any
        The new column labels. Can be a string, list, or pandas Index.

    Returns
    -------
    pd.Index
        A pandas Index named ``"columns"``.
    """
    if isinstance(value, str):
        value = [value]
    if hasattr(obj, "values") and obj.values is not None:
        obj._len_check(range(obj.values.shape[1]), value)
    elif hasattr(obj, "_axes") and (1 in obj._axes or "columns" in obj._axes):
        obj._len_check(obj.columns, value)
    return pd.Index(value, name="columns")


def _set_index(obj: Triangle, value: Any) -> pd.DataFrame:
    """
    Validate and transform assigned index into a pandas DataFrame.

    Parameters
    ----------
    obj : Triangle
        The Triangle instance being updated.
    value : Any
        The new index labels. Must be a pandas DataFrame.

    Returns
    -------
    pd.DataFrame
        A reset-index pandas DataFrame.
    """
    if not isinstance(value, pd.DataFrame):
        raise TypeError("index must be a pandas DataFrame")
    if hasattr(obj, "values") and obj.values is not None:
        obj._len_check(range(obj.values.shape[0]), value)
    elif hasattr(obj, "_axes") and (0 in obj._axes or "index" in obj._axes):
        obj._len_check(obj.index, value)
    return value.copy().reset_index(drop=True)


def _get_origin(obj: Triangle, raw: Any) -> pd.PeriodIndex | pd.Series:
    """
    Transform raw origin timestamp array into public PeriodIndex or Series.
    """
    if obj.is_pattern and len(raw) == 1:
        return pd.Series(["(All)"])
    freq = {
        "S": "2Q",
        "H": "2Q",
    }.get(obj.origin_grain, obj.origin_grain)
    freq = freq if freq == "M" else freq + "-" + obj.origin_close
    return pd.DatetimeIndex(raw, name="origin").to_period(freq=freq)


def _set_origin(obj: Triangle, value: Any) -> np.ndarray:
    """
    Validate and transform assigned origin periods into internal timestamps array.
    """
    if hasattr(obj, "values") and obj.values is not None:
        obj._len_check(range(obj.values.shape[2]), value)
    elif hasattr(obj, "_axes") and (2 in obj._axes or "origin" in obj._axes):
        obj._len_check(obj.origin, value)
    freq = {
        "S": "2Q",
    }.get(obj.origin_grain, obj.origin_grain)
    freq = freq if freq == "M" else freq + "-" + obj.origin_close
    value = pd.PeriodIndex(list(value), freq=freq)
    return value.to_timestamp().values


def _get_development(obj: Triangle, raw: Any) -> pd.Series:
    """
    Transform raw development lags/timestamps into public Series.
    """
    ddims = raw.copy()
    if obj.is_val_tri:
        formats = {"Y": "%Y", "S": "%YQ%q", "Q": "%YQ%q", "M": "%Y-%m"}
        ddims = ddims.to_period(freq=obj.development_grain.replace("S", "2Q")).strftime(
            formats[obj.development_grain]
        )
    elif obj.is_pattern:
        offset = obj._dstep()["M"][obj.development_grain]
        if obj.is_ultimate:
            ddims[-1] = ddims[-2] + offset
        if obj.is_cumulative:
            ddims = ["{}-Ult".format(ddims[i]) for i in range(len(ddims))]
        else:
            ddims = [
                "{}-{}".format(ddims[i], ddims[i] + offset) for i in range(len(ddims))
            ]
    return pd.Series(list(ddims), name="development")


def _set_development(obj: Triangle, value: Any) -> np.ndarray:
    """
    Validate and transform assigned development periods into internal lag/timestamp array.
    """
    if hasattr(obj, "values") and obj.values is not None:
        obj._len_check(range(obj.values.shape[3]), value)
    elif hasattr(obj, "_axes") and (3 in obj._axes or "development" in obj._axes):
        obj._len_check(obj.development, value)
    return np.array([value] if isinstance(value, str) else value)


def _set_axis(obj: Triangle, axis: int | str, value: Any) -> None:
    """
    Set values on the specified axis of a Triangle instance.

    Parameters
    ----------
    obj : Triangle
        The Triangle instance to update.
    axis : int or str
        Axis identifier (0..3 or 'index', 'columns', 'origin', 'development').
    value : Any
        New axis values.
    """
    axis_name = _get_axis_name(axis)
    setattr(obj, axis_name, value)
