"""Axis descriptor for Triangle dimensions, analogous to Pandas AxisProperty."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable
import numpy as np
import pandas as pd

from chainladder.core.base import TriangleBase
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

    def set_unchecked(self, obj: Triangle, value: Any) -> None:
        """
        Set the axis from a value, skipping the length check.

        Used when an axis is widened ahead of ``values`` (arithmetic broadcast)
        or when slicing/reindexing internal coordinates.
        """
        if self.fset is _set_origin or self.fset is _set_development:
            raw = self.fset(obj, value, check=False)
        else:
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
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__"):
        value = [value]
    if hasattr(obj, "_len_check"):
        if hasattr(obj, "values") and obj.values is not None:
            obj._len_check(range(obj.values.shape[1]), value)
        elif (
            not isinstance(obj, TriangleBase)
            and hasattr(obj, "_axes")
            and (1 in obj._axes or "columns" in obj._axes)
        ):
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
    if hasattr(obj, "_len_check"):
        if hasattr(obj, "values") and obj.values is not None:
            obj._len_check(range(obj.values.shape[0]), value)
        elif (
            not isinstance(obj, TriangleBase)
            and hasattr(obj, "_axes")
            and (0 in obj._axes or "index" in obj._axes)
        ):
            obj._len_check(obj.index, value)
    return value.copy().reset_index(drop=True)


def _get_origin(obj: Triangle, raw: Any) -> pd.PeriodIndex | pd.Series:
    """
    Transform raw origin timestamp array into public PeriodIndex or Series.
    """
    if obj.is_pattern and len(raw) == 1:
        return pd.Series(["(All)"])
    if raw is None:
        return None
    freq = {
        "S": "2Q",
        "H": "2Q",
    }.get(obj.origin_grain, obj.origin_grain)
    freq = freq if (freq == "M" or "-" in freq) else freq + "-" + obj.origin_close
    return pd.DatetimeIndex(raw, name="origin").to_period(freq=freq)


def _set_origin(obj: Triangle, value: Any, check: bool = True) -> np.ndarray:
    """
    Validate and transform assigned origin periods into internal timestamps array.
    """
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__"):
        value = [value]
    if check and hasattr(obj, "_len_check"):
        if hasattr(obj, "values") and obj.values is not None:
            obj._len_check(range(obj.values.shape[2]), value)
        elif (
            not isinstance(obj, TriangleBase)
            and hasattr(obj, "_axes")
            and (2 in obj._axes or "origin" in obj._axes)
        ):
            raw_orig = obj._axes.get(2, obj._axes.get("origin"))
            if raw_orig is not None:
                obj._len_check(raw_orig, value)
    if isinstance(value, pd.PeriodIndex):
        return value.to_timestamp().values
    if isinstance(value, pd.DatetimeIndex):
        return value.values
    if isinstance(value, np.ndarray) and value.dtype.kind == "M":
        return value
    if len(value) == 1 and (
        (isinstance(value[0], str) and value[0] == "(All)")
        or (hasattr(value, "iloc") and str(value.iloc[0]) == "(All)")
    ):
        if hasattr(obj, "_axes") and (2 in obj._axes or "origin" in obj._axes):
            raw_orig = obj._axes.get(2, obj._axes.get("origin"))
            if raw_orig is not None and len(raw_orig) == 1:
                return raw_orig
        return pd.date_range("2000", periods=1).values
    origin_grain = getattr(obj, "origin_grain", "Y")
    origin_close = getattr(obj, "origin_close", "DEC")
    freq = {
        "S": "2Q",
        "H": "2Q",
    }.get(origin_grain, origin_grain)
    freq = freq if (freq == "M" or "-" in freq) else freq + "-" + origin_close
    value = pd.PeriodIndex(value, freq=freq)
    return value.to_timestamp().values


def _get_development(obj: Triangle, raw: Any) -> pd.Series:
    """
    Transform raw development lags/timestamps into public Series.
    """
    if raw is None:
        return None
    dev = raw.copy()
    if obj.is_val_tri:
        formats = {"Y": "%Y", "S": "%YQ%q", "Q": "%YQ%q", "M": "%Y-%m"}
        dev = dev.to_period(freq=obj.development_grain.replace("S", "2Q")).strftime(
            formats[obj.development_grain]
        )
    elif obj.is_pattern:
        offset = obj._dstep()["M"][obj.development_grain]
        if obj.is_ultimate:
            dev[-1] = dev[-2] + offset
        if obj.is_cumulative:
            dev = ["{}-Ult".format(dev[i]) for i in range(len(dev))]
        else:
            dev = ["{}-{}".format(dev[i], dev[i] + offset) for i in range(len(dev))]
    return pd.Series(list(dev), name="development")


def _set_development(obj: Triangle, value: Any, check: bool = True) -> Any:
    """
    Validate and transform assigned development periods into internal lag/timestamp array.
    """
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__"):
        value = [value]
    if check and hasattr(obj, "_len_check"):
        if hasattr(obj, "values") and obj.values is not None:
            obj._len_check(range(obj.values.shape[3]), value)
        elif (
            not isinstance(obj, TriangleBase)
            and hasattr(obj, "_axes")
            and (3 in obj._axes or "development" in obj._axes)
        ):
            raw_dev = obj._axes.get(3, obj._axes.get("development"))
            if raw_dev is not None:
                obj._len_check(raw_dev, value)
    if isinstance(value, pd.DatetimeIndex):
        return value
    if isinstance(value, pd.PeriodIndex):
        return (
            value.to_timestamp(how="end")
            if getattr(obj, "is_val_tri", False)
            else value.to_timestamp().values
        )
    if isinstance(value, np.ndarray) and value.dtype.kind == "M":
        return pd.DatetimeIndex(value) if getattr(obj, "is_val_tri", False) else value

    val_arr = np.array(value)
    if val_arr.dtype.kind in ("i", "u"):
        return val_arr.astype(int)
    if val_arr.dtype.kind == "f":
        if np.all(np.equal(np.mod(val_arr, 1), 0)):
            return val_arr.astype(int)
        return val_arr

    if getattr(obj, "is_val_tri", False):
        # Try matching existing DatetimeIndex labels or parsing as PeriodIndex/DatetimeIndex
        if hasattr(obj, "_axes") and isinstance(obj._axes.get(3), pd.DatetimeIndex):
            current_dt = obj._axes[3]
            current_labels = list(_get_development(obj, current_dt))
            val_list_str = [str(v) for v in val_arr]
            if all(v in current_labels for v in val_list_str):
                indices = [current_labels.index(v) for v in val_list_str]
                return current_dt[indices]

        try:
            freq = getattr(obj, "development_grain", "Y").replace("S", "2Q")
            return pd.PeriodIndex(val_arr.astype(str), freq=freq).to_timestamp(
                how="end"
            )
        except (ValueError, TypeError):
            try:
                return pd.DatetimeIndex(val_arr)
            except (ValueError, TypeError):
                pass
        return val_arr

    # Non-valuation triangle: development axis represents numeric lags or pattern intervals
    val_list = [str(v).strip() for v in val_arr]
    try:
        lags = []
        for s in val_list:
            if s.lower() == "ult":
                lags.append(9999)
            elif "-" in s:
                parts = s.split("-")
                if len(parts) == 2:
                    p0 = parts[0].strip()
                    p1 = parts[1].strip()
                    try:
                        n0 = int(p0) if p0.isdigit() else float(p0)
                    except ValueError:
                        n0 = None
                    if p1.lower() == "ult":
                        n1 = 9999
                    else:
                        try:
                            n1 = int(p1) if p1.isdigit() else float(p1)
                        except ValueError:
                            n1 = None
                    if n0 is not None and n1 is not None and n0 < 1000 and n1 <= 9999:
                        lags.append(n0)
                    else:
                        raise ValueError(f"Not a development lag pattern: {s}")
                else:
                    raise ValueError(f"Not a development lag pattern: {s}")
            else:
                try:
                    val_num = int(s)
                except ValueError:
                    val_num = float(s)
                if val_num >= 1000 and val_num != 9999:
                    raise ValueError(f"Year passed to development lag: {s}")
                lags.append(val_num)

        raw_dev = getattr(obj, "_axes", {}).get(3)
        if raw_dev is None and hasattr(obj, "_axes") and isinstance(obj._axes, dict):
            raw_dev = obj._axes.get("development")
        is_ult = getattr(obj, "is_ultimate", False) or (
            raw_dev is not None
            and len(raw_dev) == len(lags)
            and len(raw_dev) > 0
            and raw_dev[-1] == 9999
        )
        if is_ult and len(lags) > 0:
            lags[-1] = 9999
        lag_arr = np.array(lags)
        if lag_arr.dtype.kind == "f" and np.all(np.equal(np.mod(lag_arr, 1), 0)):
            lag_arr = lag_arr.astype(int)
        return lag_arr
    except (ValueError, TypeError):
        pass

    # If not parseable as numeric lags / patterns, try parsing as valuation periods / dates
    try:
        freq = getattr(obj, "development_grain", "Y").replace("S", "2Q")
        return pd.PeriodIndex(val_arr.astype(str), freq=freq).to_timestamp(how="end")
    except (ValueError, TypeError):
        try:
            return pd.DatetimeIndex(val_arr)
        except (ValueError, TypeError):
            pass

    return val_arr


def _set_axis(obj: Triangle, axis: int | str, value: Any, check: bool = True) -> None:
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
    check : bool, default True
        Whether to perform length and consistency validation.
    """
    axis_name = _get_axis_name(axis)
    if not check:
        descriptor = getattr(type(obj), axis_name, None)
        if isinstance(descriptor, TriangleAxis):
            descriptor.set_unchecked(obj, value)
            return
    setattr(obj, axis_name, value)
