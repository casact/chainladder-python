"""Axis descriptor for Triangle dimensions, analogous to Pandas AxisProperty."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable
import pandas as pd

if TYPE_CHECKING:
    from chainladder.core.triangle import Triangle


class TriangleAxis:
    """
    Generalized descriptor for representing a Triangle dimension,
    analogous to Pandas AxisProperty.
    """

    def __init__(
        self,
        key: str,
        *,
        fget: Callable[[Triangle, Any], Any] | None = None,
        fset: Callable[[Triangle, Any], Any] | None = None,
        doc: str | None = None,
    ) -> None:
        self.key = key
        self.fget = fget  # raw -> public; identity if None
        self.fset = fset  # (obj, public) -> raw; identity if None
        self.__doc__ = doc

    def __get__(self, obj: Triangle | None, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        if not hasattr(obj, "_axes") or self.key not in obj._axes:
            raise AttributeError(
                f"'{type(obj).__name__}' object has no attribute '{self.key}'"
            )
        raw = obj._axes[self.key]
        return self.fget(obj, raw) if self.fget else raw

    def __set__(self, obj: Triangle, value: Any) -> None:
        raw = self.fset(obj, value) if self.fset else value
        if not hasattr(obj, "_axes"):
            obj._axes = {}
        obj._axes[self.key] = raw
        if hasattr(obj, "virtual_columns"):
            obj._set_slicers()


def _set_columns(obj: Triangle, value: Any) -> pd.Index:
    if isinstance(value, str):
        value = [value]
    if hasattr(obj, "values") and obj.values is not None:
        obj._len_check(range(obj.values.shape[1]), value)
    elif hasattr(obj, "_axes") and "columns" in obj._axes:
        obj._len_check(obj.columns, value)
    return pd.Index(value, name="columns")
