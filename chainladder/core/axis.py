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

    Parameters
    ----------
    key : str
        Key used to store and access the raw axis data in ``obj._axes``.
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
        key: str,
        *,
        fget: Callable[[Triangle, Any], Any] | None = None,
        fset: Callable[[Triangle, Any], Any] | None = None,
        doc: str | None = None,
    ) -> None:
        """
        Initialize a TriangleAxis descriptor.

        Parameters
        ----------
        key : str
            Internal storage key in ``obj._axes``.
        fget : callable, optional
            Getter callable mapping ``(obj, raw)`` to public representation.
        fset : callable, optional
            Setter callable mapping ``(obj, value)`` to internal representation.
        doc : str, optional
            Docstring for the attribute.
        """
        self.key = key
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
        if not hasattr(obj, "_axes") or self.key not in obj._axes:
            raise AttributeError(
                f"'{type(obj).__name__}' object has no attribute '{self.key}'"
            )
        raw = obj._axes[self.key]
        return self.fget(obj, raw) if self.fget else raw

    def __set__(self, obj: Triangle, value: Any) -> None:
        """
        Set the axis value on a Triangle instance.

        Transforms ``value`` using ``fset`` if defined, stores the raw value
        in ``obj._axes[self.key]``, and updates slicers if necessary.

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
        obj._axes[self.key] = raw
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
    elif hasattr(obj, "_axes") and "columns" in obj._axes:
        obj._len_check(obj.columns, value)
    return pd.Index(value, name="columns")
