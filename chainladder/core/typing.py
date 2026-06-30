from __future__ import annotations

import numpy as np
import pandas as pd

from types import EllipsisType
from typing import (
    Any,
    Literal,
    overload,
    Protocol,
    # Self,  # Make use of this once Python 3.10 is deprecated.
    TYPE_CHECKING,
    TypeAlias
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from numpy import ndarray
    from types import ModuleType
    from chainladder import Triangle
    from chainladder.core.slice import (
        At,
        Iat,
        Ilocation,
        Location,
        TriangleSlicer,
        VirtualColumns
    )
    from numpy.typing import ArrayLike
    from pandas import DataFrame, Series
    from sparse import COO

# Alias for a Triangle or any object that behaves like one.
TriangleLike: TypeAlias = "Triangle"

# The backend array stored in Triangle.values, whose concrete type
# depends on the active array_backend ("numpy" or "sparse").
BackendArray: TypeAlias = "np.ndarray | COO"

# A raw indexing expression as passed by the user to an indexer such aa .iloc[] before normalization.
IndexExpression: TypeAlias = int | slice | list | np.ndarray | tuple | EllipsisType

# A single axis key after normalization: a bounded slice, an integer position,
# or a fancy-index array.
_AxisKey: TypeAlias = slice | int | np.int64 | np.int32 | np.ndarray

# A label-based selector for a single Triangle axis, as accepted by .loc[]
# before index_key/other_key resolve it to a positional _AxisKey.
_LabelKey: TypeAlias = IndexExpression | str | pd.Series | pd.DataFrame

class TriangleProtocol(Protocol):
    """
    Common interface expected for Triangle mixins.
    """
    @property
    def shape(self) -> tuple[int, int, int, int]: ...

    @property
    def index(self) -> pd.DataFrame: ...

    @property
    def is_val_tri(self) -> bool: ...

    @property
    def columns(self) -> pd.Index: ...

    @property
    def origin(self) -> pd.PeriodIndex: ...

    @property
    def development(self) -> pd.Series: ...

    @property
    def valuation_date(self) -> pd.Timestamp: ...

    @property
    def nan_triangle(self) -> BackendArray: ...

    key_labels: list[str]
    values: BackendArray
    array_backend: str
    iloc: Ilocation
    loc: Location
    iat: Iat
    at: At
    virtual_columns: VirtualColumns

    def __len__(self) -> int: ...
    def get_array_module(self, arr: ArrayLike | None = None) -> ModuleType: ...
    def copy(self) -> Triangle: ...
    def set_backend(self, backend: str, inplace: bool = False, **kwargs) -> Triangle: ...
    def drop(self, labels: str | int | list | None = None, axis: int = 1) -> Triangle: ...
    def val_to_dev(self) -> Triangle: ...
    def _repr_format(self, origin_as_datetime: bool = False) -> pd.DataFrame: ...
    def _slice(self, key: pd.Series | np.ndarray, axis: Literal['ddims', 'odims']) -> Triangle: ...
    def _slice_valuation(self, key: np.ndarray) -> Triangle: ...
    def to_frame(
        self,
        origin_as_datetime: bool = True,
        keepdims: bool = False,
        implicit_axis: bool = False,
    ) -> DataFrame | Series: ...
    def sum(self, axis: str | int | None = None, *args, **kwargs) -> TriangleProtocol: ...  # -> Self once Python 3.10 is deprecated.
    def fillna(self, value: int | float | ndarray | None = None, inplace: bool = False) -> TriangleProtocol: ...  # -> Self once Python 3.10 is deprecated.
    def fillzero(self, inplace: bool = False) -> TriangleProtocol: ...  # -> Self once Python 3.10 is deprecated.
    def __add__(self, other: Any) -> TriangleProtocol: ...  # -> Self once Python 3.10 is deprecated.
    def __radd__(self, other: Any) -> TriangleProtocol: ...  # -> Self once Python 3.10 is deprecated.
    def __mul__(self, other: Any) -> TriangleProtocol: ...  # -> Self once Python 3.10 is deprecated.
    def __rmul__(self, other: Any) -> TriangleProtocol: ...  # -> Self once Python 3.10 is deprecated.
    @overload
    def __getitem__(self, key: pd.Series | np.ndarray | list[str]) -> Triangle: ...
    @overload
    def __getitem__(self, key: str | int) -> Triangle | Series: ...
    def __setitem__(self, key: str | int, value: int | float | TriangleSlicer | Callable[[Triangle], TriangleSlicer]) -> None: ...