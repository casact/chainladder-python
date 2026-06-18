from __future__ import annotations

import numpy as np
import pandas as pd

from types import EllipsisType
from typing import (
    Literal,
    Protocol,
    TYPE_CHECKING,
    TypeAlias
)

if TYPE_CHECKING:
    from types import ModuleType
    from chainladder import Triangle
    from chainladder.core.slice import (
        At,
        Iat,
        Ilocation,
        Location,
        VirtualColumns
    )
    from numpy.typing import ArrayLike
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

    key_labels: list[str]
    values: BackendArray
    array_backend: str
    iloc: Ilocation
    loc: Location
    iat: Iat
    at: At
    virtual_columns: VirtualColumns

    def __len__(self) -> int: ...
    def get_array_module(self, arr: ArrayLike = None) -> ModuleType: ...
    def copy(self) -> Triangle: ...
    def set_backend(self, backend: str, inplace: bool = False, **kwargs) -> Triangle: ...
    def drop(self, labels: str | int | list | None = None, axis: int = 1) -> Triangle: ...
    def _slice(self, key: pd.Series | np.ndarray, axis: Literal['ddims', 'odims']) -> Triangle: ...
    def _slice_valuation(self, key: np.ndarray) -> Triangle: ...