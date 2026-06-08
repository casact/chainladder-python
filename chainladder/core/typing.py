from __future__ import annotations

import numpy as np
import pandas as pd

from types import EllipsisType
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from chainladder import Triangle

# Alias for a Triangle or any object that behaves like one.
TriangleLike: TypeAlias = "Triangle"

# A raw indexing expression as passed by the user to an indexer such aa .iloc[] before normalization.
IndexExpression: TypeAlias = int | slice | list | np.ndarray | tuple | EllipsisType

# A single axis key after normalization: a bounded slice, an integer position,
# or a fancy-index array.
_AxisKey: TypeAlias = slice | int | np.int64 | np.int32 | np.ndarray

# A label-based selector for a single Triangle axis, as accepted by .loc[]
# before index_key/other_key resolve it to a positional _AxisKey.
_LabelKey: TypeAlias = IndexExpression | str | pd.Series | pd.DataFrame


