from __future__ import annotations

import numpy as np

from types import EllipsisType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chainladder import Triangle

# Alias for a Triangle or any object that behaves like one.
TriangleLike = "Triangle"

# A raw indexing expression as passed by the user to an indexer such as
# .iloc[] or .loc[] before normalization.
IndexExpression = int | slice | list | np.ndarray | tuple | EllipsisType

# A single axis key after normalization: a bounded slice, an integer position,
# or a fancy-index array.
_AxisKey = slice | int | np.ndarray


