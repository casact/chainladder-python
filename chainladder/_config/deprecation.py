"""
Utilities for deprecating chainladder features.
"""

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import inspect
import warnings

from typing import overload, TYPE_CHECKING

if TYPE_CHECKING:
    from types import FrameType
    from typing import Literal
del TYPE_CHECKING
del annotations


# Array backends slated for removal, mapped to the issue tracking each one.
# Selecting one of these (via set_option, ARRAY_PRIORITY, or set_backend, or by
# passing a Dask dataframe to the Triangle constructor) emits a
# DeprecationWarning.
_DEPRECATED_BACKENDS: dict[str, str] = {
    "cupy": "https://github.com/casact/chainladder-python/issues/843",
    "dask": "https://github.com/casact/chainladder-python/issues/842",
}


def _deprecated_backend_message(backend: str) -> str:
    """Build the deprecation message for a soon-to-be-removed array backend."""
    return (
        f"The '{backend}' array backend is deprecated and will be removed in a "
        f"future release. See {_DEPRECATED_BACKENDS[backend]}."
    )


class _DaskParallelWarningState:
    """
    Tracks whether the one-time dask parallel-compute deprecation warning has
    already fired this process. The dask 'bag' code paths run automatically
    whenever dask is installed, so they warn at most once instead of on every
    operation. See issue #842.
    """

    def __init__(self) -> None:
        self.warned: bool = False


_dask_parallel_state = _DaskParallelWarningState()


def _warn_dask_parallel_deprecated(stacklevel: int = 2) -> None:
    """
    Emit a one-time DeprecationWarning for dask-accelerated parallel compute.

    The dask ``bag`` scheduler is used as an optional parallel-compute engine
    for the sparse backend (groupby aggregation, grouped-triangle arithmetic,
    and incremental-to-cumulative conversion). It is deprecated alongside the
    dask array backend and will be removed in a future release. Because these
    paths run automatically on every qualifying operation, the warning fires at
    most once per process to avoid flooding output.

    Parameters
    ----------
    stacklevel: int
        Forwarded to ``warnings.warn``. Defaults to 2 so the warning points at
        the chainladder method that triggered the dask path.

    Returns
    -------
    None

    """
    if _dask_parallel_state.warned:
        return
    _dask_parallel_state.warned = True
    warnings.warn(
        "Using dask for parallel computation is deprecated and will be removed "
        f"in a future release. See {_DEPRECATED_BACKENDS['dask']}.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


_option_warning: str = "The parameter 'option' is deprecated and will be removed in a future release. Use 'pat' instead."


@overload
def _resolve_pat(
    pat: str | None, option: str | None, required: Literal[True] = ...
) -> str: ...
@overload
def _resolve_pat(
    pat: str | None, option: str | None, required: Literal[False]
) -> str | None: ...


del overload


def _resolve_pat(
    pat: str | None, option: str | None, required: bool = True
) -> str | None:
    """
    Handles backward compatibility of 'options' parameter in options functions. Checks whether option or pat is
    assigned a value and returns it. This value is meant to be assigned to the 'pat' parameter of the calling function.

    Once the 'options' parameter is fully removed, this function can be deleted or generalized as a backwards
    compatibility tool to assist in the renaming and deprecation of function parameters.

    Parameters
    ----------
    pat: str | None
        The 'pat' parameter of the calling function.
    option: str | None
        The 'option' parameter of the calling function.
    required: bool
        Whether pat or option are required parameters in the calling function. Defaults to True.

    Returns
    -------
        The value to be assigned to the 'pat' parameter of the calling function.

    """
    # Raise an error if the user accidentally assigns a value to both 'pat' and 'option'.
    if pat is not None and option is not None:
        raise TypeError("Cannot specify both 'pat' and 'option'.")
    # Raise the deprecation warning if the user assigns a value to 'option'.
    if option is not None:
        warnings.warn(_option_warning, FutureWarning, stacklevel=3)
        pat: str = option
    # Raise an error if neither 'option' nor 'pat' is assigned.
    if pat is None and required:
        # Determine the name of the calling function.
        err: str = "Unable to determine calling function."
        frame: FrameType | None = inspect.currentframe()
        if frame is None:
            raise AttributeError(err)
        else:
            f_back: FrameType | None = frame.f_back
        if f_back is None:
            raise AttributeError(err)
        else:
            caller: str = f_back.f_code.co_name
        raise TypeError(f"{caller}() missing required argument: 'pat'.")
    return pat
