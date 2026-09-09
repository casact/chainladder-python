"""
Utilities for deprecating chainladder features.
"""

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import functools
import inspect
import warnings

from typing import overload, TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from types import FrameType
    from typing import Callable, Literal
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


# Type variable ensures that decorated functions maintain their signatures.
_F = TypeVar("_F", bound="Callable[..., object]")


def deprecated_rename(
    new_name: str,
    *,
    version: str | None = None,
    category: type[Warning] = FutureWarning,
) -> Callable[[_F], _F]:
    """
    Decorator factory that marks a function as scheduled to be renamed.

    Calling the decorated function will emit a warning that the function will be renamed in a future release.

    Parameters
    ----------
    new_name: str
        The name this function will be renamed to.
    version: str | None
        The release the rename is expected to land in, e.g. "0.11.0".
        Included in the warning message when given. Optional.
    category: type[Warning]
        The warning category to emit. Defaults to FutureWarning.

    Returns
    -------
    Callable
        A decorator that wraps a function, preserving its name, docstring,
        and signature.

    Examples
    --------

    .. testcode::
        :options: +SKIP

        from chainladder._config.deprecation import deprecated_rename

        @deprecated_rename("new_func", version="0.11.0")
        def old_func(x):
            return x + 1

        old_func(1)

    .. testoutput::

        example.py:8: FutureWarning: 'old_func' is deprecated and will be renamed to 'new_func' in 0.11.0. Update your code to use 'new_func' instead.
          old_func(1)

    """

    def decorator(func: _F) -> _F:
        old_name = func.__name__
        message = f"'{old_name}' is deprecated and will be renamed to '{new_name}'"
        if version:
            message += f" in {version}"
        message += f". Update your code to use '{new_name}' instead."

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(message, category, stacklevel=2)  # noqa
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def deprecated_rename_argument(
    old_name: str,
    new_name: str,
    *,
    version: str | None = None,
    category: type[Warning] = FutureWarning,
) -> Callable[[_F], _F]:
    """
    Decorator factory that marks a keyword argument as scheduled to be
    renamed.

    Apply this to a function while it still accepts the argument under its
    *current* name, to warn callers ahead of the actual rename.

    This decorator allows you to replace the old argument with the new argument
    in the function signature. Once you are ready to deprecate, simply remove the
    decorator.

    Parameters
    ----------
    old_name: str
        The keyword argument name the function currently accepts.
    new_name: str
        The keyword argument name it will be renamed to.
    version: str | None
        The release the rename is expected to land in, e.g. "0.11.0".
        Included in the warning message when given. Optional.
    category: type[Warning]
        The warning category to emit. Defaults to FutureWarning.

    Returns
    -------
    Callable
        A decorator that wraps a function, preserving its name, docstring,
        and signature via functools.wraps.

    Examples
    --------

    .. testcode::
        :options: +SKIP

        from chainladder._config.deprecation import deprecated_rename_argument

        @deprecated_rename_argument("old_arg", "new_arg", version="0.11.0")
        def func(new_arg):
            return new_arg + 1

        print(func(old_arg=1))

    .. testoutput::

        example.py:8: FutureWarning: 'old_arg' is deprecated and will be renamed to 'new_arg' in 0.11.0. Use 'new_arg' instead.
          func(old_arg=1)

    """

    def decorator(func: _F) -> _F:
        message = f"'{old_name}' is deprecated and will be renamed to '{new_name}'"
        if version:
            message += f" in {version}"
        message += f". Use '{new_name}' instead."

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if old_name in kwargs:
                if new_name in kwargs:
                    raise TypeError(
                        f"Cannot specify both '{old_name}' and '{new_name}'."
                    )
                warnings.warn(message, category, stacklevel=2)  # noqa
                kwargs[new_name] = kwargs.pop(old_name)
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def deprecated_drop_argument(
    name: str,
    *,
    version: str | None = None,
    category: type[Warning] = FutureWarning,
) -> Callable[[_F], _F]:
    """
    Decorator factory that marks a keyword argument as scheduled for removal,
    with no replacement.

    Parameters
    ----------
    name: str
        The keyword argument scheduled for removal.
    version: str | None
        The release the removal is expected to land in, e.g. "0.11.0".
        Included in the warning message when given. Optional.
    category: type[Warning]
        The warning category to emit. Defaults to FutureWarning.

    Returns
    -------
    Callable
        A decorator that wraps a function, preserving its name, docstring,
        and signature via functools.wraps.

    Examples
    --------

    .. testcode::
        :options: +SKIP

        from chainladder._config.deprecation import deprecated_drop_argument

        @deprecated_drop_argument("verbose", version="0.11.0")
        def func(x, verbose=False):
            return x + 1

        print(func(1, verbose=True))

    .. testoutput::

        example.py:8: FutureWarning: 'verbose' is deprecated and will be removed in 0.11.0.
          func(1, verbose=True)

    """

    def decorator(func: _F) -> _F:
        message = f"'{name}' is deprecated and will be removed"
        message += f" in {version}." if version else " in a future release."

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if name in kwargs:
                warnings.warn(message, category, stacklevel=2)  # noqa
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
