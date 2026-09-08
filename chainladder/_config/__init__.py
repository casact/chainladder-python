"""
Governs configuration, options, and deprecations.
"""

from chainladder._config.deprecation import (
    _DEPRECATED_BACKENDS,
    _deprecated_backend_message,
    _dask_parallel_state,
    _warn_dask_parallel_deprecated,  # noqa (API import)
)
from chainladder._config.options import (
    __dt64_dtype__,
    __dt64_unit__,
    Options,
    options,
)

__all__: list[str] = [
    "__dt64_dtype__",
    "__dt64_unit__",
    "Options",
    "options",
    "_DEPRECATED_BACKENDS",
    "_deprecated_backend_message",
    "_dask_parallel_state",
    "_warn_dask_parallel_deprecated",
]
