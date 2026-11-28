from __future__ import annotations

from typing import TYPE_CHECKING

from chainladder.core.triangle import Triangle  # noqa (API import)
from chainladder.core.correlation import (
    DevelopmentCorrelation,
    ValuationCorrelation,
)  # noqa (API import)

if TYPE_CHECKING:
    from chainladder.core.style import Styler

__all__ = [
    "Triangle",
    "DevelopmentCorrelation",
    "ValuationCorrelation",
    "Styler",
]


def __getattr__(name: str) -> type[Styler]:
    """Import Styler on first use, so jinja2 stays optional (#1714)."""
    if name == "Styler":
        from chainladder.core.style import Styler

        return Styler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
