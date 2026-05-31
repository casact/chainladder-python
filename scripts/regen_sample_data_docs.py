#!/usr/bin/env python
"""Regenerate docs/library/sample_data.md from the sample-dataset manifest.

The sample-data documentation table used to be maintained by hand, which let it
drift out of sync with the actual datasets (missing rows, typo'd names, wrong
grain). It is now generated from the single source of truth,
``chainladder/utils/data/_manifest.py``, by way of :func:`chainladder.list_samples`.

Run from the repository root after adding or changing a sample dataset::

    python scripts/regen_sample_data_docs.py

The script overwrites ``docs/library/sample_data.md`` in place.
"""
from __future__ import annotations

from pathlib import Path

import chainladder as cl

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_PATH = REPO_ROOT / "docs" / "library" / "sample_data.md"

HEADER = """# Sample Dataset

Below is the list of all datasets that come included with the `chainladder` package, and their basic attributes.

You can load any dataset with `cl.load_sample(...)` such as `cl.load_sample("abc")`.

This table is generated from the sample-dataset manifest
(`chainladder/utils/data/_manifest.py`) via `cl.list_samples()`. To regenerate it,
run `python scripts/regen_sample_data_docs.py` from the repository root.

"""


def _fmt_list(value) -> str:
    """Render an index/columns cell."""
    if value is None:
        return "(none)"
    return "[" + ", ".join(str(v) for v in value) + "]"


# Unit noun used in the "(N units)" suffix on each grain label.
_GRAIN_UNITS: dict = {
    "Annual": "Yrs",
    "Semiannual": "Half-Yrs",
    "Quarter": "Qtrs",
    "Month": "months",
}


def _fmt_grain(label: str, periods: int) -> str:
    """Render a grain cell, e.g. 'Annual (10 Yrs)'."""
    unit = _GRAIN_UNITS.get(label, label)
    return f"{label} ({periods} {unit})"


def build_table() -> str:
    df = cl.list_samples()

    rows = [
        "| Dataset Name | Indexes | Columns | Origin Grain | Development Grain |",
        "|---|---|---|---|---|",
    ]
    for name, row in df.iterrows():
        rows.append(
            "| {name} | {index} | {columns} | {origin} | {development} |".format(
                name=name,
                index=_fmt_list(row["index"]),
                columns=_fmt_list(row["columns"]),
                origin=_fmt_grain(row["origin_grain"], row["origin_periods"]),
                development=_fmt_grain(
                    row["development_grain"], row["development_periods"]
                ),
            )
        )
    return "\n".join(rows) + "\n"


def main() -> None:
    content = HEADER + build_table()
    DOCS_PATH.write_text(content)
    print(f"Wrote {DOCS_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
