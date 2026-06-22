"""
Builds a Markdown summary comparing pyright `--verifytypes` reports for a
PR's base and head commits, for posting as a PR comment.

"Project" completeness is read directly from the head report's
typeCompleteness.completenessScore. "Patch" completeness is computed by
matching exported symbols between the base and head reports by their
dotted name: a symbol counts toward the patch if it is new in head, or if
its known/ambiguous/unknown status changed between base and head. This
avoids needing to map symbols to source line ranges, since pyright only
reports a file/line for symbols that already have a type problem.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

STATUS_ICON = {"known": "✅", "ambiguous": "⚠️", "unknown": "❌"}


def status_of(symbol: dict[str, Any]) -> str:
    if symbol["isTypeKnown"]:
        return "known"
    if symbol["isTypeAmbiguous"]:
        return "ambiguous"
    return "unknown"


def load_type_completeness(report_path: Path) -> dict[str, Any]:
    with report_path.open() as f:
        data = json.load(f)
    return data["typeCompleteness"]


def exported_symbols(type_completeness: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {s["name"]: s for s in type_completeness["symbols"] if s["isExported"]}


def other_symbol_counts(type_completeness: dict[str, Any]) -> dict[str, int]:
    other = type_completeness["otherSymbolCounts"]
    return {
        "known": other["withKnownType"],
        "ambiguous": other["withAmbiguousType"],
        "unknown": other["withUnknownType"],
    }


def counts_by_status(symbols: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"known": 0, "ambiguous": 0, "unknown": 0}
    for s in symbols:
        counts[status_of(s)] += 1
    return counts


def completeness_pct(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    return 100.0 * counts["known"] / total if total else 0.0


def render_counts_table(rows: list[tuple[str, dict[str, int]]]) -> str:
    lines = ["| | Known | Ambiguous | Unknown | Total |", "|---|---|---|---|---|"]
    for label, counts in rows:
        total = sum(counts.values())
        lines.append(
            f"| {label} | {counts['known']} | {counts['ambiguous']} | "
            f"{counts['unknown']} | {total} |"
        )
    return "\n".join(lines)


def render_patch_detail(
    patch_names: list[str],
    base: dict[str, dict[str, Any]],
    head: dict[str, dict[str, Any]],
) -> str:
    lines = ["| Symbol | Status | Change |", "|---|---|---|"]
    for name in sorted(patch_names):
        head_status = status_of(head[name])
        icon = STATUS_ICON[head_status]
        if name not in base:
            change = "new"
        else:
            base_status = status_of(base[name])
            change = f"changed (was {STATUS_ICON[base_status]} {base_status})"
        lines.append(f"| `{name}` | {icon} {head_status} | {change} |")
    return "\n".join(lines)


def build_summary(base_path: Path, head_path: Path, run_url: str | None = None) -> str:
    base_tc = load_type_completeness(base_path)
    head_tc = load_type_completeness(head_path)
    base = exported_symbols(base_tc)
    head = exported_symbols(head_tc)

    project_counts = counts_by_status(list(head.values()))
    project_pct = completeness_pct(project_counts)
    other_counts = other_symbol_counts(head_tc)

    patch_names = [
        name
        for name, symbol in head.items()
        if name not in base or status_of(base[name]) != status_of(symbol)
    ]

    sections = [
        "## Pyright Type Completeness",
        "",
    ]
    if run_url:
        sections += [
            f"[View the full `pyright --verifytypes` output for this commit]({run_url})",
            "",
        ]
    sections += [
        f"**Project (full `chainladder` package, at this PR's head):** "
        f"{project_pct:.1f}% of exported symbols fully typed "
        f"({project_counts['known']} / {sum(project_counts.values())})",
        "",
        render_counts_table([("Project (head)", project_counts)]),
        "",
        f"Other symbols referenced but not exported by `chainladder`: "
        f"{sum(other_counts.values())}",
        "",
        render_counts_table([("Other (head)", other_counts)]),
        "",
        "Symbols without documentation:",
        f"- Functions without docstring: {head_tc['missingFunctionDocStringCount']}",
        f"- Functions without default param: {head_tc['missingDefaultParamCount']}",
        f"- Classes without docstring: {head_tc['missingClassDocStringCount']}",
        "",
    ]

    if patch_names:
        patch_counts = counts_by_status([head[n] for n in patch_names])
        patch_pct = completeness_pct(patch_counts)
        sections += [
            f"**Patch (exported symbols added or changed by this PR):** "
            f"{patch_pct:.1f}% fully typed "
            f"({patch_counts['known']} / {sum(patch_counts.values())})",
            "",
            render_counts_table([("Patch", patch_counts)]),
            "",
            "<details>",
            "<summary>Patch symbol details</summary>",
            "",
            render_patch_detail(patch_names, base, head),
            "",
            "</details>",
        ]
    else:
        sections += [
            "**Patch (exported symbols added or changed by this PR):** "
            "no exported symbol type-completeness changes detected.",
        ]

    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--head", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--run-url",
        default=None,
        help="Link to the workflow run, e.g. for its step summary.",
    )
    args = parser.parse_args()

    summary = build_summary(args.base, args.head, run_url=args.run_url)
    args.output.write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
