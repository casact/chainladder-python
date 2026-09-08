"""
Checks chainladder's minimum-supported dependency versions against the Scientific
Python SPEC 0 policy (https://scientific-python.org/specs/spec-0000/) and reports
which ones are within 3 months of (or past) their recommended drop-support date.

Python itself is checked against its official end-of-life date rather than SPEC 0's
generic "3 years after release" rule, per this project's stated policy of supporting
Python through EOL.

This script targets Python 3.11+ for `tomllib`. That is a constraint on this
dev-tooling script only, independent of the `requires-python` floor chainladder
itself declares for its users.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tomllib
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Literal

# The subset of chainladder's dependencies that also appear on SPEC 0's own
# endorsed/example table. SPEC 0 leaves "core package" project-defined, so
# non-ecosystem-core dependencies (sparse, dill, patsy) are intentionally excluded
# here rather than extrapolating the 2-year rule to them.
CORE_PACKAGES = ["numpy", "pandas", "scikit-learn", "matplotlib"]

# Community-maintained release-cycle API; used for Python's official EOL dates so
# this script doesn't need a hand-maintained table updated for every new release.
PYTHON_EOL_API_URL = "https://endoflife.date/api/python.json"

WARN_DAYS = 90

Status = Literal["ok", "within_window", "past_due"]


@dataclass
class CheckResult:
    """
    The outcome of checking a single dependency (or Python itself) against its
    support-window policy.

    Attributes
    ----------
    name: str
        The dependency name, or "python".
    floor: str
        The currently-declared minimum version, as a string (e.g. "2.0").
    anchor_date: date | None
        The date the policy's window is measured from: the minor version's initial
        release date for packages, or None for Python (whose window is measured
        directly against `drop_date`, its EOL).
    drop_date: date | None
        The date support is recommended to be dropped by. None when no floor is
        declared (e.g. an unpinned dependency), in which case there's nothing to
        check.
    status: Status
        "ok", "within_window" (inside the WARN_DAYS alert window), or "past_due".
    """

    name: str
    floor: str
    anchor_date: date | None
    drop_date: date | None
    status: Status


def add_months(d: date, months: int) -> date:
    """
    Adds a number of months to a date, clamping the day to the last valid day of
    the resulting month (e.g. Jan 31 + 1 month -> Feb 28/29).

    Parameters
    ----------
    d: date
        The starting date.
    months: int
        The number of months to add.

    Returns
    -------
    date
        The resulting date.

    """
    month_index = d.month - 1 + months
    year = d.year + month_index // 12
    month = month_index % 12 + 1
    day = d.day
    while True:
        try:
            return date(year, month, day)
        except ValueError:
            day -= 1


def classify(drop_date: date, today: date) -> Status:
    """
    Classifies a drop-support date relative to today's date.

    Parameters
    ----------
    drop_date: date
        The date support is recommended to be dropped by.
    today: date
        The date to compare against.

    Returns
    -------
    Status
        "past_due" if drop_date is in the past, "within_window" if it's within
        WARN_DAYS days, else "ok".

    """
    days_until = (drop_date - today).days
    if days_until < 0:
        return "past_due"
    if days_until <= WARN_DAYS:
        return "within_window"
    return "ok"


def parse_requires_python(pyproject: dict) -> tuple[int, int]:
    """
    Extracts the minimum Python version from a parsed pyproject.toml's
    `project.requires-python` field.

    Parameters
    ----------
    pyproject: dict
        The parsed contents of pyproject.toml.

    Returns
    -------
    tuple[int, int]
        The (major, minor) floor, e.g. (3, 10).

    """
    spec = pyproject["project"]["requires-python"]
    match = re.search(r">=\s*(\d+)\.(\d+)", spec)
    if not match:
        raise ValueError(f"Could not parse a minimum version from requires-python: {spec!r}")
    return int(match.group(1)), int(match.group(2))


def parse_core_dependency_floors(pyproject: dict) -> dict[str, str | None]:
    """
    Extracts the declared minimum version for each package in CORE_PACKAGES from a
    parsed pyproject.toml's `project.dependencies` list.

    Parameters
    ----------
    pyproject: dict
        The parsed contents of pyproject.toml.

    Returns
    -------
    dict[str, str | None]
        Mapping of package name to its declared minimum version string, or None if
        the dependency has no version floor (or isn't declared at all).

    """
    floors: dict[str, str | None] = {name: None for name in CORE_PACKAGES}
    for entry in pyproject["project"]["dependencies"]:
        match = re.match(r"^\s*([A-Za-z0-9_.-]+)", entry)
        if not match:
            continue
        name = match.group(1)
        if name not in floors:
            continue
        version_match = re.search(r">=?\s*([\d.]+)", entry)
        if version_match:
            floors[name] = version_match.group(1)
    return floors


def fetch_minor_release_dates(package: str) -> dict[tuple[int, int], date]:
    """
    Queries the PyPI JSON API for a package's release history and returns the
    earliest upload date seen for each (major, minor) series.

    Parameters
    ----------
    package: str
        The PyPI project name.

    Returns
    -------
    dict[tuple[int, int], date]
        Mapping of (major, minor) to that series' earliest release date.

    """
    url = f"https://pypi.org/pypi/{package}/json"
    with urllib.request.urlopen(url, timeout=30) as response:
        data = json.load(response)

    by_minor: dict[tuple[int, int], date] = {}
    for version, files in data["releases"].items():
        if not files:
            continue
        match = re.match(r"^(\d+)\.(\d+)\.\d+$", version)
        if not match:
            continue
        key = (int(match.group(1)), int(match.group(2)))
        upload_date = min(
            datetime.fromisoformat(f["upload_time_iso_8601"]).date() for f in files
        )
        if key not in by_minor or upload_date < by_minor[key]:
            by_minor[key] = upload_date
    return by_minor


def fetch_python_eol_dates() -> dict[tuple[int, int], date]:
    """
    Queries the endoflife.date API for CPython's official end-of-life date per
    minor version, so this stays current without a hand-maintained table.

    Returns
    -------
    dict[tuple[int, int], date]
        Mapping of (major, minor) to that version's official EOL date. Cycles
        without a known EOL date yet (endoflife.date reports these as `false`)
        are omitted.

    Examples
    --------

    .. testsetup::

        from check_spec0 import fetch_python_eol_dates

    .. testcode::

        eol_dates = fetch_python_eol_dates()
        print(eol_dates[(3, 9)])

    .. testoutput::

        2025-10-05

    """
    with urllib.request.urlopen(PYTHON_EOL_API_URL, timeout=30) as response:
        data = json.load(response)

    eol_dates: dict[tuple[int, int], date] = {}
    for cycle in data:
        match = re.match(r"^(\d+)\.(\d+)$", cycle["cycle"])
        eol = cycle["eol"]
        if not match or not isinstance(eol, str):
            continue
        eol_dates[(int(match.group(1)), int(match.group(2)))] = date.fromisoformat(eol)
    return eol_dates


def check_python(
    requires_python_floor: tuple[int, int], eol_dates: dict[tuple[int, int], date], today: date
) -> CheckResult:
    """
    Checks the declared minimum Python version against its official EOL date.

    Parameters
    ----------
    requires_python_floor: tuple[int, int]
        The (major, minor) Python version floor from pyproject.toml.
    eol_dates: dict[tuple[int, int], date]
        Mapping of (major, minor) to EOL date, as produced by
        fetch_python_eol_dates().
    today: date
        The date to evaluate the check against.

    Returns
    -------
    CheckResult
        The outcome of the check.

    """
    eol = eol_dates.get(requires_python_floor)
    if eol is None:
        return CheckResult(
            name="python",
            floor=f"{requires_python_floor[0]}.{requires_python_floor[1]}",
            anchor_date=None,
            drop_date=None,
            status="ok",
        )
    return CheckResult(
        name="python",
        floor=f"{requires_python_floor[0]}.{requires_python_floor[1]}",
        anchor_date=None,
        drop_date=eol,
        status=classify(eol, today),
    )


def check_package(name: str, floor: str | None, today: date) -> CheckResult:
    """
    Checks a single core package's declared minimum version against SPEC 0's
    2-years-after-release rule.

    Parameters
    ----------
    name: str
        The PyPI package name.
    floor: str | None
        The declared minimum version string, or None if unpinned.
    today: date
        The date to evaluate the check against.

    Returns
    -------
    CheckResult
        The outcome of the check.

    """
    if floor is None:
        return CheckResult(name=name, floor="(none)", anchor_date=None, drop_date=None, status="ok")

    match = re.match(r"^(\d+)\.(\d+)", floor)
    if not match:
        raise ValueError(f"Could not parse a minor version from {name} floor: {floor!r}")
    minor_series = (int(match.group(1)), int(match.group(2)))

    release_dates = fetch_minor_release_dates(name)
    anchor_date = release_dates.get(minor_series)
    if anchor_date is None:
        return CheckResult(name=name, floor=floor, anchor_date=None, drop_date=None, status="ok")

    drop_date = add_months(anchor_date, 24)
    return CheckResult(
        name=name,
        floor=floor,
        anchor_date=anchor_date,
        drop_date=drop_date,
        status=classify(drop_date, today),
    )


def render_markdown(results: list[CheckResult], today: date) -> str:
    """
    Renders a Markdown summary table of the check results.

    Parameters
    ----------
    results: list[CheckResult]
        The results to render, one per dependency (plus Python).
    today: date
        The date the checks were evaluated against, included in the summary header.

    Returns
    -------
    str
        The rendered Markdown report.

    """
    status_label = {"ok": "OK", "within_window": "Within 3 months", "past_due": "Past due"}
    lines = [
        "## SPEC 0 dependency support check",
        "",
        f"Evaluated {today.isoformat()}. Python is checked against its official EOL date; "
        "core packages (numpy, pandas, scikit-learn, matplotlib) are checked against "
        "[SPEC 0](https://scientific-python.org/specs/spec-0000/)'s 2-years-after-release rule.",
        "",
        "| Dependency | Declared minimum | Drop date | Status |",
        "|---|---|---|---|",
    ]
    for r in results:
        drop = r.drop_date.isoformat() if r.drop_date else "n/a"
        lines.append(f"| {r.name} | {r.floor} | {drop} | {status_label[r.status]} |")
    return "\n".join(lines)


def issue_title(violation: dict) -> str:
    """
    Builds the title used both to search for an existing tracking issue and to
    create a new one, for a single violation.

    Parameters
    ----------
    violation: dict
        One entry from the `violations` list produced by main(), with keys
        "name", "floor", "drop_date", "status".

    Returns
    -------
    str
        The issue title.

    """
    verb = "is past its" if violation["status"] == "past_due" else "is within 3 months of its"
    return f"SPEC 0: {violation['name']} {verb} support drop date ({violation['drop_date']})"


def issue_body(violation: dict) -> str:
    """
    Builds the body text for a new tracking issue for a single violation.

    Parameters
    ----------
    violation: dict
        One entry from the `violations` list produced by main().

    Returns
    -------
    str
        The issue body, in Markdown.

    """
    return (
        f"The currently-declared minimum for `{violation['name']}` "
        f"(`>= {violation['floor']}`) has a SPEC 0 recommended drop-support date of "
        f"**{violation['drop_date']}** ({violation['status'].replace('_', ' ')}).\n\n"
        "Python is tracked against its official end-of-life date rather than SPEC 0's "
        "generic 3-years-after-release rule; core packages (numpy, pandas, scikit-learn, "
        "matplotlib) are tracked against SPEC 0's 2-years-after-release rule. "
        "See https://scientific-python.org/specs/spec-0000/ for the policy.\n\n"
        "This issue was opened automatically by the nightly SPEC 0 check "
        "(`.github/workflows/spec0_check.yml`). If this version should still be supported "
        "for a specific reason, close this issue with an explanation; otherwise it tracks "
        "bumping the minimum version."
    )


def create_github_issues(violations: list[dict], repo: str, dry_run: bool = False) -> None:
    """
    Opens a GitHub issue for each violation that doesn't already have an open
    tracking issue, via the `gh` CLI.

    Parameters
    ----------
    violations: list[dict]
        The violations to open issues for, as produced by main().
    repo: str
        The `owner/name` repository to open issues in.
    dry_run: bool
        When True, only prints what would be done instead of calling `gh`.

    Returns
    -------
    None

    """
    for violation in violations:
        title = issue_title(violation)
        if dry_run:
            print(f"[dry-run] would check for/open issue: {title}")
            continue

        existing = subprocess.run(
            [
                "gh", "issue", "list",
                "--repo", repo,
                "--search", f"{title} in:title",
                "--state", "open",
                "--json", "number",
                "--jq", ".[0].number",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if existing:
            print(f"Issue already open for {violation['name']} (#{existing}), skipping.")
            continue

        subprocess.run(
            ["gh", "issue", "create", "--repo", repo, "--title", title, "--body", issue_body(violation)],
            check=True,
        )
        print(f"Opened issue: {title}")


def main() -> None:
    """
    Checks pyproject.toml's minimum dependency versions against SPEC 0 (and
    Python's own EOL) and reports the results.

    Accepts command line arguments:

    --pyproject: Path to pyproject.toml. Defaults to the repo root relative to
        this script.
    --summary-output: Path to write the Markdown summary to, in addition to
        printing it. Optional.
    --violations-output: Path to write a JSON array of the within_window/past_due
        results to, for a workflow to act on. Optional.
    --create-issues: When set, opens a GitHub issue (via the `gh` CLI) for each
        violation that doesn't already have one open. Requires `gh` to be
        authenticated (e.g. via GH_TOKEN) and --repo to be set.
    --repo: The `owner/name` repository to open issues in. Required when
        --create-issues is set; defaults to the GITHUB_REPOSITORY env var.
    --dry-run: With --create-issues, print what would be done instead of calling
        `gh`.

    Returns
    -------
    None

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pyproject",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "pyproject.toml",
    )
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--violations-output", type=Path, default=None)
    parser.add_argument("--create-issues", action="store_true")
    parser.add_argument("--repo", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with args.pyproject.open("rb") as f:
        pyproject = tomllib.load(f)

    today = date.today()
    eol_dates = fetch_python_eol_dates()
    results = [check_python(parse_requires_python(pyproject), eol_dates, today)]
    floors = parse_core_dependency_floors(pyproject)
    for name in CORE_PACKAGES:
        results.append(check_package(name, floors[name], today))

    summary = render_markdown(results, today)
    print(summary)
    if args.summary_output:
        args.summary_output.write_text(summary)

    violations = [
        {
            "name": r.name,
            "floor": r.floor,
            "drop_date": r.drop_date.isoformat() if r.drop_date else None,
            "status": r.status,
        }
        for r in results
        if r.status != "ok"
    ]
    if args.violations_output:
        args.violations_output.write_text(json.dumps(violations))
    print(f"\n{len(violations)} violation(s) found.")

    if args.create_issues and violations:
        repo = args.repo or os.environ.get("GITHUB_REPOSITORY")
        if not repo:
            raise SystemExit("--create-issues requires --repo or GITHUB_REPOSITORY to be set.")
        create_github_issues(violations, repo, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
