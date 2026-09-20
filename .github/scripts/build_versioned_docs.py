"""
Build the documentation for several versions into one directory tree.

Each version is built from its own git ref, so the pages match what that
release shipped. The version switcher's assets and configuration are copied in
from the working tree first, because released versions predate them, and the
switcher has to appear on every version for its links to be useful.

The result is a directory holding one subdirectory per version plus the
switcher.json that every version reads:

    <output>/switcher.json
    <output>/main/index.html
    <output>/v0.10.1/index.html
    ...

Serve <output> over HTTP to browse it. Opening the files directly will not
work: the switcher fetches switcher.json, which a browser blocks from file://.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Copied into each version's docs/ before building, so released versions get
# the switcher they were built without.
ASSETS = (
    Path("_static/switcher.json"),
    Path("_static/custom.css"),
    Path("_templates/version-switcher.html"),
)

# Appended to each version's conf.py. Sphinx executes conf.py top to bottom, so
# these override whatever the released version set.
CONF_PATCH = """

# --- appended by .github/scripts/build_versioned_docs.py ---------------------
# Releases before v0.10 leave templates_path unset, so the copied-in
# version-switcher.html override would never be found.
templates_path = list(dict.fromkeys(["_templates", *globals().get("templates_path", [])]))
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {{
    **globals().get("html_theme_options", {{}}),
    "article_header_end": [
        "version-switcher.html",
        "article-header-buttons.html",
    ],
    "switcher": {{
        "json_url": "{json_url}",
        "version_match": "{version}",
    }},
}}
"""


def build(docs_dir: Path, output: Path, version: str, json_url: str) -> bool:
    """
    Build one version's documentation.

    Parameters
    ----------
    docs_dir: Path
        The docs directory to build, in a worktree or in the repository.
    output: Path
        Where to write the HTML.
    version: str
        The name this version is known by in switcher.json.
    json_url: str
        Where the built pages should fetch switcher.json from.

    Returns
    -------
    bool
        Whether the build succeeded.
    """
    for asset in ASSETS:
        source = REPO / "docs" / asset
        target = docs_dir / asset
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)

    conf = docs_dir / "conf.py"
    conf.write_text(
        conf.read_text(encoding="utf8")
        + CONF_PATCH.format(json_url=json_url, version=version),
        encoding="utf8",
    )

    # Notebook execution is off: this builds many versions against one
    # environment, and old notebooks need the dependencies of their own day.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-b",
            "html",
            str(docs_dir),
            str(output),
            "-D",
            "nb_execution_mode=off",
            "-q",
        ],
        cwd=docs_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # Builds run in parallel, so only failures are worth printing, and only
        # the tail: Sphinx is verbose about warnings that did not stop it.
        print(f"\n--- {version} failed ---", flush=True)
        print("\n".join((result.stdout + result.stderr).splitlines()[-15:]), flush=True)
        shutil.rmtree(output, ignore_errors=True)
        return False

    # Releases old enough to predate sphinx-book-theme build cleanly but render
    # no switcher, which would strand a reader on a version they cannot leave.
    if not any(
        "version-switcher__container" in page.read_text(errors="replace")
        for page in output.rglob("*.html")
    ):
        print(f"\n--- {version} built without a version switcher ---", flush=True)
        shutil.rmtree(output, ignore_errors=True)
        return False

    return True


def latest_per_minor(tags: list[str]) -> list[tuple[str, str]]:
    """
    Keep only the newest patch release of each minor version.

    Parameters
    ----------
    tags: list of str
        Tag names, e.g. ``v0.10.1``. Names that are not a plain
        major.minor.patch release, such as ``v0.8.11-alpha``, are dropped:
        a pre-release is not the version a reader should be sent to.

    Returns
    -------
    list of tuple of str
        ``(ref, name)`` pairs, newest first. The name drops the tag's "v":
        ``v0.10.1`` is published as ``0.10.1``.
    """
    newest: dict[tuple[int, int], tuple[int, str]] = {}
    for tag in tags:
        match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", tag)
        if not match:
            continue
        major, minor, patch = (int(part) for part in match.groups())
        if newest.get((major, minor), (-1, ""))[0] < patch:
            newest[(major, minor)] = (patch, tag)
    return [
        (tag, tag.lstrip("v")) for _, (_, tag) in sorted(newest.items(), reverse=True)
    ]


def deployable_tags() -> list[str]:
    """
    Every tag that ships a docs/conf.py, newest first.

    Returns
    -------
    list of str
        The tag names. A tag without docs cannot be built, so it is skipped.
    """
    tags = subprocess.run(
        ["git", "tag", "--sort=-v:refname"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    return [
        tag
        for tag in tags
        if subprocess.run(
            ["git", "cat-file", "-e", f"{tag}:docs/conf.py"],
            cwd=REPO,
            capture_output=True,
        ).returncode
        == 0
    ]


def build_ref(ref: str, name: str, output: Path, json_url: str) -> bool:
    """
    Check a ref out into a throwaway worktree and build its documentation.

    Parameters
    ----------
    ref: str
        The git ref to build.
    name: str
        The name this version is known by in switcher.json.
    output: Path
        The root of the version tree; the build lands in ``output / name``.
    json_url: str
        Where the built pages should fetch switcher.json from.

    Returns
    -------
    bool
        Whether the build succeeded.
    """
    with tempfile.TemporaryDirectory() as tmp:
        worktree = Path(tmp) / "repo"
        # A worktree is used even for HEAD: the build patches conf.py, which
        # must not touch the repository itself.
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree), ref],
            cwd=REPO,
            check=True,
            capture_output=True,
        )
        try:
            return build(worktree / "docs", output / name, name, json_url)
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree)],
                cwd=REPO,
                check=False,
                capture_output=True,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "docs" / "_build" / "versions",
        help="Directory to write the version tree into.",
    )
    parser.add_argument(
        "--version",
        action="append",
        default=[],
        dest="versions",
        metavar="REF[:NAME]",
        help="A git ref to build, optionally renamed, e.g. v0.10.1 or HEAD:main.",
    )
    parser.add_argument(
        "--all-tags",
        action="store_true",
        help="Build every tag that carries a docs/conf.py, newest first.",
    )
    parser.add_argument(
        "--minor-only",
        action="store_true",
        help="With --all-tags, build only the newest patch of each minor "
        "version, so the menu lists one entry per release line.",
    )
    parser.add_argument(
        "--stable",
        default="",
        metavar="NAME",
        help="The version to mark as the preferred one, labelled '(stable)' in "
        "the menu. Defaults to the newest tag built.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="How many versions to build at once.",
    )
    parser.add_argument(
        "--base-url",
        default="",
        help="Prefix the switcher's links, e.g. /chainladder-python for a "
        "GitHub Pages project site. Defaults to serving from the root.",
    )
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    base = args.base_url.rstrip("/")
    json_url = f"{base}/switcher.json"

    versions = [
        (ref, name)
        for ref, _, name in (v.partition(":") for v in args.versions)
        for name in [name or ref]
    ]
    if args.all_tags:
        tags = deployable_tags()
        versions += (
            latest_per_minor(tags) if args.minor_only else [(tag, tag) for tag in tags]
        )

    failed = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {
            pool.submit(build_ref, ref, name, output, json_url): name
            for ref, name in versions
        }
        for future in as_completed(futures):
            name = futures[future]
            ok = future.result()
            print(f"{'built  ' if ok else 'FAILED '} {name}", flush=True)
            if not ok:
                failed.append(name)

    # Written after the builds so that it lists only versions that exist: a
    # menu entry pointing at a version that failed to build is a broken link.
    built = [(ref, name) for ref, name in versions if name not in failed]
    stable = args.stable or next(
        (name for ref, name in built if ref != "HEAD"),
        "",
    )
    (output / "switcher.json").write_text(
        json.dumps(
            [
                {
                    # `version` is matched against the build's own
                    # version_match, so it stays the bare name; `name` is what
                    # the menu displays.
                    "name": f"{name} (stable)" if name == stable else name,
                    "version": name,
                    "url": f"{base}/{name}/",
                    **({"preferred": True} if name == stable else {}),
                }
                for _, name in built
            ],
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )

    print(f"\nBuilt {len(built)}/{len(versions)} versions into {output}")
    if failed:
        print(f"Failed: {', '.join(sorted(failed))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
