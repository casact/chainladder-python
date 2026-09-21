"""
Build prior releases into this build's own output directory.

--- proof of concept only ---------------------------------------------------

Read the Docs has every release registered as a version but none of them
activated, so the switcher's links to prior versions 404. Activating them is a
production decision -- it rebuilds them from their own source, in the live
project -- and this branch only needs a reviewer to be able to click through.

So the releases are built here instead and written inside this build's output,
where Read the Docs serves them alongside it:

    <version>/                 this build
    <version>/v0.10.1/         built here
    <version>/v0.9.2/          built here
    <version>/v0.8.26/         built here

Everything therefore lives on one domain, including the pull request preview's
temporary one, and nothing outside this build is touched. The branch that makes
this work for real should delete this script and activate the versions instead.
"""

import os
import shutil
import subprocess
import sys
import tempfile

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUILDER = REPO / ".github" / "scripts" / "build_versioned_docs.py"
SWITCHER = REPO / "docs" / "_static" / "switcher.json"

# Kept in step with DEMO_RELEASES in generate_switcher.py, which writes the
# links these builds sit behind.
RELEASES = ("v0.10.1", "v0.9.2", "v0.8.26")


def fetch(tag: str) -> bool:
    """
    Make one tag available in the build's checkout.

    Read the Docs clones with ``--depth 1``, so tags are absent and have to be
    fetched one at a time. Fetching shallowly keeps that cheap.

    Parameters
    ----------
    tag: str
        The tag to fetch.

    Returns
    -------
    bool
        Whether the tag is now present. A tag already in the checkout, as in a
        local run against a full clone, counts as present.
    """
    if subprocess.run(["git", "cat-file", "-e", tag], cwd=REPO).returncode == 0:
        return True

    result = subprocess.run(
        ["git", "fetch", "--depth", "1", "origin", "tag", tag, "--no-tags"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"--- could not fetch {tag} ---\n{result.stderr.strip()}", flush=True)
        return False
    return True


def main() -> int:
    output = os.environ.get("READTHEDOCS_OUTPUT")
    version = os.environ.get("READTHEDOCS_VERSION")
    if not output or not version:
        print("not a Read the Docs build; nothing to do")
        return 0

    html = Path(output) / "html"
    if not html.is_dir():
        print(f"no HTML output at {html}; nothing to do")
        return 0

    tags = [tag for tag in RELEASES if fetch(tag)]
    if not tags:
        print("no releases could be fetched; the menu's links will 404")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "versions"
        result = subprocess.run(
            [
                sys.executable,
                str(BUILDER),
                "--output",
                str(staging),
                # Links inside the built pages resolve against this build's own
                # URL, so the nested versions stay reachable on whichever domain
                # is serving them.
                "--base-url",
                f"/{version}",
                "--jobs",
                "3",
                *(arg for tag in tags for arg in ("--version", f"{tag}:{tag}")),
            ],
        )
        if result.returncode != 0:
            # The switcher still renders; its links to whatever failed will
            # 404, which is the state this script exists to improve on, not a
            # reason to fail a build that is otherwise fine.
            print("some releases did not build; continuing", flush=True)

        for tag in tags:
            built = staging / tag
            if not built.is_dir():
                continue
            shutil.copytree(built, html / tag, dirs_exist_ok=True)
            print(f"nested {tag} under {version}/", flush=True)

    # The nested builds were patched to read <version>/switcher.json, so put the
    # menu this build generated there too. One list, every page.
    if SWITCHER.is_file():
        shutil.copyfile(SWITCHER, html / "switcher.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
