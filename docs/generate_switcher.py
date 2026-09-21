"""
Write docs/_static/switcher.json from the versions Read the Docs has built.

The version switcher needs a list of versions to offer. Maintaining that list by
hand means it is wrong from the moment a release is cut, so it is generated at
build time from Read the Docs' own API: whatever RTD is hosting is what the menu
offers, and activating a version on RTD is all it takes to add an entry.

Every build reads the copy it wrote itself: prep_sphinx_conf.py points json_url
at this version's own _static. That keeps a pull request preview self-contained,
at the cost of a version listing only what existed when it was last built.

The checked-in switcher.json is the fallback. A docs build is not worth failing
over a navigation menu, so any problem here leaves that file in place.
"""

import json
import os
import re
import urllib.error
import urllib.request

from pathlib import Path

API = (
    "https://app.readthedocs.org/api/v3/projects/{slug}/versions/?active=true&limit=100"
)
SWITCHER = Path(__file__).resolve().parent / "_static" / "switcher.json"

# Read the Docs' slugs are terse. These read better in a menu.
LABELS = {"main": "main (dev)"}

# The version a reader should land on when they have expressed no preference.
PREFERRED = "stable"

# --- proof of concept only --------------------------------------------------
# Read the Docs has every release registered as a version, but none of them are
# activated, so the API reports only stable and main and the menu renders nearly
# empty. build_demo_versions.py builds these releases into this build's own
# output instead, and the links below point at them there -- so a reviewer can
# open them without anything being activated in the live project.
#
# The two lists have to agree: a release named here without a build beside it is
# a link to a 404. SWITCHER_DEMO turns this on, set in readthedocs.yaml. The
# branch that makes this work for real should delete the flag, this list and
# demo_entries(), and activate the versions on Read the Docs instead.
#
# Hardcoded rather than read from git tags: Read the Docs clones with
# --depth 1, so tags are not reliably present in the build checkout.
DEMO_RELEASES = ("v0.10.1", "v0.9.2", "v0.8.26")


def pull_entry() -> list[dict]:
    """
    A menu entry for the pull request build itself.

    Without one, the menu is a one way trip: a reviewer who opens a release has
    no way back to the build they came from. The label matches the version_match
    prep_sphinx_conf.py sets, so the theme marks this entry as the current one
    and leaves the button's text alone.

    Returns
    -------
    list of dict
        The single entry, or nothing when this is not a pull request build.
    """
    version = os.environ.get("READTHEDOCS_VERSION", "")
    if not version or os.environ.get("READTHEDOCS_VERSION_TYPE") != "external":
        return []

    label = f"{version} (pull)"
    return [{"name": label, "version": label, "url": f"/{version}/"}]


def demo_entries(existing: list[dict]) -> list[dict]:
    """
    Menu entries for the releases built into this build's output.

    Parameters
    ----------
    existing: list of dict
        The entries already built from the API, so a release that is genuinely
        active is not listed twice.

    Returns
    -------
    list of dict
        One entry per release, newest first, in the order of DEMO_RELEASES.
        Empty when the version being built is unknown, since the links are
        relative to it.
    """
    version = os.environ.get("READTHEDOCS_VERSION", "")
    if not version:
        return []

    seen = {entry["version"] for entry in existing}
    return [
        {
            "name": slug.lstrip("v"),
            "version": slug,
            # Served from inside this build, so the link resolves on the pull
            # request preview's domain as readily as on readthedocs.io.
            "url": f"/{version}/{slug}/",
        }
        for slug in DEMO_RELEASES
        if slug not in seen
    ]


def sort_key(slug: str) -> tuple:
    """
    Order the menu: stable, then main, then releases newest first.

    Parameters
    ----------
    slug: str
        A Read the Docs version slug, e.g. ``stable`` or ``v0.10.1``.

    Returns
    -------
    tuple
        Sortable ascending, so the caller needs no ``reverse``.
    """
    if slug == PREFERRED:
        return (0,)
    if slug == "main":
        return (1,)
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", slug)
    if match:
        # Negated so that the newest release sorts first.
        return (2, *(-int(part) for part in match.groups()))
    return (3, slug)


def fetch(slug: str) -> list[dict]:
    """
    Every active version Read the Docs has actually built.

    Parameters
    ----------
    slug: str
        The Read the Docs project slug.

    Returns
    -------
    list of dict
        The switcher entries, ordered for display. A version that is active but
        not yet built is left out: its URL would 404.
    """
    with urllib.request.urlopen(API.format(slug=slug), timeout=30) as response:
        payload = json.load(response)

    versions = [v for v in payload.get("results", []) if v.get("built")]
    versions.sort(key=lambda v: sort_key(v["slug"]))

    return [
        {
            # `version` is matched against the build's own version_match, which
            # prep_sphinx_conf.py sets from READTHEDOCS_VERSION -- the slug.
            "name": LABELS.get(v["slug"], v["slug"]),
            "version": v["slug"],
            "url": v["urls"]["documentation"],
            **({"preferred": True} if v["slug"] == PREFERRED else {}),
        }
        for v in versions
    ]


def main() -> None:
    slug = os.environ.get("READTHEDOCS_PROJECT", "chainladder-python")
    try:
        entries = fetch(slug)
    except (urllib.error.URLError, OSError, ValueError, KeyError) as error:
        # The demo entries below are built from this branch, not the API, so
        # they still stand. Losing stable and main is better than losing the
        # menu in front of the reviewers it was written for.
        print(f"read the docs API unavailable: {type(error).__name__}: {error}")
        entries = []

    if os.environ.get("SWITCHER_DEMO"):
        entries = pull_entry() + entries + demo_entries(entries)

    if not entries:
        print("switcher.json left as committed: no built versions reported")
        return

    SWITCHER.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf8")
    print(f"switcher.json lists {', '.join(e['version'] for e in entries)}")


if __name__ == "__main__":
    main()
