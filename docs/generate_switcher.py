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
# empty. Listing the releases anyway is what lets a reviewer see the finished
# shape of the feature; their links 404 until the versions are activated, which
# is a production concern and not this branch's.
#
# SWITCHER_DEMO turns this on, set in readthedocs.yaml. The branch that makes
# this work for real should delete the flag, this list, and demo_entries().
#
# Hardcoded rather than read from git tags: Read the Docs clones with
# --depth 1, so tags are not reliably present in the build checkout.
DEMO_RELEASES = (
    "v0.10.1",
    "v0.9.2",
    "v0.8.26",
    "v0.7.12",
    "v0.6.3",
    "v0.5.5",
    "v0.4.10",
    "v0.3.0",
    "v0.2.9",
    "v0.1.7",
)
DOCS_URL = "https://chainladder-python.readthedocs.io/{slug}/"


def demo_entries(existing: list[dict]) -> list[dict]:
    """
    Menu entries for releases Read the Docs is not yet hosting.

    Parameters
    ----------
    existing: list of dict
        The entries already built from the API, so a release that is genuinely
        active is not listed twice.

    Returns
    -------
    list of dict
        One entry per release, newest first, in the order of DEMO_RELEASES.
    """
    seen = {entry["version"] for entry in existing}
    return [
        {
            "name": slug.lstrip("v"),
            "version": slug,
            "url": DOCS_URL.format(slug=slug),
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
        print(f"switcher.json left as committed: {type(error).__name__}: {error}")
        return

    if os.environ.get("SWITCHER_DEMO"):
        entries += demo_entries(entries)

    if not entries:
        print("switcher.json left as committed: no built versions reported")
        return

    SWITCHER.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf8")
    print(f"switcher.json lists {', '.join(e['version'] for e in entries)}")


if __name__ == "__main__":
    main()
