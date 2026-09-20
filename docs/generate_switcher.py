"""
Write docs/_static/switcher.json from the versions Read the Docs has built.

The version switcher needs a list of versions to offer. Maintaining that list by
hand means it is wrong from the moment a release is cut, so it is generated at
build time from Read the Docs' own API: whatever RTD is hosting is what the menu
offers, and activating a version on RTD is all it takes to add an entry.

Only the `main` build's copy is ever read -- every version's json_url points
there, so one list serves them all and old versions do not need rebuilding to
learn about new ones.

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

    if not entries:
        print("switcher.json left as committed: no built versions reported")
        return

    SWITCHER.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf8")
    print(f"switcher.json lists {', '.join(e['version'] for e in entries)}")


if __name__ == "__main__":
    main()
