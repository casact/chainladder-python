"""
Patch the conf.py that `jupyter-book config sphinx` generates.

That command regenerates docs/conf.py from docs/_config.yml on every build
(Read the Docs runs it in `pre_build`), so whatever the checked-in conf.py does
is discarded, and whatever _config.yml cannot express is lost. Run this after
generating conf.py to put both back:

- docs/_ext on sys.path, since jupyter-book lists the local extensions but does
  not emit the path they live on.
- The version switcher's version_match, since _config.yml can only hold a
  literal and the version being built is only known at build time.
"""

from pathlib import Path

CONF = Path(__file__).resolve().parent / "conf.py"

EXT_MARKER = "_DOCS_EXT_DIR = Path(__file__).resolve().parent / '_ext'"

EXT_BLOCK = """import sys
from pathlib import Path
_DOCS_EXT_DIR = Path(__file__).resolve().parent / '_ext'
_p = str(_DOCS_EXT_DIR)
if _p not in sys.path:
    sys.path.insert(0, _p)

"""

SWITCHER_MARKER = "# --- appended by docs/prep_sphinx_conf.py ---"

# Appended rather than prepended: Sphinx runs conf.py top to bottom, so this
# wins over the html_theme_options jupyter-book wrote out from _config.yml.
# Read the Docs sets READTHEDOCS_VERSION to the version's slug ("main",
# "stable") or, on a pull request build, the pull request number. Anywhere else
# it is unset and the literal from _config.yml stands, leaving local builds
# alone.
SWITCHER_BLOCK = """

# --- appended by docs/prep_sphinx_conf.py ---
import os as _os
_version_match = _os.environ.get("READTHEDOCS_VERSION", "")
if _version_match:
    html_theme_options = {
        **globals().get("html_theme_options", {}),
        "switcher": {
            **globals().get("html_theme_options", {}).get("switcher", {}),
            "version_match": _version_match,
        },
    }
"""


def main() -> None:
    text = CONF.read_text(encoding="utf8")

    if EXT_MARKER not in text:
        text = EXT_BLOCK + text
    if SWITCHER_MARKER not in text:
        text = text + SWITCHER_BLOCK

    CONF.write_text(text, encoding="utf8")


if __name__ == "__main__":
    main()
