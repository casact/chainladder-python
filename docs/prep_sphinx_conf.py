"""
Prepend docs/_ext to sys.path in conf.py.

`jupyter-book config sphinx` lists local extensions but does not emit this path,
so standalone Sphinx (e.g. Read the Docs) cannot import them. Run after generating conf.py.
"""
from pathlib import Path

CONF = Path(__file__).resolve().parent / "conf.py"
MARKER = "_DOCS_EXT_DIR = Path(__file__).resolve().parent / '_ext'"

BLOCK = """import sys
import os
import subprocess
from pathlib import Path

_DOCS_EXT_DIR = Path(__file__).resolve().parent / '_ext'
_p = str(_DOCS_EXT_DIR)
if _p not in sys.path:
    sys.path.insert(0, _p)

def run_doctests(app):
    from sphinx.application import Sphinx

    # Only run during RTD build
    if os.environ.get("READTHEDOCS") == "True":
        import subprocess
        subprocess.run(
            ["sphinx-build", "-b", "doctest", ".", "_build/doctest"],
            check=True
        )

def setup(app):
    app.connect("builder-inited", run_doctests)

"""


def main() -> None:
    text = CONF.read_text(encoding="utf8")
    if MARKER in text:
        return
    CONF.write_text(BLOCK + text, encoding="utf8")


if __name__ == "__main__":
    main()
