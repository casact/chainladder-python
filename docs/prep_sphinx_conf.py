"""
Prepend docs/_ext to sys.path in conf.py.

`jupyter-book config sphinx` lists local extensions but does not emit this path,
so standalone Sphinx (e.g. Read the Docs) cannot import them. Run after generating conf.py.
"""
from pathlib import Path

CONF = Path(__file__).resolve().parent / "conf.py"
MARKER = "_DOCS_EXT_DIR = Path(__file__).resolve().parent / '_ext'"

BLOCK = """import sys
from pathlib import Path
_DOCS_EXT_DIR = Path(__file__).resolve().parent / '_ext'
_p = str(_DOCS_EXT_DIR)
if _p not in sys.path:
    sys.path.insert(0, _p)

"""


def main() -> None:
    text = CONF.read_text(encoding="utf8")
    if MARKER in text:
        return
    CONF.write_text(BLOCK + text, encoding="utf8")


if __name__ == "__main__":
    main()
