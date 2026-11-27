"""
Check that chainladder imports and runs without jinja2, which only Styler needs.

Each test runs in a fresh interpreter where ``sys.modules["jinja2"]`` is None,
so any import of jinja2 raises ImportError the way a missing install does.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

BLOCK_JINJA2 = 'import sys\nsys.modules["jinja2"] = None\n'


def run_without_jinja2(code: str) -> subprocess.CompletedProcess:
    """Run code in a new interpreter that cannot import jinja2."""
    return subprocess.run(
        [sys.executable, "-c", BLOCK_JINJA2 + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_import_and_fit_without_jinja2() -> None:
    """
    Check that importing chainladder and fitting a model need no jinja2.
    """
    result = run_without_jinja2(
        """
        import chainladder as cl
        tri = cl.load_sample("raa")
        print(round(float(cl.Chainladder().fit(tri).ibnr_.sum()), 2))
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "52135.23"


def test_style_accessor_is_missing_without_jinja2() -> None:
    """
    Check that Triangle.style raises AttributeError, so hasattr returns False.
    """
    result = run_without_jinja2(
        """
        import chainladder as cl
        tri = cl.load_sample("raa")
        print(hasattr(tri, "style"))
        try:
            tri.style
        except AttributeError as err:
            print(err)
        """
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == "False"
    assert "requires jinja2" in lines[1]


def test_styler_import_names_jinja2_without_it() -> None:
    """
    Check that cl.Styler and chainladder.core.Styler raise ImportError about jinja2.
    """
    result = run_without_jinja2(
        """
        import chainladder as cl
        import chainladder.core
        for owner in (cl, chainladder.core):
            try:
                owner.Styler
            except ImportError as err:
                print("jinja2" in str(err).lower())
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["True", "True"]
