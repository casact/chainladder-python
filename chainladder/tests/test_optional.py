"""
Check that chainladder imports and runs without its optional dependencies.

Each optional dependency gets one test class. Subprocess tests run a fresh
interpreter where ``sys.modules[name]`` is None, so any import of that module
raises ImportError the way a missing install does. In-process tests cover the
same code paths, since coverage does not track the subprocesses.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap

import pytest


def run_without(module_name: str, code: str) -> subprocess.CompletedProcess:
    """Run code in a new interpreter that cannot import module_name."""
    block = f"import sys\nsys.modules[{module_name!r}] = None\n"
    return subprocess.run(
        [sys.executable, "-c", block + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
    )


class TestOptionalJinja2:
    """chainladder without jinja2, which only Styler and Triangle.style need."""

    def test_import_and_fit_without_jinja2(self) -> None:
        """
        Check that importing chainladder and fitting a model need no jinja2.
        """
        result = run_without(
            "jinja2",
            """
            import chainladder as cl
            tri = cl.load_sample("raa")
            print(round(float(cl.Chainladder().fit(tri).ibnr_.sum()), 2))
            """,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "52135.23"

    def test_style_accessor_is_missing_without_jinja2(self) -> None:
        """
        Check that Triangle.style raises AttributeError, so hasattr returns False.
        """
        result = run_without(
            "jinja2",
            """
            import chainladder as cl
            tri = cl.load_sample("raa")
            print(hasattr(tri, "style"))
            try:
                tri.style
            except AttributeError as err:
                print(err)
            """,
        )
        assert result.returncode == 0, result.stderr
        lines = result.stdout.splitlines()
        assert lines[0] == "False"
        assert "requires jinja2" in lines[1]

    def test_styler_import_names_jinja2_without_it(self) -> None:
        """
        Check that cl.Styler and chainladder.core.Styler raise ImportError about jinja2.
        """
        result = run_without(
            "jinja2",
            """
            import chainladder as cl
            import chainladder.core
            for owner in (cl, chainladder.core):
                try:
                    owner.Styler
                except ImportError as err:
                    print("jinja2" in str(err).lower())
            """,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.split() == ["True", "True"]

    def test_styler_names_resolve_in_process(self):
        import chainladder
        import chainladder.core
        from chainladder.core.style import Styler

        assert chainladder.Styler is Styler
        assert chainladder.core.Styler is Styler

    @pytest.mark.parametrize("module_name", ["chainladder", "chainladder.core"])
    def test_unknown_module_attribute_raises(self, module_name):
        module = importlib.import_module(module_name)
        with pytest.raises(AttributeError, match="no_such_name"):
            getattr(module, "no_such_name")

    def test_style_raises_attribute_error_when_styler_import_fails(self, monkeypatch):
        import chainladder as cl

        tri = cl.load_sample("raa")
        monkeypatch.setitem(sys.modules, "chainladder.core.style", None)
        assert not hasattr(tri, "style")
        with pytest.raises(AttributeError, match="jinja2"):
            _ = tri.style
