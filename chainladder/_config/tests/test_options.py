from __future__ import annotations

import pytest

import chainladder as cl

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytest import CaptureFixture
    from pytest import MonkeyPatch


def test_reset_option() -> None:
    """
    Change some of the options and then reset them. Values after reset should match the original values.

    Returns
    -------
    None

    """

    original_backend = cl.options.ARRAY_BACKEND
    original_auto_sparse = cl.options.AUTO_SPARSE
    original_array_priority = cl.options.ARRAY_PRIORITY

    try:
        cl.options.set_option("ARRAY_BACKEND", "sparse")
        cl.options.set_option("AUTO_SPARSE", False)
        cl.options.set_option("ARRAY_PRIORITY", ["sparse", "dask", "numpy", "cupy"])

        cl.options.reset_option()

        assert cl.options.ARRAY_BACKEND == original_backend
        assert cl.options.AUTO_SPARSE == original_auto_sparse
        assert cl.options.ARRAY_PRIORITY == original_array_priority

    finally:
        # Manual reset in case of test failure.
        cl.options.set_option("ARRAY_BACKEND", original_backend)
        cl.options.set_option("AUTO_SPARSE", original_auto_sparse)
        cl.options.set_option("ARRAY_PRIORITY", original_array_priority)


def test_options_defaults() -> None:
    """
    When initialized, default options should be correct and accessible from the options variable.

    Returns
    -------
    None

    """
    options = cl.Options()
    assert options.ARRAY_BACKEND == "numpy"
    assert options.AUTO_SPARSE
    assert options.ARRAY_PRIORITY == ["dask", "sparse", "cupy", "numpy"]
    assert isinstance(options.ULT_VAL, str)
    assert options.get_option("display.value_format") is None
    assert options.get_option("display.pattern_format") is None
    assert options.get_option("display.html.auto_format_small") == "{0:,.4f}"
    assert options.get_option("display.html.auto_format_medium") == "{0:,.2f}"
    assert options.get_option("display.html.auto_format_large") == "{:,.0f}"
    assert options.get_option("display.html.auto_format_small_threshold") == 10
    assert options.get_option("display.html.auto_format_medium_threshold") == 1000


def test_display_format_options_dotted_names() -> None:
    """
    display.value_format and display.pattern_format use dotted names rather
    than the flat ALL_CAPS convention used by the other options, but should
    otherwise support the full get_option/set_option/reset_option API,
    since Options stores arbitrary string keys (not just valid Python
    identifiers) in its instance ``__dict__``.

    Returns
    -------
    None

    """
    try:
        cl.options.set_option("display.value_format", "{:,.0f}")
        assert cl.options.get_option("display.value_format") == "{:,.0f}"

        cl.options.set_option("display.pattern_format", "{:.4f}".format)
        assert cl.options.get_option("display.pattern_format")(1.23456) == "1.2346"
    finally:
        cl.options.reset_option("display.value_format")
        cl.options.reset_option("display.pattern_format")

    assert cl.options.get_option("display.value_format") is None
    assert cl.options.get_option("display.pattern_format") is None


def test_display_format_option_invalid_name_raises() -> None:
    """
    An unrecognized dotted option name should raise ValueError, same as any
    other invalid option.

    Returns
    -------
    None

    """
    with pytest.raises(ValueError):
        cl.options.get_option("display.not_a_real_option")


def test_describe_option_display_value_format() -> None:
    """
    describe_option should support the dotted display.value_format name and
    its multi-word type hint (str, callable, or None).

    Returns
    -------
    None

    """
    result = cl.options.describe_option("display.value_format", _print_desc=False)
    assert isinstance(result, str)
    assert "display.value_format : str, callable, or None" in result
    assert "[default: None]" in result
    assert "[currently: None]" in result


def test_get_option() -> None:
    """
    get_option should return the appropriate attribute value.

    Returns
    -------
    None

    """
    assert cl.options.get_option("ARRAY_BACKEND") == cl.options.ARRAY_BACKEND
    assert cl.options.get_option("AUTO_SPARSE") == cl.options.AUTO_SPARSE
    assert cl.options.get_option("ARRAY_PRIORITY") == cl.options.ARRAY_PRIORITY
    assert cl.options.get_option("ULT_VAL") == cl.options.ULT_VAL


def test_set_option_consistency() -> None:
    """
    When set_option changes an option value, get_option should return the new option value.

    Returns
    -------
    None

    """
    try:
        cl.options.set_option("ARRAY_BACKEND", "sparse")
        assert cl.options.ARRAY_BACKEND == "sparse"
        assert cl.options.get_option("ARRAY_BACKEND") == "sparse"
    finally:
        # Reset the options to default if the test fails.
        cl.options.reset_option("ARRAY_BACKEND")


def test_reset_single_option() -> None:
    """
    Set an option and check its value, then reset it and check its value.

    Returns
    -------
    None

    """
    cl.options.set_option("ARRAY_BACKEND", "sparse")
    assert cl.options.ARRAY_BACKEND == "sparse"
    # Return backend to original state.
    cl.options.reset_option("ARRAY_BACKEND")
    assert cl.options.ARRAY_BACKEND == "numpy"


def test_reset_option_invalid() -> None:
    """
    Supply in invalid option to cl.options.reset_option() and raise an error.

    Returns
    -------
    None
    """
    with pytest.raises(ValueError):
        cl.options.reset_option("NOT_A_REAL_OPTION")


def test_set_option_cupy_backend_deprecated() -> None:
    """
    Setting ARRAY_BACKEND to 'cupy' should emit a DeprecationWarning. See issue #843.

    Returns
    -------
    None
    """
    try:
        with pytest.warns(DeprecationWarning, match="cupy"):
            cl.options.set_option("ARRAY_BACKEND", "cupy")
    finally:
        cl.options.reset_option("ARRAY_BACKEND")


def test_set_option_dask_backend_deprecated() -> None:
    """
    Setting ARRAY_BACKEND to 'dask' should emit a DeprecationWarning. See issue #842.

    Returns
    -------
    None
    """
    try:
        with pytest.warns(DeprecationWarning, match="dask"):
            cl.options.set_option("ARRAY_BACKEND", "dask")
    finally:
        cl.options.reset_option("ARRAY_BACKEND")


def test_set_option_cupy_priority_deprecated() -> None:
    """
    Setting ARRAY_PRIORITY with 'cupy' ahead of a non-deprecated backend
    ('numpy' or 'sparse') should emit a DeprecationWarning. See issue #843.

    Returns
    -------
    None
    """
    try:
        with pytest.warns(DeprecationWarning, match="cupy"):
            cl.options.set_option("ARRAY_PRIORITY", ["cupy", "numpy", "sparse", "dask"])
    finally:
        cl.options.reset_option("ARRAY_PRIORITY")


def test_set_option_dask_priority_deprecated() -> None:
    """
    Setting ARRAY_PRIORITY with 'dask' ahead of a non-deprecated backend
    ('numpy' or 'sparse') should emit a DeprecationWarning. See issue #842.

    Returns
    -------
    None
    """
    try:
        with pytest.warns(DeprecationWarning, match="dask"):
            cl.options.set_option("ARRAY_PRIORITY", ["dask", "numpy", "sparse", "cupy"])
    finally:
        cl.options.reset_option("ARRAY_PRIORITY")


def test_set_option_deprecated_priority_last_no_warning(recwarn) -> None:
    """
    Setting ARRAY_PRIORITY with the deprecated backends ('cupy' and 'dask')
    ranked below every non-deprecated backend should not warn, since neither
    would ever be selected over a supported backend. See issues #842 and #843.

    Returns
    -------
    None
    """
    try:
        cl.options.set_option("ARRAY_PRIORITY", ["numpy", "sparse", "dask", "cupy"])
        assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]
    finally:
        cl.options.reset_option("ARRAY_PRIORITY")


def test_set_option_supported_backend_no_warning(recwarn) -> None:
    """
    Setting a non-deprecated backend ('sparse'), and a priority list where no
    deprecated backend precedes a supported one, should not emit a
    DeprecationWarning.

    Returns
    -------
    None
    """
    try:
        cl.options.set_option("ARRAY_BACKEND", "sparse")
        cl.options.set_option("ARRAY_PRIORITY", ["sparse", "numpy"])
        assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]
    finally:
        cl.options.reset_option("ARRAY_BACKEND")
        cl.options.reset_option("ARRAY_PRIORITY")


def test_describe_option(capsys: CaptureFixture[str]) -> None:
    """
    Supply an option to cl.options.describe_option(). Attribute name, type, default/current
    settings should be captured in the output.

    Parameters
    ----------
    capsys: CaptureFixture[str]
        pytest built-in fixture to capture stdout

    Returns
    -------
    None

    """
    cl.options.describe_option("ARRAY_BACKEND")
    captured = capsys.readouterr()
    assert "ARRAY_BACKEND : str" in captured.out
    assert "[default: numpy]" in captured.out
    assert "[currently: numpy]" in captured.out


def test_describe_option_multi(capsys) -> None:
    """
    Supply two options to cl.options.describe_option(). Attribute names, types, default/current
    settings should be captured in the output.

    Parameters
    ----------
    capsys: CaptureFixture[str]
        pytest built-in fixture to capture stdout

    Returns
    -------
    None

    """
    cl.options.describe_option("ARRAY_BACKEND|AUTO_SPARSE")
    captured = capsys.readouterr()
    assert "ARRAY_BACKEND : str" in captured.out
    assert "[default: numpy]" in captured.out
    assert "[currently: numpy]" in captured.out
    assert "AUTO_SPARSE : bool" in captured.out
    assert "[default: True]" in captured.out
    assert "[currently: True]" in captured.out
    assert "ARRAY_PRIORITY" not in captured.out


def test_describe_option_all(capsys) -> None:
    """
    Execute cl.options.describe_option() with default arguments. All attributes
    should be captured.

    Parameters
    ----------
    capsys: CaptureFixture[str]
        pytest built-in fixture to capture stdout

    Returns
    -------
    None

    """
    cl.options.describe_option()
    captured = capsys.readouterr()
    for key in cl.Options()._defaults:
        assert key in captured.out


def test_describe_option_return_string() -> None:
    """
    Execute cl.options.desribe_option() with _print_desc=False. Should return a string. Check
    if attribute info is in the string.

    Returns
    -------
    None

    """
    result = cl.options.describe_option("ARRAY_BACKEND", _print_desc=False)
    assert isinstance(result, str)
    assert "ARRAY_BACKEND : str" in result
    assert "[default: numpy]" in result
    assert "[currently: numpy]" in result


def test_deprecated_option_kwarg_warns() -> None:
    """
    Passing option= to get_option or set_option should emit a FutureWarning.
    """
    with pytest.warns(FutureWarning, match="'option'"):
        cl.options.get_option(option="ARRAY_BACKEND")

    try:
        with pytest.warns(FutureWarning, match="'option'"):
            cl.options.set_option(option="ARRAY_BACKEND", value="numpy")
    finally:
        cl.options.reset_option("ARRAY_BACKEND")


def test_deprecated_option_kwarg_reset_option_warns() -> None:
    """
    Passing option= to reset_option should emit a FutureWarning.
    """
    try:
        cl.options.set_option("ARRAY_BACKEND", "sparse")
        with pytest.warns(FutureWarning, match="'option'"):
            cl.options.reset_option(option="ARRAY_BACKEND")
        assert cl.options.ARRAY_BACKEND == "numpy"
    finally:
        cl.options.reset_option("ARRAY_BACKEND")


def test_get_option_missing_pat_raises() -> None:
    """
    Calling get_option() with neither pat nor option should raise TypeError.
    """
    with pytest.raises(TypeError, match="missing required argument"):
        cl.options.get_option()


def test_describe_option_no_docstring_match(monkeypatch: MonkeyPatch) -> None:
    """
    When the class docstring has no entry for an option, describe_option should fall back
    to 'No description available.' rather than raising an error.

    Parameters
    ----------
    monkeypatch: MonkeyPatch
        The pytest built-in monkeypatch fixture.

    Returns
    -------
    None
    """
    monkeypatch.setattr(cl.Options, "__doc__", "")
    result = cl.options.describe_option("ARRAY_BACKEND", _print_desc=False)
    assert "No description available." in result


def test_describe_option_invalid() -> None:
    """
    Execute cl.options.desribe_option() with an invalid argument. Should raise a ValueError.

    Returns
    -------
    None

    """
    with pytest.raises(ValueError):
        cl.options.describe_option("NOT_A_REAL_OPTION")


def test_both_pat_and_option_raises() -> None:
    """
    Passing both pat and option to get_option, set_option, or reset_option should raise TypeError.
    """
    with pytest.raises(TypeError, match="Cannot specify both"):
        cl.options.get_option(pat="ARRAY_BACKEND", option="ARRAY_BACKEND")


def test_set_option_missing_value_raises() -> None:
    """
    Calling set_option with pat but no value should raise TypeError.
    """
    with pytest.raises(TypeError, match="missing required argument"):
        cl.options.set_option("ARRAY_BACKEND")


def test_describe_option_invalid_regex() -> None:
    """
    Passing a malformed regular expression to describe_option should raise ValueError.
    """
    with pytest.raises(ValueError, match="not a valid regular expression"):
        cl.options.describe_option("[")
