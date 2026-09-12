"""
Test the deprecation tools.
"""

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import warnings

import pytest

from chainladder._config.deprecation import (
    _deprecated_drop_argument,
    _deprecated_rename,
    _deprecated_rename_argument,
)


def _warn_once(func, *args, **kwargs) -> tuple[object, warnings.WarningMessage]:
    """Calls func and returns (result, the single warning recorded)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = func(*args, **kwargs)
    assert caught is not None
    # Check that the warning was triggered.
    assert len(caught) == 1
    return result, caught[0]


class TestDeprecatedRename:
    """Test the _deprecated_rename decorator."""

    def test_warns_default_category(self) -> None:
        """Check that the default warning category is FutureWarning."""

        @_deprecated_rename("new_func")
        def old_func(x):
            return x + 1

        result, warning = _warn_once(old_func, 1)
        assert result == 2
        assert warning.category is FutureWarning

    def test_message_with_version(self) -> None:
        """Check the warning message when a version is given."""

        @_deprecated_rename("new_func", version="0.11.0")
        def old_func():
            pass

        _, warning = _warn_once(old_func)
        assert str(warning.message) == (
            "'old_func' is deprecated and will be renamed to 'new_func' in "
            "0.11.0. Update your code to use 'new_func' instead."
        )

    def test_message_without_version(self) -> None:
        """Check the warning message when no version is given."""

        @_deprecated_rename("new_func")
        def old_func():
            pass

        _, warning = _warn_once(old_func)
        assert str(warning.message) == (
            "'old_func' is deprecated and will be renamed to 'new_func'. "
            "Update your code to use 'new_func' instead."
        )

    def test_custom_category(self) -> None:
        """Check that a custom warning category is honored."""

        @_deprecated_rename("new_func", category=DeprecationWarning)
        def old_func():
            pass

        with pytest.warns(DeprecationWarning):
            old_func()

    def test_forwards_args_and_kwargs(self) -> None:
        """Check that positional and keyword arguments reach the wrapped function unchanged."""

        @_deprecated_rename("new_func")
        def old_func(a, b, *, c):
            return a, b, c

        with warnings.catch_warnings():  # noqa
            warnings.simplefilter("ignore")
            assert old_func(1, 2, c=3) == (1, 2, 3)

    def test_preserves_metadata(self) -> None:
        """Check that functools.wraps preserves the function's name and docstring."""

        @_deprecated_rename("new_func")
        def old_func():
            """Original docstring."""

        assert old_func.__name__ == "old_func"
        assert old_func.__doc__ == "Original docstring."

    def test_warns_every_call(self) -> None:
        """Check that the warning fires on every call, not just the first."""

        @_deprecated_rename("new_func")
        def old_func():
            pass

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            old_func()
            old_func()
        assert caught is not None
        assert len(caught) == 2


class TestDeprecatedRenameArgument:
    """Tests for the _deprecated_rename_argument decorator."""

    def test_old_name_translates_and_warns(self) -> None:
        """Check that the old argument name is translated to the new one and warns."""

        @_deprecated_rename_argument("old_arg", "new_arg")
        def func(new_arg):
            return new_arg

        result, warning = _warn_once(func, old_arg=1)
        assert result == 1
        assert warning.category is FutureWarning

    def test_new_name_no_warning(self) -> None:
        """Check that calling with the new argument name alone doesn't warn."""

        @_deprecated_rename_argument("old_arg", "new_arg")
        def func(new_arg):
            return new_arg

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = func(new_arg=1)
        assert result == 1
        assert caught is not None
        assert len(caught) == 0

    def test_neither_name_uses_default_no_warning(self) -> None:
        """Check that omitting both names falls back to the default without warning."""

        @_deprecated_rename_argument("old_arg", "new_arg")
        def func(new_arg="default"):
            return new_arg

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = func()
        assert result == "default"
        assert caught is not None
        assert len(caught) == 0

    def test_both_names_raises_type_error(self) -> None:
        """Check that passing both the old and new names raises a TypeError."""

        @_deprecated_rename_argument("old_arg", "new_arg")
        def func(new_arg=None):
            return new_arg

        with pytest.raises(
            TypeError, match="Cannot specify both 'old_arg' and 'new_arg'"
        ):
            # noinspection PyArgumentList
            func(old_arg=1, new_arg=2)  # pyright: ignore[reportCallIssue]

    def test_message_with_version(self) -> None:
        """Check the warning message when a version is given."""

        @_deprecated_rename_argument("old_arg", "new_arg", version="0.11.0")
        def func(new_arg=None):
            return new_arg

        _, warning = _warn_once(func, old_arg=1)
        assert str(warning.message) == (
            "'old_arg' is deprecated and will be renamed to 'new_arg' in "
            "0.11.0. Use 'new_arg' instead."
        )

    def test_message_without_version(self) -> None:
        """Check the warning message when no version is given."""

        @_deprecated_rename_argument("old_arg", "new_arg")
        def func(new_arg=None):
            return new_arg

        _, warning = _warn_once(func, old_arg=1)
        assert str(warning.message) == (
            "'old_arg' is deprecated and will be renamed to 'new_arg'. "
            "Use 'new_arg' instead."
        )

    def test_custom_category(self) -> None:
        """Check that a custom warning category is honored."""

        @_deprecated_rename_argument("old_arg", "new_arg", category=DeprecationWarning)
        def func(new_arg=None):
            return new_arg

        with pytest.warns(DeprecationWarning):
            # noinspection PyArgumentList
            func(old_arg=1)  # pyright: ignore[reportCallIssue]

    def test_positional_args_unaffected(self) -> None:
        """Check that positional arguments pass through unaffected."""

        @_deprecated_rename_argument("old_arg", "new_arg")
        def func(a, b, new_arg=None):
            return a, b, new_arg

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # noinspection PyArgumentList
            result = func(1, 2, old_arg=3)  # pyright: ignore[reportCallIssue]
        assert result == (1, 2, 3)
        assert caught is not None
        assert len(caught) == 1

    def test_preserves_metadata(self) -> None:
        """Check that functools.wraps preserves the function's name and docstring."""

        @_deprecated_rename_argument("old_arg", "new_arg")
        def func(new_arg=None):  # noqa
            """Original docstring."""

        assert func.__name__ == "func"
        assert func.__doc__ == "Original docstring."

    def test_warns_every_call(self) -> None:
        """Check that the warning fires on every call, not just the first."""

        @_deprecated_rename_argument("old_arg", "new_arg")
        def func(new_arg=None):
            return new_arg

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # noinspection PyArgumentList
            func(old_arg=1)  # pyright: ignore[reportCallIssue]
            # noinspection PyArgumentList
            func(old_arg=2)  # pyright: ignore[reportCallIssue]
        assert caught is not None
        assert len(caught) == 2


class TestDeprecatedDropArgument:
    """Tests for the _deprecated_drop_argument decorator."""

    def test_warns_and_forwards_value_unchanged(self) -> None:
        """Check that the deprecated argument still reaches the function unchanged."""

        @_deprecated_drop_argument("verbose")
        def func(x, verbose: bool = False):
            return x, verbose

        result, warning = _warn_once(func, 1, verbose=True)
        assert result == (1, True)
        assert warning.category is FutureWarning

    def test_not_passed_no_warning(self) -> None:
        """Check that omitting the deprecated argument doesn't warn."""

        @_deprecated_drop_argument("verbose")
        def func(x, verbose: bool = False):
            return x, verbose

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = func(1)
        assert result == (1, False)
        assert caught is not None
        assert len(caught) == 0

    def test_message_with_version(self) -> None:
        """Check the warning message when a version is given."""

        @_deprecated_drop_argument("verbose", version="0.11.0")
        def func(verbose: bool = False):  # noqa
            pass

        _, warning = _warn_once(func, verbose=True)
        assert str(warning.message) == (
            "'verbose' is deprecated and will be removed in 0.11.0."
        )

    def test_message_without_version(self) -> None:
        """Check the warning message when no version is given."""

        @_deprecated_drop_argument("verbose")
        def func(verbose: bool = False):  # noqa
            pass

        _, warning = _warn_once(func, verbose=True)
        assert str(warning.message) == (
            "'verbose' is deprecated and will be removed in a future release."
        )

    def test_custom_category(self) -> None:
        """Check that a custom warning category is honored."""

        @_deprecated_drop_argument("verbose", category=DeprecationWarning)
        def func(verbose: bool = False):  # noqa
            pass

        with pytest.warns(DeprecationWarning):
            func(verbose=True)

    def test_positional_args_unaffected(self) -> None:
        """Check that positional arguments pass through unaffected."""

        @_deprecated_drop_argument("verbose")
        def func(a, b, verbose: bool = False):
            return a, b, verbose

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = func(1, 2, verbose=True)
        assert result == (1, 2, True)
        assert caught is not None
        assert len(caught) == 1

    def test_preserves_metadata(self) -> None:
        """Check that functools.wraps preserves the function's name and docstring."""

        @_deprecated_drop_argument("verbose")
        def func(verbose: bool = False):  # noqa
            """Original docstring."""

        assert func.__name__ == "func"
        assert func.__doc__ == "Original docstring."

    def test_warns_every_call(self) -> None:
        """Check that the warning fires on every call, not just the first."""

        @_deprecated_drop_argument("verbose")
        def func(verbose: bool = False):  # noqa
            pass

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            func(verbose=True)
            func(verbose=True)
        assert caught is not None
        assert len(caught) == 2
