"""
The chainladder-python package was built to be able to handle all of your actuarial reserving needs in python.
It consists of popular actuarial tools, such as triangle data manipulation, link ratios calculation, and
IBNR estimates using both deterministic and stochastic models. We build this package so you no longer have to rely
on outdated software and tools when performing actuarial pricing or reserving indications.

This package strives to be minimalistic in needing its own API. The syntax mimics popular packages such as pandas for
data manipulation and scikit-learn for model construction. An actuary that is already familiar with these tools will be
able to pick up this package with ease. You will be able to save your mental energy for actual actuarial work.

The __init__.py file governs package configuration, including datetime datatypes and precision, backend and ultimate
valuation defaults, as well as package metadata such as version number.
"""
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import copy
import inspect
import re
import numpy as np
import pandas as pd
import warnings

from importlib.metadata import version
from typing import (
    overload,
    TYPE_CHECKING
)

if TYPE_CHECKING:
    from re import Match
    from types import FrameType
    from typing import Any, Literal
del TYPE_CHECKING
del annotations


# Get the default datetime64 data type and precision, extracted from Pandas installation.
# Used for cross-version compatibility between Pandas 2 and Pandas 3.
__dt64_dtype__: str = pd.to_datetime(["2000-01-01"]).dtype.name
__dt64_unit__: str = np.datetime_data(__dt64_dtype__)[0]

# Sentinel pattern used to mark a parameter as required and validate it.
_UNSET: Any = object()

_option_warning: str = (
    "The parameter 'option' is deprecated and will be removed in a future release. Use 'pat' instead."
)

# Array backends slated for removal, mapped to the issue tracking each one.
# Selecting one of these (via set_option, ARRAY_PRIORITY, or set_backend, or by
# passing a Dask dataframe to the Triangle constructor) emits a
# DeprecationWarning.
_DEPRECATED_BACKENDS: dict[str, str] = {
    "cupy": "https://github.com/casact/chainladder-python/issues/843",
    "dask": "https://github.com/casact/chainladder-python/issues/842",
}


def _deprecated_backend_message(backend: str) -> str:
    """Build the deprecation message for a soon-to-be-removed array backend."""
    return (
        f"The '{backend}' array backend is deprecated and will be removed in a "
        f"future release. See {_DEPRECATED_BACKENDS[backend]}."
    )


@overload
def _resolve_pat(pat: str | None, option: str | None, required: Literal[True] = ...) -> str: ...
@overload
def _resolve_pat(pat: str | None, option: str | None, required: Literal[False]) -> str | None: ...
del overload
def _resolve_pat(pat: str | None, option: str | None, required: bool = True) -> str | None:
    """
    Handles backward compatibility of 'options' parameter in options functions. Checks whether option or pat is
    assigned a value and returns it. This value is meant to be assigned to the 'pat' parameter of the calling function.

    Once the 'options' parameter is fully removed, this function can be deleted or generalized as a backwards
    compatibility tool to assist in the renaming and deprecation of function parameters.

    Parameters
    ----------
    pat: str | None
        The 'pat' parameter of the calling function.
    option: str | None
        The 'option' parameter of the calling function.
    required: bool
        Whether pat or option are required parameters in the calling function. Defaults to True.

    Returns
    -------
        The value to be assigned to the 'pat' parameter of the calling function.

    """
    # Raise an error if the user accidentally assigns a value to both 'pat' and 'option'.
    if pat is not None and option is not None:
        raise TypeError("Cannot specify both 'pat' and 'option'.")
    # Raise the deprecation warning if the user assigns a value to 'option'.
    if option is not None:
        warnings.warn(_option_warning, FutureWarning, stacklevel=3)
        pat: str = option
    # Raise an error if neither 'option' nor 'pat' is assigned.
    if pat is None and required:
        # Determine the name of the calling function.
        err: str = "Unable to determine calling function."
        frame: FrameType | None = inspect.currentframe()
        if frame is None:
            raise AttributeError(err)
        else:
            f_back: FrameType | None = frame.f_back
        if f_back is None:
            raise AttributeError(err)
        else:
            caller: str = f_back.f_code.co_name
        raise TypeError(f"{caller}() missing required argument: 'pat'.")
    return pat

class Options:
    """
    Used to set defaults for array backend and datetime units.

    Attributes
    ----------

    ARRAY_BACKEND: str
        The default array backend for chainladder.
    AUTO_SPARSE: bool
        Controls whether chainladder automatically converts a triangle's backing array to a sparse representation
        when it would be memory-efficient to do so.
    ARRAY_PRIORITY: list
        Determines which backend wins when two triangles with different backends interact, i.e.,
        when comparing or concatenating them.
    ULT_VAL: str
        The default ultimate valuation datetime, precision set to default of Pandas installation.

    """
    def __init__(self):
        self.ARRAY_BACKEND = "numpy"
        self.AUTO_SPARSE = True
        self.ARRAY_PRIORITY = ["dask", "sparse", "cupy", "numpy"]
        self.ULT_VAL = str(
            pd.Timestamp("2262-01-01") - \
            pd.Timedelta(1, unit=__dt64_unit__)
        )
        # Store initial values as defaults.
        self._defaults = copy.deepcopy({k: v for k, v in vars(self).items() if not k.startswith('_')})

    def get_option(
            self,
            pat: str | None = None,
            *,
            option: str | None = None
    ) -> str | bool | list:
        """
        Get the option value for the specified option.

        .. deprecated:: 0.9.3
            The ``option`` parameter is deprecated; use ``pat`` instead.

        Parameters
        ----------
        pat: str | None
            The option you wish to get the values for.
        option: str | None
            The option you wish to get the values for.

        Returns
        -------
        The option value.

        """
        pat: str = _resolve_pat(pat=pat, option=option)
        self._validate_option(pat)
        return getattr(self, pat)

    def set_option(
            self,
            pat: str | None = None,
            value: str | bool | list = _UNSET,
            *,
            option: str | None = None
    ) -> None:
        """
        Set the option value for the specified option.

        .. deprecated:: 0.9.3
            The ``option`` parameter is deprecated; use ``pat`` instead.

        Parameters
        ----------
        pat: str | None
            The option you wish to set the value for.
        value: str | bool | list
            The option value.
        option: str | None
            The option you wish to set the values for.

        Returns
        -------
        None

        """
        pat: str = _resolve_pat(pat=pat, option=option)
        self._validate_option(pat)
        if value is _UNSET:
            raise TypeError("set_option() missing required argument: 'value'.")
        if pat == "ARRAY_BACKEND" and value in _DEPRECATED_BACKENDS:
            warnings.warn(
                _deprecated_backend_message(value),
                DeprecationWarning,
                stacklevel=2,
            )
        elif pat == "ARRAY_PRIORITY" and isinstance(value, list):
            # Only warn when a deprecated backend ('cupy' or 'dask') is
            # prioritized ahead of a non-deprecated backend ('numpy' or
            # 'sparse'), i.e. it would actually be selected over a supported
            # backend. The position in the list determines precedence.
            for backend in _DEPRECATED_BACKENDS:
                if backend not in value:
                    continue
                backend_index = value.index(backend)
                if any(
                    supported in value and value.index(supported) > backend_index
                    for supported in ("numpy", "sparse")
                ):
                    warnings.warn(
                        _deprecated_backend_message(backend),
                        DeprecationWarning,
                        stacklevel=2,
                    )
        setattr(self, pat, value)

    def reset_option(
            self,
            pat: str | None = None,
            *,
            option: str | None = None
    ) -> None:
        """
        Restores the default value for the specified option. Restores default values for
        all options if pat is None.

        .. deprecated:: 0.9.3
            The ``option`` parameter is deprecated; use ``pat`` instead.

        Parameters
        ----------
        pat: str | None
            The option you wish to reset the value for.
        option: str | None
            The option you wish to reset the value for.

        Returns
        -------
        None

        """
        pat = _resolve_pat(pat=pat, option=option, required=False)
        if pat is not None:
            self._validate_option(pat)
            setattr(self, pat, copy.deepcopy(self._defaults[pat]))
        else:
            self.__init__()

    def _validate_option(self, pat: str) -> None:
        """
        Check whether string assigned to option is one of the configurable options in the Options class.

        Parameters
        ----------
        pat: str
            The option you want to check.

        Returns
        -------
        None

        """
        if pat not in self._defaults:
            raise ValueError(f"Invalid option(s): {pat}. Must be one of {list(self._defaults)}.")

    def describe_option(self, pat: str = "", _print_desc: bool=True) -> None | str:
        """
        Print the description for one or more options.

        Call with no arguments to get a listing for all options.

        Parameters
        ----------
        pat: str, default ""
            The name of the option(s) you want described. Supplying an empty string will describe all options.
            For multiple options, separate them with a pipe, |.
        _print_desc: bool, default True
            If True (default) the description(s) will be printed to stdout.
            Otherwise, the description(s) will be returned as a string.

        Returns
        -------
        The description for the specified option(s) if _print_desc=False, otherwise, `None`.

        Examples
        --------

        Describe information on a single option by passing the option name to `pat`.

        .. testsetup::

            import chainladder as cl

        .. testcode::

            cl.options.describe_option("AUTO_SPARSE")

        .. testoutput::

            AUTO_SPARSE : bool
                Controls whether chainladder automatically converts a triangle's backing array to a sparse representation
                when it would be memory-efficient to do so.
                [default: True] [currently: True]

        You can use a regexp to look up information on multiple options.

        .. testcode::

            cl.options.describe_option("AUTO_SPARSE|ARRAY_BACKEND")

        .. testoutput::

            ARRAY_BACKEND : str
                The default array backend for chainladder.
                [default: numpy] [currently: numpy]
            AUTO_SPARSE : bool
                Controls whether chainladder automatically converts a triangle's backing array to a sparse representation
                when it would be memory-efficient to do so.
                [default: True] [currently: True]

        Setting `_print_desc=False` will return a string

        .. testcode::

            res = cl.options.describe_option("AUTO_SPARSE", _print_desc=False)
            print(res)

        .. testoutput::

            AUTO_SPARSE : bool
                Controls whether chainladder automatically converts a triangle's backing array to a sparse representation
                when it would be memory-efficient to do so.
                [default: True] [currently: True]
        """
        # Match option names against pat as a regex. Empty pattern matches all.
        try:
            keys = [key for key in self._defaults if re.search(pat, key)]
        except re.error:
            raise ValueError(f"'{pat}' is not a valid regular expression.")

        if pat and not keys:
            raise ValueError(f"No option matching '{pat}'. Must be one of {list(self._defaults)}.")

        # Extract class docstring and clean up indentation.
        doc: str = inspect.cleandoc(self.__class__.__doc__)

        # Holds the output.
        lines: list[str] = []
        for key in keys:
            # Find a match for the specified option in the docstring.
            match: Match[str] | None = re.search(
                # Look for pattern matching structure of an attribute. e.g., the attribute name, followed by
                # the type name, then the attribute description indented on the next line. Search will be
                # split up into groups, specified by parentheses ().
                pattern=rf"^{re.escape(key)}:\s*(\S+)\n((?:[ \t]+.+\n?)+)",
                string=doc,
                flags=re.MULTILINE  # Needed to specify '^' as starting line anchor for each line.
            )

            # If there's a match, extract the attribute type and description.
            if match:
                type_hint: str = match.group(1)  # Type annotation captured by (\S+)
                description: str = inspect.cleandoc(match.group(2))  # Description block captured by ((?:[ \t]+.+\n?)+).
            else:
                type_hint: str = ""
                description: str = "No description available."

            # Indent the description relative to the attribute name.
            indented: str = "\n    ".join(description.splitlines())
            # Extract the default option values.
            default: str | bool | list = self._defaults[key]
            # Extract the current option values.
            current: str | bool | list = getattr(self, key)
            # Write the option followed by a type hint.
            header: str = f"{key} : {type_hint}" if type_hint else key
            # Indent the description relative to the header.
            lines.append(f"{header}\n    {indented}\n    [default: {default}] [currently: {current}]")

        output: str = "\n".join(lines)
        # Print output by default, otherwise return the string.
        if _print_desc:
            print(output)
            return None
        return output

options = Options()


from chainladder.utils import (  # noqa (API import)
    WeightedRegression,
    parallelogram_olf,
    read_csv,
    read_pickle,
    read_json,
    concat,
    load_sample,
    list_samples,
    minimum,
    maximum,
    PatsyFormula,
    model_diagnostics,
    cp,
    sp,
    dp,
)
from chainladder.core import (  # noqa (API import)
    Triangle,
    DevelopmentCorrelation,
    ValuationCorrelation,
)
from chainladder.development import (  # noqa (API import)
    DevelopmentBase,
    Development,
    MunichAdjustment,
    IncrementalAdditive,
    DevelopmentConstant,
    ClarkLDF,
    CaseOutstanding,
    DevelopmentML,
    TweedieGLM,
    BarnettZehnwirth,
)
from chainladder.adjustments import (  # noqa (API import)
    BootstrapODPSample,
    BerquistSherman,
    ParallelogramOLF,
    Trend,
    TrendConstant,
)
from chainladder.tails import (  # noqa (API import)
    TailBase,
    TailConstant,
    TailCurve,
    TailBondy,
    TailClark,
)
from chainladder.methods import (  # noqa (API import)
    MethodBase,
    Chainladder,
    MackChainladder,
    Benktander,
    BornhuetterFerguson,
    CapeCod,
    ExpectedLoss,
)
from chainladder.workflow import (  # noqa (API import)
    GridSearch,
    Pipeline,
    VotingChainladder,
)

__version__ = version("chainladder")
