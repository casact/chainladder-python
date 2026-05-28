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
from importlib.metadata import version

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from re import Match


# Get the default datetime64 data type and precision, extracted from Pandas installation.
# Used for cross-version compatibility between Pandas 2 and Pandas 3.
__dt64_dtype__: str = pd.to_datetime(["2000-01-01"]).dtype.name
__dt64_unit__: str = np.datetime_data(__dt64_dtype__)[0]


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

    def get_option(self, pat: str) -> str | bool | list:
        """
        Get the option value for the specified option.

        Parameters
        ----------
        pat: str
            The option you wish to get the values for.

        Returns
        -------
        The option value.

        """
        self._validate_option(pat)
        return getattr(self, pat)

    def set_option(
            self,
            pat: str,
            value: str | bool | list
    ) -> None:
        """
        Set the option value for the specified option.

        Parameters
        ----------
        pat: str
            The option you wish to set the value for.
        value: str | bool | list
            The option value.

        Returns
        -------
        None

        """
        self._validate_option(pat)
        setattr(self, pat, value)

    def reset_option(self, pat: str | None = None) -> None:
        """
        Restores the default value for the specified option. Restores default values for
        all options if option is None.

        Returns
        -------
        None

        """

        if pat is not None:
            self._validate_option(pat)
            setattr(self, pat, copy.deepcopy(self._defaults[pat]))
        else:
            self.__init__()

    def _validate_option(self, pat: str) -> None:
        """
        Check whether string assigned to option is one of the configurable options in the Option class.

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

    def describe_option(self, pat: str = "", _print_desc=True) -> None | str:
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
        None

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

            "AUTO_SPARSE : bool\n    Controls whether chainladder automatically converts a triangle's backing array
            to a sparse representation\n    when it would be memory-efficient to do so.\n
            [default: True] [currently: True]"
        """
        # Match option names against pat as a regex. Empty pattern matches all.
        keys: list[str] = [key for key in self._defaults if re.search(pat, key)]

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
                pattern=rf"^{key}:\s*(\S+)\n((?:[ \t]+.+\n?)+)",
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


from chainladder.utils import *  # noqa (API Import)
from chainladder.core import *  # noqa (API Import)
from chainladder.development import *  # noqa (API Import)
from chainladder.adjustments import *  # noqa (API Import)
from chainladder.tails import *  # noqa (API Import)
from chainladder.methods import *  # noqa (API Import)
from chainladder.workflow import *  # noqa (API Import)

__version__ = version("chainladder")
