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

import copy
import numpy as np
import pandas as pd
from importlib.metadata import version


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

    def get_option(self, option: str) -> str | bool | list:
        """
        Get the option value for the specified option.

        Parameters
        ----------
        option: str
            The option you wish to get the values for.

        Returns
        -------
        The option value.

        """
        self._validate_option(option)
        return getattr(self, option)

    def set_option(
            self,
            option: str,
            value: str | bool | list
    ) -> None:
        """
        Set the option value for the specified option.

        Parameters
        ----------
        option: str
            The option you wish to set the value for.
        value: str | bool | list
            The option value.

        Returns
        -------
        None

        """
        self._validate_option(option)
        setattr(self, option, value)

    def reset_option(self, option: str | None = None) -> None:
        """
        Restores the default value for the specified option. Restores default values for
        all options if option is None.

        Returns
        -------
        None

        """

        if option is not None:
            self._validate_option(option)
            setattr(self, option, copy.deepcopy(self._defaults[option]))
        else:
            self.__init__()

    def _validate_option(self, option: str) -> None:

        if option not in self._defaults:
            raise ValueError(f"Invalid option(s): {option}. Must be one of {list(self._defaults)}.")

    def describe_option(self, option: str) -> str:
        pass

options = Options()


from chainladder.utils import *  # noqa (API Import)
from chainladder.core import *  # noqa (API Import)
from chainladder.development import *  # noqa (API Import)
from chainladder.adjustments import *  # noqa (API Import)
from chainladder.tails import *  # noqa (API Import)
from chainladder.methods import *  # noqa (API Import)
from chainladder.workflow import *  # noqa (API Import)

__version__ = version("chainladder")
