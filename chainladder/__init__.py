import numpy as np
import pandas as pd
from importlib.metadata import version
from sklearn.utils import deprecated


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
    DT64_DTYPE: str
        The default datetime64 data type, extracted from Pandas installation.
    DT64_UNIT: str
        The default datetime64 precision, extracted from Pandas installation.
    ULT_VAL: str
        The default ultimate valuation datetime, precision set to default of Pandas installation.

    """
    def __init__(self):
        self.ARRAY_BACKEND = "numpy"
        self.AUTO_SPARSE = True
        self.ARRAY_PRIORITY = ["dask", "sparse", "cupy", "numpy"]
        self.DT64_DTYPE: str = pd.to_datetime(["2000-01-01"]).dtype.name
        self.DT64_UNIT: str = np.datetime_data(self.DT64_DTYPE)[0]
        self.ULT_VAL = str(
            pd.Timestamp("2262-01-01") - \
            pd.Timedelta(1, unit=self.DT64_UNIT)
        )

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

        setattr(self, option, value)

    def reset_option(self) -> None:
        """
        Restores the default options.

        Returns
        -------
        None

        """
        self.__init__()

    def describe_option(self):
        pass

options = Options()

@deprecated("In an upcoming version of the package, this function will be deprecated. Use `chainladder.options.set_option('ARRAY_BACKEND', value)` to avoid breakage.")
def array_backend(array_backend="numpy"):
    options.set_option('ARRAY_BACKEND', array_backend)

@deprecated("In an upcoming version of the package, this function will be deprecated. Use `chainladder.options.set_option('AUTO_SPARSE', value)` to avoid breakage.")
def auto_sparse(auto_sparse=True):
    options.set_option('AUTO_SPARSE', auto_sparse)


from chainladder.utils import *  # noqa (API Import)
from chainladder.core import *  # noqa (API Import)
from chainladder.development import *  # noqa (API Import)
from chainladder.adjustments import *  # noqa (API Import)
from chainladder.tails import *  # noqa (API Import)
from chainladder.methods import *  # noqa (API Import)
from chainladder.workflow import *  # noqa (API Import)

__version__ = version("chainladder")
