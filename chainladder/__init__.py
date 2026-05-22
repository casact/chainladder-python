import numpy as np
import pandas as pd
from sklearn.utils import deprecated


_DT64_DTYPE = pd.to_datetime(["2000-01-01"]).dtype
_ULT_VAL: str = str(
    pd.Timestamp("2262-01-01") - \
    pd.Timedelta(1, unit=np.datetime_data(_DT64_DTYPE)[0])
)

class Options:
    ARRAY_BACKEND = "numpy"
    AUTO_SPARSE = True
    ARRAY_PRIORITY = ["dask", "sparse", "cupy", "numpy"]
    ULT_VAL: str = _ULT_VAL
    DT64_UNIT: str = np.datetime_data(_DT64_DTYPE)[0]
    DT64_DTYPE: str = str(_DT64_DTYPE)

    @classmethod
    def get_option(cls, option=None):
        return getattr(cls, option)

    @classmethod
    def set_option(cls, option, value):
        setattr(cls, option, value)

    def reset_option(self):
        self.set_option('ARRAY_BACKEND', "numpy")
        self.set_option('AUTO_SPARSE', True)
        self.set_option('ARRAY_PRIORITY', ["dask", "sparse", "cupy", "numpy"])
        self.set_option('ULT_VAL', _ULT_VAL)
        self.set_option('DT64_UNIT', np.datetime_data(_DT64_DTYPE)[0])
        self.set_option('DT64_DTYPE', str(_DT64_DTYPE))

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

try:
    from importlib.metadata import version
    __version__ = version("chainladder")
except ImportError:
    # Fallback for Python < 3.8
    from importlib_metadata import version
    __version__ = version("chainladder")
