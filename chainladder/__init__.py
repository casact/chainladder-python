from sklearn.utils import deprecated

class Options:
    ARRAY_BACKEND = "numpy"
    AUTO_SPARSE = True
    ARRAY_PRIORITY = ["dask", "sparse", "cupy", "numpy"]
    ULT_VAL = "2261-12-31 23:59:59.999999999"

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
        self.set_option('ULT_VAL', "2261-12-31 23:59:59.999999999")

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
