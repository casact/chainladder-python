ARRAY_BACKEND = 'numpy'

def array_backend(array_backend='numpy'):
    global ARRAY_BACKEND
    ARRAY_BACKEND = array_backend

from chainladder.utils import * # noqa (API Import)
from chainladder.core import * # noqa (API Import)
from chainladder.development import * # noqa (API Import)
from chainladder.tails import * # noqa (API Import)
from chainladder.methods import * # noqa (API Import)
from chainladder.workflow import * # noqa (API Import)

__version__ = '0.7.0'
