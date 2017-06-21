"""
The datasets module has only one function.  It could probably be refactored
into the Triangle class, but I like the notation of `chainladder.datasets`.

"""
from pandas import read_pickle
import os

def load_dataset(key):
    path = os.path.dirname(os.path.abspath(__file__))
    return read_pickle(os.path.join(path,'data',key))

    