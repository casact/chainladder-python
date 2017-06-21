"""
The datasets module
"""
from pandas import read_pickle
import os

def load_dataset(key):
    path = os.path.dirname(os.path.abspath(__file__))
    return read_pickle(os.path.join(path,'data',key))

    