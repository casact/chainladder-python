"""
The datasets module has only one function.  It could probably be refactored
into the Triangle class, but I like the notation of `chainladder.datasets`.

"""
from pandas import read_pickle, DataFrame
import os
import matplotlib.pyplot as plt
import seaborn as sns
from chainladder.Triangle import Triangle
from chainladder.Chainladder import Chainladder
from chainladder.MackChainladder import MackChainladder
from statsmodels.nonparametric.smoothers_lowess import lowess
from warnings import warn





def load_dataset(key):
    path = os.path.dirname(os.path.abspath(__file__))
    return read_pickle(os.path.join(path, 'data', key))

    
