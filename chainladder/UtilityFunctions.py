"""
The datasets module has only one function.  It could probably be refactored
into the Triangle class, and may do so in the future.

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
    """ Function to load datasets included in the chainladder package.
   
    Arguments:
	key: str
	    The name of the dataset, e.g. RAA, ABC, UKMotor, GenIns, etc.
    
    Returns:
	pandas.DataFrame of the loaded dataset.
   """
    path = os.path.dirname(os.path.abspath(__file__))
    return read_pickle(os.path.join(path, 'data', key))

    
