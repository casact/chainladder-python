"""
The datasets module has only one function.  It could probably be refactored
into the Triangle class, and may do so in the future.

"""
from pandas import read_pickle
import os


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

    
