"""
This module contains various utilities shared across most of the other
*chainladder* modules.

"""
import pandas as pd
import os
from chainladder.core.triangle import Triangle


def load_dataset(key):
    """ Function to load datasets included in the chainladder package.

        Arguments:
        key: str
        The name of the dataset, e.g. RAA, ABC, UKMotor, GenIns, etc.

        Returns:
    	pandas.DataFrame of the loaded dataset.
   """
    path = os.path.dirname(os.path.abspath(__file__))
    origin = 'origin'
    development = 'development'
    values = 'values'
    keys = None
    if key.lower() in ['mcl', 'usaa', 'quarterly', 'auto']:
        values = ['incurred', 'paid']
    if key.lower() == 'casresearch':
        origin = 'AccidentYear'
        development = 'DevelopmentYear'
        keys = ['GRNAME', 'LOB']
        values = ['IncurLoss', 'CumPaidLoss', 'BulkLoss', 'EarnedPremDIR',
                  'EarnedPremCeded', 'EarnedPremNet']
    if key.lower() in ['liab', 'auto']:
        keys = ['lob']
    df = pd.read_pickle(os.path.join(path, 'data', key + '.pkl'))
    return Triangle(df, origin=origin, development=development,
                    values=values, keys=keys)
