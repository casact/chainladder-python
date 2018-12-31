"""
This module contains various utilities shared across most of the other
*chainladder* modules.

"""
import pandas as pd
import numpy as np
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
    values = ['values']
    keys = None
    if key.lower() in ['mcl', 'usaa', 'quarterly', 'auto', 'usauto']:
        values = ['incurred', 'paid']
    if key.lower() == 'clrd':
        origin = 'AccidentYear'
        development = 'DevelopmentYear'
        keys = ['GRNAME', 'LOB']
        values = ['IncurLoss', 'CumPaidLoss', 'BulkLoss', 'EarnedPremDIR',
                  'EarnedPremCeded', 'EarnedPremNet']
    if key.lower() in ['liab', 'auto']:
        keys = ['lob']
    df = pd.read_pickle(os.path.join(path, 'data', key.lower() + '.pkl'))
    return Triangle(df, origin=origin, development=development,
                    values=values, keys=keys)


def parallelogram_olf(values, date, start_date=None, end_date=None,
                      grain='M', vertical_line=False):
    """ Parallelogram approach to on-leveling.  Need to fix return grain
    """
    date = pd.to_datetime(date)
    if not start_date:
        start_date = f'{date.min().year-1}-01-01'
    if not end_date:
        end_date = f'{date.max().year+1}-12-31'
    date_idx = pd.date_range(start_date, end_date)
    y = pd.Series(np.array(values), np.array(date))
    y = y.reindex(date_idx, fill_value=0)
    idx = np.cumprod(y.values+1)
    idx = idx[-1]/idx
    y = pd.Series(idx, y.index)
    if not vertical_line:
        y = y.to_frame().rolling(365).mean()
    y = y.groupby(y.index.to_period(grain)).mean().reset_index()
    y.columns = ['origin', 'olf']
    y['origin'] = y['origin'].astype(str)
    return y
