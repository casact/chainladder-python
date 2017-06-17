import pandas as pd
import numpy as np
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, r
pandas2ri.activate()
import os.path

path = os.path.abspath('')[:-6] + 'data\\'
CL =importr('ChainLadder')
d = r('data(package=\"ChainLadder\")')

datasets = [item for item in d[2].rx(True,3)]
data_descr = [item for item in d[2].rx(True,4)]

def create_python_df(name):
    df = pd.DataFrame()
    if r('class('+ name + ')')[0] == 'data.frame':
        df = pd.DataFrame(r(name))
    elif r('class('+ name + ')')[0] == 'triangle':
        df = pd.DataFrame(r(name), index = r('rownames(' + name + ')').astype(float), columns=r('colnames(' + name + ')'))
        df = df.rename_axis("dev", axis="columns")
        df.index.name = 'origin'
    elif r('class('+ name + ')')[0] == 'list':
        df_dict = {}
        for i in range(r('length('+ name + ')')[0]):
            subname = r('names('+ name + '[' + str(i + 1) + '])')[0] # Name of subtriangle
            df = pd.DataFrame(r(name + '$' + subname))
            df.columns = df.columns.values.astype(str)
            df = df.rename_axis("dev", axis="columns")
            df = df.rename_axis("origin", axis="rows")
            df = df.replace(-2147483648, np.nan)
            df_dict[name + '$' + subname] = df
        return df_dict
    # Integer type NAs return as -2147483648 - No clue why, but these are NA is the R environment
    df = df.replace(-2147483648, np.nan)
    return df

for item in datasets:
    return_val = create_python_df(item)
    if type(return_val) == type(dict()):
        for k, v in return_val.items():
            v.to_pickle(path + k) 
    if type(return_val) == type(pd.DataFrame()):
        return_val.to_pickle(path + item)    
