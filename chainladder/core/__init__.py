""" core should store the core data structure functionality.
"""
def set_method(cls, func, k):
    ''' Assigns methods to a class '''
    func.__doc__ = 'Refer to pandas for ``{}`` functionality.'.format(k)
    func.__name__ = k
    setattr(cls, func.__name__, func)

df_passthru = ['to_clipboard', 'to_csv', 'to_excel', 'to_json',
               'to_html', 'to_dict', 'unstack', 'pivot', 'drop_duplicates',
               'describe', 'melt', 'pct_chg']
agg_funcs = ['sum', 'mean', 'median', 'max', 'min', 'prod', 'var', 'std']
agg_funcs = {item: 'nan'+item for item in agg_funcs}

from chainladder.core.triangle import Triangle
from chainladder.core.base import IO
