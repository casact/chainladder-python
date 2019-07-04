import pandas as pd
import numpy as np
import copy
from chainladder.core import set_method, agg_funcs

class TriangleGroupBy:
    def __init__(self, old_obj, by):
        obj = copy.deepcopy(old_obj)
        v1_len = len(obj.index.index)
        if by != -1:
            indices = obj.index.groupby(by).indices
            new_index = obj.index.groupby(by).count().index
        else:
            indices = {'All': np.arange(len(obj.index))}
            new_index = pd.Index(['All'], name='All')
        groups = [indices[item] for item in sorted(list(indices.keys()))]
        v2_len = len(groups)
        old_k_by_new_k = np.zeros((v1_len, v2_len))
        for num, item in enumerate(groups):
            old_k_by_new_k[:, num][item] = 1
        old_k_by_new_k = np.swapaxes(old_k_by_new_k, 0, 1)
        for i in range(3):
            old_k_by_new_k = old_k_by_new_k[..., np.newaxis]
        new_tri = obj.values
        new_tri = np.repeat(new_tri[np.newaxis], v2_len, 0)
        obj.values = new_tri
        obj.kdims = np.array(list(new_index))
        obj.key_labels = list(new_index.names)
        self.obj = obj
        self.old_k_by_new_k = old_k_by_new_k

    def quantile(self, q, axis=1, *args, **kwargs):
        """ Return values at the given quantile over requested axis.  If Triangle is
        convertible to DataFrame then pandas quantile functionality is used instead.

        Parameters
        ----------
        q: float or array-like, default 0.5 (50% quantile)
            Value between 0 <= q <= 1, the quantile(s) to compute.
        axis:  {0, 1, ‘index’, ‘columns’} (default 1)
            Equals 0 or ‘index’ for row-wise, 1 or ‘columns’ for column-wise.

        Returns
        -------
            Triangle

        """
        x = self.obj.values*self.old_k_by_new_k
        ignore_vector = np.sum(np.isnan(x), axis=1, keepdims=True) == \
            x.shape[1]
        x = np.where(ignore_vector, 0, x)
        self.obj.values = \
            getattr(np, 'nanpercentile')(x, q*100, axis=1, *args, **kwargs)
        self.obj.values[self.obj.values == 0] = np.nan
        return self.obj


def add_groupby_agg_func(cls, k, v):
    ''' Aggregate Overrides in GroupBy '''
    def agg_func(self, axis=1, *args, **kwargs):
        x = self.obj.values*self.old_k_by_new_k
        ignore_vector = np.sum(np.isnan(x), axis=1, keepdims=True) == \
            x.shape[1]
        x = np.where(ignore_vector, 0, x)
        self.obj.values = \
            getattr(np, v)(x, axis=1, *args, **kwargs)
        self.obj.values[self.obj.values == 0] = np.nan
        return self.obj
    set_method(cls, agg_func, k)

# Aggregate method overridden to the 4D Triangle Shape


for k, v in agg_funcs.items():
    add_groupby_agg_func(TriangleGroupBy, k, v)
