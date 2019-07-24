import pandas  as pd
import numpy as np
import copy


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
        old_k_by_new_k = np.zeros((v1_len, v2_len), dtype='int8')
        for num, item in enumerate(groups):
            old_k_by_new_k[:, num][item] = 1
        old_k_by_new_k = np.swapaxes(old_k_by_new_k, 0, 1)
        for i in range(3):
            old_k_by_new_k = old_k_by_new_k[..., np.newaxis]
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


class TrianglePandas:
    def to_frame(self, *args, **kwargs):
        """ Converts a triangle to a pandas.DataFrame.  Requires an individual
        index and column selection to appropriately grab the 2D DataFrame.

        Returns
        -------
            pandas.DataFrame representation of the Triangle.
        """
        axes = [num for num, item in enumerate(self.shape) if item > 1]
        if self.shape[:2] == (1, 1):
            return self._repr_format()
        elif len(axes) == 2 or len(axes) == 1:
            tri = np.squeeze(self.values)
            axes_lookup = {0: self.kdims, 1: self.vdims,
                           2: self.origin, 3: self.ddims}
            if len(axes) == 2:
                return pd.DataFrame(tri, index=axes_lookup[axes[0]],
                                    columns=axes_lookup[axes[1]]).fillna(0)
            if len(axes) == 1:
                return pd.Series(tri, index=axes_lookup[axes[0]]).fillna(0)
        else:
            raise ValueError('len(index) and len(columns) must be 1.')

    def plot(self, *args, **kwargs):
        """ Passthrough of pandas functionality """
        return self.to_frame().plot(*args, **kwargs)

    @property
    def T(self):
        return self.to_frame().T

    def quantile(self, q, *args, **kwargs):
        if self.shape[:2] == (1, 1):
            return self.to_frame().quantile(q, *args, **kwargs)
        return TriangleGroupBy(self, by=-1).quantile(q, axis=1)

    def groupby(self, by, *args, **kwargs):
        """ Group Triangle by index values.  If the triangle is convertable to a
        DataFrame, then it defaults to pandas groupby functionality.

        Parameters
        ----------
        by: str or list
            The index to group by

        Returns
        -------
            GroupBy object (pandas or Triangle)
        """
        if self.shape[:2] == (1, 1):
            return self.to_frame().groupby(*args, **kwargs)
        return TriangleGroupBy(self, by)

    def append(self, other, index):
        """ Append rows of other to the end of caller, returning a new object.

        Parameters
        ----------
        other : Triangle
            The data to append.
        index:
            The index label(s) to assign the appended data.

        Returns
        -------
            New Triangle with appended data.
        """
        return_obj = copy.deepcopy(self)
        x = pd.DataFrame(list(return_obj.kdims), columns=return_obj.key_labels)
        new_idx = pd.DataFrame([index], columns=return_obj.key_labels)
        x = x.append(new_idx, sort=True)
        x.set_index(return_obj.key_labels, inplace=True)
        return_obj.values = np.append(return_obj.values, other.values, axis=0)
        return_obj.kdims = np.array(x.index.unique())
        return return_obj

    def rename(self, axis, value):
        """ Alter axes labels.

        Parameters
        ----------
            axis: str or int
                A value of 0 <= axis <= 4 corresponding to axes 'index',
                'columns', 'origin', 'development' respectively.  Both the
                int and str representation can be used.
            value: list or str
                List of new labels to be assigned to the axis. List must be of
                same length of the specified axis.

        Returns
        -------
            Triangle with relabeled axis.
        """
        value = [value] if type(value) is str else value
        if axis == 'index' or axis == 0:
            self.index = value
        if axis == 'columns' or axis == 1:
            self.columns = value
        if axis == 'origin' or axis == 2:
            self.origin = value
        if axis == 'development' or axis == 3:
            self.development = value
        return self


def add_triangle_agg_func(cls, k, v):
    ''' Aggregate Overrides in Triangle '''
    def agg_func(self, *args, **kwargs):
        if self.shape[:2] == (1, 1):
            return getattr(pd.DataFrame, k)(self.to_frame(), *args, **kwargs)
        else:
            return getattr(TriangleGroupBy(self, by=-1), k)(axis=1)
    set_method(cls, agg_func, k)


def add_groupby_agg_func(cls, k, v):
    ''' Aggregate Overrides in GroupBy '''
    def agg_func(self, axis=1, *args, **kwargs):
        obj = copy.deepcopy(self.obj)
        x = np.repeat(self.obj.values[np.newaxis],
                      self.old_k_by_new_k.shape[0], 0) * \
            self.old_k_by_new_k
        ignore_vector = np.sum(np.isnan(x), axis=1, keepdims=True) == \
            x.shape[1]
        x = np.where(ignore_vector, 0, x)
        obj.values = \
            getattr(np, v)(x, axis=1, *args, **kwargs)
        obj.values[obj.values == 0] = np.nan
        return obj
    set_method(cls, agg_func, k)


def add_df_passthru(cls, k):
    '''Pass Through of pandas functionality '''
    def df_passthru(self, *args, **kwargs):
        return getattr(pd.DataFrame, k)(self.to_frame(), *args, **kwargs)
    set_method(cls, df_passthru, k)


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

for item in df_passthru:
    add_df_passthru(TrianglePandas, item)

for k, v in agg_funcs.items():
    add_triangle_agg_func(TrianglePandas, k, v)
    add_groupby_agg_func(TriangleGroupBy, k, v)
