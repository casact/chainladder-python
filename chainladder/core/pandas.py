# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pandas as pd
import numpy as np
from chainladder.utils.cupy import cp
import copy


class TriangleGroupBy:
    def __init__(self, old_obj, by):
        obj = copy.deepcopy(old_obj)
        if by != -1:
            indices = obj.index.groupby(by).indices
            new_index = obj.index.groupby(by).count().index
        else:
            indices = {'All': np.arange(len(obj.index))}
            new_index = pd.Index(['All'], name='All')
        groups = [indices[item] for item in sorted(list(indices.keys()))]
        xp = cp.get_array_module(obj.values)
        old_k_by_new_k = xp.zeros(
            (len(obj.index.index), len(groups)), dtype='bool')
        for num, item in enumerate(groups):
            old_k_by_new_k[:, num][item] = True
        old_k_by_new_k = xp.swapaxes(old_k_by_new_k, 0, 1)
        for i in range(3):
            old_k_by_new_k = old_k_by_new_k[..., np.newaxis]
        self.old_k_by_new_k = old_k_by_new_k
        obj.kdims = np.array(list(new_index))
        obj.key_labels = list(new_index.names)
        self.obj = obj

    def quantile(self, q, axis=1, *args, **kwargs):
        """ Return values at the given quantile over requested axis.  If
            Triangle is convertible to DataFrame then pandas quantile
            functionality is used instead.

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
        xp = cp.get_array_module(self.obj.values)
        x = self.obj.values*self.old_k_by_new_k
        ignore_vector = xp.sum(xp.isnan(x), axis=1, keepdims=True) == \
            x.shape[1]
        x = xp.where(ignore_vector, 0, x)
        self.obj.values = \
            getattr(xp, 'nanpercentile')(x, q*100, axis=1, *args, **kwargs)
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
        xp = cp.get_array_module(self.values)
        axes = [num for num, item in enumerate(self.shape) if item > 1]
        if self.shape[:2] == (1, 1):
            return self._repr_format()
        elif len(axes) in [1, 2]:
            odims, ddims = self._repr_date_axes()
            tri = xp.squeeze(self.values)
            axes_lookup = {0: self.kdims, 1: self.vdims,
                           2: odims, 3: ddims}
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

    def hvplot(self, *args, **kwargs):
        """ Passthrough of pandas functionality """
        df = self.to_frame()
        if type(df.index) == pd.PeriodIndex and len(df.columns)>1:
            df.index = df.index.to_timestamp(how='s')
        return df.hvplot(*args, **kwargs)

    def _get_axis(self, axis):
        ax = {0: 0, 1: 1,2: 2, 3: 3, -1: 3, -2: 2, -3: 1, -4: 0,
             'index': 0,'columns': 1, 'origin': 2, 'development':3}
        return ax.get(axis, 0)

    def dropna(self):
        """  Method that removes orgin/development vectors from edge of a
        triangle that are all missing values. This may come in handy for a
        new line of business that doesn't have origins/developments of an
        existing line in the same triangle.
        """
        xp = cp.get_array_module(self.values)
        obj = self.sum(axis=0).sum(axis=1)
        odim = list(obj.sum(axis=-1).values[0, 0, :, 0]*0+1)
        min_odim = obj.origin[odim.index(1)]
        max_odim = obj.origin[::-1][odim[::-1].index(1)]
        if obj.shape[-1] != 1:
            if xp.__name__ == 'cupy':
                ddim = cp.asnumpy(xp.nan_to_num((obj.sum(axis=-2).values*0+1)[0, 0, 0]))
            else:
                ddim = np.nan_to_num((obj.sum(axis=-2).values*0+1)[0, 0, 0])
            ddim = obj.development.iloc[:, 0][pd.Series(ddim).astype(bool)]
            obj = self[(self.development >= ddim.min()) &
                  (self.development <= ddim.max())]
            return obj[(self.origin >= min_odim) & (self.origin <= max_odim)]
        obj = self[(self.origin >= min_odim) & (self.origin <= max_odim)]
        return obj

    def drop(self, labels=None, axis=1):
        """ Drop specified labels from rows or columns.

        Remove rows or columns by specifying label names and corresponding axis,
        or by specifying directly index or column names.

        Parameters
        -----------

        label: single label or list-like
            Index or column labels to drop.

        axis: {0 or ‘index’, 1 or ‘columns’}, default 1
            Whether to drop labels from the index (0 or ‘index’)
            or columns (1 or ‘columns’).

        Returns
        -------
        Triangle

        """
        if axis==1:
            return self[list(self._idx_table().drop(labels, axis=axis).columns)]
        else:
            raise NotImplementedError('drop only inpemented for column axis')


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
        try:
            return self.to_frame().groupby(*args, **kwargs)
        except:
            return TriangleGroupBy(self, by)

    def append(self, other):
        """ Append rows of other to the end of caller, returning a new object.

        Parameters
        ----------
        other : Triangle
            The data to append.

        Returns
        -------
            New Triangle with appended data.
        """
        xp = cp.get_array_module(self.values)
        return_obj = copy.deepcopy(self)
        return_obj.kdims = (return_obj.index.append(other.index)).values
        try:
            return_obj.values = xp.concatenate((return_obj.values, other.values), axis=0)
        except:
            # For misaligned triangle support
            self.values = xp.concatenate(
                (return_obj.values,
                (return_obj.iloc[:, 0]*0+other.values).values), axis=1)

        return_obj._set_slicers()
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

    def astype(self, dtype, inplace=True):
        '''
        Copy of the array, cast to a specified type.

        Parameters
        ----------
            dtype : str or dtype
                Typecode or data-type to which the array is cast.
            copy : bool, optional
                By default, astype always returns a newly allocated array.
        Returns
        -------
            Triangle as new datatype.
        '''
        obj = copy.deepcopy(self) if inplace is True else self
        obj.values = obj.values.astype(dtype)
        return obj


def add_triangle_agg_func(cls, k, v):
    ''' Aggregate Overrides in Triangle '''
    def agg_func(self, axis=None, *args, **kwargs):
            obj = copy.deepcopy(self)
            if axis is None:
                axis = min([num for num, _ in enumerate(obj.shape) if _ != 1])
            else:
                axis = self._get_axis(axis)
            xp = cp.get_array_module(obj.values)
            func = getattr(xp, v)
            kwargs.update({'keepdims': True})
            obj.values = func(obj.values, axis=axis, *args, **kwargs)
            if axis == 0 and obj.values.shape[axis] == 1:
                obj.kdims = np.array([None])
                obj.key_labels = [None]
            if axis == 1 and obj.values.shape[axis] == 1:
                obj.vdims = np.array([None])
            if axis == 2 and obj.values.shape[axis] == 1:
                obj.odims = np.array([None])
            if axis == 3 and obj.values.shape[axis] == 1:
                obj.ddims = np.array([None])
            obj._set_slicers()
            obj.values = obj.values * obj._expand_dims(obj._nan_triangle())
            obj.values[obj.values == 0] = np.nan
            if obj.shape == (1, 1, 1, 1):
                return obj.values[0, 0, 0, 0]
            else:
                return obj
    set_method(cls, agg_func, k)


def add_groupby_agg_func(cls, k, v):
    ''' Aggregate Overrides in GroupBy '''
    def agg_func(self, axis=1, *args, **kwargs):
        obj = copy.deepcopy(self.obj)
        xp = cp.get_array_module(obj.values)
        x = xp.broadcast_to(
            self.obj.values,
            (self.old_k_by_new_k.shape[0], *self.obj.values.shape)) * \
            self.old_k_by_new_k
        ignore_vector = xp.sum(np.isnan(x), axis=1, keepdims=True) == \
            x.shape[1]
        x = xp.where(ignore_vector, 0, x)
        x[~xp.isfinite(x)] = np.nan
        obj.values = \
            getattr(xp, v)(x, axis=1, *args, **kwargs)
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
               'describe', 'melt', 'pct_chg', 'round']
agg_funcs = ['sum', 'mean', 'median', 'max', 'min', 'prod',
             'var', 'std', 'cumsum']
agg_funcs = {item: 'nan'+item for item in agg_funcs}
more_aggs= ['diff']
agg_funcs = {**agg_funcs, **{item: item for item in more_aggs}}

for item in df_passthru:
    add_df_passthru(TrianglePandas, item)

for k, v in agg_funcs.items():
    add_triangle_agg_func(TrianglePandas, k, v)
    add_groupby_agg_func(TriangleGroupBy, k, v)
