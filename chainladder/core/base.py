# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pandas as pd
import numpy as np
from chainladder.utils.cupy import cp
from chainladder.utils.sparse import sp
from chainladder.utils.dask import dp
import warnings

from chainladder.core.display import TriangleDisplay
from chainladder.core.dunders import TriangleDunders
from chainladder.core.pandas import TrianglePandas
from chainladder.core.slice import TriangleSlicer
from chainladder.core.io import TriangleIO
from chainladder.core.common import Common
from chainladder import options
from chainladder.utils.utility_functions import num_to_nan, concat


class TriangleBase(TriangleIO, TriangleDisplay, TriangleSlicer,
                   TriangleDunders, TrianglePandas, Common):
    """ This class handles the initialization of a triangle """

    @property
    def shape(self):
        return self.values.shape

    @staticmethod
    def _input_validation(data, index, columns, origin, development):
        """ Validate/sanitize inputs """
        def str_to_list(arg):
            if arg is None:
                return
            if type(arg) in [str, pd.Period]:
                return [arg]
            else:
                return list(arg)
        index = str_to_list(index)
        columns = str_to_list(columns)
        origin = str_to_list(origin)
        development = str_to_list(development)
        if "object" in data[columns].dtypes:
            raise TypeError("column attribute must be numeric.")
        if data[columns].shape[1] != len(columns):
            raise AttributeError("Columns are required to have unique names")
        return index, columns, origin, development

    @staticmethod
    def _set_development(data, development, development_format, origin_date):
        """ Initialize development and its grain """
        if development:
            development_date = TriangleBase._to_datetime(
                data, development, period_end=True, format=development_format)
        else:
            o_max = pd.Period(
                origin_date.max(),
                freq=TriangleBase._get_grain(origin_date)).to_timestamp(how='e')
            development_date = pd.Series([o_max] * len(origin_date))
        development_date.name = "__development__"
        if pd.Series(development_date).dt.year.min() == \
           pd.Series(development_date).dt.year.max() == 1970:
            raise ValueError(
                'Development lags could not be determined. This may be because development'
                'is expressed as an age where a date-like vector is required')
        return development_date

    @staticmethod
    def _set_index(col, unique):
        return col.map(dict(zip(unique, range(len(unique))))).values[None].T

    @staticmethod
    def _aggregate_data(data, origin_date, development_date, index, columns):
        """ Summarize dataframe to the level specified in axes """
        if type(data) != pd.DataFrame:
            # Dask dataframes are mutated
            data["__origin__"] = origin_date
            data["__development__"] = development_date
            key_gr = ["__origin__", "__development__"] + [
                data[item] for item in ([] if not index else index)]
            data_agg = data.groupby(key_gr)[columns].sum().reset_index().fillna(0)
            data = data.drop(['__origin__', '__development__'], axis=1)
        else:
            # Summarize dataframe to the level specified in axes
            key_gr = [origin_date, development_date] + [
                data[item] for item in ([] if not index else index)]
            data_agg = data[columns].groupby(key_gr).sum().reset_index().fillna(0)
            data_agg["__origin__"] = data_agg[origin_date.name]
            data_agg["__development__"] = data_agg[development_date.name]
        # origin <= development is required - truncate bad records if not true
        valid = data_agg["__origin__"] <= data_agg["__development__"]
        if sum(~valid) > 0:
            warnings.warn("""
                Observations with development before
                origin start have been removed.""")
            valid = valid.compute() if hasattr(valid, 'compute') else valid
            data_agg = data_agg[valid]
        return data_agg

    @staticmethod
    def _set_kdims(data_agg, index):
        kdims = data_agg[index].drop_duplicates().reset_index(drop=True).reset_index()
        key_idx = (data_agg[index].merge(
            kdims, how="left", on=index)["index"].values[None].T)
        return kdims.drop("index", axis=1).values, key_idx

    @staticmethod
    def _set_odims(data_agg, date_axes):
        odims = np.sort(date_axes["__origin__"].unique())
        orig_idx = TriangleBase._set_index(data_agg["__origin__"], odims)
        return odims, orig_idx

    @staticmethod
    def _set_ddims(data_agg, date_axes):
        if date_axes["__development__"].nunique() > 1:
            dev_lag = TriangleBase._development_lag(
                data_agg["__origin__"], data_agg["__development__"])
            ddims = np.sort(
                TriangleBase._development_lag(
                    date_axes["__origin__"], date_axes["__development__"]
                ).unique())
            dev_idx = TriangleBase._set_index(dev_lag, ddims)
        else:
            ddims = pd.DatetimeIndex(
                [data_agg['__development__'].max()], name='valuation')
            dev_idx = np.zeros((len(data_agg), 1))
        return ddims, dev_idx

    @staticmethod
    def _set_values(data_agg, key_idx, columns, orig_idx, dev_idx):
        val_idx = (
            ((np.ones(len(data_agg))[None].T) * range(len(columns)))
            .reshape((1, -1), order="F").T)
        coords = np.concatenate(
            tuple([np.concatenate((orig_idx, dev_idx), 1)] * len(columns)), 0)
        coords = np.concatenate(
            (np.concatenate(tuple([key_idx] * len(columns)), 0), val_idx, coords), 1)
        amts = np.concatenate(
            [data_agg[col].fillna(0).values for col in data_agg[columns]]).astype("float64")
        return coords.T.astype('int64'), amts

    def _len_check(self, x, y):
        if len(x) != len(y):
            raise ValueError(
                "Length mismatch: Expected axis has",
                "{} elements, new values have".format(len(x)),
                "{} elements".format(len(y)),
            )

    def _get_date_axes(self, origin_date, development_date, origin_grain, development_grain):
        """ Function to find any missing origin dates or development dates that
            would otherwise mess up the origin/development dimensions.
        """
        o = pd.period_range(
            start=origin_date.min(), end=origin_date.max(),
            freq=origin_grain).to_timestamp(how='s')
        d = pd.period_range(
            start=development_date.min(), end=development_date.max(),
            freq=development_grain).to_timestamp(how='e')
        c = pd.DataFrame(TriangleBase._cartesian_product(o, d),
                         columns=['__origin__', '__development__'])
        return c[c['__development__']>c['__origin__']]


    @property
    def nan_triangle(self):
        """Given the current triangle shape and valuation, it determines the
           appropriate placement of NANs in the triangle for future valuations.
           This becomes useful when managing array arithmetic.
        """
        xp = self.get_array_module()
        if min(self.values.shape[2:]) == 1:
            return xp.ones(self.values.shape[2:], dtype="float16")
        val_array = np.array(self.valuation).reshape(self.shape[-2:], order="f")
        nan_triangle = np.array(pd.DataFrame(val_array) > self.valuation_date)
        nan_triangle = xp.array(np.where(nan_triangle, xp.nan, 1), dtype="float16")
        return nan_triangle

    @staticmethod
    def _to_datetime(data, fields, period_end=False, format=None):
        """For tabular form, this will take a set of data
        column(s) and return a single date array.  This function heavily
        relies on pandas, but does two additional things:
        1. It extends the automatic inference using date_inference_list
        2. it allows pd_to_datetime on a set of columns
        """
        # Concat everything into one field
        if len(fields) > 1:
            target_field = data[fields].astype(str).apply(lambda x: "-".join(x), axis=1)
        else:
            target_field = data[fields].iloc[:, 0]
        if hasattr(target_field, "dt"):
            target = target_field
            if type(target.iloc[0]) == pd.Period:
                return target.dt.to_timestamp(how={1: "e", 0: "s"}[period_end])
        else:
            datetime_arg = target_field.unique()
            format = [{"arg": datetime_arg, "format": format}] if format else []
            date_inference_list = format + [
                {"arg": datetime_arg, "format": "%Y%m"},
                {"arg": datetime_arg, "format": "%Y"},
                {"arg": datetime_arg, "infer_datetime_format": True},
            ]
            for item in date_inference_list:
                try:
                    arr = dict(zip(datetime_arg, pd.to_datetime(**item)))
                    break
                except:
                    pass
            target = target_field.map(arr)
        return target

    @staticmethod
    def _development_lag(origin, development):
        """ For tabular format, this will convert the origin/development
            difference to a development lag """
        return ((development - origin) /
                np.timedelta64(1, 'M')).round(0).astype(int)

    @staticmethod
    def _get_grain(dates, trailing=False, kind='origin'):
        """ Determines Grain of origin or valuation vector
        
        Parameters:
        
        dates: pd.Series[datetime64[ns]]
            A Datetime Series
        trailing:
            Set to False if you want to treat December as period end. Set
            to True if you want it inferred from the data.
        """
        months = dates.dt.month.unique()
        diffs = np.diff(np.sort(months))
        if len(dates.unique()) == 1:
            grain = 'M'
        elif len(months) == 1:
            grain = 'A'
        elif np.all(diffs == 6):
            grain = '2Q'
        elif np.all(diffs == 3):
            grain = 'Q'
        else:
            grain = 'M'
        if trailing and grain != 'M':
            if kind == 'origin':
                end = (dates.min() - pd.DateOffset(days=1)).strftime('%b').upper()
            else:
                # If inferred to beginning of calendar period, 1/1 from YYYY, 4/1 from YYYYQQ
                if dates.dt.strftime('%m%d').isin(['0101', '0401', '0701', '1001']).any():
                    end = (dates.min() - pd.DateOffset(days=1, years=-1)).strftime('%b').upper()
                else:
                    end = dates.max().strftime('%b').upper()
            grain = grain + '-' + end
        return grain

    @staticmethod
    def _cartesian_product(*arrays):
        """A fast implementation of cartesian product, used for filling in gaps
        in triangles (if any)"""
        arr = np.empty(
            [len(a) for a in arrays] + [len(arrays)],
            dtype=np.result_type(*arrays))
        for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a
        arr = arr.reshape(-1, len(arrays))
        return arr

    def get_array_module(self, arr=None):
        backend = (self.array_backend if arr is None
                   else arr.__class__.__module__.split(".")[0])
        modules = {"cupy": cp, "sparse": sp, "numpy": np, "dask": dp}
        if modules.get(backend, None):
            return modules.get(backend, None)
        else:
            raise ValueError("Array backend is invalid or not properly set.")

    def _auto_sparse(self):
        """ Auto sparsifies at 30Mb or more and 20% density or less """
        if not options.AUTO_SPARSE:
            return self
        n = np.prod(list(self.shape) + [8 / 1e6])
        if (self.array_backend == "numpy" and n > 30
            and 1 - np.isnan(self.values).sum() / n * (8 / 1e6) < 0.2):
            self.set_backend("sparse", inplace=True)
        if self.array_backend == "sparse" and not (
           self.values.density < 0.2 and n > 30):
            self.set_backend("numpy", inplace=True)
        return self


    @property
    def valuation(self):
        ddims = self.ddims
        if self.is_val_tri:
            out = pd.DataFrame(np.repeat(self.ddims.values[None],
                               len(self.odims), 0))
            return pd.DatetimeIndex(out.unstack().values)
        ddim_arr = ddims - ddims[0]
        origin = np.minimum(self.odims, np.datetime64(self.valuation_date))
        val_array = origin.astype("datetime64[M]") + np.timedelta64(ddims[0], "M")
        val_array = val_array.astype("datetime64[ns]") - np.timedelta64(1, "ns")
        val_array = val_array[:, None]
        s = slice(None, -1) if ddims[-1] == 9999 else slice(None, None)
        val_array = (
            val_array.astype("datetime64[M]") + ddim_arr[s][None, :] + 1
        ).astype("datetime64[ns]") - np.timedelta64(1, "ns")
        if ddims[-1] == 9999:
            ult = np.repeat(np.datetime64(options.ULT_VAL), val_array.shape[0])[:, None]
            val_array = np.concatenate((val_array, ult,), axis=1,)
        return pd.DatetimeIndex(val_array.reshape(1, -1, order="F")[0])

    def _drop_subtriangles(self):
        """ Removes subtriangles from a Triangle instance """
        sub_tris = [k for k, v in vars(self).items()
                    if isinstance(v, TriangleBase)]
        if 'ldf_' in sub_tris:
            del self.ldf_
        if 'sigma_' in sub_tris:
            del self.sigma_
        if 'std_err_' in sub_tris:
            del self.std_err_

    @property
    def subtriangles(self):
        """ Lists subtriangles from a Triangle instance """
        return  [k for k, v in vars(self).items() if isinstance(v, TriangleBase)]

    def __array__(self):
        return self.values

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        obj = self.copy()
        if method == '__call__':
            inputs = [i.values if hasattr(i, 'values') else i for i in inputs]
            obj.values = ufunc(*inputs, **kwargs)
            return obj
        else:
            raise NotImplementedError()

    def __array_function__(self, func, types, args, kwargs):
        from chainladder.utils.utility_functions import concat
        methods_as_funcs = list(set(dir(np)).intersection(set(dir(self))) -
                                {'__dir__', '__doc__'})
        methods_as_funcs = {getattr(np, i): getattr(self, i)
                            for i in methods_as_funcs}
        HANDLED_FUNCTIONS = {np.concatenate: concat, np.round: self.__round__}
        HANDLED_FUNCTIONS = {**HANDLED_FUNCTIONS, **methods_as_funcs}
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        if func in methods_as_funcs:
            args = args[1:]
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def compute(self, *args, **kwargs):
        if hasattr(self.values, 'chunks'):
            obj = self.copy()
            obj.values = obj.values.compute(*args, **kwargs)
            m = obj.get_array_module(obj.values)
            if m == sp:
                obj.array_backend = 'sparse'
            if m == cp:
                obj.array_backend = 'cupy'
            if m == np:
                obj.array_backend = 'numpy'
            return obj
        return self


def is_chainladder(estimator):
    """Return True if the given estimator is a chainladder based method.
    Parameters
    ----------
    estimator : object
        Estimator object to test.
    Returns
    -------
    out : bool
        True if estimator is a chainladder based method and False otherwise.
    """
    return getattr(estimator, "_estimator_type", None) == "chainladder"
