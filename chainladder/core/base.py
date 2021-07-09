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
from chainladder.core.slice import TriangleSlicer, VirtualColumns
from chainladder.core.io import TriangleIO
from chainladder.core.common import Common
from chainladder import AUTO_SPARSE, ULT_VAL
from chainladder.utils.utility_functions import num_to_nan, concat


class TriangleBase(
    TriangleIO, TriangleDisplay, TriangleSlicer, TriangleDunders, TrianglePandas, Common
):
    """ This class handles the initialization of a triangle """

    def __init__(
        self,
        data=None,
        origin=None,
        development=None,
        columns=None,
        index=None,
        origin_format=None,
        development_format=None,
        cumulative=None,
        array_backend=None,
        pattern=False,
        *args,
        **kwargs
    ):
        # Allow Empty Triangle so that we can piece it together programatically
        if data is None:
            return

        # Check whether all columns are unique and numeric
        check = data[columns].dtypes
        check = [check] if isinstance(check, np.dtype) else check.to_list()
        columns = [columns] if type(columns) is not list else columns
        if "object" in check:
            raise TypeError("column attribute must be numeric.")
        if data[columns].shape[1] != len(columns):
            raise AttributeError("Columns are required to have unique names")

        # Sanitize all axis inputs to lists
        str_to_list = lambda *args: tuple(
            [arg] if type(arg) in [str, pd.Period] else arg for arg in args
        )
        index, columns, origin, development = str_to_list(
            index, columns, origin, development
        )

        # Determine desired array backend of the Triangle
        if array_backend is None:
            from chainladder import ARRAY_BACKEND

            array_backend = ARRAY_BACKEND
        if (
            development
            and len(development) == 1
            and data[development[0]].dtype == "<M8[ns]"
        ):
            u = data[data[development[0]] == ULT_VAL].copy()
            if len(u) > 0 and len(u) != len(data):
                u = TriangleBase(
                    u,
                    origin=origin,
                    development=development,
                    columns=columns,
                    index=index,
                )
                data = data[data[development[0]] != ULT_VAL]
            else:
                u = None
        else:
            u = None
        # Initialize origin and its grain
        origin = development if origin is None else origin
        origin_date = TriangleBase._to_datetime(data, origin, format=origin_format)
        self.origin_grain = TriangleBase._get_grain(origin_date)
        date_offset = {
            'M': (pd.offsets.MonthEnd(0), pd.offsets.MonthBegin(1)),
            'Q': (pd.offsets.QuarterEnd(0), pd.offsets.MonthBegin(3)),
            'Y': (pd.offsets.YearEnd(0), pd.offsets.MonthBegin(12))
        }[self.origin_grain]
        origin_date = origin_date + date_offset[0] - date_offset[1]

        # Initialize development and its grain
        m_cnt = {"Y": 12, "Q": 3, "M": 1}
        has_dev = development and len(np.unique(data[development])) > 1
        if has_dev:
            development_date = TriangleBase._to_datetime(
                data, development, period_end=True, format=development_format
            )
            self.development_grain = TriangleBase._get_grain(development_date)
        else:
            development_date = pd.PeriodIndex(
                origin_date + pd.tseries.offsets.MonthEnd(m_cnt[self.origin_grain]),
                freq={"Y": "A"}.get(self.origin_grain, self.origin_grain),
            ).to_timestamp(how="e")
            self.development_grain = self.origin_grain
        development_date.name = "development"
        if pd.Series(development_date).dt.year.min() == pd.Series(development_date).dt.year.max() == 1970:
            raise ValueError(
                'Development lags could not be determined. This may be because development'
                ' is expressed as an age where a date-like vector is required')

        # Summarize dataframe to the level specified in axes
        data["__origin__"] = origin_date
        data["__development__"] = development_date
        key_gr = ["__origin__", "__development__"] + [
            data[item] for item in ([] if not index else index)
        ]
        data_agg = data.groupby(key_gr)[columns].sum().reset_index().fillna(0)
        data = data.drop(['__origin__', '__development__'], 1)
        if not index:
            index = ["Total"]
            data_agg[index[0]] = "Total"

        # Fill in any gaps in origin/development
        date_axes = self._get_date_axes(
            data_agg["__origin__"], data_agg["__development__"]
        )  # cartesian product
        dev_lag = TriangleBase._development_lag(
            data_agg["__origin__"], data_agg["__development__"]
        )

        # Grab unique index, origin, development
        dev_lag_unique = np.sort(
            TriangleBase._development_lag(
                date_axes["__origin__"], date_axes["__development__"]
            ).unique()
        )

        orig_unique = np.sort(date_axes["__origin__"].unique())
        kdims = data_agg[index].drop_duplicates().reset_index(drop=True).reset_index()

        # Map index, origin, development indices to data
        set_idx = (
            lambda col, unique: col.map(dict(zip(unique, range(len(unique)))))
            .values[None]
            .T
        )
        orig_idx = set_idx(data_agg["__origin__"], orig_unique)
        dev_idx = set_idx(dev_lag, dev_lag_unique)
        key_idx = (
            data_agg[index].merge(kdims, how="left", on=index)["index"].values[None].T
        )

        # origin <= development is required - truncate bad records if not true
        valid = data_agg["__origin__"] <= data_agg["__development__"]
        if sum(~valid) > 0:
            warnings.warn(
                "Observations with development before "
                + "origin start have been removed."
            )
        valid = valid.compute() if hasattr(valid, 'compute') else valid
        data_agg, orig_idx = data_agg[valid], orig_idx[valid]
        dev_idx, key_idx = dev_idx[valid], key_idx[valid]

        # All Triangles start out as sparse arrays
        val_idx = (
            ((np.ones(len(data_agg))[None].T) * range(len(columns)))
            .reshape((1, -1), order="F")
            .T
        )
        coords = np.concatenate(
            tuple([np.concatenate((orig_idx, dev_idx), 1)] * len(columns)), 0
        )
        coords = np.concatenate(
            (np.concatenate(tuple([key_idx] * len(columns)), 0), val_idx, coords), 1
        )
        amts = np.concatenate([data_agg[col].fillna(0).values for col in data_agg[columns]]).astype("float64")
        self.array_backend = "sparse"
        self.values = num_to_nan(
            sp(
                coords.T.astype('int64'),
                amts,
                prune=True,
                has_duplicates=False,
                sorted=True,
                shape=(
                    len(kdims),
                    len(columns),
                    len(orig_unique),
                    len(dev_lag_unique) if has_dev else 1,
                ),
            )
        )

        # Set all axis values
        val_date = data_agg["__development__"].max()
        val_date = val_date.compute() if hasattr(val_date, 'compute') else val_date
        self.valuation_date = val_date
        self.kdims = kdims.drop("index", 1).values
        self.odims = orig_unique
        self.ddims = dev_lag_unique if has_dev else dev_lag[0:1].values
        self.ddims = self.ddims * (m_cnt[self.development_grain])
        if development and not has_dev:
            self.ddims = pd.DatetimeIndex(TriangleBase._to_datetime(
                data, development, period_end=True, format=development_format)[0:1])
            self.valuation_date = self.ddims[0]
        self.vdims = np.array(columns)

        # Set remaining triangle properties
        self.key_labels = index
        self.is_cumulative = cumulative
        self.virtual_columns = VirtualColumns(self)
        self.is_pattern = pattern
        if not AUTO_SPARSE or array_backend == "cupy":
            self.set_backend(array_backend, inplace=True)
        else:
            self = self._auto_sparse()
        self._set_slicers()
        if self.is_pattern:
            obj = self.dropna()
            self.odims = obj.odims
            self.ddims = obj.ddims
            self.values = obj.values
        if u:
            obj = concat((self.dev_to_val().iloc[..., : len(u.odims), :], u), -1)
            obj = obj.val_to_dev()
            self.odims = obj.odims
            self.ddims = obj.ddims
            self.values = obj.values
            self.valuation_date = pd.Timestamp(ULT_VAL)

    @property
    def shape(self):
        return self.values.shape

    def _len_check(self, x, y):
        if len(x) != len(y):
            raise ValueError(
                "Length mismatch: Expected axis has ",
                "{} elements, new values have".format(len(x)),
                " {} elements".format(len(y)),
            )

    def _get_date_axes(self, origin_date, development_date):
        """ Function to find any missing origin dates or development dates that
            would otherwise mess up the origin/development dimensions.
        """

        def complete_date_range(
            origin_date, development_date, origin_grain, development_grain
        ):
            """ Determines origin/development combinations in full.  Useful for
                when the triangle has holes in it. """
            o_min = origin_date.min()
            o_max = origin_date.max()
            d_max = development_date.max()
            c = lambda x : x.compute() if hasattr(x, 'compute') else x
            origin_unique = pd.period_range(
                start=c(o_min), end=max(c(o_max), c(d_max)), freq=origin_grain,
            ).to_timestamp()
            development_unique = pd.period_range(
                start=c(o_min), end=c(d_max), freq=development_grain,
            ).to_timestamp(how="e")
            # Let's get rid of any development periods before origin periods
            cart_prod = TriangleBase._cartesian_product(
                origin_unique, development_unique
            )
            cart_prod = cart_prod[cart_prod[:, 0] <= cart_prod[:, 1], :]
            return pd.DataFrame(cart_prod, columns=["__origin__", "__development__"])


        cart_prod_o = complete_date_range(
            pd.Series(origin_date.min()),
            development_date,
            self.origin_grain,
            self.development_grain,
        )
        cart_prod_d = complete_date_range(
            origin_date,
            pd.Series(origin_date.max()),
            self.origin_grain,
            self.development_grain,
        )
        cart_prod_t = pd.DataFrame(
            {"__origin__": origin_date, "__development__": development_date}
        )
        cart_prod = (
            cart_prod_o.append(cart_prod_d, sort=True)
            .append(cart_prod_t, sort=True)
            .drop_duplicates()
        )
        cart_prod = cart_prod[cart_prod["__development__"] >= cart_prod["__origin__"]]
        return cart_prod

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
        if period_end:
            target = target.dt.to_period(
                "M" if len(target.unique()) == 1 else TriangleBase._get_grain(target),
            ).dt.to_timestamp(how="e")
        return target

    @staticmethod
    def _development_lag(origin, development):
        """ For tabular format, this will convert the origin/development
            difference to a development lag """
        year_diff = development.dt.year - origin.dt.year
        quarter_diff = development.dt.quarter - origin.dt.quarter
        month_diff = development.dt.month - origin.dt.month
        all = dp.all if hasattr(origin, 'compute') else np.all
        if all(origin != development):
            development_grain = TriangleBase._get_grain(development)
        else:
            development_grain = "M"
        return dict(
            Y=year_diff + 1,
            Q=year_diff * 4 + quarter_diff + 1,
            M=year_diff * 12 + month_diff + 1,
        )[development_grain]

    @staticmethod
    def _get_grain(array):
        return {1: "Y", 4: "Q"}.get(len(set(array.dt.month)), "M")

    @staticmethod
    def _cartesian_product(*arrays):
        """A fast implementation of cartesian product, used for filling in gaps
        in triangles (if any)"""
        arr = np.empty(
            [len(a) for a in arrays] + [len(arrays)], dtype=np.result_type(*arrays)
        )
        for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a
        arr = arr.reshape(-1, len(arrays))
        return arr

    def get_array_module(self, arr=None):
        backend = (
            self.array_backend
            if arr is None
            else arr.__class__.__module__.split(".")[0]
        )
        modules = {"cupy": cp, "sparse": sp, "numpy": np, "dask": dp}
        if modules.get(backend, None):
            return modules.get(backend, None)
        else:
            raise ValueError("Array backend is invalid or not properly set.")

    def _auto_sparse(self):
        """ Auto sparsifies at 30Mb or more and 20% density or less """
        from chainladder import AUTO_SPARSE

        if not AUTO_SPARSE:
            return self
        n = np.prod(list(self.shape) + [8 / 1e6])
        if (
            self.array_backend == "numpy"
            and n > 30
            and 1 - np.isnan(self.values).sum() / n * (8 / 1e6) < 0.2
        ):
            self.set_backend("sparse", inplace=True)
        if self.array_backend == "sparse" and not (
            self.values.density < 0.2 and n > 30
        ):
            self.set_backend("numpy", inplace=True)
        return self

    def copy(self):
        X = TriangleBase()
        X.__dict__.update(vars(self))
        X._set_slicers()
        return X

    @property
    def valuation(self):
        ddims = self.ddims
        if self.is_val_tri:
            out = pd.DataFrame(np.repeat(self.ddims.values[None], len(self.odims), 0))
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
            ult = np.repeat(np.datetime64(ULT_VAL), val_array.shape[0])[:, None]
            val_array = np.concatenate((val_array, ult,), axis=1,)
        return pd.DatetimeIndex(val_array.reshape(1, -1, order="F")[0])

    def _drop_subtriangles(self):
        """ Removes subtriangles from a Triangle instance """
        sub_tris = [k for k, v in vars(self).items() if isinstance(v, TriangleBase)]
        if 'ldf_' in sub_tris:
            del self.ldf_
        if 'sigma_' in sub_tris:
            del self.sigma_
        if 'std_err_' in sub_tris:
            del self.std_err_

    @property
    def subtriangles(self):
        """ Removes subtriangles from a Triangle instance """
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
        HANDLED_FUNCTIONS = {
            np.concatenate: concat
        }
        if func not in HANDLED_FUNCTIONS:
           return NotImplemented
        if not all(issubclass(t, self.__class__) for t in types):
           return NotImplemented
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
