# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import copy
import dill
import json
import os
import numpy as np
import pandas as pd

from chainladder.utils.sparse import sp
from io import StringIO
from patsy import dmatrix # noqa
from sklearn.base import (
    BaseEstimator,
    TransformerMixin
)

from typing import (
    Iterable,
    Union,
    Optional,
    TYPE_CHECKING
)

if TYPE_CHECKING:
    from chainladder import Triangle
    from numpy.typing import ArrayLike
    from pandas import DataFrame
    from pandas.core.interchange.dataframe_protocol import DataFrame as DataFrameXchg
    from sparse import COO
    from types import ModuleType
    from typing import AnyStr
    from pandas._typing import (
        FilePath,
        ReadCsvBuffer
    )


def load_sample(
        key: str,
        *args,
        **kwargs
) -> Triangle:
    """Function to load datasets included in the chainladder package. These consist of CSV
    files located in the repository directory chainladder/utils/data.

    Parameters
    ----------
    key: str
        The name of the dataset, e.g. RAA, ABC, UKMotor, GenIns, etc. The name should match the
        file name, without extension, of one of the files in the sample data folder.

    Returns
    -------
        chainladder.Triangle of the loaded dataset.

    """
    from chainladder import Triangle


    # Set base path to be the parent directory of this file, e.g., the utils folder.
    utils_path: AnyStr = os.path.dirname(os.path.abspath(__file__))

    # Validate that the file indicated by the key argument exists.
    dataset_path: str = os.path.join(utils_path, "data", key.lower() + ".csv")

    if not os.path.exists(dataset_path):
        raise ValueError(
            """
            Invalid key supplied. The key should match the name, without extension, of one of the file names
            in the sample data set folder. Please refer to the documentation page on sample data sets to see 
            what data are available.
            """
         )

    # Set initial values for arguments to Triangle __init__. These may be overridden by
    # values specific to the data set.
    origin: str = "origin"
    development: str = "development"
    columns: list = ["values"]
    index: list | None = None
    cumulative: bool = True

    if key.lower() in [
        "mcl",
        "usaa",
        "quarterly",
        "auto",
        "usauto",
        "tail_sample"
    ]:
        columns: list = [
            "incurred",
            "paid"
        ]
    if key.lower() == "clrd":
        origin: str = "AccidentYear"
        development: str = "DevelopmentYear"
        index: list = [
            "GRNAME",
            "LOB"
        ]
        columns: list = [
            "IncurLoss",
            "CumPaidLoss",
            "BulkLoss",
            "EarnedPremDIR",
            "EarnedPremCeded",
            "EarnedPremNet",
        ]
    if key.lower() == "berqsherm":
        origin: str = "AccidentYear"
        development: str = "DevelopmentYear"
        index: list = ["LOB"]
        columns: list = [
            "Incurred",
            "Paid",
            "Reported",
            "Closed"
        ]
    if key.lower() == "xyz":
        origin: str = "AccidentYear"
        development: str = "DevelopmentYear"
        columns: list = [
            "Incurred",
            "Paid",
            "Reported",
            "Closed",
            "Premium"
        ]
    if key.lower() in [
        "liab",
        "auto"
    ]:
        index: list = ["lob"]
    if key.lower() in [
        "cc_sample",
        "ia_sample"
    ]:
        columns: list = [
            "loss",
            "exposure"
        ]
    if key.lower() in ["prism"]:
        columns: list = [
            "reportedCount",
            "closedPaidCount",
            "Paid",
            "Incurred"
        ]
        index: list = [
            "ClaimNo",
            "Line",
            "Type",
            "ClaimLiability",
            "Limit",
            "Deductible"
        ]
        origin: str = "AccidentDate"
        development: str = "PaymentDate"
        cumulative: bool = False

    df = pd.read_csv(filepath_or_buffer=dataset_path)

    return Triangle(
        data=df,
        origin=origin,
        development=development,
        index=index,
        columns=columns,
        cumulative=cumulative,
        *args,
        **kwargs
    )


def read_pickle(path):
    with open(path, "rb") as pkl:
        return dill.load(pkl)

def read_csv(
        filepath_or_buffer: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
        origin: Optional[str | list] = None,
        development: Optional[str | list] = None,
        columns: Optional[str | list] = None,
        index: Optional[str | list] = None,
        origin_format: Optional[str] = None,
        development_format: Optional[str] = None,
        cumulative: Optional[bool] = None,
        array_backend: str = None,
        pattern=False,
        trailing: bool = True,
        *args, 
        **kwargs
        ) -> Triangle:
    """
    Funtion that creates Triangle directly from input. Wrapper for pandas dataframe:
    https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html

    Parameters
    ----------
    filepath_or_buffer: str, path object or file-like object
        Any valid string path is acceptable. The string could be a URL. Valid URL schemes 
        include http, ftp, s3, gs, and file. For file URLs, a host is expected. A local 
        file could be: file://localhost/path/to/table.csv.
        If you want to pass in a path object, pandas accepts any os.PathLike.
        By file-like object, we refer to objects with a read() method, such as a 
        file handle (e.g. via builtin open function) or StringIO.
    origin: str or list
         A representation of the accident, reporting or more generally the
         origin period of the triangle that will map to the Origin dimension
    development: str or list
        A representation of the development/valuation periods of the triangle
        that will map to the Development dimension
    columns: str or list
        A representation of the numeric data of the triangle that will map to
        the columns dimension.  If None, then a single 'Total' key will be
        generated.
    index: str or list or None
        A representation of the index of the triangle that will map to the
        index dimension.  If None, then a single 'Total' key will be generated.
    origin_format: optional str
        A string representation of the date format of the origin arg. If
        omitted then date format will be inferred by pandas.
    development_format: optional str
        A string representation of the date format of the development arg. If
        omitted then date format will be inferred by pandas.
    cumulative: bool
        Whether the triangle is cumulative or incremental.  This attribute is
        required to use the ``grain`` and ``dev_to_val`` methods and will be
        automatically set when invoking ``cum_to_incr`` or ``incr_to_cum`` methods.
    trailing: bool
        When partial origin periods are present, setting trailing to True will
        ensure the most recent origin period is a full period and the oldest
        origin is partial. If full origin periods are present in the data, then
        trailing has no effect.

    Attributes
    ----------
    index: Series
        Represents all available levels of the index dimension.
    columns: Series
        Represents all available levels of the value dimension.
    origin: DatetimeIndex
        Represents all available levels of the origin dimension.
    development: Series
        Represents all available levels of the development dimension.
    key_labels: list
        Represents the ``index`` axis labels
    virtual_columns: Series
        Represents the subset of columns of the triangle that are virtual.
    valuation: DatetimeIndex
        Represents all valuation dates of each cell in the Triangle.
    origin_grain: str
        The grain of the origin vector ('Y', 'S', 'Q', 'M')
    development_grain: str
        The grain of the development vector ('Y', 'S', 'Q', 'M')
    shape: tuple
        The 4D shape of the triangle instance with axes corresponding to (index, columns, origin, development)
    link_ratio, age_to_age
        Displays age-to-age ratios for the triangle.
    valuation_date : date
        The latest valuation date of the data
    loc: Triangle
        pandas-style ``loc`` accessor
    iloc: Triangle
        pandas-style ``iloc`` accessor
    latest_diagonal: Triangle
        The latest diagonal of the triangle
    is_cumulative: bool
        Whether the triangle is cumulative or not
    is_ultimate: bool
        Whether the triangle has an ultimate valuation
    is_full: bool
        Whether lower half of Triangle has been filled in
    is_val_tri:
        Whether the triangle development period is expressed as valuation
        periods.
    values: array
        4D numpy array underlying the Triangle instance
    T: Triangle
        Transpose index and columns of object.  Only available when Triangle is
        convertible to DataFrame.
    """


    from chainladder import Triangle

    #Chainladder implementation of: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
    #This will allow the user to create a trignel directly from csv instead in csv -> dataframe -> triangle

    #create a data frame using the *args and **kwargs that the user specified
    local_dataframe = pd.read_csv(filepath_or_buffer,*args, **kwargs)

    #pass the created local_dataframe in the Triangle constructor 
    local_triangle = Triangle(
        data = local_dataframe, 
        origin=origin,
        development=development,
        columns=columns,
        index=index,
        origin_format=origin_format,
        development_format=development_format,
        cumulative=cumulative,
        array_backend=array_backend,
        pattern=pattern,
        trailing = trailing
    )

    return local_triangle

def read_json(json_str, array_backend=None):
    from chainladder import Triangle
    from chainladder.workflow import Pipeline

    if array_backend is None:
        from chainladder import options

        array_backend = options.ARRAY_BACKEND
    json_dict = json.loads(json_str)
    if type(json_dict) is list:
        import chainladder as cl

        return Pipeline(
            steps=[
                (
                    item["name"],
                    cl.__dict__[item["__class__"]]().set_params(**item["params"]),
                )
                for item in json_dict
            ]
        )
    elif "metadata" in json_dict.keys():
        j = json.loads(json_str)
        y = pd.read_json(StringIO(j["data"]), orient="split", date_unit="ns")
        y["origin"] = pd.to_datetime(y["origin"])
        y.columns = [c if c != "valuation" else "development" for c in y.columns]
        y["development"] = pd.to_datetime(y["development"])
        index = list(y.columns[: list(y.columns).index("origin")])
        columns = list(y.columns[list(y.columns).index("development") + 1 :])
        tri = Triangle(
            y,
            origin="origin",
            development="development",
            index=index,
            columns=columns,
            pattern=json.loads(j["metadata"])["is_pattern"],
            cumulative=False,
        )
        if json.loads(j["metadata"])["is_val_tri"]:
            tri = tri.dev_to_val()
        if json.loads(j["metadata"])["is_cumulative"]:
            tri = tri.incr_to_cum()
        tri = tri[json.loads(j["metadata"])["columns"]].sort_index()
        if "sub_tris" in json_dict.keys():
            for k, v in json_dict["sub_tris"].items():
                setattr(tri, k, read_json(v, array_backend))
                setattr(getattr(tri, k), "origin_grain", tri.origin_grain)
                setattr(getattr(tri, k), "development_grain", tri.development_grain)
        if "dfs" in json_dict.keys():
            for k, v in json_dict["dfs"].items():
                df = pd.read_json(v)
                if len(df.columns) == 1:
                    df = df.iloc[:, 0]
                setattr(tri, k, df)
        if array_backend:
            return tri.set_backend(array_backend)
        else:
            return tri
    else:
        import chainladder as cl

        return cl.__dict__[json_dict["__class__"]]().set_params(**json_dict["params"])


def parallelogram_olf(
    values,
    date,
    start_date=None,
    end_date=None,
    grain="Y",
    approximation_grain="M",
    vertical_line=False,
):
    """Parallelogram approach to on-leveling."""
    date = pd.to_datetime(date)
    if not start_date:
        start_date = "{}-01-01".format(date.min().year)
    if not end_date:
        end_date = "{}-12-31".format(date.max().year)
    start_date = pd.to_datetime(start_date) - pd.tseries.offsets.DateOffset(days=1)

    date_freq = {
        "M": "MS",
        "D": "D",
    }
    if approximation_grain not in ['M', 'D']:
        raise ValueError("approximation_grain must be " "M" " or " "D" "")
    date_idx = pd.date_range(
        start_date - pd.tseries.offsets.DateOffset(years=1),
        end_date,
        freq=date_freq[approximation_grain],
    )

    rate_changes = pd.Series(np.array(values), np.array(date))
    rate_changes = rate_changes.reindex(date_idx, fill_value=0)

    cum_rate_changes = np.cumprod(1 + rate_changes.values)
    cum_rate_changes = pd.Series(cum_rate_changes, rate_changes.index)
    crl = cum_rate_changes.iloc[-1]

    cum_avg_rate_non_leaps = cum_rate_changes
    cum_avg_rate_leaps = cum_rate_changes

    if not vertical_line:
        rolling_num = {
            "M": 12,
            "D": 365,
        }

        cum_avg_rate_non_leaps = cum_rate_changes.rolling(
            rolling_num[approximation_grain]
        ).mean()
        cum_avg_rate_non_leaps = (
            cum_avg_rate_non_leaps + cum_avg_rate_non_leaps.shift(1).values
        ) / 2

        cum_avg_rate_leaps = cum_rate_changes.rolling(
            rolling_num[approximation_grain] + 1
        ).mean()
        cum_avg_rate_leaps = (
            cum_avg_rate_leaps + cum_avg_rate_leaps.shift(1).values
        ) / 2

    dropdates_num = {
        "M": 12,
        "D": 366,
    }
    cum_avg_rate_non_leaps = cum_avg_rate_non_leaps.iloc[
        dropdates_num[approximation_grain] :
    ]
    cum_avg_rate_leaps = cum_avg_rate_leaps.iloc[
        dropdates_num[approximation_grain] + 1 :
    ]

    fcrl_non_leaps = (
        cum_avg_rate_non_leaps.groupby(cum_avg_rate_non_leaps.index.to_period(grain))
        .mean()
        .reset_index()
    )
    fcrl_non_leaps.columns = ["Origin", "OLF"]
    fcrl_non_leaps["Origin"] = fcrl_non_leaps["Origin"].astype(str)
    fcrl_non_leaps["OLF"] = crl / fcrl_non_leaps["OLF"]

    fcrl_leaps = (
        cum_avg_rate_leaps.groupby(cum_avg_rate_leaps.index.to_period(grain))
        .mean()
        .reset_index()
    )
    fcrl_leaps.columns = ["Origin", "OLF"]
    fcrl_leaps["Origin"] = fcrl_leaps["Origin"].astype(str)
    fcrl_leaps["OLF"] = crl / fcrl_leaps["OLF"]

    combined = fcrl_non_leaps.join(fcrl_leaps, lsuffix="_non_leaps", rsuffix="_leaps")
    combined["is_leap"] = pd.to_datetime(
        combined["Origin_non_leaps"], format="%Y" + ("-%M" if grain == "M" else "")
    ).dt.is_leap_year
    

    if approximation_grain == "M":
        combined["final_OLF"] = combined["OLF_non_leaps"]
    else:
        combined["final_OLF"] = np.where(
            combined["is_leap"], combined["OLF_leaps"], combined["OLF_non_leaps"]
        )

    combined.drop(
        ["OLF_non_leaps", "Origin_leaps", "OLF_leaps", "is_leap"],
        axis=1,
        inplace=True,
    )
    combined.columns = ["Origin", "OLF"]

    return combined.set_index("Origin")


def set_common_backend(objs):
    from chainladder import options

    priority = options.ARRAY_PRIORITY
    backend = priority[np.min([priority.index(i.array_backend) for i in objs])]
    return [i.set_backend(backend) for i in objs]


def concat(
    objs: Iterable,
    axis: Union[int, str],
    ignore_index: bool = False,
    sort: bool = False,
):
    """Concatenate Triangle objects along a particular axis.

    Parameters
    ----------
    objs: list or tuple
        A list or tuple of Triangle objects to concat. All non-concat axes must
        be identical and all elements of the concat axes must be unique.
    axis: string or int
        The axis to concatenate along.
    ignore_index: bool, default False
        If True, do not use the index values along the concatenation axis. The
        resulting axis will be labeled 0, …, n - 1. This is useful if you are
        concatenating objects where the concatenation axis does not have
        meaningful indexing information. Note the index values on the other
        axes are still respected in the join.

    Returns
    -------
    Updated triangle
    """
    if type(objs) not in (list, tuple):
        raise TypeError("objects to be concatenated must be in a list or tuple")
    if type(objs) is tuple:
        objs = list(objs)
    if len(objs) == 0:
        raise ValueError("objs must contain at least one element")
    xp = objs[0].get_array_module()
    axis = objs[0]._get_axis(axis)

    if axis != 1:
        all_columns = []
        for obj in objs:
            for col in obj.columns:
                if col not in all_columns:
                    all_columns.append(col)
        for num, obj in enumerate(objs):
            for col in all_columns:
                if col not in obj.columns:
                    obj = copy.deepcopy(obj)
                    obj[col] = xp.nan
                    objs[num] = obj
    objs = set_common_backend(objs)
    mapper = {0: "kdims", 1: "vdims", 2: "odims", 3: "ddims"}
    for k in mapper.keys():
        if k != axis and k != 1:  # All non-concat axes must be identical
            a = np.array([getattr(obj, mapper[k]) for obj in objs])
            assert np.all(a == a[0])
        else:  # All elements of concat axis must be unique
            if ignore_index:
                new_axis = np.arange(
                    np.sum([len(getattr(obj, mapper[axis])) for obj in objs])
                )
                new_axis = new_axis[:, None] if axis == 0 else new_axis
            else:
                new_axis = np.concatenate([getattr(obj, mapper[axis]) for obj in objs])
            if axis == 0:
                assert len(pd.DataFrame(new_axis).drop_duplicates()) == len(new_axis)
            else:
                assert len(new_axis) == len(set(new_axis))
    out = copy.deepcopy(objs[0])
    out.values = xp.concatenate([obj.values for obj in objs], axis=axis)
    setattr(out, mapper[axis], new_axis)
    if ignore_index and axis == 0:
        out.key_labels = ["Index"]
    out.valuation_date = pd.Series([obj.valuation_date for obj in objs]).max()
    if out.ddims.dtype == "datetime64[ns]" and type(out.ddims) == np.ndarray:
        out.ddims = pd.DatetimeIndex(out.ddims)
    out._set_slicers()
    if sort:
        return out.sort_axis(axis)
    else:
        return out


def num_to_value(
        arr: ArrayLike,
        value
) -> ArrayLike:
    """
    Function that turns all zeros to nan values in an array.
    """
    backend = arr.__class__.__module__.split(".")[0]
    if backend == "sparse":
        if arr.fill_value == 0 or sp.isnan(arr.fill_value):
            arr.coords = arr.coords[:, arr.data != 0]
            arr.data = arr.data[arr.data != 0]

            arr: COO = sp(
                coords=arr.coords,
                data=arr.data,
                fill_value=sp.nan, # noqa
                shape=arr.shape
            )
        else:
            arr: COO = sp(
                num_to_nan(np.nan_to_num(arr.todense())),
                fill_value=value
            )
    else:
        arr[arr == 0] = value
    return arr


def num_to_nan(arr: ArrayLike) -> ArrayLike:
    """
    Function that turns all zeros to nan values in an array.

    Parameters
    ----------
    arr: ArrayLike
        An array-like object. For example, the values used in a Triangle.

    Returns
    -------
        The supplied array with all zeros converted to nan values. These nans
        are specific to the Triangle backend used.
    """

    from chainladder import Triangle

    # Take the nan specific to the module of the backend used. e.g., numpy, cupy, sparse, etc.
    xp: ModuleType = Triangle.get_array_module(
        None,
        arr=arr
    )

    return num_to_value(arr, xp.nan)


def minimum(x1, x2):
    return x1.minimum(x2)


def maximum(x1, x2):
    return x1.maximum(x2)

def to_period(dateseries: pd.Series, freq:str):
    if freq[:2] != '2Q':
        return dateseries.dt.to_period(freq)
    else:
        return dateseries.where(dateseries.dt.to_period(freq).dt.strftime('%q').isin(['1','3']),dateseries.dt.date + pd.DateOffset(months=-3)).dt.to_period(freq)

class PatsyFormula(BaseEstimator, TransformerMixin):
    """A sklearn-style Transformer for patsy formulas.

    PatsyFormula allows for R-style formula preprocessing of the ``design_matrix``
    of a machine learning algorithm. It's particularly useful with the `DevelopmentML`
    and `TweedieGLM` estimators.

    Parameters
    ----------

    formula: str
        A string representation of the regression model X features.

    Attributes
    ----------
    design_info_:
        The patsy instructions for generating the design_matrix, X.

    """

    def __init__(self, formula=None):
        self.formula = formula

    def _check_X(self, X):
        from chainladder.core import Triangle

        if isinstance(X, Triangle):
            raise AttributeError("X must be a pandas dataframe, not a Triangle")

    def fit(self, X, y=None, sample_weight=None):
        self._check_X(X)
        self.design_info_ = dmatrix(self.formula, X).design_info
        return self

    def transform(self, X):
        self._check_X(X)
        return dmatrix(self.design_info_, X)


def model_diagnostics(model, name=None, groupby=None):
    """A helper function that summarizes various vectors of an
    IBNR model as columns of a Triangle

    Parameters
    ----------
    model:
        A chainladder IBNR estimator or Pipeline
    name:
        An alias to give the model. This will be added to the index of the return
        triangle.
    groupby:
        The index level at which the model should be summarized

    Returns
    -------
    Triangle up select origin vectors, IBNR, ultimate, Latest diagonal, etc.
    """
    from chainladder import Pipeline

    if isinstance(model, Pipeline):
        obj = copy.deepcopy(model.steps[-1][-1])
    else:
        obj = copy.deepcopy(model)
    if groupby is not None:
        obj.X_ = obj.X_.groupby(groupby).sum().cum_to_incr()
        obj.ultimate_ = obj.ultimate_.groupby(groupby).sum()
        if hasattr(obj, "expectation_"):
            obj.expectation_ = obj.expectation_.groupby(groupby).sum()
    else:
        obj.X_ = obj.X_.incr_to_cum()
    val = obj.X_.valuation
    latest = obj.X_.sum("development")
    run_off = obj.full_expectation_.iloc[..., :-1].dev_to_val().cum_to_incr()
    run_off = run_off[run_off.development > str(obj.X_.valuation_date)]
    run_off = run_off.iloc[
        ..., : {"M": 12, "S": 6, "Q": 4, "Y": 1}[obj.X_.development_grain]
    ]

    triangles = []
    for col in obj.ultimate_.columns:
        idx = latest.index
        idx["Measure"] = col
        idx["Model"] = obj.__class__.__name__ if name is None else name
        idx = idx[list(idx.columns[-2:]) + list(idx.columns[:-2])]
        out = latest[col].rename("columns", ["Latest"])
        if obj.X_.development_grain in ["M"]:
            out["Month Incremental"] = obj.X_[col][val == obj.X_.valuation_date].sum(
                "development"
            )
        if (
            obj.X_.development_grain in ["M", "Q"]
            and pd.Period(out.valuation_date, freq="Q").to_timestamp(how="s")
            > val.min()
        ):
            out["Quarter Incremental"] = (
                obj.X_
                - obj.X_[
                    val
                    < pd.Period(out.valuation_date, freq="Q")
                    .to_timestamp(how="s")
                    .strftime("%Y-%m")
                ]
            ).sum("development")[col]
        else:
            out["Quarter Incremental"] = 0
        if pd.Period(out.valuation_date, freq="Y").to_timestamp(how="s") > val.min():
            out["Year Incremental"] = (
                obj.X_ - obj.X_[val < str(obj.X_.valuation_date.year)]
            ).sum("development")[col]
        else:
            out["Year Incremental"] = 0
        out["IBNR"] = obj.ibnr_[col]
        out["Ultimate"] = obj.ultimate_[col]
        for i in range(run_off.shape[-1]):
            out["Run Off " + str(i + 1)] = run_off[col].iloc[..., i]
        if hasattr(obj, "expectation_"):
            out["Apriori"] = (
                obj.expectation_
                if obj.expectation_.shape[1] == 1
                else obj.expectation_[col]
            )
        out.index = idx
        triangles.append(out)
    return concat(triangles, 0)


def PTF_formula(alpha: list = None, gamma: list = None, iota: list = None,dgrain: int = 12):
    """ Helper formula that builds a patsy formula string for the BarnettZehnwirth 
    estimator.  Each axis's parameters can be grouped together. Groups of origin 
    parameters (alpha) are set equal, and are specified by the first period in each bin. 
    Groups of development (gamma) and valuation (iota) parameters are fit to 
    separate linear trends, specified a list denoting the endpoints of the linear pieces.
    In other words, development and valuation trends are fit to a piecewise linear model.
    A triangle must be supplied to provide some critical information.
    """
    formula_parts=[]
    if(alpha):
        # The intercept term takes the place of the first alpha
        for ind,a in enumerate(alpha):
            if(a==0):
                alpha=alpha[:ind]+alpha[(ind+1):]
        formula_parts += ['+'.join([f'I({x} <= origin)' for x in alpha])]
    if(gamma): 
        # preprocess gamma to align with grain
        graingamma = [(i+1)*dgrain for i in gamma]
        for ind in range(1,len(graingamma)):
            formula_parts += ['+'.join([f'I((np.minimum({graingamma[ind]},development) - np.minimum({graingamma[ind-1]},development))/{dgrain})'])]
    if(iota):
        for ind in range(1,len(iota)):
            formula_parts += ['+'.join([f'I(np.minimum({iota[ind]},valuation) - np.minimum({iota[ind-1]},valuation))'])]
    if(formula_parts):
        return '+'.join(formula_parts)
    return ''
