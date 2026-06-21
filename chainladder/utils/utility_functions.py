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

from chainladder import __dt64_unit__, __dt64_dtype__
from chainladder.utils.sparse import sp
from chainladder.utils.data._manifest import SAMPLES
from io import StringIO
from patsy import dmatrix  # noqa
from sklearn.base import BaseEstimator, TransformerMixin

from typing import Iterable, Union, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from chainladder import Triangle, MethodBase, Pipeline
    from numpy.typing import ArrayLike
    from pandas import DataFrame
    from sparse import COO
    from types import ModuleType
    from typing import AnyStr
    from pandas._typing import FilePath, ReadCsvBuffer


def load_sample(key: str, *args, **kwargs) -> Triangle:
    """Function to load a dataset already included in the chainladder package. These consist of CSV
    files located in the repository directory chainladder/utils/data.

    Parameters
    ----------
    key: str (not case sensitive)
        The name of the dataset. The name should match the file name, without extension, of one of
        the files in the sample data folder.

        Datasets that are commonly used in examples are: raa, clrd, and prism.

        For the complete list of available datasets, call :func:`list_samples`.

    Returns
    -------
        chainladder.Triangle of the loaded dataset.


    Examples
    --------

    Loading "raa" as an example.

    .. testsetup::

        import chainladder as cl

    .. testcode::

        tr = cl.load_sample("raa")
        print(tr)

    .. testoutput::

                 12       24       36       48       60       72       84       96       108      120
        1981  5012.0   8269.0  10907.0  11805.0  13539.0  16181.0  18009.0  18608.0  18662.0  18834.0
        1982   106.0   4285.0   5396.0  10666.0  13782.0  15599.0  15496.0  16169.0  16704.0      NaN
        1983  3410.0   8992.0  13873.0  16141.0  18735.0  22214.0  22863.0  23466.0      NaN      NaN
        1984  5655.0  11555.0  15766.0  21266.0  23425.0  26083.0  27067.0      NaN      NaN      NaN
        1985  1092.0   9565.0  15836.0  22169.0  25955.0  26180.0      NaN      NaN      NaN      NaN
        1986  1513.0   6445.0  11702.0  12935.0  15852.0      NaN      NaN      NaN      NaN      NaN
        1987   557.0   4020.0  10946.0  12314.0      NaN      NaN      NaN      NaN      NaN      NaN
        1988  1351.0   6947.0  13112.0      NaN      NaN      NaN      NaN      NaN      NaN      NaN
        1989  3133.0   5395.0      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN
        1990  2063.0      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN



    """
    from chainladder import Triangle

    # Set base path to be the parent directory of this file, e.g., the utils folder.
    utils_path: AnyStr = os.path.dirname(os.path.abspath(__file__))

    # Validate the key against the sample-dataset manifest. The manifest is the
    # authoritative list of available samples; every entry has a matching CSV in
    # the data folder.
    if key.lower() not in SAMPLES:
        raise ValueError(
            """
            Invalid key supplied. The key should match the name, without extension, of one of the file names
            in the sample data set folder. Please refer to the documentation page on sample data sets to see 
            what data are available.
            
            You supplied: {}
            """.format(
                key
            )
        )

    dataset_path: str = os.path.join(utils_path, "data", key.lower() + ".csv")

    # Look up the Triangle configuration for this sample from the central
    # manifest (chainladder/utils/data/_manifest.py). The manifest is the
    # single source of truth for sample-dataset metadata, replacing the long
    # per-dataset if/elif chain that previously lived here and duplicated the
    # column names already present in the tests and the sample-data docs.
    config: dict = SAMPLES[key.lower()]
    origin = config["origin"]
    development = config["development"]
    index = config["index"]
    columns = config["columns"]
    cumulative = config["cumulative"]

    development_format = config.get("development_format", None)

    df = pd.read_csv(filepath_or_buffer=dataset_path)

    return Triangle(
        data=df,
        origin=origin,
        development=development,
        index=index,
        columns=columns,
        cumulative=cumulative,
        development_format=development_format,
        *args,
        **kwargs,
    )


# Human-readable labels for the single-character grain codes a Triangle exposes
# via ``origin_grain`` / ``development_grain``.
_GRAIN_LABELS: dict = {
    "Y": "Annual",
    "S": "Semiannual",
    "Q": "Quarter",
    "M": "Month",
}


def list_samples(include_grain: bool = True) -> DataFrame:
    """List the sample datasets bundled with the chainladder package.

    The returned table is driven by the sample-dataset manifest
    (``chainladder/utils/data/_manifest.py``), the same source
    :func:`load_sample` reads, so it always reflects exactly what is loadable.

    Parameters
    ----------
    include_grain: bool
        If ``True`` (default), load each sample to report its origin and
        development grain (and the number of origin/development periods). This
        is the slower path because every Triangle is built. Set to ``False`` to
        return just the manifest metadata (name, index, columns, cumulative)
        without loading any data.

    Returns
    -------
        pandas.DataFrame indexed by sample name, with columns ``index``,
        ``columns``, ``cumulative`` and, when ``include_grain`` is ``True``,
        ``origin_grain``, ``development_grain``, ``origin_periods`` and
        ``development_periods``.

    Examples
    --------

    .. code-block:: python

        import chainladder as cl
        cl.list_samples()                    # full table, grain included
        cl.list_samples(include_grain=False) # fast, metadata only
    """
    records: list = []
    for name in sorted(SAMPLES):
        config: dict = SAMPLES[name]
        record: dict = {
            "name": name,
            "index": config["index"],
            "columns": config["columns"],
            "cumulative": config["cumulative"],
        }
        if include_grain:
            triangle = load_sample(name)
            record["origin_grain"] = _GRAIN_LABELS.get(
                triangle.origin_grain, triangle.origin_grain
            )
            record["development_grain"] = _GRAIN_LABELS.get(
                triangle.development_grain, triangle.development_grain
            )
            record["origin_periods"] = len(triangle.origin)
            record["development_periods"] = triangle.development.shape[0]
        records.append(record)

    return pd.DataFrame.from_records(records).set_index("name")


def read_pickle(path):
    """Load an object serialized with ``to_pickle`` (``dill`` format).

    Parameters
    ----------
    path : str or path-like
        Path to the pickle file.

    Returns
    -------
    object
        The deserialized triangle or estimator.

    Examples
    --------
    Pickling preserves all fitted parameters, including non-default settings.
    A ``Development`` configured with ``average='simple'`` and ``n_periods=4``
    produces identical factors before and after a round-trip through disk, and
    the restored estimator can still ``transform`` new data.

    .. testsetup::

        import tempfile
        import os

    .. testcode::

        import chainladder as cl

        tri = cl.load_sample("raa")
        dev = cl.Development(average="simple", n_periods=4).fit(tri)
        fd, p = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)
        dev.to_pickle(p)
        restored = cl.read_pickle(p)
        os.remove(p)
        print(dev.ldf_.values[0, 0, 0, :].round(4))
        print(restored.ldf_.values[0, 0, 0, :].round(4))
        print(restored.transform(tri).ldf_.values[0, 0, 0, :].round(4))

    .. testoutput::

        [4.5853 2.0204 1.2448 1.1646 1.1099 1.0433 1.0344 1.018  1.0092]
        [4.5853 2.0204 1.2448 1.1646 1.1099 1.0433 1.0344 1.018  1.0092]
        [4.5853 2.0204 1.2448 1.1646 1.1099 1.0433 1.0344 1.018  1.0092]

    """
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
    **kwargs,
) -> Triangle:
    """
    Function that creates Triangle directly from input. Wrapper for pandas dataframe:
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
    """

    from chainladder import Triangle

    # Chainladder implementation of: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
    # This will allow the user to create a trignel directly from csv instead in csv -> dataframe -> triangle

    # create a data frame using the *args and **kwargs that the user specified
    local_dataframe = pd.read_csv(filepath_or_buffer, *args, **kwargs)

    # pass the created local_dataframe in the Triangle constructor
    local_triangle = Triangle(
        data=local_dataframe,
        origin=origin,
        development=development,
        columns=columns,
        index=index,
        origin_format=origin_format,
        development_format=development_format,
        cumulative=cumulative,
        array_backend=array_backend,
        pattern=pattern,
        trailing=trailing,
    )

    return local_triangle


def read_json(json_str, array_backend=None):
    """Deserialize JSON produced by ``to_json`` (triangle, estimator, or pipeline).

    Examples
    --------
    ``to_json`` serializes an estimator's parameters as a JSON string that
    can be stored in a database, config file, or REST API. ``read_json``
    reconstructs the estimator with all parameters intact.

    .. testsetup::

        import chainladder as cl

    .. testcode::

        dev = cl.Development(average="simple", n_periods=4)
        json_str = dev.to_json()
        print(json_str)
        dev2 = cl.read_json(json_str)
        print(dev2.get_params()["average"])
        print(dev2.get_params()["n_periods"])

    .. testoutput::

        {"params": {"average": "simple", "drop": null, "drop_above": Infinity, "drop_below": 0.0, "drop_high": null, "drop_low": null, "drop_valuation": null, "fillna": null, "groupby": null, "n_periods": 4, "preserve": 1, "sigma_interpolation": "log-linear"}, "__class__": "Development"}
        simple
        4

    """
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
                df = pd.read_json(StringIO(v))
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
    dates,
    start_date=None,
    end_date=None,
    grain="Y",
    policy_length=12,
    approximation_grain="M",
    vertical_line=False,
):
    """Parallelogram approach to on-leveling."""
    if approximation_grain not in ["M", "D"]:
        raise ValueError("approximation_grain must be M or D")

    dates = pd.to_datetime(dates)

    if start_date:
        start_date = pd.to_datetime(start_date) - pd.tseries.offsets.DateOffset(days=1)
    else:
        start_date = pd.to_datetime("{}-01-01".format(dates.min().year))

    if not end_date:
        end_date = pd.to_datetime("{}-12-31".format(dates.max().year))

    lookback_years = max(1, -(-policy_length // 12))

    date_idx = pd.date_range(
        start_date - pd.tseries.offsets.DateOffset(years=lookback_years),
        end_date,
        freq={"M": "MS", "D": "D"}[approximation_grain],
    )

    rate_changes = pd.Series(np.array(values), np.array(dates)).reindex(
        date_idx, fill_value=0
    )
    cum_rate_changes = pd.Series(
        np.cumprod(1 + rate_changes.values), rate_changes.index
    )
    crl = cum_rate_changes.iloc[-1]

    rolling_num_base = {
        "M": policy_length,
        "D": int(365 * policy_length / 12),
    }[approximation_grain]
    dropdates_base = {
        "M": 12 * lookback_years,
        "D": 366 * lookback_years,
    }[approximation_grain]

    def _fcrl_for_leap(is_leap_year: bool):
        # In monthly mode every month is treated as an equal length period, so a
        # leap day has no effect on the rolling window or the lookback drop.
        if approximation_grain == "M":
            is_leap_year = False

        if is_leap_year:
            leap_day = 1
        else:
            leap_day = 0

        if vertical_line:  # rectangle method, rate change impact is immediate
            cum_avg = cum_rate_changes

        else:  # parallelogram method, rate change impact is overtime
            #
            average_period = max(rolling_num_base + leap_day, 1)

            cum_avg = cum_rate_changes.rolling(average_period).mean()
            cum_avg = (cum_avg + cum_avg.shift(1).values) / 2

        cum_avg = cum_avg.iloc[dropdates_base + leap_day :]

        fcrl = cum_avg.groupby(cum_avg.index.to_period(grain)).mean().reset_index()
        fcrl.columns = ["Origin", "OLF"]
        fcrl["Origin"] = fcrl["Origin"].astype(str)
        fcrl["OLF"] = crl / fcrl["OLF"]

        return fcrl

    fcrl_non_leaps = _fcrl_for_leap(False)
    fcrl_leaps = fcrl_non_leaps if approximation_grain == "M" else _fcrl_for_leap(True)

    combined = fcrl_non_leaps.join(fcrl_leaps, lsuffix="_non_leaps", rsuffix="_leaps")
    combined["is_leap"] = pd.to_datetime(
        combined["Origin_non_leaps"], format="%Y" + ("-%m" if grain == "M" else "")
    ).dt.is_leap_year

    combined["final_OLF"] = np.where(
        combined["is_leap"], combined["OLF_leaps"], combined["OLF_non_leaps"]
    )

    combined.drop(
        ["OLF_non_leaps", "Origin_leaps", "OLF_leaps", "is_leap"], axis=1, inplace=True
    )
    combined.columns = ["Origin", "OLF"]

    return combined.set_index("Origin")


def set_common_backend(objs):
    from chainladder import options

    priority = options.ARRAY_PRIORITY
    backend = priority[np.min([priority.index(i.array_backend) for i in objs])]
    return [i.set_backend(backend) for i in objs]


def concat(
    objs: list | tuple,
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
    sort: bool
        If True, sort the result along the desired axis.

    Returns
    -------
    Updated triangle

    Examples
    --------
    When paid and incurred triangles are constructed separately, ``concat``
    along ``axis=1`` combines them into one multi-column triangle, giving
    downstream methods access to both columns at once.

    .. testsetup::

        import chainladder as cl

    .. testcode::

        clrd = cl.load_sample("clrd").groupby("LOB").sum()
        wkcomp = clrd.iloc[5:6]
        paid = wkcomp["CumPaidLoss"]
        incurred = wkcomp["IncurLoss"]
        combined = cl.concat([paid, incurred], axis=1)
        print(list(combined.columns))

    .. testoutput::

        ['CumPaidLoss', 'IncurLoss']

    When two triangles possess a column that the other does not have, their concatenation will fill in the
    missing values of each sub-triangle with xp.nan.

    .. testsetup::

        import chainladder as cl

    .. testcode::

        clrd = cl.load_sample('clrd')
        clrd = clrd.groupby("LOB").sum()
        t1 = clrd.loc["wkcomp"][["IncurLoss"]].rename("columns", ["A"])
        t2 = clrd.loc["comauto"][["CumPaidLoss"]].rename("columns", ["B"])
        result = cl.concat([t1, t2], axis=0)
        print(result.loc["wkcomp"]["B"])

    .. testoutput::

              12   24   36   48   60   72   84   96   108  120
        1988  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
        1989  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
        1990  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
        1991  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
        1992  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
        1993  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
        1994  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
        1995  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
        1996  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
        1997  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
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
            # Make sure columns are in the same order for all objs to ensure proper indexing.
            if list(objs[num].columns) != all_columns:
                objs[num] = objs[num][all_columns]
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
    if out.ddims.dtype == __dt64_dtype__ and type(out.ddims) == np.ndarray:
        out.ddims = pd.DatetimeIndex(out.ddims)
    out._set_slicers()
    if sort:
        return out.sort_axis(axis)
    else:
        return out


def num_to_value(arr: ArrayLike, value) -> ArrayLike:
    """
    Function that turns all zeros to nan values in an array.
    """
    backend = arr.__class__.__module__.split(".")[0]
    if backend == "sparse":
        if arr.fill_value == 0 or sp.isnan(arr.fill_value):
            arr.coords = arr.coords[:, arr.data != 0]
            arr.data = arr.data[arr.data != 0]

            arr: COO = sp.COO(
                coords=arr.coords,
                data=arr.data,
                fill_value=sp.COO.nan,  # noqa
                shape=arr.shape,
            )
        else:
            arr: COO = sp.COO(
                num_to_nan(np.nan_to_num(arr.todense())), fill_value=value
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
    xp: ModuleType = Triangle.get_array_module(None, arr=arr)

    return num_to_value(arr, xp.nan)


def minimum(x1, x2):
    """Element-wise minimum of two triangles or a triangle and a scalar
    (delegates to ``Triangle.minimum``).

    Parameters
    ----------
    x1 : Triangle
        The first triangle operand.
    x2 : Triangle or scalar
        The second operand. If a scalar, each element of ``x1`` is compared
        against that constant value.

    Examples
    --------
    When two chainladder runs use different development factor selections,
    the ultimates may disagree at each origin. ``minimum`` picks the lower
    ultimate at each origin, producing the low-side scenario.

    .. testsetup::

        import chainladder as cl

    .. testcode::

        tri = cl.load_sample("raa")
        ult_vol = cl.Chainladder().fit(
            cl.Development(average="volume").fit_transform(tri)
        ).ultimate_
        ult_sim = cl.Chainladder().fit(
            cl.Development(average="simple").fit_transform(tri)
        ).ultimate_
        print(ult_vol.values[0, 0, -5:, 0].round(0))
        print(ult_sim.values[0, 0, -5:, 0].round(0))
        low_side = cl.minimum(ult_vol, ult_sim)
        print(low_side.values[0, 0, -5:, 0].round(0))

    .. testoutput::

        [19501. 17749. 24019. 16045. 18402.]
        [19807. 18201. 25475. 17776. 55781.]
        [19501. 17749. 24019. 16045. 18402.]

    """
    return x1.minimum(x2)


def maximum(x1, x2):
    """Element-wise maximum of two triangles or a triangle and a scalar
    (delegates to ``Triangle.maximum``).

    Parameters
    ----------
    x1 : Triangle
        The first triangle operand.
    x2 : Triangle or scalar
        The second operand. If a scalar, each element of ``x1`` is compared
        against that constant value.

    Examples
    --------
    ``maximum`` picks the higher ultimate at each origin, producing the
    high-side scenario. This is useful for stress testing or setting a
    conservative reserve when two methods produce different estimates.

    .. testsetup::

        import chainladder as cl

    .. testcode::

        tri = cl.load_sample("raa")
        ult_vol = cl.Chainladder().fit(
            cl.Development(average="volume").fit_transform(tri)
        ).ultimate_
        ult_sim = cl.Chainladder().fit(
            cl.Development(average="simple").fit_transform(tri)
        ).ultimate_
        high_side = cl.maximum(ult_vol, ult_sim)
        print(high_side.values[0, 0, -5:, 0].round(0))

    .. testoutput::

        [19807. 18201. 25475. 17776. 55781.]

    """
    return x1.maximum(x2)


def to_period(dateseries: pd.Series, freq: str):
    if freq[:2] != "2Q":
        return dateseries.dt.to_period(freq)
    else:
        return dateseries.where(
            dateseries.dt.to_period(freq).dt.strftime("%q").isin(["1", "3"]),
            dateseries.dt.date + pd.DateOffset(months=-3),
        ).dt.to_period(freq)


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

    Examples
    --------
    If a development-only Poisson GLM produces residuals that vary
    systematically by accident year, adding ``C(origin)`` to the formula
    introduces origin-level intercepts and reduces that structure. The
    expanded model matrix has more columns (one per development period plus one
    per origin), which ``PatsyFormula`` builds from the same R-style string.

    .. testsetup::

        import chainladder as cl

    .. testcode::

        genins = cl.load_sample("genins")
        by_dev = cl.TweedieGLM(design_matrix="C(development)").fit(genins)
        by_both = cl.TweedieGLM(
            design_matrix="C(development) + C(origin)"
        ).fit(genins)
        print(len(by_dev.coef_))
        print(len(by_both.coef_))
        print(by_dev.ldf_.values[0, 0, 0, :].round(4))
        print(by_both.ldf_.values[0, 0, 0, :].round(4))

    .. testoutput::

        10
        19
        [3.5085 1.7436 1.4379 1.1656 1.0991 1.0832 1.0511 1.0693 1.0135]
        [3.491  1.7474 1.4574 1.1739 1.1038 1.0863 1.0539 1.0766 1.0177]

    When ``TweedieGLM`` is not flexible enough (for example, when you need a
    non-Tweedie model or a continuous origin term), build a custom
    ``DevelopmentML`` pipeline and use ``PatsyFormula`` as the preprocessing
    step with the same formula syntax.

    .. testcode::

        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline
        from chainladder.utils.utility_functions import PatsyFormula

        genins = cl.load_sample("genins")
        col = genins.columns[0]
        dev_only = cl.DevelopmentML(
            Pipeline(
                [
                    ("design_matrix", PatsyFormula("C(development)")),
                    ("model", LinearRegression(fit_intercept=False)),
                ]
            ),
            y_ml=col,
            fit_incrementals=False,
        ).fit(genins)
        print(dev_only.ldf_.values[0, 0, 0, :].round(4))

    .. testoutput::

        [3.515  1.735  1.3993 1.152  1.0988 1.0926 1.0332 1.0245 0.8507]

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


def model_diagnostics(
        model:Triangle|MethodBase|Pipeline, 
        name:str|None=None, 
        groupby:str|list(str)|None=None) -> Triangle:
    """A helper function that summarizes various vectors of an
    IBNR model as columns of a Triangle

    Parameters
    ----------
    model: Triangle|MethodBase|Pipeline
        A predicted Triangle, chainladder IBNR estimator or Pipeline
    name: str, optional (default=None)
        An alias to give the model. This will be added to the index of the return
        triangle.
    groupby:
        The index level at which the model should be summarized

    Returns
    -------
    Triangle with relevant figures as columns, including 
    - ``Latest``: Cumulative value at the latest valuation date, equivalent to ``latest_diagonal``
    - ``Month/Quarter/Year Incremental``: Actual emergence between the latest valuation and the one prior valuation date
    - ``LDF``: Age-to-age loss development factor to the next development/valuation period (from ``ldf_``); ignored if ``groupby`` is supplied
    - ``CDF``: Cumulative loss development factor from current age to ultimate (from ``cdf_``); ignored if ``groupby`` is supplied
    - ``Ultimate``: Projected ultimate loss from the fitted IBNR model (``ultimate_``)
    - ``IBNR``: Ultimate - Latest
    - ``Run Off 1/2/3...``: Expected incremental emergence in successive future valuation periods (from ``full_expectation_``)
    - ``Apriori``: Expected ultimate for Benktander family of methods (from ``expectation_``)

    Columns from the original Triangle are cross-joined into the index. ``Measure`` will contain all the columns from the original Triangle. 
    """
    from chainladder import Pipeline, Triangle

    if isinstance(model, Pipeline):
        obj = copy.deepcopy(model.steps[-1][-1])
    else:
        obj = copy.deepcopy(model)
    if not (hasattr(obj,"ultimate_") & hasattr(obj,"ibnr_") & hasattr(obj,"ldf_")):
        raise ValueError("model does not have ultimate_/ibnr_/ldf_")
    if isinstance(model, Triangle):
        obj.X_ = obj
    if groupby is not None:
        obj.X_ = obj.X_.groupby(groupby).sum().cum_to_incr()
        obj.ultimate_ = obj.ultimate_.groupby(groupby).sum()
        if hasattr(obj, "expectation_"):
            obj.expectation_ = obj.expectation_.groupby(groupby).sum()
    obj.X_ = obj.X_.cum_to_incr()
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
        if groupby is None:
            out["LDF"] = obj.ldf_.align_pattern(obj.X_.incr_to_cum(),sample_weight = obj.ultimate_[col])[col]
            out["CDF"] = obj.cdf_.align_pattern(obj.X_.incr_to_cum(),sample_weight = obj.ultimate_[col])[col]
        out["Ultimate"] = obj.ultimate_[col]
        out["IBNR"] = out["Ultimate"] - out["Latest"]
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


def PTF_formula(
    alpha: list = None, gamma: list = None, iota: list = None, dgrain: int = 12
):
    """Helper formula that builds a patsy formula string for the BarnettZehnwirth
    estimator.  Each axis's parameters can be grouped together. Groups of origin
    parameters (alpha) are set equal, and are specified by the first period in each bin.
    Groups of development (gamma) and valuation (iota) parameters are fit to
    separate linear trends, specified a list denoting the endpoints of the linear pieces.
    In other words, development and valuation trends are fit to a piecewise linear model.
    A triangle must be supplied to provide some critical information.
    """
    formula_parts = []
    if alpha:
        # The intercept term takes the place of the first alpha
        for ind, a in enumerate(alpha):
            if a == 0:
                alpha = alpha[:ind] + alpha[(ind + 1) :]
        formula_parts += ["+".join([f"I({x} <= origin)" for x in alpha])]
    if gamma:
        # preprocess gamma to align with grain
        graingamma = [(i + 1) * dgrain for i in gamma]
        for ind in range(1, len(graingamma)):
            formula_parts += [
                "+".join(
                    [
                        f"I((np.minimum({graingamma[ind]},development) - np.minimum({graingamma[ind-1]},development))/{dgrain})"
                    ]
                )
            ]
    if iota:
        for ind in range(1, len(iota)):
            formula_parts += [
                "+".join(
                    [
                        f"I(np.minimum({iota[ind]},valuation) - np.minimum({iota[ind-1]},valuation))"
                    ]
                )
            ]
    if formula_parts:
        return "+".join(formula_parts)
    return ""


def date_delta_adjustment(date: str) -> str:
    """
    Subtracts the default pandas datetime delta from a date in "YYYY-MM-DD" string format.

    Parameters
    ----------
    date: str
        A date in "YYYY-MM-DD" format.

    Returns
    -------
    The original date, minus one unit of the default precision level of pandas, e.g., nanosecond for Pandas 2
    or microsecond for Pandas 3.

    Examples
    --------

    .. testcode::
        :options: +SKIP

        import pandas as pd

        print(date_delta_adjustment("2025-11-01"))

    If using Pandas 2:

    .. testoutput::

        '2025-10-31 23:59:59.999999999'

    If using Pandas 3:

    .. testoutput::

        '2025-10-31 23:59:59.999999'
    """

    res: str = str(pd.Timestamp(date) - pd.Timedelta(1, unit=__dt64_unit__))

    return res
