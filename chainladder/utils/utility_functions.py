# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pandas as pd
import numpy as np
from chainladder.utils.cupy import cp
from chainladder.utils.sparse import sp
from scipy.sparse import coo_matrix
import joblib
import dill
import json
import os
import copy
from sklearn.utils import deprecated
from patsy import dmatrix
from sklearn.base import BaseEstimator, TransformerMixin


def load_sample(key, *args, **kwargs):
    """ Function to load datasets included in the chainladder package.

        Parameters
        ----------
        key: str
            The name of the dataset, e.g. RAA, ABC, UKMotor, GenIns, etc.

        Returns
        --------
            pandas.DataFrame of the loaded dataset.

    """
    from chainladder import Triangle

    path = os.path.dirname(os.path.abspath(__file__))
    origin = "origin"
    development = "development"
    columns = ["values"]
    index = None
    cumulative = True
    if key.lower() in ["mcl", "usaa", "quarterly", "auto", "usauto", "tail_sample"]:
        columns = ["incurred", "paid"]
    if key.lower() == "clrd":
        origin = "AccidentYear"
        development = "DevelopmentYear"
        index = ["GRNAME", "LOB"]
        columns = [
            "IncurLoss",
            "CumPaidLoss",
            "BulkLoss",
            "EarnedPremDIR",
            "EarnedPremCeded",
            "EarnedPremNet",
        ]
    if key.lower() == "berqsherm":
        origin = "AccidentYear"
        development = "DevelopmentYear"
        index = ["LOB"]
        columns = ["Incurred", "Paid", "Reported", "Closed"]
    if key.lower() == "xyz":
        origin = "AccidentYear"
        development = "DevelopmentYear"
        columns = ["Incurred", "Paid", "Reported", "Closed", "Premium"]
    if key.lower() in ["liab", "auto"]:
        index = ["lob"]
    if key.lower() in ["cc_sample", "ia_sample"]:
        columns = ["loss", "exposure"]
    if key.lower() in ["prism"]:
        columns = ["reportedCount", "closedPaidCount", "Paid", "Incurred"]
        index = ["ClaimNo", "Line", "Type", "ClaimLiability", "Limit", "Deductible"]
        origin = "AccidentDate"
        development = "PaymentDate"
        cumulative = False
    df = pd.read_csv(os.path.join(path, "data", key.lower() + ".csv"))
    return Triangle(
        df,
        origin=origin,
        development=development,
        index=index,
        columns=columns,
        cumulative=cumulative,
        *args,
        **kwargs
    )


@deprecated("Use load_sample instead.")
def load_dataset(key, *args, **kwargs):
    return load_sample(key, *args, **kwargs)


def read_pickle(path):
    with open(path, "rb") as pkl:
        return dill.load(pkl)


def read_json(json_str, array_backend=None):
    from chainladder import Triangle
    from chainladder.workflow import Pipeline

    if array_backend is None:
        from chainladder import ARRAY_BACKEND

        array_backend = ARRAY_BACKEND
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
        y = pd.read_json(j["data"], orient="split", date_unit="ns")
        y["origin"] = pd.to_datetime(y["origin"])
        y.columns= [
            c if c != 'valuation' else 'development' for c in y.columns]
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
        )
        if json.loads(j["metadata"])["is_val_tri"]:
            tri = tri.dev_to_val()
        if json.loads(j["metadata"])["is_cumulative"]:
            tri = tri.incr_to_cum()
        tri = tri[json.loads(j["metadata"])["columns"]].sort_index()
        if "sub_tris" in json_dict.keys():
            for k, v in json_dict["sub_tris"].items():
                setattr(tri, k, read_json(v, array_backend))
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
    values, date, start_date=None, end_date=None, grain="M", vertical_line=False
):
    """ Parallelogram approach to on-leveling.

    Ar
    """
    date = pd.to_datetime(date)
    if not start_date:
        start_date = "{}-01-01".format(date.min().year)
    if not end_date:
        end_date = "{}-12-31".format(date.max().year)
    start_date = pd.to_datetime(start_date) - pd.tseries.offsets.DateOffset(days=1)
    date_idx = pd.date_range(
        start_date - pd.tseries.offsets.DateOffset(years=1), end_date
    )
    y = pd.Series(np.array(values), np.array(date))
    y = y.reindex(date_idx, fill_value=0)
    idx = np.cumprod(y.values + 1)
    idx = idx[-1] / idx
    y = pd.Series(idx, y.index)
    y = y[~((y.index.day == 29) & (y.index.month == 2))]
    if not vertical_line:
        y = y.rolling(365).mean()
        y = (y + y.shift(1).values) / 2
    y = y.iloc[366:]
    y = y.groupby(y.index.to_period(grain)).mean().reset_index()
    y.columns = ["Origin", "OLF"]
    y["Origin"] = y["Origin"].astype(str)
    return y.set_index("Origin")


def set_common_backend(objs):
    from chainladder import ARRAY_PRIORITY as priority

    backend = priority[np.min([priority.index(i.array_backend) for i in objs])]
    return [i.set_backend(backend) for i in objs]


def concat(objs, axis, ignore_index=False, sort=False):
    """ Concatenate Triangle objects along a particular axis.

    Parameters
    ----------
    objs : list or tuple
        A list or tuple of Triangle objects to concat. All non-concat axes must
        be identical and all elements of the concat axes must be unique.
    axis : string or int
        The axis to concatenate along.
    ignore_index : bool, default False
        If True, do not use the index values along the concatenation axis. The
        resulting axis will be labeled 0, …, n - 1. This is useful if you are
        concatenating objects where the concatenation axis does not have
        meaningful indexing information. Note the index values on the other
        axes are still respected in the join.

    Returns
    -------
    Updated triangle
    """
    xp = objs[0].get_array_module()
    axis = objs[0]._get_axis(axis)

    if axis != 1:
        all_columns = []
        for obj in objs:
            for col in obj.columns:
                if col not in all_columns:
                    all_columns.append(col)
        for obj in objs:
            for col in all_columns:
                if col not in obj.columns:
                    obj[col] = xp.nan
    objs = set_common_backend(objs)
    mapper = {0: "kdims", 1: "vdims", 2: "odims", 3: "ddims"}
    for k, v in mapper.items():
        if k != axis and k !=1:  # All non-concat axes must be identical
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


def num_to_nan(arr):
    """ Function that turns all zeros to nan values in an array """
    backend = arr.__class__.__module__.split(".")[0]
    if backend == "sparse":
        if arr.fill_value == 0 or sp.isnan(arr.fill_value):
            arr.coords = arr.coords[:, arr.data != 0]
            arr.data = arr.data[arr.data != 0]
            arr = sp(coords=arr.coords, data=arr.data, fill_value=sp.nan, shape=arr.shape)
        else:
            arr = sp(num_to_nan(np.nan_to_num(arr.todense())), fill_value=sp.nan)
    else:
        nan = np.nan if backend == "numpy" else cp.nan
        arr[arr == 0] = nan
    return arr


def minimum(x1, x2):
    return x1.minimum(x2)


def maximum(x1, x2):
    return x1.maximum(x2)

class PatsyFormula(BaseEstimator, TransformerMixin):
    """ A sklearn-style Transformer for patsy formulas.

    PatsyFormula allows for R-style formula preprocessing of the ``design_matrix``
    of a machine learning algorithm. It's particularly useful with the `DevelopmentML`
    and `TweedieGLM` estimators.

    Parameters
    -----------

    formula : str
        A string representation of the regression model X features.

    Attributes
    ------------
    design_info_ :
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


def model_diagnostics(model, name=None,  groupby=None):
    """ A helper function that summarizes various vectors of an
    IBNR model as columns of a Triangle

    Parameters
    ----------
    model :
        A chainladder IBNR estimator or Pipeline
    name :
        An alias to give the model. This will be added to the index of the return
        triangle.
    groupby :
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
        if hasattr(obj, 'expectation_'):
            obj.expectation_ = obj.expectation_.groupby(groupby).sum()
    else:
        obj.X_ = obj.X_.incr_to_cum()
    val = obj.X_.valuation
    latest = obj.X_.sum('development')
    run_off = obj.full_expectation_.iloc[..., :-1].dev_to_val().cum_to_incr()
    run_off = run_off[run_off.development>str(obj.X_.valuation_date)]
    run_off = run_off.iloc[..., :{'M': 12, 'Q': 4, 'Y': 1}[obj.X_.development_grain]]

    triangles = []
    for col in obj.ultimate_.columns:
        idx = latest.index
        idx['Measure'] = col
        idx['Model'] = obj.__class__.__name__ if name is None else name
        idx = idx[list(idx.columns[-2:]) + list(idx.columns[:-2])]
        out = latest[col].rename('columns', ['Latest'])
        if obj.X_.development_grain in ['M']:
            out['Month Incremental'] = obj.X_[col][val==obj.X_.valuation_date].sum('development')
        if obj.X_.development_grain in ['M', 'Q']:
            out['Quarter Incremental'] = (obj.X_ - obj.X_[val < pd.Timestamp(np.sort(val[val<=obj.X_.valuation_date].unique())[-3])]).sum('development')[col]
        out['Year Incremental'] = (obj.X_ - obj.X_[val<str(obj.X_.valuation_date.year)]).sum('development')[col]
        out['IBNR'] = obj.ibnr_[col]
        out['Ultimate'] = obj.ultimate_[col]
        for i in range(run_off.shape[-1]):
            out['Run Off ' + str(i+1)] = run_off[col].iloc[..., i]
        if hasattr(obj, 'expectation_'):
            out['Apriori'] = obj.expectation_ if obj.expectation_.shape[1] == 1 else obj.expectation_[col]
        out.index = idx
        triangles.append(out)
    return concat(triangles,0)
