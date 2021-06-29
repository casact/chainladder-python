""" utils should store all utility functions and classes, i.e. things that
    are used by various modules in the package.
"""
from sklearn.utils import deprecated
from chainladder.utils.weighted_regression import (
    WeightedRegression,
)  # noqa (API import)
from xlcompose import (
    DataFrame,
    Series,
    Row,
    Column,
    Tabs,
    CSpacer,
    RSpacer,
    Title,
    Image,
    VSpacer,
    HSpacer,
    Sheet,
    load_yaml,
)
import os

@deprecated('The load_template class is deprecated and will be removed in a future version of chainladder. Use xlcompose library directly for the same functionality')
def load_template(template, env=None, **kwargs):
    path = os.path.dirname(os.path.abspath(__file__))
    try:
        return load_yaml(template, env, **kwargs)
    except:
        template = os.path.join(path, "templates", template.lower() + ".yaml")
        return load_yaml(template, env, **kwargs)

DatFrame = deprecated('The DataFrame class is deprecated and will be removed in a future version of chainladder. Use xlcompose library directly for the same functionality')(DataFrame)
Series = deprecated('The Series class is deprecated and will be removed in a future version of chainladder. Use xlcompose library directly for the same functionality')(Series)
Row = deprecated('The Row class is deprecated and will be removed in a future version of chainladder. Use xlcompose library directly for the same functionality')(Row)
Column = deprecated('The Column class is deprecated and will be removed in a future version of chainladder. Use xlcompose library directly for the same functionality')(Column)
Tabs = deprecated('The Tabs class is deprecated and will be removed in a future version of chainladder. Use xlcompose library directly for the same functionality')(Tabs)
CSpacer = deprecated('The CSpacer class is deprecated and will be removed in a future version of chainladder. Use xlcompose library directly for the same functionality')(CSpacer)
RSpacer = deprecated('The RSpacer class is deprecated and will be removed in a future version of chainladder. Use xlcompose library directly for the same functionality')(RSpacer)
Title = deprecated('The Title class is deprecated and will be removed in a future version of chainladder. Use xlcompose library directly for the same functionality')(Title)
Image = deprecated('The Image class is deprecated and will be removed in a future version of chainladder. Use xlcompose library directly for the same functionality')(Image)
VSpacer = deprecated('The VSpacer class is deprecated and will be removed in a future version of chainladder. Use xlcompose library directly for the same functionality')(VSpacer)
HSpacer = deprecated('The HSpacer class is deprecated and will be removed in a future version of chainladder. Use xlcompose library directly for the same functionality')(HSpacer)
Sheet = deprecated('The Sheet class is deprecated and will be removed in a future version of chainladder. Use xlcompose library directly for the same functionality')(Sheet)
load_template = (load_template)

from chainladder.utils.utility_functions import (  # noqa (API import)
    load_dataset,
    parallelogram_olf,
    read_pickle,
    read_json,
    concat,
    load_sample,
    minimum,
    maximum,
    PatsyFormula,
    model_diagnostics
)
from chainladder.utils.cupy import cp
from chainladder.utils.sparse import sp
from chainladder.utils.dask import dp
