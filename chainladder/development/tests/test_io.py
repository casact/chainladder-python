"""Tests for saving and loading development estimators with JSON."""

import json

import pytest

import chainladder as cl


@pytest.mark.parametrize(
    "params, name",
    [({"average": "bogus"}, "average"), ({"n_periods": -5}, "n_periods")],
)
def test_read_json_rejects_invalid_params(params, name):
    """Check that ``read_json`` checks the parameter values it loads (#1569)."""
    json_str = json.dumps({"params": params, "__class__": "Development"})
    with pytest.raises(ValueError, match=name):
        cl.read_json(json_str)
    with pytest.raises(ValueError, match=name):
        cl.read_json(json.dumps([{"name": "dev", **json.loads(json_str)}]))


@pytest.mark.parametrize("name", ["Trend", "BerquistSherman", "ParallelogramOLF"])
def test_read_json_loads_estimators_without_the_hook(name):
    """
    Check that ``read_json`` still loads estimators that do not define
    ``_check_param_values``.
    """
    estimator = getattr(cl, name)()
    assert cl.read_json(estimator.to_json()).get_params() == estimator.get_params()


def test_read_json_leaves_dict_params_to_fit():
    """
    Check that ``read_json`` loads a dict assumption without a triangle to
    resolve it against, leaving its values to be checked in ``fit``.
    """
    dev = cl.Development(n_periods={12: 2, 24: -1})
    loaded = cl.read_json(dev.to_json())
    assert isinstance(loaded, cl.Development)
    assert list(loaded.n_periods.values()) == [2, -1]
