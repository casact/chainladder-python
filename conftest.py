from __future__ import annotations

import chainladder as cl
import functools
import pytest

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from chainladder import Triangle
    from typing import (
        Any,
        Callable,
    )


FIXTURES = {
    "raa": {
        "data": "raa",
        "runs": ["normal_run", "sparse_only_run"],
    },
    "qtr": {
        "data": "quarterly",
        "runs": ["normal_run", "sparse_only_run"],
    },
    "clrd": {
        "data": "clrd",
        "runs": ["normal_run", "sparse_only_run"],
    },
    "genins": {
        "data": "genins",
        "runs": ["normal_run", "sparse_only_run"],
    },
    "monthly": {
        "data": "prism",
        "runs": ["normal_run", "sparse_only_run"],
        "transform": (lambda t: t.sum()),
    },
    "prism": {
        "data": "prism",
        "runs": ["sparse_only_run"],
    },
    "tail_sample": {
        "data": "tail_sample",
        "runs": ["normal_run", "sparse_only_run"],
    },
    "xyz": {
        "data": "xyz",
        "runs": ["normal_run", "sparse_only_run"],
    },
}


def pytest_generate_tests(metafunc):
    for x in metafunc.fixturenames:
        if x in FIXTURES.keys():
            metafunc.parametrize(x, FIXTURES[x]["runs"], indirect=True)


@functools.lru_cache(maxsize=None)
def _cached_load_sample(sample: str) -> Triangle:
    """
    Create a cache of the requested sample Triangle when called initially,
    then load the cache when called again.

    The cache is preserved throughout the test session.

    Parameters
    ----------
    sample: str
        The requested triangle, e.g., "clrd", "raa", etc.

    Returns
    -------
    Triangle
        The cached triangle.
    """
    return cl.load_sample(sample)


def make_fixture(
    sample: str,
    transform: Callable[[Triangle], Triangle] | None = None,
) -> Callable:
    """
    Common template fixture for using sample data in unit tests.

    Parameters
    ----------
    sample: str
        The name of the sample data set to be loaded, e.g., raa, clrd, etc.
    transform: Callable[[Triangle], Triangle] | None
        An optional transformation to be applied to the triangle supplied as a lambda function.

    Yields
    -------
    A Triangle, with backend set according to request.param.

    """

    @pytest.fixture
    def _sample_fixture(request: Any) -> Iterator[Triangle]:
        # Load a copy of cached sample data.
        tri = _cached_load_sample(sample).copy()
        # Apply a transformation if supplied
        tri = transform(tri) if transform else tri
        # Set the backend to sparse for a sparse-only-run, then yield the triangle to the test.
        yield tri.set_backend(
            "sparse" if request.param == "sparse_only_run" else "numpy"
        )

    return _sample_fixture


for x, v in FIXTURES.items():
    globals()[x] = make_fixture(v["data"], v.get("transform"))


@pytest.fixture
def atol():
    return 1e-4


@pytest.fixture
def empty_triangle():
    return cl.Triangle()
