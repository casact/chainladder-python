from __future__ import annotations

import pytest
import chainladder as cl

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from chainladder import Triangle
    from typing import (
        Any,
        Callable
    )


def pytest_generate_tests(metafunc):
    if "raa" in metafunc.fixturenames:
        metafunc.parametrize("raa", ["normal_run", "sparse_only_run"], indirect=True)
    if "qtr" in metafunc.fixturenames:
        metafunc.parametrize("qtr", ["normal_run", "sparse_only_run"], indirect=True)
    if "clrd" in metafunc.fixturenames:
        metafunc.parametrize("clrd", ["normal_run", "sparse_only_run"], indirect=True)
    if "genins" in metafunc.fixturenames:
        metafunc.parametrize("genins", ["normal_run", "sparse_only_run"], indirect=True)
    if "prism_dense" in metafunc.fixturenames:
        metafunc.parametrize(
            "prism_dense", ["normal_run", "sparse_only_run"], indirect=True
        )
    if "prism" in metafunc.fixturenames:
        metafunc.parametrize("prism", ["normal_run"], indirect=True)
    if "xyz" in metafunc.fixturenames:
        metafunc.parametrize("xyz", ["normal_run", "sparse_only_run"], indirect=True)


def _sample_fixture(
        request: Any,
        sample: str,
        transform: Callable[[Triangle], Triangle] | None = None
    ) -> Iterator[Triangle]:
    """
    Common template fixture for using sample data in unit tests.

    Parameters
    ----------
    request:Any
        The pytest request built-in.
    sample: str
        The name of the sample data set to be loaded, e.g., raa, clrd, etc.
    transform: Callable[[Triangle], Triangle] | None
        An optional transformation to be applied to the triangle supplied as a lambda function.

    Yields
    -------
    A Triangle, with backend set according to request.param.

    """

    # Set the backend to sparse for a sparse-only-run.
    cl.options.set_option("ARRAY_BACKEND", "sparse" if request.param == "sparse_only_run" else "numpy")
    # Load the sample data.
    tri = cl.load_sample(sample)
    # Apply a transformation if supplied, then yield the triangle to the test.
    yield transform(tri) if transform else tri
    # After the test, reset the backend to default numpy.
    cl.options.set_option("ARRAY_BACKEND", "numpy")


@pytest.fixture
def raa(request):
    yield from _sample_fixture(request, "raa")


@pytest.fixture
def qtr(request):
    yield from _sample_fixture(request, "quarterly")


@pytest.fixture
def clrd(request):
    yield from _sample_fixture(request, "clrd")


@pytest.fixture
def genins(request):
    yield from _sample_fixture(request, "genins")


@pytest.fixture
def prism(request):
    yield from _sample_fixture(request, "prism")


@pytest.fixture
def prism_dense(request):
    yield from _sample_fixture(request, "prism", transform=lambda t: t.sum())


@pytest.fixture
def xyz(request):
    yield from _sample_fixture(request, "xyz")


@pytest.fixture
def atol():
    return 1e-4

@pytest.fixture
def empty_triangle():
    return cl.Triangle()
