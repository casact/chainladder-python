import pytest
import chainladder as cl


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


@pytest.fixture
def raa(request):
    if request.param == "sparse_only_run":
        cl.options.set_option("ARRAY_BACKEND", "sparse")
    else:
        cl.options.set_option("ARRAY_BACKEND", "numpy")
    yield cl.load_sample("raa")
    cl.options.set_option("ARRAY_BACKEND", "numpy")


@pytest.fixture
def qtr(request):
    if request.param == "sparse_only_run":
        cl.options.set_option("ARRAY_BACKEND", "sparse")
    else:
        cl.options.set_option("ARRAY_BACKEND", "numpy")
    yield cl.load_sample("quarterly")
    cl.options.set_option("ARRAY_BACKEND", "numpy")


@pytest.fixture
def clrd(request):
    if request.param == "sparse_only_run":
        cl.options.set_option("ARRAY_BACKEND", "sparse")
    else:
        cl.options.set_option("ARRAY_BACKEND", "numpy")
    yield cl.load_sample("clrd")
    cl.options.set_option("ARRAY_BACKEND", "numpy")


@pytest.fixture
def genins(request):
    if request.param == "sparse_only_run":
        cl.options.set_option("ARRAY_BACKEND", "sparse")
    else:
        cl.options.set_option("ARRAY_BACKEND", "numpy")
    yield cl.load_sample("genins")
    cl.options.set_option("ARRAY_BACKEND", "numpy")


@pytest.fixture
def prism(request):
    cl.options.set_option("ARRAY_BACKEND", "numpy")
    yield cl.load_sample("prism")
    cl.options.set_option("ARRAY_BACKEND", "numpy")


@pytest.fixture
def prism_dense(request):
    if request.param == "sparse_only_run":
        cl.options.set_option("ARRAY_BACKEND", "sparse")
    else:
        cl.options.set_option("ARRAY_BACKEND", "numpy")
    yield cl.load_sample("prism").sum()
    cl.options.set_option("ARRAY_BACKEND", "numpy")


@pytest.fixture
def xyz(request):
    if request.param == "sparse_only_run":
        cl.options.set_option("ARRAY_BACKEND", "sparse")
    else:
        cl.options.set_option("ARRAY_BACKEND", "numpy")
    yield cl.load_sample("xyz")
    cl.options.set_option("ARRAY_BACKEND", "numpy")


@pytest.fixture
def atol():
    return 1e-4

@pytest.fixture
def empty_triangle():
    return cl.Triangle()
