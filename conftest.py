import pytest
import chainladder as cl

def pytest_generate_tests(metafunc):
    if "raa" in metafunc.fixturenames:
        metafunc.parametrize(
            "raa", ["normal_run", "sparse_only_run"], indirect=True)
    if "qtr" in metafunc.fixturenames:
        metafunc.parametrize(
                "qtr", ["normal_run", "sparse_only_run"], indirect=True)
    if "clrd" in metafunc.fixturenames:
        metafunc.parametrize(
                "clrd", ["normal_run", "sparse_only_run"], indirect=True)
    if "genins" in metafunc.fixturenames:
        metafunc.parametrize(
                "genins", ["normal_run", "sparse_only_run"], indirect=True)
    if "prism_dense" in metafunc.fixturenames:
        metafunc.parametrize(
                "prism_dense", ["normal_run", "sparse_only_run"], indirect=True)
    if "prism" in metafunc.fixturenames:
        metafunc.parametrize("prism", ["normal_run"], indirect=True)


@pytest.fixture
def raa(request):
    if request.param == "sparse_only_run":
        cl.options.set_option('ARRAY_BACKEND', 'sparse')
    else:
        cl.options.set_option('ARRAY_BACKEND', 'numpy')
    return cl.load_sample('raa')


@pytest.fixture
def qtr(request):
    if request.param == "sparse_only_run":
        cl.options.set_option('ARRAY_BACKEND', 'sparse')
    else:
        cl.options.set_option('ARRAY_BACKEND', 'numpy')
    return cl.load_sample('quarterly')


@pytest.fixture
def clrd(request):
    if request.param == "sparse_only_run":
        cl.options.set_option('ARRAY_BACKEND', 'sparse')
    else:
        cl.options.set_option('ARRAY_BACKEND', 'numpy')
    return cl.load_sample('clrd')


@pytest.fixture
def genins(request):
    if request.param == "sparse_only_run":
        cl.options.set_option('ARRAY_BACKEND', 'sparse')
    else:
        cl.options.set_option('ARRAY_BACKEND', 'numpy')
    return cl.load_sample('genins')


@pytest.fixture
def prism(request):
    cl.options.set_option('ARRAY_BACKEND', 'numpy')
    return cl.load_sample('prism')


@pytest.fixture
def prism_dense(request):
    if request.param == "sparse_only_run":
        cl.options.set_option('ARRAY_BACKEND', 'numpy')
    else:
        cl.options.set_option('ARRAY_BACKEND', 'numpy')
    return cl.load_sample('prism').sum()


@pytest.fixture
def atol(): return 1e-4
