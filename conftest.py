import pytest
import chainladder as cl

@pytest.fixture
def raa(request):
    return cl.load_sample('raa')


@pytest.fixture
def qtr(request):
    return cl.load_sample('quarterly')


@pytest.fixture
def clrd(request):
    return cl.load_sample('clrd')


@pytest.fixture
def genins(request):
    return cl.load_sample('genins')


@pytest.fixture
def prism(request):
    return cl.load_sample('prism')


@pytest.fixture
def prism_dense(request):
    return cl.load_sample('prism').sum()


@pytest.fixture
def atol(): return 1e-4
