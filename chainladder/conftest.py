import chainladder as cl
import pytest
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, r

pandas2ri.activate()
CL = importr('ChainLadder')
ATOL = 1e-5

# 'mcl'
# 'auto':'auto',
# 'liab':'liab',
# 'quarterly':'quarterly',
# 'USAA':'USAA'


@pytest.fixture
def datasets():
    datasets = {'ABC': 'ABC',
                'GenIns': 'GenIns',
                'M3IR5': 'M3IR5',
                'Mortgage': 'Mortgage',
                'MW2008': 'MW2008',
                'MW2014': 'MW2014',
                'RAA': 'RAA',
                'UKMotor': 'UKMotor'}
    return datasets


@pytest.fixture
def atol():
    return 1e-5


def mack_r(alpha):
    r_dict = dict(datasets())
    for k, v in r_dict.items():
        r_dict[k] = r(f'mack<-MackChainLadder({v},alpha={alpha})')
    return r_dict


def mack_p(avg_type):
    p_dict = dict(datasets())
    for k, v in p_dict.items():
        p_dict[k] = cl.Development(avg_type=avg_type).fit(cl.load_dataset(v))
    return p_dict


@pytest.fixture
def mack_r_simple():
    return mack_r(0)


@pytest.fixture
def mack_p_simple():
    return mack_p('simple')

@pytest.fixture
def mack_r_volume():
    return mack_r(1)


@pytest.fixture
def mack_p_volume():
    return mack_p('volume')


@pytest.fixture
def mack_r_reg():
    return mack_r(2)


@pytest.fixture
def mack_p_reg():
    return mack_p('regression')
