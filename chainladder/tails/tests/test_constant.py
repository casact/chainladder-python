import chainladder as cl
import numpy as np


def test_constant_balances():
    raa = cl.load_dataset('quarterly')
    assert np.prod(cl.TailConstant(1.05, decay=0.8)
                     .fit(raa).ldf_.iloc[0, 1].values[0, 0, 0, -5:]) == 1.05
