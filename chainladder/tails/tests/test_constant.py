import chainladder as cl
import numpy as np
from chainladder.utils.cupy import cp


def test_constant_balances(qtr):
    xp = qtr.get_array_module()
    assert (
        round(
            float(
                xp.prod(
                    cl.TailConstant(1.05, decay=0.8)
                    .fit(qtr)
                    .ldf_.iloc[0, 1]
                    .values[0, 0, 0, -5:]
                )
            ),
            3,
        )
        == 1.050
    )
