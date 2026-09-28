from __future__ import annotations

import chainladder as cl

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chainladder import Triangle


def test_bondy1(tail_sample: Triangle) -> None:
    tri = tail_sample["paid"]
    dev = cl.Development(average="simple").fit_transform(tri)
    assert (
        round(float(cl.TailBondy(earliest_age=12).fit(dev).cdf_.values[0, 0, 0, -2]), 3)
        == 1.028
    )
