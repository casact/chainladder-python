from __future__ import annotations

import numpy as np
import chainladder as cl

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chainladder import Triangle


def test_truncation_age(genins: Triangle, atol: float) -> None:
    """
    Validate that sufficiently distant truncation age is equivalent to
    not truncating
    """
    long_truncation = (
        cl.TailClark(truncation_age=99999).fit(cl.ClarkLDF().fit_transform(genins)).cdf_
    )
    no_truncation = cl.TailClark().fit(cl.ClarkLDF().fit_transform(genins)).cdf_
    assert np.allclose(
        long_truncation.values[..., -1], no_truncation.values[..., -1], atol=atol
    )


def test_clark_g_list_of_ages(raa: Triangle) -> None:
    """
    Validate that passing a Python list of ages to G_ works without length mismatch.
    """
    model = cl.ClarkLDF().fit(raa)
    g = model.G_([12, 24, 36])
    assert g.shape[-1] == 3
    assert list(g.development) == [12.0, 24.0, 36.0]
