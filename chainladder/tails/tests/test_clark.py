from __future__ import annotations

import numpy as np
import chainladder as cl
import pytest

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chainladder import Triangle


def test_truncation_age(genins: Triangle, atol: float) -> None:
    """
    Validate that sufficiently distant truncation age is equivalent to
    not truncating
    """
    long_truncation = cl.TailClark(truncation_age=99999).fit(
        cl.ClarkLDF().fit_transform(genins)
    ).cdf_
    no_truncation = cl.TailClark().fit(cl.ClarkLDF().fit_transform(genins)).cdf_
    assert np.allclose(
        long_truncation.values,
        no_truncation.values,
        atol=atol
    )
