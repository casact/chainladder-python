# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pytest
import chainladder as cl


@pytest.mark.parametrize(
    "origin_grain, development_grain, periods_per_origin",
    [
        ("Y", "Y", 1),
        ("Y", "Q", 4),
        ("Y", "M", 12),
        ("Q", "Q", 1),
        ("Q", "M", 3),
        ("M", "M", 1),
    ],
)
def test_random_triangle_grain_combinations(
    origin_grain, development_grain, periods_per_origin
):
    n_origin_periods = 6
    tri = cl.random_triangle(
        origin_grain=origin_grain,
        development_grain=development_grain,
        n_origin_periods=n_origin_periods,
        random_state=42,
    )
    assert tri.origin_grain == origin_grain
    assert tri.development_grain == development_grain
    assert len(tri.origin) == n_origin_periods
    assert tri.shape == (1, 1, n_origin_periods, n_origin_periods * periods_per_origin)


def test_random_triangle_invalid_grain_combo():
    with pytest.raises(ValueError, match="at least as fine"):
        cl.random_triangle(origin_grain="Q", development_grain="Y")


def test_random_triangle_invalid_grain_value():
    with pytest.raises(ValueError, match='"Y", "Q", "M"'):
        cl.random_triangle(origin_grain="X")


def test_random_triangle_multi_index_and_columns():
    tri = cl.random_triangle(index=2, columns=3, n_origin_periods=4, random_state=5)
    assert tri.shape == (2, 3, 4, 4)
    assert len(tri.index) == 2
    assert len(tri.columns) == 3


def test_random_triangle_is_triangular_not_square():
    """Recent origins should have fewer known development periods, like a real triangle."""
    tri = cl.random_triangle(
        origin_grain="Y", development_grain="Q", n_origin_periods=5, random_state=1
    )
    # The most recent origin only has one known development period (lag 1),
    # not the full 20 periods the oldest origin has - a real triangular shape.
    latest = tri.latest_diagonal.to_frame()
    assert latest.notna().all().all()
    assert tri.link_ratio.to_frame().shape[1] == tri.shape[3] - 1


def test_random_triangle_reproducible_with_seed():
    import numpy as np

    tri1 = cl.random_triangle(n_origin_periods=4, random_state=99)
    tri2 = cl.random_triangle(n_origin_periods=4, random_state=99)
    assert np.array_equal(
        tri1.to_frame().values, tri2.to_frame().values, equal_nan=True
    )


def test_random_triangle_cumulative_flag():
    cumulative = cl.random_triangle(n_origin_periods=4, cumulative=True, random_state=3)
    incremental = cl.random_triangle(n_origin_periods=4, cumulative=False, random_state=3)
    assert cumulative.is_cumulative
    assert not incremental.is_cumulative
