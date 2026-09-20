from __future__ import annotations

import chainladder as cl

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chainladder import Triangle


class TestFullTri:
    """Test weight generation on full triangles"""

    def test_triangleweight_full_triangle(self, raa: Triangle) -> None:
        """
        Testing new path that allows weights on full triangles
        """
        ult = cl.Chainladder().fit(raa)
        tw = cl.TriangleWeight(n_periods=4).fit(raa)
        tw_full = cl.TriangleWeight(n_periods=4).fit(ult.full_triangle_)
        assert tw.w_.iloc[:, :, :, 0] == tw_full.w_.iloc[:, :, :, 0]

    def test_triangleweight_full_irregular_triangle(self) -> None:
        """
        Testing unequal grains
        """
        prism = cl.load_sample("prism_oydq")["Paid"]
        ult = cl.Chainladder().fit(prism)
        tw = cl.TriangleWeight(n_periods=4).fit(prism)
        tw_full = cl.TriangleWeight(n_periods=4).fit(ult.full_triangle_)
        assert tw.w_.iloc[:, :, :, 0] == tw_full.w_.iloc[:, :, :, 0]
