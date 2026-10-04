from __future__ import annotations

import numpy as np
import chainladder as cl
import dill
import pytest

from pathlib import Path

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chainladder import Triangle


class TestJSON:
    """Test JSON IO"""

    def test_triangle_json_io(self, clrd: Triangle) -> None:
        """
        test JSON round trip for Triangle

        Parameters
        ----------
        clrd: Triangle
            The CLRD sample data set fixture

        Returns
        -------
        None
        """
        clrd2 = cl.read_json(clrd.to_json(), array_backend=clrd.array_backend)
        assert clrd == clrd2
        assert np.all(clrd.kdims == clrd2.kdims)
        assert np.all(clrd.vdims == clrd2.vdims)
        assert np.all(clrd.odims == clrd2.odims)
        assert np.all(clrd.ddims == clrd2.ddims)
        assert np.all(clrd.valuation == clrd2.valuation)

    def test_json_for_val(self, raa: Triangle) -> None:
        """
        test JSON round trip for val Triangle

        Parameters
        ----------
        raa: Triangle
            The CLRD sample data set fixture

        Returns
        -------
        None
        """
        x = raa.dev_to_val().to_json()
        assert cl.read_json(x) == raa.dev_to_val()

    def test_estimator_json_io(self) -> None:
        """
        test JSON round trip for Development estimator

        Returns
        -------
        None
        """
        assert (
            cl.read_json(cl.Development().to_json()).get_params()
            == cl.Development().get_params()
        )

    def test_pipeline_json_io(self) -> None:
        """
        test JSON round trip for an estimator pipeline

        Returns
        -------
        None
        """
        pipe = cl.Pipeline(
            steps=[("dev", cl.Development()), ("model", cl.BornhuetterFerguson())]
        )
        pipe2 = cl.read_json(pipe.to_json())
        assert {
            item[0]: item[1].get_params() for item in pipe.get_params()["steps"]
        } == {item[0]: item[1].get_params() for item in pipe2.get_params()["steps"]}

    def test_json_subtri(self, raa: Triangle, atol: float) -> None:
        """
        test JSON round trip preserves sub-Triangles

        Parameters
        ----------
        raa: Triangle
            The CLRD sample data set fixture

        atol: float
            the absolute tolerance for this test

        Returns
        -------
        None
        """
        a = cl.read_json(cl.Chainladder().fit_predict(raa).to_json()).full_triangle_
        b = cl.Chainladder().fit_predict(raa).full_triangle_
        # since a and b are meant to be identical, (a - b) should be 0 everywhere
        # 0's are turned to NaN's in Triangle arithmetic, resulting in an All-NaN
        # warning when aggregating
        with pytest.warns(RuntimeWarning, match="All-NaN"):
            assert abs(a - b).max().max() < atol

    def test_json_df(self, atol: float) -> None:
        """
        test JSON round trip preserves sub-Triangles from Munich Adjustment

        Parameters
        ----------
        atol: float
            the absolute tolerance for this test

        Returns
        -------
        None
        """
        x = cl.MunichAdjustment(paid_to_incurred=("paid", "incurred")).fit_transform(
            cl.load_sample("mcl")
        )
        assert abs(cl.read_json(x.to_json()).lambda_ - x.lambda_).sum() < atol


class TestPickle:
    """Test JSON IO"""

    def test_to_pickle_read_pickle(self, raa: Triangle) -> None:
        """
        test JSON round trip for val Triangle

        Parameters
        ----------
        raa: Triangle
            The CLRD sample data set fixture

        Returns
        -------
        None
        """
        import tempfile
        import os

        dev = cl.Development(average="simple", n_periods=4).fit(raa)
        fd, path = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)
        try:
            dev.to_pickle(path)
            restored = cl.read_pickle(path)
            assert restored.average == dev.average
            assert restored.n_periods == dev.n_periods
            np.testing.assert_array_almost_equal(restored.ldf_.values, dev.ldf_.values)
        finally:
            os.remove(path)

    def test_read_pickle_triangle(self, raa: Triangle, tmp_path: Path) -> None:
        """
        Create a triangle, dump a pickle of it, and then read it back in. The ingested pickle should result
        in an equal copy of the triangle that was dumped.

        Parameters
        ----------
        raa: Triangle
            The raa sample data set.
        tmp_path: Path
            The builtin pytest tmp_path fixture, provides a temporary path to dump the pickle to.

        Returns
        -------
        None

        """
        pkl_path = tmp_path / "triangle.pkl"
        with open(pkl_path, "wb") as f:
            dill.dump(raa, f)
        assert cl.read_pickle(str(pkl_path)) == raa

    def test_triangle_to_pickle(
        self, raa: Triangle, clrd: Triangle, tmp_path: Path
    ) -> None:
        """
        Dump a pickle of a triangle and read it back in. The read-in triangle should
        equal the one that was dumped.

        Parameters
        ----------
        raa: Triangle
            The raa sample data set Triangle.
        clrd: Triangle
            The clrd sample data set Triangle.
        tmp_path: Path
            The builtin pytest tmp_path fixture, provides a temporary path to dump the pickle to.

        Returns
        -------
        None

        """
        # Single-dimension case.
        raa_path = tmp_path / "raa.pkl"
        raa.to_pickle(str(raa_path))
        assert raa_path.is_file()
        assert cl.read_pickle(str(raa_path)) == raa

        # Multidimensional case.
        clrd_path = tmp_path / "clrd.pkl"
        clrd.to_pickle(str(clrd_path))
        assert clrd_path.is_file()
        assert cl.read_pickle(str(clrd_path)) == clrd

    def test_read_pickle_estimator(self, raa: Triangle, tmp_path: Path) -> None:
        """
        Create an estimator, dump a pickle of it, and then read it back in. The ingested pickle should result
        produce the same LDFs that the original estimator does.

        Parameters
        ----------
        raa: Triangle
            The raa sample data set.
        tmp_path: Path
            The builtin pytest tmp_path fixture, provides a temporary path to dump the pickle to.

        Returns
        -------
        None

        """

        pkl_path = tmp_path / "estimator.pkl"
        dev = cl.Development().fit(raa)
        with open(pkl_path, "wb") as f:
            dill.dump(dev, f)
        assert dev.ldf_ == cl.read_pickle(str(pkl_path)).ldf_

    def test_pickle_backward_compatibility(self, raa: Triangle) -> None:
        """
        Tests backward compatibility with pickles from older versions

        Parameters
        ----------
        raa: Triangle
            The raa sample data set fixture

        Returns
        -------
        None
        """
        data_dir: Path = Path(__file__).parent.parent / "data"
        raa_pickles = list(data_dir.glob("raa*.pkl"))

        for raa_pickle in raa_pickles:
            test = cl.read_pickle(raa_pickle)
            assert test == raa
            assert repr(test) == repr(raa)
            assert cl.Development().fit(test).ldf_ == cl.Development().fit(raa).ldf_
