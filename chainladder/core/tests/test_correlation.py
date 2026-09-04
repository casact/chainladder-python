import chainladder as cl
import pytest

raa = cl.load_sample("RAA")


def test_val_corr_total_true():
    assert raa.valuation_correlation(p_critical=0.5, total=True)


def test_val_corr_total_false():
    assert raa.valuation_correlation(p_critical=0.5, total=False)


def test_dev_corr():
    assert raa.development_correlation(p_critical=0.5)


def test_dev_corr_sparse():
    assert raa.set_backend("sparse").development_correlation(p_critical=0.5)


def test_validate_critical():
    with pytest.raises(ValueError):
        raa.valuation_correlation(p_critical=1.5, total=True)


def test_val_corr_incomplete_triangle(xyz):
    # GH #320: a triangle missing its earliest diagonals raised a
    # ValueError on repr because z_critical dropped all-NaN diagonals
    # while its values kept one entry per link-ratio diagonal.
    z_critical = (
        xyz["Paid"].valuation_correlation(p_critical=0.1, total=False).z_critical
    )
    assert z_critical.values.shape[-1] == len(z_critical.ddims)
    assert repr(z_critical)
