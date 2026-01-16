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
    assert raa.set_backend('sparse').development_correlation(p_critical=0.5)

def test_validate_critical():
    with pytest.raises(ValueError):
        raa.valuation_correlation(p_critical=1.5, total=True)