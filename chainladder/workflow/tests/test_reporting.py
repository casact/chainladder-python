import pandas as pd
import pytest

import chainladder as cl


def test_reserving_report_renders_sections_and_assumptions():
    report = cl.ReservingReport(
        title="Q4 reserve review", assumptions={"Method": "Chainladder"}
    ).fit({"Quality check": pd.DataFrame({"check": ["missing"], "count": [0]})})

    assert "Q4 reserve review" in report.html_
    assert "Quality check" in report.html_
    assert "Chainladder" in report.html_


def test_reserving_report_validates_sections():
    with pytest.raises(ValueError, match="dictionary"):
        cl.ReservingReport().fit({})
