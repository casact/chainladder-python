# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from typing import Literal

import numpy as np
import pandas as pd
import pytest
import chainladder as cl


@pytest.mark.parametrize(
    "approximation_grain, expected",
    [
        (
            "M",
            [
                10284813 / 9015625,
                934983 / 846250,
                201663 / 186040,
                792 / 721,
                792 / 767,
            ],
        ),
        (
            "D",
            [
                38269789173 / 33561250000,
                184537941 / 167147840,
                119406903 / 110160745,
                1465475 / 1334172,
                368379 / 356827,
            ],
        ),
    ],
)
def test_documented_rate_history_independent_geometry(
    approximation_grain: Literal["M", "D"], expected: list[float]
) -> None:
    """
    Pin the rate-history example to an independent earned-exposure integral.

    Uniform issuance of equal exposure, uniform earning during the policy term;
    OLF = final rate / average earned rate, not the average of individual OLFs.

    Monthly convention: equal-length months, term=12; an off-grid date changes
    the next month-start rate (Aug 2016 / Nov 2019 in this example).
    Daily convention: actual days with the current algorithm's term=365 days
    for non-leap origins and 366 days for leap origins. This regression does
    not assert exact calendar-anniversary policy earning across leap years.

    For an increase in rate level delta at d, the contribution to the average
    earned rate over [a,b] is delta * (F(b-d)-F(a-d))/(T*(b-a)), where
    F(u)=0 if u<=0, u*u/2 if 0<u<T, and T*u-T*T/2 otherwise.
    The constants above were derived with rational arithmetic from this formula
    and checked by separately integrating each policy cohort's earned overlap.
    """
    rate_history = pd.DataFrame({
        "EffDate": ["2016-07-15", "2017-03-01", "2018-01-01", "2019-10-31"],
        "RateChange": [0.02, 0.05, -0.03, 0.1],
    })
    data = pd.DataFrame({"Year": range(2016, 2021), "EarnedPremium": [10000] * 5})
    premium = cl.Triangle(data, origin="Year", columns="EarnedPremium", cumulative=True)
    result = (
        cl
        .ParallelogramOLF(
            rate_history,
            change_col="RateChange",
            date_col="EffDate",
            approximation_grain=approximation_grain,
            policy_length=12,
        )
        .fit_transform(premium)
        .olf_.values.flatten()
    )
    np.testing.assert_allclose(result, expected, rtol=1e-13, atol=1e-14)
