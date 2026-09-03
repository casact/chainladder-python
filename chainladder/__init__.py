"""
The chainladder-python package was built to be able to handle all of your actuarial reserving needs in python.
It consists of popular actuarial tools, such as triangle data manipulation, link ratios calculation, and
IBNR estimates using both deterministic and stochastic models. We build this package so you no longer have to rely
on outdated software and tools when performing actuarial pricing or reserving indications.

This package strives to be minimalistic in needing its own API. The syntax mimics popular packages such as pandas for
data manipulation and scikit-learn for model construction. An actuary that is already familiar with these tools will be
able to pick up this package with ease. You will be able to save your mental energy for actual actuarial work.
"""

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from importlib.metadata import version

del annotations


from chainladder._config import (  # noqa (API import)
    __dt64_dtype__,
    __dt64_unit__,
    Options,
    options,
)

# noinspection PyProtectedMember
from chainladder._config import (  # noqa (API import)
    _DEPRECATED_BACKENDS,
    _deprecated_backend_message,
    _dask_parallel_state,
    _warn_dask_parallel_deprecated,
)
from chainladder.utils import (  # noqa (API import)
    WeightedRegression,
    TriangleWeight,
    parallelogram_olf,
    read_csv,
    read_pickle,
    read_json,
    concat,
    load_sample,
    list_samples,
    minimum,
    maximum,
    PatsyFormula,
    model_diagnostics,
    cp,
    sp,
    dp,
)
from chainladder.core import (  # noqa (API import)
    Triangle,
    DevelopmentCorrelation,
    ValuationCorrelation,
)
from chainladder.development import (  # noqa (API import)
    DevelopmentBase,
    Development,
    MunichAdjustment,
    IncrementalAdditive,
    DevelopmentConstant,
    ClarkLDF,
    CaseOutstanding,
    DevelopmentML,
    TweedieGLM,
    BarnettZehnwirth,
)
from chainladder.adjustments import (  # noqa (API import)
    BootstrapODPSample,
    BerquistSherman,
    ParallelogramOLF,
    Trend,
    TrendConstant,
    DisposalRate,
)
from chainladder.tails import (  # noqa (API import)
    TailBase,
    TailConstant,
    TailCurve,
    TailBondy,
    TailClark,
)
from chainladder.methods import (  # noqa (API import)
    MethodBase,
    Chainladder,
    MackChainladder,
    Benktander,
    BornhuetterFerguson,
    CapeCod,
    ExpectedLoss,
)
from chainladder.workflow import (  # noqa (API import)
    GridSearch,
    Pipeline,
    VotingChainladder,
    TriangleSelector,
)

__version__ = version("chainladder")
