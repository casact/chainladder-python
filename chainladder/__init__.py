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
    Options as Options,
    options as options,
)

# noinspection PyProtectedMember
from chainladder._config import (  # noqa (API import)
    _DEPRECATED_BACKENDS,
    _deprecated_backend_message,
    _dask_parallel_state,
    _warn_dask_parallel_deprecated,
    _deprecated_rename,
    _deprecated_rename_argument,
    _deprecated_drop_argument,
)
from chainladder.utils import (  # noqa: E402
    WeightedRegression as WeightedRegression,
    TriangleWeight as TriangleWeight,
    parallelogram_olf as parallelogram_olf,
    read_csv as read_csv,
    read_pickle as read_pickle,
    read_json as read_json,
    concat as concat,
    load_sample as load_sample,
    list_samples as list_samples,
    minimum as minimum,
    maximum as maximum,
    PatsyFormula as PatsyFormula,
    model_diagnostics as model_diagnostics,
    cp as cp,
    sp as sp,
    dp as dp,
)
from chainladder.core import (  # noqa: E402
    Triangle as Triangle,
    DevelopmentCorrelation as DevelopmentCorrelation,
    ValuationCorrelation as ValuationCorrelation,
    Styler as Styler,
)
from chainladder.development import (  # noqa: E402
    DevelopmentBase as DevelopmentBase,
    Development as Development,
    MunichAdjustment as MunichAdjustment,
    IncrementalAdditive as IncrementalAdditive,
    DevelopmentConstant as DevelopmentConstant,
    ClarkLDF as ClarkLDF,
    CaseOutstanding as CaseOutstanding,
    ImpliedCaseDevelopment as ImpliedCaseDevelopment,
    ImpliedCase as ImpliedCase,
    DevelopmentML as DevelopmentML,
    TweedieGLM as TweedieGLM,
    BarnettZehnwirth as BarnettZehnwirth,
)
from chainladder.adjustments import (  # noqa: E402
    BootstrapODPSample as BootstrapODPSample,
    BerquistSherman as BerquistSherman,
    ParallelogramOLF as ParallelogramOLF,
    Trend as Trend,
    DisposalRate as DisposalRate,
)
from chainladder.tails import (  # noqa: E402
    TailBase as TailBase,
    TailConstant as TailConstant,
    TailCurve as TailCurve,
    TailBondy as TailBondy,
    TailClark as TailClark,
)
from chainladder.methods import (  # noqa: E402
    MethodBase as MethodBase,
    Chainladder as Chainladder,
    MackChainladder as MackChainladder,
    Benktander as Benktander,
    BornhuetterFerguson as BornhuetterFerguson,
    CapeCod as CapeCod,
    ExpectedLoss as ExpectedLoss,
    HurlimannCLR as HurlimannCLR,
)
from chainladder.workflow import (  # noqa: E402
    GridSearch as GridSearch,
    Pipeline as Pipeline,
    VotingChainladder as VotingChainladder,
    TriangleSelector as TriangleSelector,
)

__version__ = version("chainladder")
