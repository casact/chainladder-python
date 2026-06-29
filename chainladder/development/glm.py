# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pandas as pd
import numpy as np
from chainladder.development.base import DevelopmentBase
from chainladder.development.learning import DevelopmentML
from sklearn.linear_model import TweedieRegressor
from sklearn.pipeline import Pipeline
from chainladder.utils.utility_functions import PatsyFormula


class TweedieGLM(DevelopmentBase):
    """ GLM reserving with scikit-learn's Tweedie distribution.

    Implements the GLM reserving structure of Taylor and McGuire. The Tweedie
    family covers normal, ODP Poisson, gamma, and related targets via ``power``
    and ``link``. Covariates from any triangle axis can enter through a patsy
    ``design_matrix`` while staying close to traditional chainladder methods when
    origin and development are coded categorically.

    Triangles are converted to long-format tables internally (as with
    ``Triangle.to_frame(keepdims=True)``); origin periods are restated as years
    from the earliest origin for sklearn compatibility, and the response is
    converted to an incremental basis before fitting. This class is a special
    case of :class:`~chainladder.DevelopmentML` that uses only
    :class:`~sklearn.linear_model.TweedieRegressor` behind a
    :class:`~chainladder.utils.utility_functions.PatsyFormula` step.

    .. versionadded:: 0.8.1

    Parameters
    ----------
    drop: tuple or list of tuples
        Drops specific origin/development combination(s)
    drop_valuation: str or list of str (default = None)
        Drops specific valuation periods. str must be date convertible.
    design_matrix: formula-like
        A patsy formula describing the independent variables, X of the GLM
    response:  str
        Column name for the response variable of the GLM. If omitted, then the
        first column of the Triangle will be used.
    power: float, default=1
        The power determines the underlying target distribution according
        to the following table:

        .. list-table::
           :header-rows: 1
        
           * - Power
             - Distribution
           * - 0
             - Normal
           * - 1
             - Poisson
           * - (1,2)
             - Compound Poisson Gamma
           * - 2
             - Gamma
           * - 3
             - Inverse Gaussian
            
        For ``0 < power < 1``, no distribution exists.
    alpha: float, default=1
        Constant that multiplies the penalty term and thus determines the
        regularization strength. ``alpha = 0`` is equivalent to unpenalized
        GLMs. In this case, the design matrix `X` must have full column rank
        (no collinearities).
    link: {'auto', 'identity', 'log'}, default='log'
        The link function of the GLM, i.e. mapping from linear predictor
        `X @ coeff + intercept` to prediction `y_pred`. Option 'auto' sets
        the link depending on the chosen family as follows:
        - 'identity' for Normal distribution
        - 'log' for Poisson,  Gamma and Inverse Gaussian distributions
    max_iter: int, default=100
        The maximal number of iterations for the solver.
    tol: float, default=1e-4
        Stopping criterion. For the lbfgs solver,
        the iteration will stop when ``max{|g_j|, j = 1, ..., d} <= tol``
        where ``g_j`` is the j-th component of the gradient (derivative) of
        the objective function.
    warm_start: bool, default=False
        If set to ``True``, reuse the solution of the previous call to ``fit``
        as initialization for ``coef_`` and ``intercept_`` .
    verbose: int, default=0
        For the lbfgs solver set verbose to any positive number for verbosity.

    Attributes
    ----------
    model_: sklearn.Pipeline
        A scikit-learn Pipeline of the GLM

    Examples
    --------
    With its default parameters (a Poisson GLM with log link and the
    categorical ``design_matrix`` of ``"C(development) + C(origin)"``),
    ``TweedieGLM`` replicates volume-weighted chainladder development:

    .. testsetup::

        import chainladder as cl

    .. testcode::

        import numpy as np

        tri = cl.load_sample("genins")
        glm = cl.TweedieGLM().fit(tri)
        trad = cl.Development(average="volume").fit(tri)
        print(np.round(glm.ldf_.values[0, 0, 0, :], 4))
        print(np.round(trad.ldf_.values[0, 0, 0, :], 4))

    .. testoutput::

        [3.491  1.7474 1.4574 1.1739 1.1038 1.0863 1.0539 1.0766 1.0177]
        [3.4906 1.7473 1.4574 1.1739 1.1038 1.0863 1.0539 1.0766 1.0177]

    On some triangles the solver needs more than the default 100 iterations
    and sklearn raises a ``ConvergenceWarning``; increase ``max_iter`` until
    the warning goes away. The ``comauto`` line of ``clrd`` is one such
    triangle:

    .. testcode::

        tri = cl.load_sample("clrd")["CumPaidLoss"].groupby("LOB").sum()
        comauto = tri[tri["LOB"] == "comauto"]
        m_cat = cl.TweedieGLM(power=1, max_iter=1000).fit(comauto)
        print(np.round(m_cat.ldf_.values[0, 0, 0, :], 4))

    .. testoutput::

        [2.0459 1.352  1.1739 1.088  1.0403 1.021  1.0093 1.0062 1.007 ]

    Patsy's ``C()`` wraps a column in categorical dummies, with one
    coefficient per level. Removing ``C()`` converts that axis to a
    continuous linear trend with a single slope, a more parsimonious fit
    that can be preferable when data are sparse. Compare the continuous
    fit below to the categorical ``m_cat`` above on the same triangle:

    .. testcode::

        m_cont = cl.TweedieGLM(
            power=1, design_matrix="development + origin"
        ).fit(comauto)
        print(np.round(m_cont.ldf_.values[0, 0, 0, :], 4))

    .. testoutput::

        [1.6993 1.2878 1.1563 1.0945 1.0604 1.0398 1.0268 1.0182 1.0125]

    On multi-LOB triangles, interaction terms can keep the model parsimonious
    (10 coefficients here versus 18+ in a full categorical chainladder). In
    the ``design_matrix`` below, ``LOB`` gives each line its own level, and
    ``LOB:C(np.minimum(development, 36))`` gives each line separate
    categorical factors for the early ages where development is steepest,
    with ``np.minimum`` pooling every age beyond 36 months into a single
    level. ``LOB:development`` and ``LOB:origin`` then carry the later ages
    and origin years as continuous linear trends per line. The percent
    difference in ``cdf_`` versus :class:`~chainladder.Development`
    stays within about 1% at each ultimate lag:

    .. testcode::

        import numpy as np

        clrd = cl.load_sample("clrd")["CumPaidLoss"].groupby("LOB").sum()
        clrd = clrd[clrd["LOB"].isin(["ppauto", "comauto"])]
        dev = cl.TweedieGLM(
            design_matrix=(
                "LOB+LOB:C(np.minimum(development, 36))"
                "+LOB:development+LOB:origin"
            ),
            max_iter=1000,
        ).fit(clrd)
        trad = cl.Development().fit(clrd)
        pct = ((dev.cdf_.iloc[..., 0, :] / trad.cdf_) - 1).to_frame().round(3)
        print(len(dev.coef_))
        print(np.round(pct.loc["comauto"].values, 3))
        print(np.round(pct.loc["ppauto"].values, 3))

    .. testoutput::

        10
        [ 0.002  0.003 -0.01   0.003  0.011  0.008  0.005 -0.    -0.002]
        [ 0.006  0.003 -0.     0.001  0.002  0.001  0.001  0.001  0.001]

    """

    def __init__(self, design_matrix='C(development) + C(origin)',
                 response=None, power=1.0, alpha=1.0, link='log',
                 max_iter=100, tol=0.0001, warm_start=False, verbose=0, drop=None,drop_valuation=None):
        self.drop = drop
        self.drop_valuation = drop_valuation
        self.response=response
        self.design_matrix = design_matrix
        self.power=power
        self.alpha=alpha
        self.link=link
        self.max_iter=max_iter
        self.tol=tol
        self.warm_start=warm_start
        self.verbose=verbose

    def fit(self, X, y=None, sample_weight=None):
        response = X.columns[0] if not self.response else self.response
        if sample_weight is None:
            weight = None
        else:
            weight = 'model'
        self.model_ = DevelopmentML(Pipeline(steps=[
            ('design_matrix', PatsyFormula(self.design_matrix)),
            ('model', TweedieRegressor(
                    link=self.link, power=self.power, max_iter=self.max_iter,
                    tol=self.tol, warm_start=self.warm_start,
                    verbose=self.verbose, fit_intercept=False))]),
                    y_ml=response, weighted_step = weight,
                    drop=self.drop, drop_valuation=self.drop_valuation).fit(X = X, sample_weight = sample_weight)
        return self

    @property
    def ldf_(self):
        return self.model_.ldf_

    @property
    def triangle_ml_(self):
        return self.model_.triangle_ml_

    @property
    def coef_(self):
        return pd.Series(
            self.model_.estimator_ml.named_steps.model.coef_, name='coef_',
            index=list(self.model_.estimator_ml.named_steps.design_matrix.
                            design_info_.column_name_indexes.keys())
        ).to_frame()

    def transform(self, X):
        return self.model_.transform(X)
