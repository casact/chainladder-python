# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from abc import abstractmethod

import numpy as np
from chainladder.methods.base import MethodBase
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble._base import _fit_single_estimator, _BaseHeterogeneousEnsemble
from sklearn.ensemble._voting import _BaseVoting
from sklearn.utils import Bunch
from sklearn.utils.validation import (_deprecate_positional_args,
                                      check_is_fitted)

from ..core.base import is_chainladder

class _BaseTriangleEnsemble(_BaseHeterogeneousEnsemble):
    """Base class for ensemble of triangle methods."""
    def __init__(self, estimators):
        super().__init__(estimators)


    def _validate_estimators(self):
        if self.estimators is None or len(self.estimators) == 0:
            raise ValueError(
                "Invalid 'estimators' attribute, 'estimators' should be a list"
                " of (string, estimator) tuples."
            )
        names, estimators = zip(*self.estimators)
        # defined by MetaEstimatorMixin in _BaseHeterogeneousEnsemble
        self._validate_names(names)

        for est in estimators:
            if est != 'drop' and not is_chainladder(est):
                raise ValueError(
                    f"The estimator {est.__class__.__name__}"
                    f" should be a chainladder method."
                )

        return names, estimators

class _BaseChainladderVoting(_BaseVoting, _BaseTriangleEnsemble):
    """Base class for voting between chainladder methods.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    def _broadcast_weights(self, X):
        if self.weights_.ndim == 3:
            weights = np.repeat(self.weights_[np.newaxis, ...], X.shape[1], axis=0)
        if self.weights_.ndim == 4:
            weights = np.repeat(self.weights_[np.newaxis, ...], X.shape[0], axis=0)
        return weights

    def _predict(self, X, sample_weight=None):
        """Collect results from clf.predict calls."""
        return [est.predict(X, sample_weight=sample_weight) for est in self.estimators_]

    @abstractmethod
    def fit(self, X, y, sample_weight=None):
        """Get common fit operations."""
        names, clfs = self._validate_estimators()
        self.weights_ = self.weights
        if self.weights_ is not None:
            self.weights_ = self.weights_[..., np.newaxis]
            if self.weights_.shape[-3] != X.shape[2]:
                raise ValueError('Length of weight arrays do not equal'
                                 f' number of accident periods; found'
                                 f' {self.weights_.shape[-3]} weights'
                                 f' and {X.shape[2]} accident periods.')
            if self.weights_.shape[-2] != len(self.estimators):
                raise ValueError('Number of weight arrays does not equal'
                                 f' number of estimators array; '
                                 f' found {self.weights_.shape[-2]} weights'
                                 f' arrays and {len(self.estimators)}'
                                 ' estimators.')
        else:
            self.weights_ = np.ones(X.shape[:3] + (len(self.estimators), 1))

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_single_estimator)(
                        clone(clf), X, y,
                        sample_weight=sample_weight,
                        message_clsname='VotingChainladder',
                        message=self._log_message(names[idx],
                                                  idx + 1, len(clfs))
                )
                for idx, clf in enumerate(clfs) if clf != 'drop'
            )

        self.named_estimators_ = Bunch()

        # Uses 'drop' as placeholder for dropped estimators
        est_iter = iter(self.estimators_)
        for name, est in self.estimators:
            current_est = est if est == 'drop' else next(est_iter)
            self.named_estimators_[name] = current_est

        return self


class VotingChainladder(_BaseChainladderVoting, MethodBase):
    """Prediction voting chainladder method for unfitted estimators.

    A voting chainladder is an ensemble meta-estimator that fits several base
    chainladder methods, each on the whole triangle. Then it combines the
    individual predictions based on a matrix of weights to form a
    final prediction.

    Read more in the :ref:`User Guide <voting>`.

    .. versionadded:: 0.8.0

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        Invoking the ``fit`` method on the ``VotingChainladder`` will fit clones
        of those original estimators that will be stored in the class attribute
        ``self.estimators_``. An estimator can be set to ``'drop'`` using
        ``set_params``.

    weights : array-like of shape (index, columns, origin, n_estimators), default=None
        Minimum shape required is (origin, n_estimators). Lower dimensional weight arrays
        will have missing dimensions repeated to match the shape of the triangle. If None,
        takes the average of all predictions.

    n_jobs : int, default=None
        The number of jobs to run in parallel for ``fit``.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
        for more details.

    verbose : bool, default=False
        If True, the time elapsed while fitting will be printed as it
        is completed.

    Attributes
    ----------
    estimators_ : list of chainladder estimators
        The collection of fitted sub-estimators as defined in ``estimators``
        that are not 'drop'.

    named_estimators_ : Bunch
        Attribute to access any fitted sub-estimators by name.

    Examples
    --------
    >>> import numpy as np
    >>> import chainladder as cl
    >>> raa = cl.load_sample('RAA')
    >>> cl_ult = cl.Chainladder().fit(raa).ultimate_  # Chainladder Ultimate
    >>> apriori = cl_ult * 0 + (float(cl_ult.sum()) / 10)  # Mean Chainladder Ultimate
    >>> bcl = cl.Chainladder()
    >>> bf = cl.BornhuetterFerguson()
    >>> cc = cl.CapeCod()
    >>> estimators = [('bcl', bcl), ('bf', bf), ('cc', cc)]
    >>> weights = np.array([[0.6, 0.2, 0.2]] * 4 + [[0, 0.5, 0.5]] * 3 + [[0, 0, 1]] * 3)
    >>> vot = cl.VotingChainladder(estimators=estimators, weights=weights)
    >>> vot.fit(raa, sample_weight=apriori)
    >>> print(vot.ultimate_)
                      2262
        1981  18834.000000
        1982  16875.500226
        1983  24058.534810
        1984  28542.580970
        1985  28236.843134
        1986  19905.317262
        1987  18947.245455
        1988  23106.943030
        1989  20004.502125
        1990  21605.832631
    """
    @_deprecate_positional_args
    def __init__(self, estimators, *, weights=None, n_jobs=None,
                 verbose=False):
        super().__init__(estimators=estimators)
        self.weights = weights
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y=None, sample_weight=None):
        """Fit the estimators.

        Parameters
        ----------
        X : Triangle
            Loss data to which the voting will be applied.

        y : None
            Ignored

        sample_weight : Triangle, default=None
            Exposure to be used in the calculation. Required if any
            of the estimators are exposure based.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.X_ = self.validate_X(X)
        super().fit(X, y, sample_weight)
        self.ultimate_ = self._get_ultimate(X, sample_weight=sample_weight)
        return self

    def predict(self, X, sample_weight=None):
        """Predicts the voting chainladder ultimate on a new triangle **X**

        Predicts the ultimate for each of the estimators and combines them
        into a single ultimate based on the weights given.

        Parameters
        ----------
        X : Triangle
            Loss data to which the model will be applied.
        sample_weight : Triangle, default=None
            Exposure to be used in the calculation. Required if any
            of the estimators are exposure based.

        Returns
        -------
        X_new: Triangle
            Loss data with VotingChainladder ultimate applied
        """
        check_is_fitted(self)
        obj = X.copy()
        # obj.ldf_ = self.ldf_
        self.validate_weight(X, sample_weight)
        if sample_weight:
            sample_weight = sample_weight.set_backend(obj.array_backend)
        obj.ultimate_ = self._get_ultimate(obj, sample_weight=sample_weight)
        return obj

    def transform(self, X, sample_weight=None):
        """Return predictions for VotingChainladder

        Parameters
        ----------
        X : Triangle
            Loss data to which the model will be applied.

        sample_weight : Triangle, default=None
            Exposure to be used in the calculation. Required if any
            of the estimators are exposure based.

        Returns
        -------
        X_new: Triangle
            Loss data with VotingChainladder ultimate applied
        """
        return self.predict(X, sample_weight=sample_weight)

    def fit_transform(self, X, y=None, sample_weight=None):
        """Fit and return predictions for VotingChainladder

        Parameters
        ----------
        X : Triangle
            Loss data to which the model will be applied.

        y : None
            Ignored

        sample_weight : Triangle, default=None
            Exposure to be used in the calculation. Required if any
            of the estimators are exposure based.

        Returns
        -------
        X_new: Triangle
            Loss data with VotingChainladder ultimate applied
        """
        return self.fit(X, y, sample_weight).transform(X, sample_weight=sample_weight)

    def _get_ultimate(self, X, sample_weight=None):
        if self.weights_.ndim < 5:
            weights = self._broadcast_weights(X)
        elif self.weights_.ndim == 5:
            if self.weights_.shape[0] != X.shape[0]:
                raise ValueError('Index length (axis 0) of weights does'
                                 ' not equal index length of X; found'
                                 f' {self.weights_.shape[0]} for weights'
                                 f' and {X.shape[0]} for X.')
            if self.weights_.shape[1] != X.shape[1]:
                raise ValueError('Column length (axis 1) of weights does'
                                 f' not equal column length of X; found'
                                 f' {self.weights_.shape[1]} for weights'
                                 f' and {X.shape[1]} for X.')
            weights = self.weights_

        ultimate = sum([
                est.predict(X, sample_weight).ultimate_ * weights[..., i, :]
                for i, est in enumerate(self.estimators_)
            ]) / weights.sum(axis=-2)
        return ultimate
