# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from abc import ABCMeta, abstractmethod

from sklearn.base import MetaEstimatorMixin
from sklearn.ensemble._base import _BaseHeterogeneousEnsemble
from sklearn.utils import Bunch
from sklearn.utils.metaestimators import _BaseComposition

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
