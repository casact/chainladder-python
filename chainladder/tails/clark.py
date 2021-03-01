# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from chainladder.tails import TailBase
from chainladder.development.clark import ClarkLDF


class TailClark(TailBase):
    """Allows for extraploation of LDFs to form a tail factor.

    .. versionadded:: 0.6.4

    Parameters
    ----------
    growth : {'loglogistic', 'weibull'}
        The growth function to be used in curve fitting development patterns.
        Options are 'loglogistic' and 'weibull'
    truncation_age : int
        The age at which you wish to stop extrapolating development
    attachment_age: int (default=None)
        The age at which to attach the fitted curve.  If None, then the latest
        age is used. Measures of variability from original ``ldf_`` are retained
        when being used in conjunction with the MackChainladder method.


    Attributes
    ----------
    ldf_ :
        ldf with tail applied.
    cdf_ :
        cdf with tail applied.
    tail_ : DataFrame
        Point estimate of tail at latest maturity available in the Triangle.
    theta_ : DataFrame
        Estimates of the theta parameter of the growth curve.
    omega_ : DataFrame
        Estimates of the omega parameter of the growth curve.
    elr_ : DataFrame
        The Expected Loss Ratio parameter. This only exists when a ``sample_weight``
        is provided to the Estimator.
    scale_ : DataFrame
        The scale parameter of the model.
    norm_resid_ : Triangle
        The "Normalized" Residuals of the model according to Clark.
    """

    def __init__(self, growth="loglogistic", truncation_age=None, attachment_age=None):
        self.growth = growth
        self.truncation_age = truncation_age
        self.attachment_age = attachment_age

    def fit(self, X, y=None, sample_weight=None):
        """Fit the model with X.

        Parameters
        ----------
        X : Triangle-like
            Set of LDFs to which the tail will be applied.
        y : Ignored
        sample_weight : Triangle-like
            Exposure vector used to invoke the Cape Cod method.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        backend = X.array_backend
        if backend != "numpy":
            X = X.set_backend("numpy")
        else:
            X = X.copy()
        xp = X.get_array_module()
        super().fit(X, y, sample_weight)
        model = ClarkLDF(growth=self.growth).fit(X, sample_weight=sample_weight)
        xp = X.get_array_module()
        age_offset = {"Y": 6.0, "Q": 1.5, "M": 0.5}[X.development_grain]
        fitted = 1 / model.G_(
            xp.array(
                [
                    self._ave_period[1] + X.ddims - age_offset
                    for item in range(self._ave_period[0] + 2)
                ]
            )[0]
        )
        fitted = xp.concatenate(
            (
                fitted.values[..., :-1] / fitted.values[..., -1:],
                fitted.values[..., -1:],
            ),
            -1,
        )
        fitted = xp.repeat(fitted, self.ldf_.values.shape[2], 2)
        attachment_age = self.attachment_age if self.attachment_age else X.ddims[-2]
        self.ldf_.values = xp.concatenate(
            (
                self.ldf_.values[..., : sum(X.ddims <= attachment_age)],
                fitted[..., -sum(X.ddims >= attachment_age) :],
            ),
            axis=-1,
        )
        self.omega_ = model.omega_
        self.theta_ = model.theta_
        self.G_ = model.G_
        self.scale_ = model.scale_
        self._G = model._G
        self.incremental_fits_ = model.incremental_fits_
        if hasattr(model, "elr_"):
            self.elr_ = model.elr_
        self.norm_resid_ = model.norm_resid_
        if self.truncation_age:
            self.ldf_.values[..., -1:] = self.ldf_.values[..., -1:] * self.G_(self.truncation_age).values
        # self._get_tail_stats(self)
        if backend == "cupy":
            self = self.set_backend("cupy", inplace=True)
        return self

    def transform(self, X):
        """Transform X.

        Parameters
        ----------
        X : Triangle
            Triangle must contain the ``ldf_`` development attribute.

        Returns
        -------
        X_new : Triangle
            New Triangle with tail factor applied to its development
            attributes.
        """
        X_new = super().transform(X)
        triangles = [
            "omega_",
            "theta_",
            "incremental_fits_",
            "G_",
            "_G",
            "growth",
            "scale_",
        ]
        for item in triangles:
            setattr(X_new, item, getattr(self, item))
        X_new._set_slicers()
        if hasattr(self, "elr_"):
            X_new.elr_ = self.elr_
        return X_new
