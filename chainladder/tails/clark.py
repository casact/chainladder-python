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
    growth: {'loglogistic', 'weibull'}
        The growth function to be used in curve fitting development patterns.
        Options are 'loglogistic' and 'weibull'
    truncation_age: int
        The age at which you wish to stop extrapolating development
    attachment_age: int (default=None)
        The age at which to attach the fitted curve.  If None, then the latest
        age is used. Measures of variability from original ``ldf_`` are retained
        when being used in conjunction with the MackChainladder method.
    projection_period: int
        The number of months beyond the latest available development age the
        `ldf_` and `cdf_` vectors should extend.

    Attributes
    ----------
    ldf_:
        ldf with tail applied.
    cdf_:
        cdf with tail applied.
    tail_: DataFrame
        Point estimate of tail at latest maturity available in the Triangle.
    theta_: DataFrame
        Estimates of the theta parameter of the growth curve.
    omega_: DataFrame
        Estimates of the omega parameter of the growth curve.
    elr_: DataFrame
        The Expected Loss Ratio parameter. This only exists when a ``sample_weight``
        is provided to the Estimator.
    scale_: DataFrame
        The scale parameter of the model.
    norm_resid_: Triangle
        The "Normalized" Residuals of the model according to Clark.

    Examples
    --------
    Clark's method fits a parametric growth curve to the percent of ultimate
    loss reported at each age, so a single fit both smooths the in-triangle
    development and extrapolates it beyond the edge of the triangle. The
    choice of growth curve is the key judgment because the two supported
    families have fundamentally different right tails:

    - ``loglogistic`` decays polynomially (a power-law tail). The unreported
      fraction shrinks slowly with age, so material development persists to
      very old ages, as is typical of excess casualty and other long-tailed
      business.
    - ``weibull`` decays exponentially. The growth curve flattens quickly,
      implying development is essentially complete shortly beyond the edge
      of the triangle.

    .. testsetup::

        import chainladder as cl

    .. testcode::

        dev = cl.Development().fit_transform(cl.load_sample("raa"))
        log = cl.TailClark(growth="loglogistic").fit(dev)
        wei = cl.TailClark(growth="weibull").fit(dev)
        print(log.ldf_.values[0, 0, 0, :].round(3))
        print(wei.ldf_.values[0, 0, 0, :].round(3))
        print(round(float(log.tail_.iloc[0, 0]), 3))
        print(round(float(wei.tail_.iloc[0, 0]), 3))

    .. testoutput::

        [2.999 1.624 1.271 1.172 1.113 1.042 1.033 1.017 1.029 1.023 1.189]
        [2.999 1.624 1.271 1.172 1.113 1.042 1.033 1.017 1.014 1.009 1.014]
        1.216
        1.022

    Inside the triangle, where the observed factors constrain both curves,
    the two fits barely differ. Beyond age 120 only the assumed shape of the
    curve matters: the loglogistic projects 21.6% of additional development
    while the weibull projects 2.2%. The gap is not noise. It is the direct
    consequence of the loglogistic's power-law decay continuing to release
    losses for decades, while the weibull's exponential decay extinguishes
    development almost immediately.

    Because the loglogistic tail dies out so slowly, Clark recommends
    establishing a truncation age at which losses can be considered fully
    developed rather than extrapolating indefinitely. Capping the same
    loglogistic fit at age 240 gives a more defensible provision:

    .. testcode::

        trunc = cl.TailClark(growth="loglogistic", truncation_age=240).fit(dev)
        print(round(float(trunc.tail_.iloc[0, 0]), 3))

    .. testoutput::

        1.127

    To assess whether the selected growth curve is consistent with the data,
    review the ``norm_resid_`` attribute: residuals scattered randomly
    around zero indicate an adequate fit, while systematic patterns by age
    suggest the other family, a different development estimator, or a
    truncated projection should be considered.

    """

    def __init__(self, growth="loglogistic", truncation_age=None,
                 attachment_age=None, projection_period=12):
        self.growth = growth
        self.truncation_age = truncation_age
        self.attachment_age = attachment_age
        self.projection_period = projection_period

    def fit(self, X, y=None, sample_weight=None):
        """Fit the model with X.

        Parameters
        ----------
        X: Triangle-like
            Set of LDFs to which the tail will be applied.
        y: Ignored
        sample_weight : Triangle-like
            Exposure vector used to invoke the Cape Cod method.

        Returns
        -------
        self: object
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
        age_offset = {"Y": 6.0, "S": 3, "Q": 1.5, "M": 0.5}[X.development_grain]
        fitted = 1 / model.G_(self.ldf_.ddims - age_offset)
        fitted = xp.concatenate(
            (
                fitted.values[..., :-1] / fitted.values[..., 1:],
                fitted.values[..., -1:],
            ),
            -1,
        )
        fitted = xp.repeat(fitted, self.ldf_.values.shape[2], 2)
        attachment_age = self.attachment_age if self.attachment_age else X.ddims[-2]
        self.ldf_.values = xp.concatenate((
            self.ldf_.values[..., : sum(self.ldf_.ddims < attachment_age)],
            fitted[..., -sum(self.ldf_.ddims >= attachment_age) :],),
            axis=-1,)
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
        X: Triangle
            Triangle must contain the ``ldf_`` development attribute.

        Returns
        -------
        X_new: Triangle
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
        if hasattr(self, "elr_"):
            X_new.elr_ = self.elr_
        return X_new
