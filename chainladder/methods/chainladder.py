# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from chainladder.methods import MethodBase


class Chainladder(MethodBase):
    """
    The basic deterministic chainladder method.

    Parameters
    ----------
    None

    Attributes
    ----------
    X_:
        returns **X** used to fit the triangle
    ultimate_:
        The ultimate losses per the method
    ibnr_:
        The IBNR per the method
    full_expectation_:
        The ultimates back-filled to each development period in **X** replacing
        the known data
    full_triangle_:
        The ultimates back-filled to each development period in **X** retaining
        the known data

    Examples
    --------
    Fit the chainladder method to a loss triangle and inspect the projected
    ultimates.

    .. testsetup::

        import chainladder as cl

    .. testcode::

        tr = cl.load_sample('ukmotor')
        model = cl.Chainladder().fit(tr)
        print(model.ultimate_)

    .. testoutput::

                      2261
        2007  12690.000000
        2008  13096.902024
        2009  14030.536767
        2010  13137.859861
        2011  13880.404483
        2012  16812.150646
        2013  20679.919151

    The ``ibnr_`` attribute is ``ultimate_ - latest_diagonal``. The 2007 origin
    is fully developed in the data, so its IBNR is ``NaN``.

    .. testcode::

        print(model.ibnr_)

    .. testoutput::

                      2261
        2007           NaN
        2008    350.902024
        2009   1037.536767
        2010   2044.859861
        2011   3663.404483
        2012   7162.150646
        2013  14396.919151

    ``full_triangle_`` projects each origin to ultimate while preserving the
    known cells. Showing the last three origins and the first five development
    periods makes the data-to-projection boundary visible: whole-number cells
    are observed, decimal cells are projected.

    .. testcode::

        print(model.full_triangle_.iloc[..., -3:, :5])

    .. testoutput::

                  12            24            36            48            60
        2011  4150.0   7897.000000  10217.000000  11719.970266  12853.969769
        2012  5102.0   9650.000000  12374.981102  14195.400857  15568.917781
        2013  6283.0  11870.058983  15221.943585  17461.165333  19150.670711

    ``full_expectation_`` is similar but replaces every cell, including the
    known ones, with the model's expectation. Compare the ``12`` column above
    against the same slice below: the observed values have been overwritten.

    .. testcode::

        print(model.full_expectation_.iloc[..., -3:, :5])

    .. testoutput::

                       12            24            36            48            60
        2011  4217.162588   7967.208127  10217.000000  11719.970266  12853.969769
        2012  5107.889530   9650.000000  12374.981102  14195.400857  15568.917781
        2013  6283.000000  11870.058983  15221.943585  17461.165333  19150.670711
    """

    def fit(self, X, y=None, sample_weight=None):
        """Fit the model with X.

        Parameters
        ----------
        X: Triangle-like
            Data to which the model will be applied.
        y: Ignored
        sample_weight : Ignored

        Returns
        -------
        self: object
            Returns the instance itself.

        Examples
        --------
        Fitting returns the estimator itself, so it can be chained with
        attribute access.

        .. testsetup::
            import chainladder as cl

        .. testcode::

            tr = cl.load_sample('ukmotor')
            cl.Chainladder().fit(tr)

        .. testoutput::

            Chainladder()
            
        """
        super().fit(X, y, sample_weight)
        self.ultimate_ = self._get_ultimate(self.X_)
        self.process_variance_ = self._include_process_variance()
        return self

    def predict(self, X, sample_weight=None):
        """Predicts the chainladder ultimate on a new triangle **X**

        Parameters
        ----------
        X: Triangle
            Loss data to which the model will be applied.
        sample_weight: Triangle
            Required exposure to be used in the calculation.

        Returns
        -------
        X_new: Triangle
            Loss data with chainladder ultimate applied

        Examples
        --------
        ``predict`` applies the fitted development patterns to a different
        Triangle. A common workflow is to fit on a prior-period view of the
        data (one diagonal removed) and then apply that model to the current
        Triangle. The ultimates differ from a freshly-fit model because the
        patterns reflect the older view.

        .. testsetup::

            import chainladder as cl

        .. testcode::

            tr = cl.load_sample('ukmotor')
            tr_prior = tr[tr.valuation < tr.valuation_date]
            model = cl.Chainladder().fit(tr_prior)
            print(model.predict(tr).ultimate_)

        .. testoutput::

                          2261
            2007  12690.000000
            2008  12746.000000
            2009  13641.379750
            2010  12719.871218
            2011  13485.986574
            2012  16296.783586
            2013  20040.175415
        """
        X_new = super().predict(X, sample_weight)
        X_new.ultimate_ = self._get_ultimate(X_new, sample_weight)
        return X_new

    def _get_ultimate(self, X, sample_weight=None):
        """ Private method that uses CDFs to obtain an ultimate vector """
        ld = X.incr_to_cum().latest_diagonal
        ultimate = X.incr_to_cum().copy()
        cdf = self._align_cdf(ultimate, sample_weight) 
        ultimate = ld * cdf 
        return self._set_ult_attr(ultimate)
