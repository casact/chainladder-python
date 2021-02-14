"""
===============================
Attachment Age Smoothing
===============================

This simple example demonstrates how a Tail ``attachment_age`` can be used to
smooth over development patterns within the triangle.  Regardless of where
the ``attachment_age`` is set, the patterns will always extrapolate one year
past the highest known lag of the `Triangle` before applying a terminal tail
factor.

"""

import chainladder as cl
import pandas as pd

raa = cl.load_sample('raa')

pd.concat((
    cl.TailCurve().fit(raa).ldf_.T.iloc[:, 0].rename('Unsmoothed'),
    cl.TailCurve(attachment_age=12).fit(raa).ldf_.T.iloc[:, 0].rename('Curve Fit')
), 1).plot(grid=True, title='Exponential Smoothing of LDF');
