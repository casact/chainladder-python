"""
======================================
Tail Curve Fit Comparison
======================================

This example demonstrates how the ``inverse_power`` curve generally produces more
conservative tail factors than the ``exponential`` fit.
"""
import chainladder as cl
import pandas as pd

clrd = cl.load_sample('clrd').groupby('LOB').sum()['CumPaidLoss']
cdf_ip = cl.TailCurve(curve='inverse_power').fit(clrd)
cdf_xp = cl.TailCurve(curve='exponential').fit(clrd)

pd.concat((cdf_ip.tail_.rename("Inverse Power"),
           cdf_xp.tail_.rename("Exponential")), axis=1).plot(
        kind='bar', grid=True, title='Curve Fit Comparison').set(
        xlabel='Industry', ylabel='Tail Factor');
