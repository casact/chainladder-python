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
cdf_ip = cl.TailCurve(curve='inverse_power').fit(clrd).cdf_
cdf_xp = cl.TailCurve(curve='exponential').fit(clrd).cdf_
tail_ip = cdf_ip[cdf_ip.development==cdf_ip.development.iloc[-2]][cdf_ip.origin==cdf_ip.origin.max()]
tail_xp = cdf_xp[cdf_xp.development==cdf_xp.development.iloc[-2]][cdf_xp.origin==cdf_xp.origin.max()]

pd.concat((tail_ip.to_frame().rename("Inverse Power"),
           tail_xp.to_frame().rename("Exponential")), axis=1).plot(
        kind='bar', grid=True, title='Curve Fit Comparison').set(
        xlabel='Industry', ylabel='Tail Factor');
