"""
==========================================
Berquist-Sherman Disposal Rate Adjustment
==========================================

This example demonstrates the adjustment to paid amounts and closed claim
counts using the Berquist-Sherman method.  The method calculates a `disposal_rate_`
using the `report_count_estimator`.  The disposal rates of the latest diagonal
are then used to infer adjustments to the inner diagonals of both the closed
claim triangle as well as the paid amount triangle.

"""
import chainladder as cl
import matplotlib.pyplot as plt


# Load data
triangle = cl.load_sample('berqsherm').loc['Auto']
# Specify Berquist-Sherman model
self = cl.BerquistSherman(
    paid_amount='Paid', incurred_amount='Incurred',
    reported_count='Reported', closed_count='Closed',
    reported_count_estimator=cl.Chainladder())

# Adjust our triangle data
berq_triangle = self.fit_transform(triangle)
berq_cdf = cl.Development().fit(berq_triangle).cdf_
orig_cdf = cl.Development().fit(triangle).cdf_

# Plot data
fig, ((ax0, ax1)) = plt.subplots(ncols=2, figsize=(15,5))
(berq_cdf['Paid'] / orig_cdf['Paid']).T.plot(
    kind='bar', grid=True, legend=False, ax=ax0,
    title='Berquist Sherman Paid to Unadjusted Paid').set(
    xlabel='Age to Ultimate', ylabel='Paid CDF Adjustment');

(berq_cdf['Closed'] / orig_cdf['Closed']).T.plot(
    kind='bar', grid=True, legend=False, ax=ax1,
    title='Berquist Sherman Closed Count to Unadjusted Closed Count').set(
    xlabel='Age to Ultimate', ylabel='Closed Count CDF Adjustment');
fig.suptitle("Berquist-Sherman Closure Rate Adjustments");
