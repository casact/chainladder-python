"""
===========================
Actual Vs Expected Analysis
===========================

This example demonstrates how you can slice triangle objects to perform a
typical 'Actual vs Expected' analysis.  We will use Medical Malpractice
payment patterns for the demo.
"""

import chainladder as cl
import seaborn as sns
sns.set_style('whitegrid')

# Load the data
tri_1997 = cl.load_dataset('clrd')
tri_1997 = tri_1997.groupby('LOB').sum().loc['medmal']['CumPaidLoss']

# Create a triangle as of the previous valuation and build IBNR model
tri_1996 = tri_1997[tri_1997.valuation < '1997']
model_1996 = cl.Chainladder().fit(cl.TailCurve().fit_transform(tri_1996))

# Slice the expected losses from the 1997 calendar period of the model
ave = model_1996.full_triangle_.dev_to_val()
ave = ave[ave.development == '1997'].rename('columns', 'Expected')

# Slice the actual losses from the 1997 calendar period for prior AYs
ave['Actual'] = tri_1997.latest_diagonal[tri_1997.origin < '1997']
ave['Actual - Expected'] = ave['Actual'] - ave['Expected']

# Plotting
ave.to_frame().T.plot(y='Actual - Expected', kind='bar', legend=False) \
                .set(title='Calendar Period 1997 Performance',
                     xlabel='Accident Period', ylabel='Actual - Expected');
