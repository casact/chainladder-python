"""
===========================
Actual Vs Expected Analysis
===========================

This example demonstrates how you can slice triangle objects to perform a
typical 'Actual vs Expected' analysis.  We will use Medical Malpractice
payment patterns for the demo.
"""

import chainladder as cl
import matplotlib.pyplot as plt

# Load the data
tri_1997 = cl.load_sample('clrd')
tri_1997 = tri_1997.groupby('LOB').sum().loc['medmal']['CumPaidLoss']

# Create a triangle as of the previous valuation and build IBNR model
tri_1996 = tri_1997[tri_1997.valuation < '1997']
model_1996 = cl.Chainladder().fit(cl.TailCurve().fit_transform(tri_1996))

# Slice the expected losses from the 1997 calendar period of the model
ave = model_1996.full_triangle_.dev_to_val()
ave = ave[ave.valuation==tri_1997.valuation_date].rename('columns', 'Expected')

# Slice the actual losses from the 1997 calendar period for prior AYs
ave['Actual'] = tri_1997.latest_diagonal[tri_1997.origin < '1997']
df = ave.to_frame().T.iloc[::-1]

# Plotting
fig, ax = plt.subplots()
ax.grid(axis='x')
plt.hlines(y=df.index.astype(str), xmin=df['Actual'], xmax=df['Expected'],
           color='grey', alpha=0.4)
plt.scatter(df['Actual'], df.index.astype(str), color='navy', alpha=1, label='Actual')
plt.scatter(df['Expected'], df.index.astype(str), color='red', alpha=0.8 , label='Expected')
plt.legend()
plt.title("Actual vs Expected results in 1997")
plt.xlabel('Difference')
plt.ylabel('Origin')
