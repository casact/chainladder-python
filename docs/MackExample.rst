
Mack chainladder model
======================

We will explore the properties and methods underlying the
MackChainladder class.

As usual, we we import the chainladder package as well as the popular
pandas package. For plotting purposes, we will also be using Jupyter's
``%matplotlib inline`` magic function.

Load package and data
~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import chainladder as cl
    import pandas as pd
    %matplotlib inline


.. parsed-literal::

    C:\Users\jboga\Anaconda3\lib\site-packages\statsmodels\compat\pandas.py:56: FutureWarning: The pandas.core.datetools module is deprecated and will be removed in a future version. Please use the pandas.tseries module instead.
      from pandas.core import datetools
    

We will be exploring the MackChainladder class on the ``GenIns`` dataset
included in the **chainladder** package. Let's load the triangle and
look at it.

.. code:: ipython3

    GI = cl.load_dataset('GenIns')
    GI_tri = cl.Triangle(GI)
    GI_tri.data




.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th>dev</th>
          <th>1</th>
          <th>2</th>
          <th>3</th>
          <th>4</th>
          <th>5</th>
          <th>6</th>
          <th>7</th>
          <th>8</th>
          <th>9</th>
          <th>10</th>
        </tr>
        <tr>
          <th>origin</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>1.0</th>
          <td>357848</td>
          <td>1124788.0</td>
          <td>1735330.0</td>
          <td>2218270.0</td>
          <td>2745596.0</td>
          <td>3319994.0</td>
          <td>3466336.0</td>
          <td>3606286.0</td>
          <td>3833515.0</td>
          <td>3901463.0</td>
        </tr>
        <tr>
          <th>2.0</th>
          <td>352118</td>
          <td>1236139.0</td>
          <td>2170033.0</td>
          <td>3353322.0</td>
          <td>3799067.0</td>
          <td>4120063.0</td>
          <td>4647867.0</td>
          <td>4914039.0</td>
          <td>5339085.0</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>3.0</th>
          <td>290507</td>
          <td>1292306.0</td>
          <td>2218525.0</td>
          <td>3235179.0</td>
          <td>3985995.0</td>
          <td>4132918.0</td>
          <td>4628910.0</td>
          <td>4909315.0</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>4.0</th>
          <td>310608</td>
          <td>1418858.0</td>
          <td>2195047.0</td>
          <td>3757447.0</td>
          <td>4029929.0</td>
          <td>4381982.0</td>
          <td>4588268.0</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>5.0</th>
          <td>443160</td>
          <td>1136350.0</td>
          <td>2128333.0</td>
          <td>2897821.0</td>
          <td>3402672.0</td>
          <td>3873311.0</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>6.0</th>
          <td>396132</td>
          <td>1333217.0</td>
          <td>2180715.0</td>
          <td>2985752.0</td>
          <td>3691712.0</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>7.0</th>
          <td>440832</td>
          <td>1288463.0</td>
          <td>2419861.0</td>
          <td>3483130.0</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>8.0</th>
          <td>359480</td>
          <td>1421128.0</td>
          <td>2864498.0</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>9.0</th>
          <td>376686</td>
          <td>1363294.0</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>10.0</th>
          <td>344014</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
      </tbody>
    </table>
    </div>



Create the MackChainladder model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create a MackChainladder model, we can specify up to four elements. A
triangle is the only non-optional element that needs to be specified to
create the model. Another parameter of interest we will be using here is
the alpha parameter.

| Thomas Mack establishes a parameter alpha as a way of generalizing the
  chainladder formula into a weighted least squares regression that
  works for:
| *alpha = 0* : straight average of link-ratios
| *alpha = 1* : volume weighted chainladder
| *alpha = 2* : ordinary least squares regression with intercept 0

The default parameter is *alpha = 1*

For all other parameters, please refer to the documentation of the
MackChainladder class.

**Load the Data**

.. code:: ipython3

    GI_mack = cl.MackChainladder(tri = GI_tri)

There are a variety of attributes and methods available in the
MackChainladder class. Most of these borrow notation similar to that of
the **`R chainladder <https://github.com/mages/ChainLadder>`__**
package, but there are a few differences. A complete list of attributes
and methods are shown below. Details on these are contained in the
`documentation <MackChainLadder.html>`__ of this module.

\*\* Available attributes and methods \*\*

.. code:: ipython3

    [item for item in dir(GI_mack) if item[:1]!='_']




.. parsed-literal::

    ['Fse',
     'age_to_age',
     'alpha',
     'chainladder',
     'dict_plot',
     'f',
     'fse',
     'full_triangle',
     'get_Fse',
     'get_parameter_risk',
     'get_process_risk',
     'get_tail_se',
     'get_tail_sigma',
     'get_tail_weighted_time_period',
     'get_total_parameter_risk',
     'is_exponential_tail_appropriate',
     'mack_se',
     'plot',
     'sigma',
     'summary',
     'total_mack_se',
     'total_parameter_risk',
     'total_process_risk',
     'triangle',
     'weights']



Mack model summary
~~~~~~~~~~~~~~~~~~

A useful method is the summary() method. This will produce, by origin
period, the IBNR estimate based off of the MackChainladder model as well
as its corresponding standard error. This is useful in gaining deeper
insight into the uncertainty in the model.

.. code:: ipython3

    GI_mack.summary().round(3)




.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }
    
        .dataframe thead th {
            text-align: left;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Latest</th>
          <th>Dev to Date</th>
          <th>Ultimate</th>
          <th>IBNR</th>
          <th>Mack S.E.</th>
          <th>CV(IBNR)</th>
        </tr>
        <tr>
          <th>origin</th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>1.0</th>
          <td>3901463.0</td>
          <td>1.000</td>
          <td>3901463.000</td>
          <td>0.000</td>
          <td>0.000</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2.0</th>
          <td>5339085.0</td>
          <td>0.983</td>
          <td>5433718.815</td>
          <td>94633.815</td>
          <td>71835.187</td>
          <td>0.759</td>
        </tr>
        <tr>
          <th>3.0</th>
          <td>4909315.0</td>
          <td>0.913</td>
          <td>5378826.290</td>
          <td>469511.290</td>
          <td>119473.736</td>
          <td>0.254</td>
        </tr>
        <tr>
          <th>4.0</th>
          <td>4588268.0</td>
          <td>0.866</td>
          <td>5297905.821</td>
          <td>709637.821</td>
          <td>131572.833</td>
          <td>0.185</td>
        </tr>
        <tr>
          <th>5.0</th>
          <td>3873311.0</td>
          <td>0.797</td>
          <td>4858199.639</td>
          <td>984888.639</td>
          <td>260530.015</td>
          <td>0.265</td>
        </tr>
        <tr>
          <th>6.0</th>
          <td>3691712.0</td>
          <td>0.722</td>
          <td>5111171.458</td>
          <td>1419459.458</td>
          <td>410406.890</td>
          <td>0.289</td>
        </tr>
        <tr>
          <th>7.0</th>
          <td>3483130.0</td>
          <td>0.615</td>
          <td>5660770.620</td>
          <td>2177640.620</td>
          <td>557795.542</td>
          <td>0.256</td>
        </tr>
        <tr>
          <th>8.0</th>
          <td>2864498.0</td>
          <td>0.422</td>
          <td>6784799.012</td>
          <td>3920301.012</td>
          <td>874882.218</td>
          <td>0.223</td>
        </tr>
        <tr>
          <th>9.0</th>
          <td>1363294.0</td>
          <td>0.242</td>
          <td>5642266.263</td>
          <td>4278972.263</td>
          <td>970959.785</td>
          <td>0.227</td>
        </tr>
        <tr>
          <th>10.0</th>
          <td>344014.0</td>
          <td>0.069</td>
          <td>4969824.694</td>
          <td>4625810.694</td>
          <td>1362981.070</td>
          <td>0.295</td>
        </tr>
      </tbody>
    </table>
    </div>



Plotting the Mack model
~~~~~~~~~~~~~~~~~~~~~~~

In many cases, we prefer a visual representation of the model, and can
represent much of the same data contained in the summary() method by
calling the plot() method.

The plot() method can be passed a list of desired plots or it can be
generically called to plot all available plots.

\*\* Individual plot \*\*

.. code:: ipython3

    GI_mack.plot(plots=['summary'])



.. parsed-literal::

    <matplotlib.figure.Figure at 0x19124f995c0>



.. image:: output_11_1.png


\*\* Plotting default (all plots) \*\*

.. code:: ipython3

    GI_mack.plot()



.. parsed-literal::

    <matplotlib.figure.Figure at 0x19125898d68>



.. image:: output_13_1.png

