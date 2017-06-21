
Welcome to the 'chainladder' quickstart guide
=============================================

This tutorial will walk you through basic functionality of the
**chainladder** package.

Import the package just like any other. We will also include the popular
pandas and numpy packages to complete this tutorial.

.. code:: ipython3

    import chainladder as cl
    
    import numpy as np
    import pandas as pd

Loading data
~~~~~~~~~~~~

In order to use the functionality of the package, we need data.
Specifically, we need loss data that can be represented in triangle
form. The current release of **chainladder** does not include any data
pre-processing functionality. Instead it relies on data in the form of
the popular **pandas.DataFrame**. Data can be either tabular or already
summarized in triangular form.

The **chainladder** package comes pre-installed with a few generic
datasets which we will access using the :func:``load_dataset`` function.
For the complete list, you can find them here. We will be using the
*Reinsurance Association of America* (RAA) triangle.

.. code:: ipython3

    RAA = cl.load_dataset('RAA')
    RAA.round(0)




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
          <th>1981.0</th>
          <td>5012</td>
          <td>8269.0</td>
          <td>10907.0</td>
          <td>11805.0</td>
          <td>13539.0</td>
          <td>16181.0</td>
          <td>18009.0</td>
          <td>18608.0</td>
          <td>18662.0</td>
          <td>18834.0</td>
        </tr>
        <tr>
          <th>1982.0</th>
          <td>106</td>
          <td>4285.0</td>
          <td>5396.0</td>
          <td>10666.0</td>
          <td>13782.0</td>
          <td>15599.0</td>
          <td>15496.0</td>
          <td>16169.0</td>
          <td>16704.0</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1983.0</th>
          <td>3410</td>
          <td>8992.0</td>
          <td>13873.0</td>
          <td>16141.0</td>
          <td>18735.0</td>
          <td>22214.0</td>
          <td>22863.0</td>
          <td>23466.0</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1984.0</th>
          <td>5655</td>
          <td>11555.0</td>
          <td>15766.0</td>
          <td>21266.0</td>
          <td>23425.0</td>
          <td>26083.0</td>
          <td>27067.0</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1985.0</th>
          <td>1092</td>
          <td>9565.0</td>
          <td>15836.0</td>
          <td>22169.0</td>
          <td>25955.0</td>
          <td>26180.0</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1986.0</th>
          <td>1513</td>
          <td>6445.0</td>
          <td>11702.0</td>
          <td>12935.0</td>
          <td>15852.0</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1987.0</th>
          <td>557</td>
          <td>4020.0</td>
          <td>10946.0</td>
          <td>12314.0</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1988.0</th>
          <td>1351</td>
          <td>6947.0</td>
          <td>13112.0</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1989.0</th>
          <td>3133</td>
          <td>5395.0</td>
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
          <th>1990.0</th>
          <td>2063</td>
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



Building our first triangle
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The RAA data is still just a DataFrame, and needs to be turned into a
triangle object. Once we do this, we can perform various calculations on
the triangle, such as converting a cumulative triangle to incremental or
or representing the data in tabular form.

**Incremental triangle**

.. code:: ipython3

    RAA_Triangle = cl.Triangle(RAA)
    RAA_Triangle.cum2incr()




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
          <th>1981.0</th>
          <td>5012</td>
          <td>3257.0</td>
          <td>2638.0</td>
          <td>898.0</td>
          <td>1734.0</td>
          <td>2642.0</td>
          <td>1828.0</td>
          <td>599.0</td>
          <td>54.0</td>
          <td>172.0</td>
        </tr>
        <tr>
          <th>1982.0</th>
          <td>106</td>
          <td>4179.0</td>
          <td>1111.0</td>
          <td>5270.0</td>
          <td>3116.0</td>
          <td>1817.0</td>
          <td>-103.0</td>
          <td>673.0</td>
          <td>535.0</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1983.0</th>
          <td>3410</td>
          <td>5582.0</td>
          <td>4881.0</td>
          <td>2268.0</td>
          <td>2594.0</td>
          <td>3479.0</td>
          <td>649.0</td>
          <td>603.0</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1984.0</th>
          <td>5655</td>
          <td>5900.0</td>
          <td>4211.0</td>
          <td>5500.0</td>
          <td>2159.0</td>
          <td>2658.0</td>
          <td>984.0</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1985.0</th>
          <td>1092</td>
          <td>8473.0</td>
          <td>6271.0</td>
          <td>6333.0</td>
          <td>3786.0</td>
          <td>225.0</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1986.0</th>
          <td>1513</td>
          <td>4932.0</td>
          <td>5257.0</td>
          <td>1233.0</td>
          <td>2917.0</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1987.0</th>
          <td>557</td>
          <td>3463.0</td>
          <td>6926.0</td>
          <td>1368.0</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1988.0</th>
          <td>1351</td>
          <td>5596.0</td>
          <td>6165.0</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1989.0</th>
          <td>3133</td>
          <td>2262.0</td>
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
          <th>1990.0</th>
          <td>2063</td>
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



\*\* Triangle in tabular form\*\*

.. code:: ipython3

    RAA_Triangle.dataAsTable().head()




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
          <th>dev</th>
          <th>values</th>
        </tr>
        <tr>
          <th>origin</th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>1981.0</th>
          <td>1</td>
          <td>5012.0</td>
        </tr>
        <tr>
          <th>1982.0</th>
          <td>1</td>
          <td>106.0</td>
        </tr>
        <tr>
          <th>1983.0</th>
          <td>1</td>
          <td>3410.0</td>
        </tr>
        <tr>
          <th>1984.0</th>
          <td>1</td>
          <td>5655.0</td>
        </tr>
        <tr>
          <th>1985.0</th>
          <td>1</td>
          <td>1092.0</td>
        </tr>
      </tbody>
    </table>
    </div>



Performing chainladder calculations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use basic chainladder functionality, we will rely on the
**ChainLadder** class. This is a class that expands on the triangle
class and includes features about loss development (*using chainladder
techniques, of course*). To create a chainladder object, you will need
to supply a triangle object.

From above, we will supply our RAA\_Triangle object, and look at a quick
age-to-age summary using the ata() method.

.. code:: ipython3

    RAA_CL = cl.ChainLadder(RAA_Triangle)
    RAA_CL.ata()




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
          <th>1-2</th>
          <th>2-3</th>
          <th>3-4</th>
          <th>4-5</th>
          <th>5-6</th>
          <th>6-7</th>
          <th>7-8</th>
          <th>8-9</th>
          <th>9-10</th>
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
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>1981.0</th>
          <td>1.650</td>
          <td>1.319</td>
          <td>1.082</td>
          <td>1.147</td>
          <td>1.195</td>
          <td>1.113</td>
          <td>1.033</td>
          <td>1.003</td>
          <td>1.009</td>
        </tr>
        <tr>
          <th>1982.0</th>
          <td>40.425</td>
          <td>1.259</td>
          <td>1.977</td>
          <td>1.292</td>
          <td>1.132</td>
          <td>0.993</td>
          <td>1.043</td>
          <td>1.033</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1983.0</th>
          <td>2.637</td>
          <td>1.543</td>
          <td>1.163</td>
          <td>1.161</td>
          <td>1.186</td>
          <td>1.029</td>
          <td>1.026</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1984.0</th>
          <td>2.043</td>
          <td>1.364</td>
          <td>1.349</td>
          <td>1.102</td>
          <td>1.113</td>
          <td>1.038</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1985.0</th>
          <td>8.759</td>
          <td>1.656</td>
          <td>1.400</td>
          <td>1.171</td>
          <td>1.009</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1986.0</th>
          <td>4.260</td>
          <td>1.816</td>
          <td>1.105</td>
          <td>1.226</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1987.0</th>
          <td>7.217</td>
          <td>2.723</td>
          <td>1.125</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1988.0</th>
          <td>5.142</td>
          <td>1.887</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1989.0</th>
          <td>1.722</td>
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
          <th>smpl</th>
          <td>8.206</td>
          <td>1.696</td>
          <td>1.315</td>
          <td>1.183</td>
          <td>1.127</td>
          <td>1.043</td>
          <td>1.034</td>
          <td>1.018</td>
          <td>1.009</td>
        </tr>
        <tr>
          <th>vwtd</th>
          <td>2.999</td>
          <td>1.624</td>
          <td>1.271</td>
          <td>1.172</td>
          <td>1.113</td>
          <td>1.042</td>
          <td>1.033</td>
          <td>1.017</td>
          <td>1.009</td>
        </tr>
      </tbody>
    </table>
    </div>



The ChainLadder class has a parameter delta. This is described by
Barnett/Zenwirth, and the default value is 1 and corresponds to a
volume-weighted loss development factor (LDFpick.

You can directly play with the chainladder model attributes to get
things such as ldfs, cdfs, and complete triangles.

**LDF**\ s

.. code:: ipython3

    LDF = pd.Series([ldf.coef_ for ldf in RAA_CL.models], index=RAA_CL.ata().columns)
    LDF.round(4)




.. parsed-literal::

    1-2     2.9994
    2-3     1.6235
    3-4     1.2709
    4-5     1.1717
    5-6     1.1134
    6-7     1.0419
    7-8     1.0333
    8-9     1.0169
    9-10    1.0092
    dtype: float64



**CDF**\ s

.. code:: ipython3

    CDF = LDF[::-1].cumprod()[::-1]
    CDF.round(4)




.. parsed-literal::

    1-2     8.9202
    2-3     2.9740
    3-4     1.8318
    4-5     1.4414
    5-6     1.2302
    6-7     1.1049
    7-8     1.0604
    8-9     1.0263
    9-10    1.0092
    dtype: float64



**Completed Triangle**

.. code:: ipython3

    RAA_CL.predict().round(0)




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
          <th>1981.0</th>
          <td>5012</td>
          <td>8269.0</td>
          <td>10907.0</td>
          <td>11805.0</td>
          <td>13539.0</td>
          <td>16181.0</td>
          <td>18009.0</td>
          <td>18608.0</td>
          <td>18662.0</td>
          <td>18834.0</td>
        </tr>
        <tr>
          <th>1982.0</th>
          <td>106</td>
          <td>4285.0</td>
          <td>5396.0</td>
          <td>10666.0</td>
          <td>13782.0</td>
          <td>15599.0</td>
          <td>15496.0</td>
          <td>16169.0</td>
          <td>16704.0</td>
          <td>16858.0</td>
        </tr>
        <tr>
          <th>1983.0</th>
          <td>3410</td>
          <td>8992.0</td>
          <td>13873.0</td>
          <td>16141.0</td>
          <td>18735.0</td>
          <td>22214.0</td>
          <td>22863.0</td>
          <td>23466.0</td>
          <td>23863.0</td>
          <td>24083.0</td>
        </tr>
        <tr>
          <th>1984.0</th>
          <td>5655</td>
          <td>11555.0</td>
          <td>15766.0</td>
          <td>21266.0</td>
          <td>23425.0</td>
          <td>26083.0</td>
          <td>27067.0</td>
          <td>27967.0</td>
          <td>28441.0</td>
          <td>28703.0</td>
        </tr>
        <tr>
          <th>1985.0</th>
          <td>1092</td>
          <td>9565.0</td>
          <td>15836.0</td>
          <td>22169.0</td>
          <td>25955.0</td>
          <td>26180.0</td>
          <td>27278.0</td>
          <td>28185.0</td>
          <td>28663.0</td>
          <td>28927.0</td>
        </tr>
        <tr>
          <th>1986.0</th>
          <td>1513</td>
          <td>6445.0</td>
          <td>11702.0</td>
          <td>12935.0</td>
          <td>15852.0</td>
          <td>17649.0</td>
          <td>18389.0</td>
          <td>19001.0</td>
          <td>19323.0</td>
          <td>19501.0</td>
        </tr>
        <tr>
          <th>1987.0</th>
          <td>557</td>
          <td>4020.0</td>
          <td>10946.0</td>
          <td>12314.0</td>
          <td>14428.0</td>
          <td>16064.0</td>
          <td>16738.0</td>
          <td>17294.0</td>
          <td>17587.0</td>
          <td>17749.0</td>
        </tr>
        <tr>
          <th>1988.0</th>
          <td>1351</td>
          <td>6947.0</td>
          <td>13112.0</td>
          <td>16664.0</td>
          <td>19525.0</td>
          <td>21738.0</td>
          <td>22650.0</td>
          <td>23403.0</td>
          <td>23800.0</td>
          <td>24019.0</td>
        </tr>
        <tr>
          <th>1989.0</th>
          <td>3133</td>
          <td>5395.0</td>
          <td>8759.0</td>
          <td>11132.0</td>
          <td>13043.0</td>
          <td>14521.0</td>
          <td>15130.0</td>
          <td>15634.0</td>
          <td>15898.0</td>
          <td>16045.0</td>
        </tr>
        <tr>
          <th>1990.0</th>
          <td>2063</td>
          <td>6188.0</td>
          <td>10046.0</td>
          <td>12767.0</td>
          <td>14959.0</td>
          <td>16655.0</td>
          <td>17353.0</td>
          <td>17931.0</td>
          <td>18234.0</td>
          <td>18402.0</td>
        </tr>
      </tbody>
    </table>
    </div>



Conclusion
~~~~~~~~~~

Well done on getting through the quickstart tutorial where we've covered
basic triangle data and chainladder functionality. A more generalized
framework to the Chainladder class is the MackChainLadder class which we
will review in the next example.
