
Quickstart Guide
================

Welcome to the **chainladder** quickstart guide. This tutorial will walk
you through basic functionality of the **chainladder** package.

Import the package just like any other. We will also include the popular
pandas and numpy packages to complete this tutorial.

.. code:: python

    import chainladder as cl
    import chainladder.deterministic as cld

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
datasets which we will access using the ``load_dataset`` function. For
the complete list, you can find them `here <Datasets.html>`__. We will
be using the *Reinsurance Association of America* (RAA) triangle.

.. code:: python

    RAA = cl.load_dataset('RAA')
    RAA.round(0)




==========  ==== ======== ======== ======== ======== ======== ======== ======== ======== ========
dev            1        2        3        4        5        6        7        8        9       10
==========  ==== ======== ======== ======== ======== ======== ======== ======== ======== ========
**origin**
**1981**    5012   8269.0  10907.0  11805.0  13539.0  16181.0  18009.0  18608.0  18662.0  18834.0
**1982**     106   4285.0   5396.0  10666.0  13782.0  15599.0  15496.0  16169.0  16704.0      NaN
**1983**    3410   8992.0  13873.0  16141.0  18735.0  22214.0  22863.0  23466.0      NaN      NaN
**1984**    5655  11555.0  15766.0  21266.0  23425.0  26083.0  27067.0      NaN      NaN      NaN
**1985**    1092   9565.0  15836.0  22169.0  25955.0  26180.0      NaN      NaN      NaN      NaN
**1986**    1513   6445.0  11702.0  12935.0  15852.0      NaN      NaN      NaN      NaN      NaN
**1987**     557   4020.0  10946.0  12314.0      NaN      NaN      NaN      NaN      NaN      NaN
**1988**    1351   6947.0  13112.0      NaN      NaN      NaN      NaN      NaN      NaN      NaN
**1989**    3133   5395.0      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN
**1990**    2063      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN      NaN
==========  ==== ======== ======== ======== ======== ======== ======== ======== ======== ========


Building our first triangle
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The RAA data is still just a DataFrame, and needs to be turned into a
triangle object. Once we do this, we can perform various calculations on
the triangle, such as converting a cumulative triangle to incremental or
or representing the data in tabular form.

**Incremental triangle**

.. code:: python

    RAA_Triangle = cl.Triangle(RAA)
    RAA_Triangle.cum_to_incr()




==========  ==== ======= ======= ======= ======= ======= ======= ====== ====== ======
dev            1       2       3       4       5       6       7      8      9     10
==========  ==== ======= ======= ======= ======= ======= ======= ====== ====== ======
**origin**
**1981**    5012  3257.0  2638.0   898.0  1734.0  2642.0  1828.0  599.0   54.0  172.0
**1982**     106  4179.0  1111.0  5270.0  3116.0  1817.0  -103.0  673.0  535.0    NaN
**1983**    3410  5582.0  4881.0  2268.0  2594.0  3479.0   649.0  603.0    NaN    NaN
**1984**    5655  5900.0  4211.0  5500.0  2159.0  2658.0   984.0    NaN    NaN    NaN
**1985**    1092  8473.0  6271.0  6333.0  3786.0   225.0     NaN    NaN    NaN    NaN
**1986**    1513  4932.0  5257.0  1233.0  2917.0     NaN     NaN    NaN    NaN    NaN
**1987**     557  3463.0  6926.0  1368.0     NaN     NaN     NaN    NaN    NaN    NaN
**1988**    1351  5596.0  6165.0     NaN     NaN     NaN     NaN    NaN    NaN    NaN
**1989**    3133  2262.0     NaN     NaN     NaN     NaN     NaN    NaN    NaN    NaN
**1990**    2063     NaN     NaN     NaN     NaN     NaN     NaN    NaN    NaN    NaN
==========  ==== ======= ======= ======= ======= ======= ======= ====== ====== ======



**Triangle in tabular form**

.. code:: python

    RAA_Triangle.data_as_table().head()




========== ==== =======
origin      dev  values
========== ==== =======
**1981**      1  5012.0
**1982**      1   106.0
**1983**      1  3410.0
**1984**      1  5655.0
**1985**      1  1092.0
========== ==== =======



Performing chainladder calculations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use basic chainladder functionality, we will rely on the
**ChainLadder** class. This is a class that expands on the triangle
class and includes features about loss development (*using chainladder
techniques, of course*). To create a chainladder object, you will need
to supply a triangle object.

From above, we will supply our RAA\_Triangle object, and look at a quick
age-to-age summary using the ata() method.

.. code:: python

    RAA_CL = cld.Chainladder(RAA_Triangle)
    RAA_CL.age_to_age()




============ ========= ========= ========= ========= ========= ========= ========= ========= ========= =======
origin             1-2       2-3       3-4       4-5       5-6       6-7       7-8       8-9      9-10  10-Ult
============ ========= ========= ========= ========= ========= ========= ========= ========= ========= =======
**1981**      1.649840  1.319023  1.082332  1.146887  1.195140  1.112972  1.033261  1.002902  1.009217     1.0
**1982**     40.424528  1.259277  1.976649  1.292143  1.131839  0.993397  1.043431  1.033088       NaN     NaN
**1983**      2.636950  1.542816  1.163483  1.160709  1.185695  1.029216  1.026374       NaN       NaN     NaN
**1984**      2.043324  1.364431  1.348852  1.101524  1.113469  1.037726       NaN       NaN       NaN     NaN
**1985**      8.759158  1.655619  1.399912  1.170779  1.008669       NaN       NaN       NaN       NaN     NaN
**1986**      4.259749  1.815671  1.105367  1.225512       NaN       NaN       NaN       NaN       NaN     NaN
**1987**      7.217235  2.722886  1.124977       NaN       NaN       NaN       NaN       NaN       NaN     NaN
**1988**      5.142117  1.887433       NaN       NaN       NaN       NaN       NaN       NaN       NaN     NaN
**1989**      1.721992       NaN       NaN       NaN       NaN       NaN       NaN       NaN       NaN     NaN
**Average**     volume    volume    volume    volume    volume    volume    volume    volume    volume    tail
**LDF**        2.99936   1.62352   1.27089   1.17167   1.11338   1.04193   1.03326   1.01694   1.00922     1.0
**CDF**        8.92023   2.97405   1.83185   1.44139    1.2302   1.10492   1.06045   1.02631   1.00922     1.0
============ ========= ========= ========= ========= ========= ========= ========= ========= ========= =======



The ChainLadder class has a parameter delta. This is described by
Barnett/Zenwirth, and the default value is 1 and corresponds to a
volume-weighted loss development factor (LDF) pick.

You can directly play with the chainladder model attributes to get
things such as ldfs, cdfs, and complete triangles.

**LDF**\ s

.. code:: python

    round(RAA_CL.LDF,4)




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

.. code:: python

    round(RAA_CL.CDF,4)




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



**Ultimates**

.. code:: python

    RAA_CL.full_triangle['Ult'].astype(int)



.. parsed-literal::

    1981    18834
    1982    16857
    1983    24083
    1984    28703
    1985    28926
    1986    19501
    1987    17749
    1988    24019
    1989    16044
    1990    18402
    dtype: int32



Conclusion
~~~~~~~~~~

Well done on getting through the quickstart tutorial where we've covered
basic triangle data and chainladder functionality. A more generalized
framework to the Chainladder class is the MackChainLadder class which we
will review in the next example.
