.. _installation-instructions:

=======================
Installing chainladder
=======================

General Installation
======================

``chainladder-python`` is available through ``pip`` and ``conda``.


.. tabbed:: pip

    |PyPI version|

    Installing `chainladder` from the `Pipi` channel can be achieved by using ``pip``:
    
    ```
    pip install chainladder
    ```

    Alternatively, if you have git and want to play with unreleased features, you can
    install directly from [Github](https://github.com/casact/chainladder-python/):

    ```
    pip install git+https://github.com/casact/chainladder-python/
    ```


.. tabbed:: conda

    |Conda Version| |Conda Downloads|

    Installing `chainladder` from the `conda-forge` channel can be achieved by adding `conda-forge` to 
    your channels with:

    ```
    conda config --add channels conda-forge
    conda config --set channel_priority strict
    ```

    Once the `conda-forge` channel has been enabled, `chainladder` can be installed with:

    ```
    conda install chainladder
    ```

    It is possible to list all of the versions of `chainladder` available on your platform with:

    ```
    conda search chainladder --channel conda-forge
    ```


Developer Installation
============================

If you're interested in contributing, please refer to :ref:`Contributing <contributing>`
for information on the developer environment.


.. |Conda Downloads| image:: https://img.shields.io/conda/dn/conda-forge/chainladder.svg
   :target: https://anaconda.org/conda-forge/chainladder

.. |PyPI version| image:: https://badge.fury.io/py/chainladder.svg
   :target: https://badge.fury.io/py/chainladder

.. |Conda Version| image:: https://img.shields.io/conda/vn/conda-forge/chainladder.svg
   :target: https://anaconda.org/conda-forge/chainladder
