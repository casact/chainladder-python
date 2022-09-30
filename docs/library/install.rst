.. _installation-instructions:

=======================
Installation
=======================

General Installation
======================

There are two ways to install the chainladder package, using `pip` or `conda`:

````{tab-set}
```{tab-item} pip

|PyPI version| |Pypi Downloads|

Installing `chainladder` using `pip`:

`pip install chainladder`

Alternatively, if you have git and want to enjoy unreleased features, you can
install directly from `Github`_:

`pip install git+https://github.com/casact/chainladder-python/`
```

```{tab-item} conda

|Conda Version| |Conda Downloads|

Installing `chainladder` using `conda`:

`conda install -c conda-forge chainladder`
```
````

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

.. |Pypi Downloads| image:: https://pepy.tech/badge/chainladder
   :target: https://pepy.tech/project/chainladder

.. _Github: https://github.com/casact/chainladder-python/

Keeping Packages Updated
============================

Depends on how you first install the package, to update the package through `pip` and `conda`:

.. tabbed:: pip

  If you want to use ``pip``, the code is a bit messy, as there isn't a built-in flag yet.

  ``pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1  | xargs -n1 pip install -U``


.. tabbed:: conda

  Using ``conda`` is simple:

  ``conda update --all``
