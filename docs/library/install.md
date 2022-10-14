(installation-instructions)=
# Installation


## General Installation


There are two ways to install the chainladder package, using `pip` or `conda`:

````{tab-set}
```{tab-item} pip

[![](https://badge.fury.io/py/chainladder.svg)](https://anaconda.org/conda-forge/chainladder) 
[![](https://pepy.tech/badge/chainladder)](https://pepy.tech/project/chainladder)


Installing `chainladder` using `pip`:

`pip install chainladder`

Alternatively, if you have git and want to enjoy unreleased features, you can
install directly from `Github`:

`pip install git+https://github.com/casact/chainladder-python/`
```

```{tab-item} conda

[![](https://img.shields.io/conda/vn/conda-forge/chainladder.svg)](https://anaconda.org/conda-forge/chainladder) 
[![](https://img.shields.io/conda/dn/conda-forge/chainladder.svg)](https://anaconda.org/conda-forge/chainladder)


Installing `chainladder` using `conda`:

`conda install -c conda-forge chainladder`
```
````

## Developer Installation


If you're interested in contributing, please refer to [Contributing](contributing)
for information on the developer environment.



## Keeping Packages Updated


Depends on how you first install the package, to update the package through `pip` and `conda`:

````{tab-set}
```{tab-item} pip

  If you want to use ``pip``, the code is a bit messy, as there isn't a built-in flag yet.

  ``pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1  | xargs -n1 pip install -U``
```
```{tab-item} conda

  Using ``conda`` is simple:

  ``conda update --all``
```
````