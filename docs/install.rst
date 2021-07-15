.. _installation-instructions:

=======================
Installing chainladder
=======================

Basic Installation
============================

There are different ways to install chainladder,  The easiest way is with ``pip``:

    ``pip install chainladder``

Alternatively, if you have git and want to play with unreleased features, you can
install from git

  ``pip install git+https://github.com/casact/chainladder-python/``

Finally, ``chainladder`` is also hosted as a conda package in the conda-forge channel

  ``conda install -c conda-forge chainladder``

Keeping Packages Updated
============================

If you want to use ``pip``, the code is a bit messy, as there isn't a built-in flag yet:

  ``pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1  | xargs -n1 pip install -U``

Alternatively, you can use conda:

  ``conda update --all``

Developer Installation
============================

If you're interested in contributing, please refer to :ref:`Contributing <contributing>`
for information on the developer environment.
