#!/bin/bash

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a

PKGS="python=${TRAVIS_PYTHON_VERSION}";

if [ ${FLAKE8} == true ]; then
  conda create -q -n chainladder ${PKGS} flake8;
  source activate chainladder;
else
  PKGS="${PKGS} pandas"; if [ ${PANDAS} ]; then PKGS="${PKGS}=${PANDAS}"; fi;
  PKGS="${PKGS} matplotlib"; if [ ${MATPLOTLIB} ]; then PKGS="${PKGS}=${MATPLOTLIB}"; fi;

  conda create -q -n chainladder --file ci_scripts/conda_requirements.txt ${PKGS};
  source activate chainladder;

  conda install -q -c conda-forge codecov;
  conda install -q -c r --file ci_scripts/conda_r_requirements.txt;

  # Install the R ChainLadder package for comparisons
  R -e "install.packages('ChainLadder', repos = 'http://cran.us.r-project.org')";
fi;
