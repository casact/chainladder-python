# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 07:01:54 2017

@author: jboga
"""

descr = "Chainladder Package"

from distutils.core import setup

setup(
    name='chainladder',
    version='0.0.1',
    maintainer='John Bogaardt',
    maintainer_email='jbogaardt@gmail.com',
    packages=['chainladder'],
    scripts=[],
    url='https://github.com/jbogaardt/chainladder-python',
    download_url='https://github.com/jbogaardt/chainladder-python/archive/v0.1.0.tar.gz'
    license='LICENSE',
    include_package_data=True,
    description= descr,
    #long_description=open('README.rst').read(),
    install_requires=[
        "pandas>=0.20.2",
        "numpy>=1.12.1",
        "matplotlib>=2.0.2",
        "seaborn>=0.7.1",
        "scipy>=0.19.0",
        "statsmodels>=0.8.0"
    ],
)
