# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from setuptools import setup, find_packages
from os import listdir

with open('requirements.txt', 'r') as f:
    dependencies = f.read().splitlines()

with open("README.rst") as f:
    long_desc = f.read()

descr = "Chainladder Package - P&C Loss Reserving package "
name = 'chainladder'
url = 'https://github.com/casact/chainladder-python'
version='0.8.5' # Put this in __init__.py

data_path = ''
setup(
    name=name,
    version=version,
    maintainer='John Bogaardt',
    maintainer_email='jbogaardt@gmail.com',
    packages=find_packages(include=["chainladder", "chainladder.*"]),
    scripts=[],
    url=url,
    download_url='{}/archive/v{}.tar.gz'.format(url, version),
    license='MPL-2.0',
    include_package_data=True,
    package_data={'data': [data_path + item
                           for item in listdir('chainladder{}'.format(data_path))]},
    description=descr,
    long_description=long_desc,
    install_requires=dependencies,
)
