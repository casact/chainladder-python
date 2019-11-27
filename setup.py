from setuptools import setup, find_packages
from os import listdir
# import chainladder

descr = "Chainladder Package - P&C Loss Reserving package "
name = 'chainladder'
url = 'https://github.com/jbogaardt/chainladder-python'
version='0.4.1' # Put this in __init__.py

data_path = ''
setup(
    name=name,
    version=version,
    maintainer='John Bogaardt',
    maintainer_email='jbogaardt@gmail.com',
    packages=['{}.{}'.format(name, p) for p in find_packages(where=name)]+['chainladder'],
    scripts=[],
    url=url,
    download_url='{}/archive/v{}.tar.gz'.format(url, version),
    license='LICENSE',
    include_package_data=True,
    package_data={'data': [data_path+item
                           for item in listdir('chainladder{}'.format(data_path))]},
    description=descr,
    # long_description=open('README.md').read(),
    install_requires=[
        "pandas>=0.23.0",
        "numpy>=1.12.0",
        "scikit-learn>=0.18.0",
        "joblib"
    ],
)
