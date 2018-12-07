descr = "Chainladder Package - P&C actuarial package modeled after the R package of the same name"

#from distutils.core import setup
from setuptools import setup

setup(
    name='chainladder',
    version='0.1.7',
    maintainer='John Bogaardt',
    maintainer_email='jbogaardt@gmail.com',
    packages=['chainladder', 'chainladder.tails', 'chainladder.development',
              'chainladder.methods', 'chainladder.core', 'chainladder.utils'],
    scripts=[],
    url='https://github.com/jbogaardt/chainladder-python',
    download_url='https://github.com/jbogaardt/chainladder-python/archive/v0.1.7.tar.gz',
    license= 'LICENSE',
    include_package_data=True,
    package_data = {'data':[
                    '/utils/data/ABC',
                    '/utils/data/M3IR5',
                    '/utils/data/mcl',
                    '/utils/data/Mortgage',
                    '/utils/data/MW2008',
                    '/utils/data/MW2014',
                    '/utils/data/quarterly',
                    '/utils/data/RAA',
                    '/utils/data/UKMotor',
                    '/utils/data/USAA',
                    '/utils/data/auto',
                    '/utils/data/GenIns',
                    '/utils/data/liab',
                    '/utils/data/casresearch']},
    description= descr,
    #long_description=open('README.md').read(),
    install_requires=[
        "pandas>=0.21.0",
        "numpy>=1.12.1"
    ],
)
