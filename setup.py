descr = "Chainladder Package - P&C actuarial package modeled after the R \
         package of the same name"

from setuptools import setup, find_packages, findall
name='chainladder'

setup(
    name=name,
    version='0.2.0',
    maintainer='John Bogaardt',
    maintainer_email='jbogaardt@gmail.com',
    packages=[f'{name}.{p}' for p in find_packages(where=name)],
    scripts=[],
    url='https://github.com/jbogaardt/chainladder-python',
    download_url='https://github.com/jbogaardt/chainladder-python/archive/v0.2.0.tar.gz',
    license='LICENSE',
    include_package_data=True,
    package_data={'data': [
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
    description=descr,
    # long_description=open('README.md').read(),
    install_requires=[
        "pandas>=0.23.0",
        "numpy>=1.12.0",
        "scikit-learn>=0.19.0"
    ],
)
