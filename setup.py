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
                    '/utils/data/abc.pkl',
                    '/utils/data/m3ir5.pkl',
                    '/utils/data/mcl.pkl',
                    '/utils/data/mortgage.pkl',
                    '/utils/data/mw2008.pkl',
                    '/utils/data/mw2014.pkl',
                    '/utils/data/quarterly.pkl',
                    '/utils/data/raa.pkl',
                    '/utils/data/ukmotor.pkl',
                    '/utils/data/usaa.pkl',
                    '/utils/data/auto.pkl',
                    '/utils/data/genins.pkl',
                    '/utils/data/liab.pkl',
                    '/utils/data/clrd.pkl']},
    description=descr,
    # long_description=open('README.md').read(),
    install_requires=[
        "pandas>=0.23.0",
        "numpy>=1.12.0",
        "scikit-learn>=0.19.0"
    ],
)
