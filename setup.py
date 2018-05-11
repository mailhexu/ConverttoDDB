#!/usr/bin/env python
from setuptools import setup

setup(
    name='ConverttoDDB',
    version='0.1',
    description='convert Vasp/phonopy data to DDB files',
    packages=['ConverttoDDB'],
    scripts=[],
    install_requires=['numpy','matplotlib','scipy','ase', 'spglib', 'phonopy'],
    classifiers=[
        #'Development Status :: 3 - Alpha',
        #'License :: OSI Approved :: GPLv3 license',
    ])
