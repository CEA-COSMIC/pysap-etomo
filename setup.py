#! /usr/bin/env python
##########################################################################
# pySAP - Copyright (C) CEA, 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# System import
from __future__ import print_function
import os
from setuptools import setup, find_packages
try:
    from pip._internal.main import main as pip_main
except:
    from pip._internal import main as pip_main

# Global parameters
CLASSIFIERS = [
    "Development Status :: 1 - Planning",
    "Environment :: Console",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering"]
AUTHOR = """
Martin Jacob <martin.jacob@cea.fr>
Guillaume Biagi <guillaume.biagi@cea.fr>
Philippe Ciuciu <philippe.ciuciu@cea.fr>
Zineb Saghi <zineb.saghi@cea.fr>
Patrick Quemere <patrick.quemere@cea.fr>
"""
# Write setup
setup_requires = ["numpy>=1.16.4", "cython>=0.27.3", "pytest-runner"]

pip_main(['install'] + setup_requires)

setup(
    name="pysap-etomo",
    description="Python Sparse data Analysis Package external ETOMO plugin.",
    long_description="Python Sparse data Analysis Package external ETOMO plugin.",
    license="CeCILL-B",
    classifiers="CLASSIFIERS",
    author=AUTHOR,
    author_email="XXX",
    version="0.0.1",
    url="https://github.com/CEA-COSMIC/pysap-etomo",
    packages=find_packages(),
    setup_requires=setup_requires,
    install_requires=[
        "scikit-learn>=0.19.1",
        "progressbar2>=3.34.3",
        "joblib",
        "scipy>=1.3.0",
        "scikit-image",
    ],
    tests_require=['pytest>=5.0.1', 'pytest-cov>=2.7.1', 'pytest-pep8', 'pytest-runner'],
    platforms="OS Independent"
)
