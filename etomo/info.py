# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

# Module current version
version_major = 0
version_minor = 0
version_micro = 1

# Expected by setup.py: string of form "X.Y.Z"
__version__ = "{0}.{1}.{2}".format(version_major, version_minor, version_micro)

# Expected by setup.py: the status of the project
CLASSIFIERS = ["Development Status :: 1 - Planning",
               "Environment :: Console",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Project descriptions
description = """
Python Sparse data Analysis Package external ETOMO plugin.
"""
SUMMARY = """
.. container:: summary-carousel

    Python Sparse data Analysis Package external electron tomography plugin.
"""
long_description = """
Python Sparse data Analysis Package external electron tomography (ETOMO) plugin.
"""

# Main setup parameters
NAME = "pysap-etomo"
ORGANISATION = "CEA"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
EXTRANAME = "COSMIC webPage"
EXTRAURL = "http://cosmic.cosmostat.org/"
URL = "https://github.com/CEA-COSMIC/pysap-etomo"
DOWNLOAD_URL = "https://github.com/CEA-COSMIC/pysap-etomo"
LICENSE = "CeCILL-B"
CLASSIFIERS = CLASSIFIERS
AUTHOR = """
Martin Jacob
Guillaume Biagi
Zineb Saghi
Patrick Quemere
Philippe Ciuciu
"""
AUTHOR_EMAIL = """
<martin.jacob@cea.fr>
<guillaume.biagi@cea.fr>
<zineb.saghi@cea.fr>
<patrick.quemere@cea.fr>
<philippe.ciuciu@cea.fr>
"""
PLATFORMS = "Linux,OSX"
ISRELEASE = True
VERSION = __version__
