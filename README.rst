|CI|_ |CodeCov|_ |Doc|_ |CircleCI|_

.. |CI| image:: https://github.com/CEA-COSMIC/pysap-etomo/actions/workflows/ci-build.yml/badge.svg?branch=master
.. _CI: https://github.com/CEA-COSMIC/pysap-etomo/actions?query=workflow%3ACI

.. |CodeCov| image:: https://codecov.io/gh/CEA-COSMIC/pysap-etomo/branch/master/graph/badge.svg?token=673YPRB88V
.. _CodeCov: https://codecov.io/gh/CEA-COSMIC/pysap-etomo

.. |Doc| image:: https://readthedocs.org/projects/pysap-etomo/badge/?version=latest
.. _Doc: https://pysap-etomo.readthedocs.io/en/latest/?badge=latest

.. |CircleCI| image:: https://circleci.com/gh/CEA-COSMIC/pysap-etomo.svg?style=svg
.. _CircleCI: https://circleci.com/gh/CEA-COSMIC/pysap-etomo

pySAP-etomo
===========

Python Sparse data Analysis Package external electron tomography (etomo) plugin.

Pysap-etomo is a plugin of the open-source Python library called PySAP (https://github.com/cea-cosmic/pySAP)
for compressed sensing tomographic reconstruction using gradient-based and wavelet-based
regularizations.
It was developed in the context of an interdisciplinary project involving CEA-Leti
(Martin Jacob, Jyh-Miin Lin, Guillaume Biagi, Zineb Saghi), CEA-NeuroSpin
(Loubna El Gueddari, Philippe Ciuciu) and CEA-CosmoStat (Samuel Farrens, Jean-Luc Starck).
We acknowledge the financial support of the Cross-Disciplinary Program on Numerical Simulation of CEA,
the French Alternative Energies and Atomic Energy Commission.

The current version of pysap-etomo contains 2D and 3D implementations of:

- Total variation (TV)
- Higher order TV (HOTV)
- Wavelets from PyWavelets library.
- Radon operators
- Fourier operators using the projection-slice theorem

Installation instructions
=========================

As of v0.0.6 of pysap, this plugin can be installed via

``pip install pysap --nosparse2D --only=pysap-etomo``

The ``--nosparse2D`` option is not mandatory, but the C++ Sparse2D library is not used in this plugin.

Special Installations
=====================

For Radon operators in 2D/3D (both CPU & GPU implementations
`astratoolbox <https://www.astra-toolbox.com/>`_

Linux/Mac:
``````````

``git clone github: https://github.com/astra-toolbox/astra-toolbox``

``cd build/linux``

``./autogen.sh   # when building a git version``

``./configure --with-cuda=/usr/local/cuda --with-python --with-install-type=module``

``make``

``make install``

Windows:
````````

``conda install -c astra-toolbox/label/dev astra-toolbox``
