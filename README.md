# pysap-etomo
PySAP plugin for electron tomography

Pysap-etomo is a plugin of the open-source Python library called PySAP (https://github.com/cea-cosmic/pySAP)
for compressed sensing tomographic reconstruction using gradient-based and wavelet-based
regularizations.
It was developed in the context of an interdisciplinary project involving CEA-Leti
(Martin Jacob, Jyh-Miin Lin, Guillaume Biagi, Zineb Saghi), CEA-NeuroSpin
(Loubna El Gueddari, Philippe Ciuciu) and CEA-CosmoStat (Samuel Farrens,
  Jean-Luc Starck). We acknowledge the financial support of the Cross-Disciplinary
  Program on Numerical Simulation of CEA, the French Alternative Energies and
  Atomic Energy Commission.

The current version of pysap-etomo contains 2D and 3D implementations of:

- Total variation (TV)
- Higher order TV (HOTV)
- Wavelets from PyWavelets library.
- Radon operators
- Fourier operators using the projection-slice theorem


# Dependencies
Prior to installation, make sure you have installed the astra toolbox that provides
the Radon operators for tomographic imaging: See installation details here:
https://www.astra-toolbox.com/ (For Linux/MacOs users,  install from sources on
  github: https://github.com/astra-toolbox/astra-toolbox). If you want to get
  access to GPU implementations of 2D/3D Radon operators, CUDA

During installation, Pyetomo will automatically install the following packages :
 - scipy
 - numpy
 - matplotlib
 - scikit-image
 - pywavelets
 - modopt


# Installation

To use the package, clone it or download it locally with the following command:

$ git clone https://github.com/cea-cosmic/pysap-etomo

Then run:

$ python setup.py install
