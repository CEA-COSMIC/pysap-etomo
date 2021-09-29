"""
Wavelet class from pyWavelet module
"""
import numpy as np
import pywt
from .utils import (flatten_swtn, unflatten_swtn, flatten_wave, unflatten_wave)
from .base import LinearBase


class WaveletPywt(LinearBase):
    """ The 3D wavelet transform class from pyWavelets package"""

    def __init__(self, wavelet_name, nb_scale=4, undecimated=False,
                 mode='zero', **kwargs):
        """
        Initialize the 'pyWavelet3' class.

        Parameters
        ----------
        wavelet_name: str
            the wavelet name to be used during the decomposition.
        nb_scales: int, default 4
            the number of scales in the decomposition.
        undecimated: bool, default False
            enable use undecimated wavelet transform.
        mode : str or tuple of str, optional
            Signal extension mode, see :ref:`Modes <ref-modes>`. This can also
            be a tuple containing a mode to apply along each axis in ``axes``.
        kwargs:
            Keyword arguments for LinearBase initialization
        """
        super().__init__(**kwargs)
        self.name = wavelet_name
        if wavelet_name not in pywt.wavelist():
            raise ValueError("Unknown transformation '{}'".format(wavelet_name))
        self.pywt_transform = pywt.Wavelet(wavelet_name)
        self.nb_scale = nb_scale
        self.undecimated = undecimated
        self.unflatten = unflatten_swtn if undecimated else unflatten_wave
        self.flatten = flatten_swtn if undecimated else flatten_wave
        self.coeffs = None
        self.coeffs_shape = None
        self.mode = mode

    def get_coeff(self):
        """
        Return the wavelet coefficients

        Return:
        -------
        The wavelet coefficients values
        """
        return self.coeffs

    def set_coeff(self, coeffs):
        """ Set wavelets decomposition coefficients values"""
        self.coeffs = coeffs  # XXX: TODO: add some checks

    def _op(self, data):
        """
        Define the wavelet operator.
        This method returns the input data convolved with the wavelet filter.

        Parameters
        ----------
        data: np.ndarray(m', n') or np.ndarray(m', n', p')
            input 2D or 3D data array.

        Returns
        -------
        coeffs: np.ndarray
            the wavelet coefficients.
        """
        if self.undecimated:
            coeffs_dict = pywt.swtn(data, self.pywt_transform,
                                    level=self.nb_scale)
            coeffs, self.coeffs_shape = self.flatten(coeffs_dict)
            return coeffs
        else:
            coeffs_dict = pywt.wavedecn(data,
                                        self.pywt_transform,
                                        level=self.nb_scale,
                                        mode=self.mode)
            self.coeffs, self.coeffs_shape = self.flatten(coeffs_dict)
            return self.coeffs

    def _adj_op(self, coeffs):
        """
        Define the wavelet adjoint operator.
        This method returns the reconsructed image.

        Parameters
        ----------
        coeffs: np.ndarray
            the wavelet coefficients.

        Returns
        -------
        data: np.ndarray((m, n)) or np.ndarray((m, n, p))
            the 2D or 3D reconstructed data.
        """
        self.coeffs = coeffs
        if self.undecimated:
            coeffs_dict = self.unflatten(coeffs, self.coeffs_shape)
            data = pywt.iswtn(coeffs_dict,
                              self.pywt_transform)
        else:
            coeffs_dict = self.unflatten(coeffs, self.coeffs_shape)
            data = pywt.waverecn(
                coeffs=coeffs_dict,
                wavelet=self.pywt_transform,
                mode=self.mode)
        return data

    def __str__(self):
        return (self.name + ' wavelet - number of scales = ' +
                str(self.nb_scale))
