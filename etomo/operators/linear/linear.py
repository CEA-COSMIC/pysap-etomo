# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
"""
This module contains sparsity operators classes.
"""
import numpy as np
import pywt
import scipy
from joblib import Parallel, delayed
from modopt.opt.linear import LinearParent

from .utils import (flatten_swtn, unflatten_swtn, flatten_wave, unflatten_wave)


class LinearBase(LinearParent):
    """
    Creates a base class for all linear operators. Ensures that the operator
    is defined in a way that is compatible with modopt reconstruction
    algorithms and handles multithreading for multichannel reconstructions.
    Any linear operator in pytetomo should inherit from this class. To create
    an operator based on this class, the direct and adjoint operators have to
    be implemented in the _op and _adj_op methods respectively.
    """

    def __init__(self, n_jobs=4, backend='threading', n_channels=1,
                 verbose=False):
        """
        Initializes the class

        Parameters
        ----------
        n_jobs: int
            Number of threads
        backend: string, default 'threading'
            Backend to use for parallel computations
        n_channels: int
            Number of channels
        verbose: bool
            Verbosity level
        """
        self.n_jobs = n_jobs
        self.backend = backend
        self.verbose = verbose
        self.n_coils = n_channels

    def _op(self, data):
        raise NotImplementedError('A direct operator _op is needed')

    def _adj_op(self, coeffs):
        raise NotImplementedError('An adjoint operator _adj_op is needed')

    def op(self, data):
        """
        Direct operator

        Parameters
        ----------
        data: np.ndarray of size ((n_channels,) (nb_proj,) img_size, img_size)
            2D or 3D images to compute operator on.

        Returns
        -------
        coeffs: np.ndarray
            Linear coefficients of imput images
        """
        if self.n_coils == 1:
            return self._op(data)
        else:
            coeffs = Parallel(n_jobs=self.n_jobs, backend=self.backend,
                              verbose=self.verbose)(
                delayed(self._op)(data[i])
                for i in np.arange(self.n_coils)
            )
            return np.asarray(coeffs)

    def adj_op(self, coeffs):
        """
        Adjoint operator

        Parameters
        ----------
        coeffs: np.ndarray
            Linear coefficients

        Returns
        -------
        images: np.ndarray
            Reconstructed data from imput coefficients
        """
        if self.n_coils == 1:
            return self._adj_op(coeffs)
        else:
            images = Parallel(n_jobs=self.n_jobs, backend=self.backend,
                              verbose=self.verbose)(
                delayed(self._adj_op)(coeffs[i])
                for i in np.arange(self.n_coils)
            )
            return np.asarray(images)

    def l2norm(self, shape):
        """
        Compute the L2 norm.

        Parameters
        ----------
        shape: tuple of int
            the 2D or 3D data shape.

        Returns
        -------
        norm: float
            the L2 norm.
        """
        # Create fake data
        if self.n_coils > 1:
            shape = shape[1:]
        shape = np.asarray(shape)
        shape += shape % 2
        fake_data = np.zeros(shape)
        np.put(fake_data, fake_data.size // 2, 1)

        # Call mr_transform
        data = self._op(fake_data)

        # Compute the L2 norm
        return np.linalg.norm(data)


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


class HOTV(LinearBase):
    """ The HOTV computation class for 2D image decomposition

    .. note:: At the moment, assumed that the image is square
    """

    def __init__(self, img_shape, order=1, **kwargs):
        """
        Initialize the 'HOTV' class.

        Parameters
        ----------
        img_shape: tuple of int
            image dimensions
        order: int, optional
            order of the differential operator used for the HOTV computation
        kwargs:
            Keyword arguments for LinearBase initialization
        """
        super().__init__(**kwargs)
        assert (img_shape[0] == img_shape[1])

        self.order = order
        self.img_size = img_shape[0]
        self.filter = np.zeros((order + 1, 1))
        for k in range(order + 1):
            self.filter[k] = (-1) ** (order - k) * scipy.special.binom(order, k)

        offsets_x = np.arange(order + 1)
        offsets_y = self.img_size * np.arange(order + 1)
        shape = (self.img_size ** 2,) * 2
        sparse_mat_x = scipy.sparse.diags(self.filter,
                                          offsets=offsets_x, shape=shape)
        sparse_mat_y = scipy.sparse.diags(self.filter,
                                          offsets=offsets_y, shape=shape)

        self.op_matrix = scipy.sparse.vstack([sparse_mat_x, sparse_mat_y])

    def _op(self, data):
        """
        Define the HOTV operator.
        This method returns the input data convolved with the HOTV filter.

        Parameters
        ----------
        data: np.ndarray((m', m'))
            input 2D data array.

        Returns
        -------
        coeffs: np.ndarray((2 * m' * m'))
            the variation values.
        """
        return self.op_matrix * (data.flatten())

    def _adj_op(self, coeffs):
        """
        Define the HOTV adjoint operator.
        This method returns the adjoint of HOTV computed image.

        Parameters
        ----------
        coeffs: np.ndarray((2 * m' * m'))
            the HOTV coefficients.

        Returns
        -------
        data: np.ndarray((m', m'))
            the reconstructed data.
        """
        return np.reshape(self.op_matrix.T * coeffs,
                          (self.img_size, self.img_size))

    def __str__(self):
        return ('HOTV, order ' + str(self.order))


class HOTV_3D(LinearBase):
    """
    The HOTV computation class for 3D image decomposition

    .. note:: At the moment, assumed that the image is square in x-y directions
    """

    def __init__(self, img_shape, nb_slices, order=1, **kwargs):
        """
        Initialize the 'HOTV' class.

        Parameters
        ----------
        img_shape: tuple of int
            image dimensions (assuming that the image is square)
        nb_slices: int
            number of slices in the 3D reconstructed image
        order: int, default is 1
            order of the differential operator used for the HOTV computation
        kwargs:
            Keyword arguments for LinearBase initialization
        """
        super().__init__(**kwargs)
        # assert (img_shape[0] == img_shape[1])

        self.order = order
        self.img_size = img_shape[0]
        self.nb_slices = nb_slices
        self.filter = np.zeros((order + 1, 1))
        for k in range(order + 1):
            self.filter[k] = (-1) ** (order - k) * scipy.special.binom(order, k)

        offsets_x = np.arange(order + 1)
        offsets_y = self.nb_slices * np.arange(order + 1)
        offsets_z = (self.img_size * self.nb_slices) * np.arange(order + 1)
        shape = ((self.img_size ** 2) * self.nb_slices,) * 2
        sparse_mat_x = scipy.sparse.diags(self.filter,
                                          offsets=offsets_x, shape=shape)
        sparse_mat_y = scipy.sparse.diags(self.filter,
                                          offsets=offsets_y, shape=shape)
        sparse_mat_z = scipy.sparse.diags(self.filter,
                                          offsets=offsets_z, shape=shape)

        self.op_matrix = scipy.sparse.vstack(
            [sparse_mat_x, sparse_mat_y, sparse_mat_z])

    def _op(self, data):
        """
        Define the HOTV operator.
        This method returns the input data convolved with the HOTV filter.

        Parameters
        ----------
        data: np.ndarray((m', m', p'))
            input 3D data array.

        Returns
        -------
        coeffs: np.ndarray((3 * m' * m' * p'))
            the variation values.
        """
        return self.op_matrix * (data.flatten())

    def _adj_op(self, coeffs):
        """
        Define the HOTV adjoint operator.
        This method returns the adjoint of HOTV computed image.

        Parameters
        ----------
        coeffs: np.ndarray((3 * m' * m' * p'))
            the HOTV coefficients.

        Returns
        -------
        data: np.ndarray((m', m', p'))
            the reconstructed data.
        """
        return np.reshape(self.op_matrix.T * coeffs,
                          (self.img_size, self.img_size, self.nb_slices))

    def __str__(self):
        return ('HOTV order ' + str(self.order))


class LinearCombo(LinearBase):
    """
    Wraps several linear operators in a single operator to use it in
    reconstructions

    Examples
    --------
    With PyETomo's linear operators:

    >>> haar = linear.pyWavelet('haar', nb_scale=3)
    >>> HOTV = linear.HOTV(phantom.shape, order=3)
    >>> combo = linear.LinearCombo(image.shape, (haar, 0.8), (HOTV, 0.2))

    Notes
    -----
    This class is used to solve reconstruction with several linear operators,
    but all of them will have the same regularization and this regularization
    should furthermore be applied coefficient by coefficient (for instance L1
    or squared L2 norm) for the reconstruction to be meaningful.

    It is not mandatory but recommended to have a sum of weights equal to 1
    to control the relative importance of each linear operator and then
    control the intensity of the penalty function with the regularization
    parameter in the reconstruction.
    """

    def __init__(self, img_shape, *args, **kwargs):
        """
        Creates the operator

        Parameters
        ----------
        img_shape: tuple
            shape of the data
        args: tuples (linear_op, weight)
            linear operators to be used and associated weight.
            Weights should sum to 1 to ensure a good convergence speed in
            modopt algorithms.
        kwargs:
            Keyword arguments for LinearBase initialization
        """
        super().__init__(**kwargs)
        self.img_shape = img_shape
        self.operators = args
        self.adj_indices = None
        self._create_indices()

    def _create_indices(self):
        """
        Creates a list of indices used for adjoint operator computations
        """
        self.adj_indices = [0]
        for (linear_op, _) in self.operators:
            self.adj_indices.append(self.adj_indices[-1] +
                                    np.prod(linear_op._op(np.zeros(
                                        self.img_shape)).shape))

    def _op(self, data):
        """
        Direct operator

        Parameters
        ----------
        data: np.ndarray
            Input image

        Returns
        -------
        coeffs: np.ndarray
            weighted results of operators
        """
        return np.concatenate([weight * linear_op._op(data) for (linear_op,
                                                                 weight) in
                               self.operators])

    def _adj_op(self, coeffs):
        """
        Adjoint operator

        Parameters
        ----------
        coeffs: np.ndarray
            Adjoint data

        Returns
        -------
        img: np.ndarray
            The reconstructed image
        """
        return np.sum([weight * linear_op._adj_op(coeffs[self.adj_indices[i]:
                                                         self.adj_indices[i +
                                                                          1]])
                       for i, (linear_op, weight) in enumerate(
                self.operators)], axis=0)

    def __str__(self):
        return ('Combination of:\n\t- ' + '\n\t- '.join([x[0].__str__() for x in
                                                         self.operators]))
