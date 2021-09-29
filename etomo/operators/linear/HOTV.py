"""
TV and HOTV class
"""
import numpy as np
import scipy
from .base import LinearBase


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
