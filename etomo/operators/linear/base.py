"""
Base class for linear operators
"""
import numpy as np
from joblib import Parallel, delayed
from modopt.opt.linear import LinearParent


class LinearBase(LinearParent):
    """
    Creates a base class for all linear operators. Ensures that the operator
    is defined in a way that is compatible with modopt reconstruction
    algorithms and handles multithreading for multichannel reconstructions.
    Any linear operator in pysap-etomo inherits from this class. To create
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
        data: np.ndarray of size ((n_channels,) (img_size,) img_size, img_size)
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
        fake_data = np.zeros(shape)
        fake_data[0] = 1

        # Call mr_transform
        data = self._op(fake_data)

        # Compute the L2 norm
        return np.linalg.norm(data)
