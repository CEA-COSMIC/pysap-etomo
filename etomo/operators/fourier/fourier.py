"""
These operators are not necessary as the reconstruction is faster and more
direct with Radon operator, but they can be used to compare CS with ML
approaches as it is easier to find an implementation of fourier transform in
tensorflow than an implementation of the Radon transform
gpuNUFFT is taken directly from pysap-mri
"""
import warnings
import numpy as np

PYNUFFT_AVAILABLE = True
GPUNUFFT_AVAILABLE = True
try:
    import pynufft
except ImportError:
    PYNUFFT_AVAILABLE = False
    warnings.warn('pynufft not installed')
try:
    from gpuNUFFT import NUFFTOp
except ImportError:
    GPUNUFFT_AVAILABLE = False
    warnings.warn('gpuNUFFT not installed')


class FourierBase:
    """ Base Fourier transform operator class"""

    def op(self, img):
        """ This method calculates a Fourier transform"""
        raise NotImplementedError("'op' is an abstract method.")

    def adj_op(self, x):
        """ This method calculates the adjoint Fourier transform of a real or
        complex sequence"""
        raise NotImplementedError("'adj_op' is an abstract method.")


class NUFFT2(FourierBase):
    """ Standard 2D non catesian Fast Fourrier Transform class

    Attributes
    ----------
    samples: np.ndarray((m' * n, 2))
        samples coordinates in the Fourier domain.
    shape: tuple of int
        shape of the final reconstructed image (m, n) (not necessarily a
        square matrix).
    normalized: bool, default False
        tells if operator is normalized or not.
    """

    def __init__(self, samples, shape, normalized=True):
        """ Initialize the 'NUFFT2' class.

        Parameters
        ----------
        samples: np.ndarray((m' * n', 3))
            samples coordinates in the Fourier domain.
        shape: tuple of int
            shape of the final reconstructed image (m, n) (not necessarily a
            square matrix).
        normalized: bool, default False
            tells if operator is normalized or not.
        """
        self.dtype = float
        if not PYNUFFT_AVAILABLE:
            raise ValueError('pynufft is not installed. Please install it of '
                             'use pynfft instead.')
        self.n_coils = 1
        self.dim = samples.shape[1]
        self.plan = pynufft.NUFFT()
        shape_fourier = []
        for dim_size in shape:
            shape_fourier.append(int(2 * dim_size))
        self.Jd = (6, 6) if self.dim == 2 else (5, 5, 5)
        self.plan.plan(samples, tuple(shape), tuple(shape_fourier), self.Jd)
        self.shape = shape
        self.samples = samples
        self.normalized = normalized
        self.norm_const = np.sqrt(samples.shape[0]) if normalized else 1

    def op(self, img):
        """ This method calculates the masked non-cartesian Fourier transform
        of a 2-D image.

        Parameters
        ----------
        img: np.ndarray((m, n))
            input 2D array with the same shape as the mask.

        Returns
        -------
        x: np.ndarray((m * n))
            masked Fourier transform of the input image.
        """
        return np.real(self.plan.forward(img)) / self.norm_const

    def adj_op(self, x):
        """ This method calculates the adjoint non-cartesian Fourier
        transform of a 2-D coefficients array.

        Parameters
        ----------
        x: np.ndarray((m' * n'))
            masked non-cartesian Fourier transform 2D data.

        Returns
        -------
        img: np.ndarray((m, n))
            adjoint 2D discrete Fourier transform of the input coefficients.
        """
        return np.real(self.plan.adjoint(x)) * np.prod(self.plan.Kd) / \
               self.norm_const


class gpuNUFFT(FourierBase):
    """  GPU implementation of N-D non uniform Fast Fourrier Transform class.
    Attributes
    ----------
    samples: np.ndarray
        the normalized kspace location values in the Fourier domain.
    shape: tuple of int
        shape of the image
    operator: The NUFFTOp object
        to carry out operation
    n_coils: int default 1
            Number of coils used to acquire the signal in case of multiarray
            receiver coils acquisition. If n_coils > 1, please organize data as
            n_coils X data_per_coil
    """
    def __init__(self, samples, shape, density_comp=None, n_coils=1,
                 kernel_width=3, sector_width=8, osf=2, balance_workload=True,
                 smaps=None):
        """ Initilize the 'NUFFT' class.
        Parameters
        ----------
        samples: np.ndarray
            the kspace sample locations in the Fourier domain,
            normalized between -0.5 and 0.5
        shape: tuple of int
            shape of the image
        density_comp: np.ndarray default None.
            k-space weighting, density compensation, if not specified
            equal weightage is given.
        kernel_width: int default 3
            interpolation kernel width (usually 3 to 7)
        sector_width: int default 8
            sector width to use
        osf: int default 2
            oversampling factor (usually between 1 and 2)
        balance_workload: bool default True
            whether the workloads need to be balanced
        """
        self.dtype = "complex128"
        if not GPUNUFFT_AVAILABLE:
            raise ValueError('gpuNUFFT library is not installed, '
                             'please refer to README')

        self.shape = shape
        if density_comp is None:
            density_comp = np.ones(samples.shape[0])
        if smaps is None:
            self.uses_sense = False
        else:
            smaps = np.asarray(
                [np.reshape(smap_ch.T, smap_ch.size) for smap_ch in smaps]
            ).T
            self.uses_sense = True
        self.operator = NUFFTOp(
            np.reshape(samples, samples.shape[::-1], order='F'),
            shape,
            n_coils,
            smaps,
            density_comp,
            kernel_width,
            sector_width,
            osf,
            balance_workload
        )
        self.samples = samples

    def op(self, image, interpolate_data=False):
        """ This method calculates the masked non-cartesian Fourier transform
        of a 2D / 3D image.
        Parameters
        ----------
        image: np.ndarray
            input array with the same shape as shape.
        interpolate_data: bool, default False
            if set to True, the image is just apodized and interpolated to
            kspace locations. This is used for density estimation.
        Returns
        -------
        np.ndarray
            Non Uniform Fourier transform of the input image.
        """
        coeff = self.operator.op(
            np.reshape(image.T, image.size),
            interpolate_data
        )
        # Data is always returned as num_channels X coeff_array,
        # so for single channel, we extract single array
        if not self.uses_sense:
            coeff = coeff[0]
        return coeff

    def adj_op(self, coeff, grid_data=False):
        """ This method calculates adjoint of non-uniform Fourier
        transform of a 1-D coefficients array.
        Parameters
        ----------
        coeff: np.ndarray
            masked non-uniform Fourier transform 1D data.
        grid_data: bool, default False
            if True, the kspace data is gridded and returned,
            this is used for density compensation
        Returns
        -------
        np.ndarray
            adjoint operator of Non Uniform Fourier transform of the
            input coefficients.
        """
        image = self.operator.adj_op(coeff, grid_data)

        image = np.squeeze(image).T
        # The received data from gpuNUFFT is num_channels x Nx x Ny x Nz,
        # hence we use squeeze
        return np.squeeze(image)
