"""
Radon operator based on Astra Toolbox
"""
import warnings
import numpy as np
from joblib import Parallel, delayed
ASTRA_AVAILABLE = True
try:
    import astra
except ImportError:
    ASTRA_AVAILABLE = False
    warnings.warn('astra not installed')


class RadonBase:
    """
    Base class for all Radon operators
    """

    def op(self, img):
        """
        Computes Radon direct operator

        Parameters
        ----------
        img: np.array
            input image

        Returns
        -------
        result: np.array
            sinogram of img
        """
        raise NotImplementedError('op not implemented')

    def adj_op(self, x):
        """
        Computes Radon adjoint operator on Radon space data x

        Parameters
        ----------
        x: np.array
            input sinogram

        Returns
        -------
        result: np.array
            back projection of x
        """
        raise NotImplementedError('adj_op not implemented')


class Radon2D(RadonBase):
    """
    Radon operator based on Astra Toolbox
    """

    def __init__(self, angles, img_size, n_channels=1, normalized=True,
                 gpu=False):
        """
        Initializes operator

        Parameters
        ----------
        angles: np.array
            angles acquired (in degrees)
        img_size: int
            size of the images (only square images)
        normalized: bool, default True
            tells if the operator is normalized or not.
        n_channels: int, default 1
            Number of images for multichannel reconstructions
        gpu: bool, default False
            use cuda implementation if True
        """
        if not ASTRA_AVAILABLE:
            raise ValueError(
                'astra-toolbox is not installed, this package must be '
                + 'installed to use this class.'
            )
        self.dtype = float
        self.n_coils = n_channels
        self.img_size = img_size
        self.shape = (img_size,) * 2
        if self.n_coils != 1:
            self.shape = (self.n_coils,) + self.shape
        self.angles = angles
        self.normalized = normalized
        self.norm_const = np.sqrt(len(angles)) if normalized else 1.
        self.implementation = 'cuda' if gpu else 'line'

        # Astra projector
        self.proj_geom = astra.create_proj_geom('parallel', 1.0, img_size,
                                                np.deg2rad(self.angles))
        self.vol_geom = astra.create_vol_geom(img_size, img_size)
        self.proj_id = astra.create_projector(self.implementation,
                                              self.proj_geom,
                                              self.vol_geom)

    def _op(self, img):
        """
        Computes sino of single image
        """
        return (
            astra.create_sino(data=np.array(img), proj_id=self.proj_id)[1]
            / self.norm_const
        )

    def op(self, img):
        """
        Returns sinogram of img as a vector

        Parameters
        ----------
        img: np.array of size ((nchannels,) img_size, img_size)
            input image

        Returns
        -------
        sinogram: np.array of size ((n_channels,) len(theta), img_size)
            sinogram of the image
        """
        # Single channel
        if self.n_coils == 1:
            return self._op(img)
        # Multithreading for cpu multichannel
        elif self.implementation != 'cuda':
            return np.array(Parallel(
                n_jobs=4,
                backend='threading',
                verbose=False
            )(
                delayed(self._op)
                (img[i])
                for i in np.arange(self.n_coils)
            ))
        # gpu multichannel
        else:
            return np.asarray([self._op(img[i]) for i in range(
                self.n_coils)])

    def _adj_op(self, x):
        """
        Computes backprojection of single set of coefficients
        """
        return (
            astra.creators.create_backprojection(
                data=np.array(x),
                proj_id=self.proj_id
            )[1]
            / self.norm_const
        )

    def adj_op(self, x):
        """
        Returns backprojection of a sinogram x

        Parameters
        ----------
        x: np.array of size ((n_channels,) len(theta), img_size)
            sinogram

        Returns
        -------
        img: np.array of size ((n_channels,) img_size, img_size)
            the backprojection
        """
        # Single channel
        if self.n_coils == 1:
            return self._adj_op(x)
        # Multithreading for cpu multichannel
        elif self.implementation != 'cuda':
            return np.array(Parallel(
                n_jobs=4,
                backend='threading',
                verbose=False
            )(
                delayed(self._adj_op)
                (x[i])
                for i in np.arange(self.n_coils)
            ))
        # gpu multichannel
        else:
            return np.asarray(
                [self._adj_op(x[i]) for i in range(self.n_coils)]
            )


class Radon3D(RadonBase):
    """
    Radon operator based on Astra Toolbox
    """

    def __init__(self, angles, img_size, nb_slices=None, n_channels=1,
                 normalized=True):
        """
        Initializes operator

        Parameters
        ----------
        angles: np.array
            angles acquired (in degrees)
        img_size: int
            size of the images (only square images)
        nb_slices: int, optional
            Number of slices. If None, assumes equal to img_size
        n_channels: int, default 1
            Number of images for multichannel reconstructions
        normalized: bool, default True
            tells if the operator is normalized or not.
        """
        if not ASTRA_AVAILABLE:
            raise ValueError(
                'astra-toolbox is not installed, this package must be '
                + 'installed to use this class.'
            )
        if nb_slices is None:
            nb_slices = img_size
        self.n_coils = n_channels
        self.dtype = float
        self.img_size = img_size
        self.shape = (nb_slices, img_size, img_size)
        if self.n_coils != 1:
            self.shape = (self.n_coils,) + self.shape
        self.angles = angles
        self.normalized = normalized
        self.norm_const = np.sqrt(len(angles)) if normalized else 1.

        # Astra projector
        self.proj_geom = astra.create_proj_geom('parallel3d', 1.0, 1.0,
                                                nb_slices, img_size,
                                                np.deg2rad(self.angles))
        self.vol_geom = astra.create_vol_geom(img_size, img_size, nb_slices)

        self.sino = astra.create_sino3d_gpu
        self.back_projection = astra.creators.create_backprojection3d_gpu

    def _op(self, img):
        """
        Computes sino of a single image
        """
        return self.sino(data=img, proj_geom=self.proj_geom,
                         vol_geom=self.vol_geom)[1] / self.norm_const

    def op(self, img):
        """
        Returns sinogram of img as a vector

        Parameters
        ----------
        img: np.array of size ((n_channels,) img_size, img_size, img_size)
            input image

        Returns
        -------
        sinogram: np.array of size ((n_channels,) img_size, len(theta),
                                    img_size,)
            sinogram of the image
        """
        if self.n_coils == 1:
            return self._op(img)
        else:
            return np.asarray([self._op(img[i]) for i in range(
                self.n_coils)])

    def _adj_op(self, x):
        """
        Computes back projection of single set of coefficients
        """
        return self.back_projection(
            data=x,
            proj_geom=self.proj_geom,
            vol_geom=self.vol_geom
        )[1] / self.norm_const

    def adj_op(self, x):
        """
        Returns backprojection of a sinogram x

        Parameters
        ----------
        x: np.array of size ((n_channels,) img_size, len(theta), img_size)
            sinogram

        Returns
        -------
        img: np.array of size ((n_channels,) img_size, img_size, img_size)
            the backprojection
        """
        if self.n_coils == 1:
            return self._adj_op(x)
        else:
            return np.asarray(
                [self._adj_op(x[i]) for i in range(self.n_coils)]
            )
