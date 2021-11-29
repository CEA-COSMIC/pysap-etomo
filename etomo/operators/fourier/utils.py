"""
Functions to transform data from radon space to fourier space (only in 2D for
now). Density compensation taken from pysap-mri.
"""
import numpy as np
import scipy.fftpack as pfft


def generate_locations_etomo_2D(size_x, angles):
    """
    This function generates the list of the samples coordinate in the k-space.

    Parameters
    ----------
    size_x: int
        image size along the x-axis
    angles: np.ndarray((q))
        array countaining the acquisition angles

    Returns
    -------
    samples: np.ndarray((size_x * q, 2))
        Fourier space locations generated from the given angles and data image
        size
   """
    diag_x = int(np.floor(np.sqrt(2) * size_x))
    rho = np.linspace(-0.5, 0.5, diag_x, endpoint=False)
    for t, angle in enumerate(angles):
        sample = np.zeros((diag_x, 2))
        sample[:, 0] = rho * np.sin(angle * 1.0 * np.pi / 180.)
        sample[:, 1] = rho * np.cos(angle * 1.0 * np.pi / 180.)
        if t == 0:
            samples = sample
        else:
            samples = np.concatenate([samples, sample], axis=0)

    return samples


def generate_kspace_etomo_2D(sinogram):
    """
    This function generates the list of the kspace observations.

    Parameters
    ----------
    sinogram: np.ndarray((q, m))
        sinogram with size nb_angles and size_x (m)

    Returns
    -------
    kspace_obs: np.ndarray((q*int(m*sqrt(2)))
        Fourier space values from the given sinogram
    """
    nb_angles, size_x = sinogram.shape
    diag_x = int(np.floor(np.sqrt(2) * size_x))
    jmin = int(np.floor((np.floor(np.sqrt(2) * size_x) - size_x) / 2))
    jmax = int(np.ceil((size_x - np.floor(np.sqrt(2) * size_x)) / 2))
    sinograms_zp = np.zeros((nb_angles, diag_x))
    sinograms_zp[:, jmin:jmax] = sinogram

    ft_sinogram = []
    for t in range(sinogram.shape[0]):
        ft_sinogram = np.concatenate([ft_sinogram, pfft.fftshift(pfft.fft(
            pfft.ifftshift(sinograms_zp[t])))])

    return np.real(ft_sinogram)


def estimate_density_compensation(kspace_loc, volume_shape, num_iterations=10):
    """ Utils function to obtain the density compensator for a
    given set of kspace locations.

    Parameters
    ----------
    kspace_loc: np.ndarray
        the kspace locations
    volume_shape: np.ndarray
        the volume shape
    num_iterations: int default 10
        the number of iterations for density estimation
    """
    from .fourier import gpuNUFFT
    grid_op = gpuNUFFT(
        samples=kspace_loc,
        shape=volume_shape,
        osf=1
    )
    density_comp = np.ones(kspace_loc.shape[0])
    for _ in range(num_iterations):
        density_comp = (
                density_comp /
                np.abs(grid_op.op(grid_op.adj_op(density_comp, True), True))
        )
    return density_comp
