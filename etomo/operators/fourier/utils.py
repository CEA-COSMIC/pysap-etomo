# -*- coding: utf-8 -*-
##########################################################################
# pySAP - Copyright (C) CEA, 2017 - 2018
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################
"""
This module contains usefull methods for electron tomography applications.
"""
import numpy as np
import scipy.fftpack as pfft

def generate_locations_etomo_3D(size_x, size_z, angles):
    """
    This function generate the list of the samples coordinate in the k-space.

    Parameters
    ----------
    size_x: int
        image size along the x-axis
    size_z: int
        image size along the z-axis (rotation axis)
    angles: np.ndarray((q))
        array countaining the acquisition angles

    Returns
    -------
    samples: np.ndarray((int(np.floor(np.sqrt(2) * size_x)) * size_z * q, 3))
        Fourier space locations generated from the given angles and data image
        size

    """
    diag_x = int(np.floor(np.sqrt(2) * size_x))
    rho = np.tile(np.linspace(-0.5, 0.5, diag_x, endpoint=False), size_z)
    k_z = np.tile(np.reshape(np.linspace(-0.5, 0.5, size_z, endpoint=False),
                             (size_z, 1)), (1, diag_x)).flatten()
    for t, angle in enumerate(angles):
        sample = np.zeros((diag_x * size_z, 3))
        sample[:, 0] = rho * np.cos(angle * 1.0 * np.pi / 180.)
        sample[:, 1] = rho * np.sin(angle * 1.0 * np.pi / 180.)
        sample[:, 2] = k_z
        if t == 0:
            samples = sample
        else:
            samples = np.concatenate([samples, sample])
        samples = np.asarray(samples)

    return samples


def generate_locations_etomo_2D_SL(size_x, angles):
    """
    This function generate the list of the samples coordinate in the k-space.

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
    rho = np.linspace(-0.5, 0.5, size_x, endpoint=False)
    for t, angle in enumerate(angles):
        sample = np.zeros((size_x, 2))
        sample[:, 0] = rho * np.cos(angle * 1.0 * np.pi / 180.)
        sample[:, 1] = rho * np.sin(angle * 1.0 * np.pi / 180.)
        if t == 0:
            samples = sample
        else:
            samples = np.concatenate([samples, sample])
        samples = np.asarray(samples)

    return samples


def generate_locations_etomo_2D(size_x, angles):
    """
    This function generate the list of the samples coordinate in the k-space.

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
        sample[:, 0] = rho * np.cos(angle * 1.0 * np.pi / 180.)
        sample[:, 1] = rho * np.sin(angle * 1.0 * np.pi / 180.)
        if t == 0:
            samples = sample
        else:
            samples = np.concatenate([samples, sample])
        samples = np.asarray(samples)

    return samples


# def generate_mask_etomo_2D(size_x, angles):
#     """This function generates the mask locations of the sample.
#
#     Parameters
#     ----------
#     size_x: int
#         image size along the x-axis
#
#     angles: np.ndarray((q))
#         array countaining the acquisition angles
#
#     Returns
#     -------
#     mask: np.ndarray(())
#         Mask locations of the Fourier space data corresponding to the
#         acquired angles
#    """
#     mask = np.zeros((size_x, size_x))
#     Xx = []
#     Yy = []
#     for angle in angles:
#         rho = size_x * np.linspace(-0.5, 0.5, size_x, endpoint=False)
#         X = rho * np.cos(angle * np.pi / 180) + size_x / 2
#         Y = rho * np.sin(angle * np.pi / 180) + size_x / 2
#         X_mask = np.where(X >= size_x)
#         Y_mask = np.where(Y >= size_x)
#         X = np.delete(X, X_mask, 0)
#         X = np.delete(X, Y_mask, 0)
#         Y = np.delete(X, X_mask, 0) # BUG TO FIX ?
#         Y = np.delete(X, Y_mask, 0)
#         Xx = np.concatenate([Xx, X])
#         Yy = np.concatenate([Yy, Y])
#     mask[Xx.astype(int), Yy.astype(int)] = 1
#     return mask


def generate_kspace_etomo_2D(sinogram):
    """
    This function generate the list of the kspace observations.

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
    jmax = -int(np.ceil((np.floor(np.sqrt(2) * size_x) - size_x) / 2))
    sinograms_zp = np.zeros((nb_angles, diag_x))
    sinograms_zp[:, jmin:jmax] = sinogram

    # nb_angles, size_x = sinogram.shape
    # diag_x = int(np.floor(np.sqrt(2) * size_x))
    # sinograms_zp = np.zeros((nb_angles, diag_x))
    # sinograms_zp[:, (diag_x - size_x) / 2):-int(
    # np.ceil((np.floor(np.sqrt(2) * size_x) - size_x) / 2))] = sinogram

    ft_sinogram = []
    for t in range(sinogram.shape[0]):
        ft_sinogram.append(pfft.fftshift(pfft.fft(pfft.ifftshift(
            sinograms_zp[t].astype("complex128")))))

    ft_sinogram = np.asarray(ft_sinogram).flatten()
    kspace_obs = ft_sinogram.flatten()
    return kspace_obs


def generate_kspace_etomo_2D_SL(sinogram):
    """
    This function generate the list of the kspace observations.

    Parameters
    ----------
    sinogram: np.ndarray((q, m))
        sinogram with size nb_angles and size_x (m)

    Returns
    -------
    kspace_obs: np.ndarray((q*m))
        Fourier space values from the given sinogram
    """
    # nb_angles, Nx = sinogram.shape
    # sinograms_zp = np.zeros((nb_angles, int(np.floor(np.sqrt(2) * Nx))))
    # sinograms_zp[:, int(np.floor((np.floor(np.sqrt(2) * Nx) - Nx) / 2)):-int(
    #     np.ceil((np.floor(np.sqrt(2) * Nx) - Nx) / 2))] = sinogram

    ft_sinogram = []
    for t in range(sinogram.shape[0]):
        ft_sinogram.append(pfft.fftshift(pfft.fft(pfft.ifftshift(
            sinogram[t].astype("complex128")))))

    ft_sinogram = np.asarray(ft_sinogram).flatten()
    kspace_obs = ft_sinogram.flatten()
    return kspace_obs


def generate_kspace_etomo_3D(sinograms):
    """
    This function generates the list of the kspace observations.

    Parameters
    ----------
    sinogram: np.ndarray((q, m, p))
        sinogram with size nb_angles and size_x, size_z (m, p)

    Returns
    -------
    kspace_obs: np.ndarray((q*int(m*sqrt(2)*p))
        Fourier space values from the given sinogram
    """
    nb_angles, size_x, size_z = sinograms.shape
    diag_x = int(np.floor(np.sqrt(2) * size_x))
    jmin = int(np.floor((np.floor(np.sqrt(2) * size_x) - size_x) / 2))
    jmax = -int(np.ceil((np.floor(np.sqrt(2) * size_x) - size_x) / 2))
    sinograms_zp = np.zeros((nb_angles, diag_x, size_z))
    sinograms_zp[:, jmin:jmax, :] = sinograms

    ft_sinograms = []
    for t in range(nb_angles):
        ft_sinograms.append(pfft.fftshift(pfft.fft2(pfft.ifftshift(
            sinograms_zp[t].astype("complex128")))).T.flatten())

    ft_sinograms = np.asarray(ft_sinograms).flatten()
    kspace_obs = ft_sinograms.flatten()
    return kspace_obs
