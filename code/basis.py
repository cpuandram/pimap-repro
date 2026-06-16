"""Low-dimensional basis helpers."""

from __future__ import annotations

import numpy as np

from spectral import SpectralData


def gaussian_basis(wl: np.ndarray, mu: float, sigma: float, dl: float) -> np.ndarray:
    """Evaluate one normalized Gaussian basis row.

    Args:
        wl: Wavelength grid in nanometers.
        mu: Gaussian center in nanometers.
        sigma: Gaussian standard deviation in nanometers.
        dl: Wavelength step in nanometers.
    """

    g = np.exp(-0.5 * ((wl - mu) / sigma) ** 2)
    return (g / (np.sum(g) * dl)) * dl


def gaussian_basis_matrix(wl: np.ndarray, centers, sigmas) -> np.ndarray:
    """Build normalized Gaussian basis rows.

    Args:
        wl: Wavelength grid in nanometers.
        centers: Gaussian centers in nanometers, one per channel.
        sigmas: Gaussian standard deviations in nanometers, one per channel.
    """

    rows = []
    for mu, sigma in zip(centers, sigmas):
        rows.append(gaussian_basis(wl, float(mu), float(sigma), float(wl[1] - wl[0])))
    return np.vstack(rows)


def xyz_reflectance_basis(data: SpectralData) -> np.ndarray:
    """Return the fixed 3-channel XYZ/albedo reflectance basis.

    Args:
        data: Spectral context containing the wavelength-weighted XYZ CMFs.
    """

    z = np.sum(data.xyz_cmf_d, axis=0)
    return (data.xyz_cmf_d / z[None, :]).T


def darling_basis_matrix(data: SpectralData) -> np.ndarray:
    """Return the fixed 6-Gaussian Darling basis.

    Args:
        data: Spectral context that supplies the wavelength grid and step size.

    Returns:
        Basis matrix shaped `(6,M)`, with rows normalized to sum to one.
    """

    centers = np.array([447.0, 481.5, 519.6, 543.1, 572.9, 622.4])
    sigmas = np.array([16.9, 4.3, 7.8, 9.4, 15.5, 18.0])
    return gaussian_basis_matrix(data.wl, centers, sigmas)
