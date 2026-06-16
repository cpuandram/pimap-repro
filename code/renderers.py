"""Renderer implementations used by the paper reproduction."""

from __future__ import annotations

import numpy as np

from basis import darling_basis_matrix, xyz_reflectance_basis
from spectral import SpectralData, _as_list, _path_list


class Renderer:
    label = "Renderer"

    def integrate_xyz(self, data: SpectralData, ill_indices, path) -> np.ndarray:
        """Render one reflectance path under one or more illuminants.

        Args:
            data: Spectral context for normalized illuminants, spectra, and CMFs.
            ill_indices: Illuminant index, slice, list, or `None` for all illuminants.
            path: Reflectance index or sequence of reflectance indices. Repeated
                entries represent repeated bounces and are intentionally preserved.

        Returns:
            XYZ array shaped `(Nill,3)` in the same Y=100 convention as `data.xyzw`.
        """

        raise NotImplementedError


class ExactRenderer(Renderer):
    label = "Exact"

    def integrate_xyz(self, data: SpectralData, ill_indices, path) -> np.ndarray:
        ill = _as_list(ill_indices, data.illuminants.shape[0])
        r_ids = _path_list(path)
        r_prod = np.prod(data.reflectances[r_ids], axis=0)
        e = data.illuminants[ill]
        xyz = (e * r_prod[None, :]) @ data.xyz_cmf_d
        return xyz


class RGBAlbedoRenderer(Renderer):
    label = "RGB"

    def __init__(self, data: SpectralData):
        """Create the fixed 3-channel XYZ/albedo baseline.

        Args:
            data: Spectral context used to derive the fixed XYZ reflectance basis.
        """

        self.basis = xyz_reflectance_basis(data)

    def integrate_xyz(self, data: SpectralData, ill_indices, path) -> np.ndarray:
        ill = _as_list(ill_indices, data.illuminants.shape[0])
        r_ids = _path_list(path)
        e_feat = data.illuminants[ill] @ data.xyz_cmf_d
        x = np.ones(3, float)
        for r in r_ids:
            x *= self.basis @ data.reflectances[int(r)]
        return e_feat * x[None, :]


class DarlingSharedBasisRenderer(Renderer):
    label = "Darling"

    def __init__(self, data: SpectralData):
        """Create the original Darling-style shared 6-channel renderer.

        Args:
            data: Spectral context used to build the fixed Darling Gaussian basis.
        """

        self.basis = darling_basis_matrix(data)
        self.matrix_xyz: np.ndarray | None = None

    def _features(self, data: SpectralData, ill: int, path) -> np.ndarray:
        r_ids = _path_list(path)
        mu_e = self.basis @ data.illuminants[int(ill)]
        x = np.ones(self.basis.shape[0], float)
        for r in r_ids:
            x *= self.basis @ data.reflectances[int(r)]
        return mu_e * x

    def fit(self, data: SpectralData, ill_indices, rfl_indices, *, max_pairs: int, seed: int) -> None:
        """Fit the global 3x6 XYZ decode matrix from sampled one-bounce pairs.

        Args:
            data: Spectral context containing spectra and exact integration data.
            ill_indices: Illuminants used for fitting; `None` means all illuminants.
            rfl_indices: Reflectance indices used for fitting.
            max_pairs: Maximum illuminant/reflectance training pairs after sampling.
            seed: Random seed used when the full pair grid must be downsampled.
        """

        rng = np.random.default_rng(seed)
        pairs = [(int(ill), int(r)) for ill in _as_list(ill_indices, data.illuminants.shape[0]) for r in rfl_indices]
        if len(pairs) > max_pairs:
            keep = rng.choice(len(pairs), size=max_pairs, replace=False)
            pairs = [pairs[int(i)] for i in keep]

        x = np.vstack([self._features(data, ill, [r]) for ill, r in pairs])
        y = []
        for ill, r in pairs:
            e = data.illuminants[ill]
            rr = data.reflectances[r]
            y.append((e * rr) @ data.xyz_cmf_d)
        coeff, *_ = np.linalg.lstsq(x, np.vstack(y), rcond=None)
        self.matrix_xyz = coeff.T

    def integrate_xyz(self, data: SpectralData, ill_indices, path) -> np.ndarray:
        if self.matrix_xyz is None:
            raise RuntimeError("Call fit() before using DarlingSharedBasisRenderer.")
        ill = _as_list(ill_indices, data.illuminants.shape[0])
        feats = np.vstack([self._features(data, i, path) for i in ill])
        return feats @ self.matrix_xyz.T


class KernelDecoderBasisRenderer(Renderer):
    """Reflectance-only basis transport with per-illuminant GLS decoders."""

    def __init__(self, label: str, basis: np.ndarray, covariance: np.ndarray, jitter: float = 1e-8):
        """Create a renderer with a fixed reflectance basis and kernel-fit decoders.

        Args:
            label: Human-readable model name used in figures/tables.
            basis: Reflectance transport basis shaped `(K,M)`, usually nonnegative
                rows that map a spectrum to K coefficients.
            covariance: Reflectance-path covariance matrix `C_R` shaped `(M,M)`.
            jitter: Diagonal regularization added to `basis @ C_R @ basis.T`.
        """

        self.label = label
        self.basis = np.asarray(basis, float)
        self.covariance = np.asarray(covariance, float)
        self.jitter = float(jitter)
        self.decoders_xyz: np.ndarray | None = None

    def fit(self, data: SpectralData, ill_indices=None) -> None:
        """Fit one closed-form XYZ decoder per illuminant.

        Args:
            data: Spectral context containing illuminants and XYZ CMFs.
            ill_indices: Illuminants to fit; `None` fits every illuminant.
        """

        ill = _as_list(ill_indices, data.illuminants.shape[0])
        bcb = self.basis @ self.covariance @ self.basis.T
        bcb.flat[:: bcb.shape[0] + 1] += self.jitter
        inv = np.linalg.inv(bcb)
        decoders = np.zeros((data.illuminants.shape[0], 3, self.basis.shape[0]), float)
        c = data.xyz_cmf_d.T
        for i in ill:
            target = c * data.illuminants[i][None, :]
            decoders[i] = (target @ self.covariance @ self.basis.T) @ inv
        self.decoders_xyz = decoders

    def _path_feature(self, data: SpectralData, path) -> np.ndarray:
        x = np.ones(self.basis.shape[0], float)
        for r in _path_list(path):
            x *= self.basis @ data.reflectances[int(r)]
        return x

    def integrate_xyz(self, data: SpectralData, ill_indices, path) -> np.ndarray:
        if self.decoders_xyz is None:
            raise RuntimeError("Call fit() before using KernelDecoderBasisRenderer.")
        ill = _as_list(ill_indices, data.illuminants.shape[0])
        x = self._path_feature(data, path)
        return np.einsum("ijk,k->ij", self.decoders_xyz[ill], x)


class WandellWPcaRenderer(Renderer):
    """Uncentered weighted-PCA Wandell coefficient transport.

    This matches the cleaned Wandell WPCA model from the scratch code: no mean
    term, no centering, and exactly `k` transported coefficients.
    """

    def __init__(
        self,
        label: str,
        data: SpectralData,
        k: int,
        weights: np.ndarray | None = None,
        *,
        fit_indices=None,
        w_normalize: str = "mean1",
        w_eps: float = 1e-12,
    ):
        """Fit the uncentered weighted PCA basis and lighting matrices.

        Args:
            label: Human-readable model name used in output artifacts.
            data: Spectral context that supplies reflectances, illuminants, and CMFs.
            k: Number of PCA coefficients transported through the path.
            weights: Optional wavelength weights for the PCA inner product. When
                provided, PCs are orthonormal under `diag(weights)`.
            fit_indices: Reflectance indices used to fit the PCA basis. `None`
                fits on all reflectances in `data`.
            w_normalize: Weight normalization, one of `"mean1"`, `"sum1"`, or `"none"`.
            w_eps: Small floor used when normalizing weights and dividing by sqrt weights.
        """

        self.label = label
        self.k = int(k)
        self.wl = np.asarray(data.wl, float)
        self.m = self.wl.size

        if fit_indices is None:
            reflectances = np.asarray(data.reflectances, float)
        else:
            reflectances = np.asarray(data.reflectances[np.asarray(fit_indices, int)], float)
        if reflectances.ndim != 2 or reflectances.shape[1] != self.m:
            raise ValueError(f"reflectances must be (N,{self.m}), got {reflectances.shape}")

        self.use_weighted_pca = weights is not None
        if self.use_weighted_pca:
            w = np.asarray(weights, float).reshape(-1)
            if w.shape[0] != self.m:
                raise ValueError(f"weights must have length {self.m}, got {w.shape[0]}")
            if np.any(w < 0):
                raise ValueError("weights must be nonnegative.")
            w = np.maximum(w, 0.0)
            if w_normalize == "mean1":
                w = w / max(float(w.mean()), w_eps)
            elif w_normalize == "sum1":
                w = w / max(float(w.sum()), w_eps)
            elif w_normalize == "none":
                pass
            else:
                raise ValueError("w_normalize must be 'mean1', 'sum1', or 'none'")
            self.weights = w
            self.sqrt_weights = np.sqrt(np.maximum(w, w_eps))
        else:
            self.weights = None
            self.sqrt_weights = None

        self.pc = self._fit_pca_uncentered(reflectances)
        self.a_all = self._build_lighting_matrices(data)

    def _fit_pca_uncentered(self, reflectances: np.ndarray) -> np.ndarray:
        """Return the uncentered PCA rows from the full reflectance matrix.

        Args:
            reflectances: LuxPy reflectance matrix shaped `(Nref,M)`.
        """

        if not self.use_weighted_pca:
            _, _, vt = np.linalg.svd(reflectances, full_matrices=False)
            return vt[: self.k].copy()

        scaled = reflectances * self.sqrt_weights[None, :]
        _, _, vt = np.linalg.svd(scaled, full_matrices=False)
        return vt[: self.k] / self.sqrt_weights[None, :]

    def _encode(self, spectrum: np.ndarray) -> np.ndarray:
        """Project one reflectance spectrum into the K Wandell coefficients.

        Args:
            spectrum: Reflectance spectrum shaped `(M,)`.
        """

        spectrum = np.asarray(spectrum, float).reshape(self.m)
        if not self.use_weighted_pca:
            return self.pc @ spectrum
        return self.pc @ (self.weights * spectrum)

    def _path_coeff(self, data: SpectralData, path) -> np.ndarray:
        """Compute the coefficient-space product for a multi-bounce path.

        Args:
            data: Spectral context containing reflectance spectra.
            path: Reflectance index or sequence of indices. Repeats are kept.
        """

        r_ids = _path_list(path)
        if not r_ids:
            return self._encode(np.ones(self.m, float))

        coeff = np.ones(self.k, float)
        for r in r_ids:
            coeff *= self._encode(data.reflectances[int(r)])
        return coeff

    def _build_lighting_matrices(self, data: SpectralData) -> np.ndarray:
        """Precompute illuminant-specific Wandell lighting matrices.

        Args:
            data: Spectral context containing normalized illuminants and XYZ CMFs.

        Returns:
            Array shaped `(Nill,3,K)`.
        """

        a_all = np.zeros((data.illuminants.shape[0], 3, self.k), float)
        for ill, illuminant in enumerate(data.illuminants):
            for j in range(self.k):
                a_all[ill, :, j] = (illuminant * self.pc[j]) @ data.xyz_cmf_d
        return a_all

    def reconstruct_from_coeff(self, coeff: np.ndarray) -> np.ndarray:
        """Reconstruct one reflectance spectrum from K coefficients.

        Args:
            coeff: Coefficient vector shaped `(K,)`.
        """

        return np.asarray(coeff, float).reshape(self.k) @ self.pc

    def reconstruct_reflectances(self, data: SpectralData, reflectances: np.ndarray | None = None) -> np.ndarray:
        """Encode and reconstruct a batch of reflectances for diagnostics.

        Args:
            data: Spectral context used when `reflectances` is omitted.
            reflectances: Optional reflectance batch shaped `(N,M)`.
        """

        if reflectances is None:
            reflectances = np.asarray(data.reflectances, float)
        else:
            reflectances = np.asarray(reflectances, float)
        if reflectances.ndim != 2 or reflectances.shape[1] != self.m:
            raise ValueError(f"reflectances must be (N,{self.m}), got {reflectances.shape}")
        coeffs = np.stack([self._encode(reflectances[i]) for i in range(reflectances.shape[0])], axis=0)
        return coeffs @ self.pc

    def integrate_xyz(self, data: SpectralData, ill_indices, path) -> np.ndarray:
        ill = _as_list(ill_indices, data.illuminants.shape[0])
        coeff = self._path_coeff(data, path)
        xyz = np.einsum("ijk,k->ij", self.a_all[ill], coeff)
        return xyz


class WandellFourierRenderer(Renderer):
    """Full-domain Fourier Wandell coefficient transport with a DC term."""

    def __init__(
        self,
        label: str,
        data: SpectralData,
        k: int,
        *,
        fourier_wl_range: tuple[float, float] | None = (400.0, 700.0),
    ):
        """Build the Fourier basis and illuminant lighting matrices.

        Args:
            label: Human-readable model name used in output artifacts.
            data: Spectral context that supplies wavelength grid, illuminants, and CMFs.
            k: Number of Fourier coefficients. The DC term is always coefficient 1.
            fourier_wl_range: Optional wavelength interval used only to fit
                coefficients. The basis itself remains defined on the full grid.
        """

        self.label = label
        self.k = int(k)
        if self.k < 1:
            raise ValueError("k must be at least 1 because the DC term is included.")
        self.wl = np.asarray(data.wl, float)
        self.m = self.wl.size

        if fourier_wl_range is None:
            self.fit_mask = np.ones(self.m, dtype=bool)
        else:
            lo, hi = map(float, fourier_wl_range)
            self.fit_mask = (self.wl >= lo) & (self.wl <= hi)
            if not np.any(self.fit_mask):
                raise ValueError("fourier_wl_range produced an empty fitting mask.")

        self.basis = self._build_full_domain_fourier_basis()
        self.basis_fit_pinv = np.linalg.pinv(self.basis[:, self.fit_mask].T)
        self.a_all = self._build_lighting_matrices(data)

    def _build_full_domain_fourier_basis(self) -> np.ndarray:
        """Build normalized Fourier rows on the full wavelength grid.

        Returns:
            Basis shaped `(K,M)` ordered as DC, cos1, sin1, cos2, sin2, ...
        """

        t = (self.wl - self.wl.min()) / (self.wl.max() - self.wl.min())
        modes = [np.ones_like(t)]
        freq = 1
        while len(modes) < self.k:
            modes.append(np.cos(2.0 * np.pi * freq * t))
            if len(modes) < self.k:
                modes.append(np.sin(2.0 * np.pi * freq * t))
            freq += 1

        basis = np.stack(modes[: self.k], axis=0)
        norms = np.linalg.norm(basis, axis=1, keepdims=True)
        return basis / (norms + 1e-30)

    def _encode(self, spectrum: np.ndarray) -> np.ndarray:
        """Least-squares fit one reflectance spectrum to Fourier coefficients.

        Args:
            spectrum: Reflectance spectrum shaped `(M,)`; only `fit_mask` samples
                are used for the coefficient fit.
        """

        spectrum = np.asarray(spectrum, float).reshape(self.m)
        return self.basis_fit_pinv @ spectrum[self.fit_mask]

    def _path_coeff(self, data: SpectralData, path) -> np.ndarray:
        """Compute the coefficient-space product for a multi-bounce path.

        Args:
            data: Spectral context containing reflectance spectra.
            path: Reflectance index or sequence of indices. Repeats are kept.
        """

        r_ids = _path_list(path)
        if not r_ids:
            return self._encode(np.ones(self.m, float))

        coeff = np.ones(self.k, float)
        for r in r_ids:
            coeff *= self._encode(data.reflectances[int(r)])
        return coeff

    def _build_lighting_matrices(self, data: SpectralData) -> np.ndarray:
        """Precompute illuminant-specific Fourier lighting matrices.

        Args:
            data: Spectral context containing illuminants and XYZ CMFs.

        Returns:
            Array shaped `(Nill,3,K)`.
        """

        a_all = np.zeros((data.illuminants.shape[0], 3, self.k), float)
        for ill, illuminant in enumerate(data.illuminants):
            for j in range(self.k):
                a_all[ill, :, j] = (illuminant * self.basis[j]) @ data.xyz_cmf_d
        return a_all

    def reconstruct_from_coeff(self, coeff: np.ndarray) -> np.ndarray:
        """Reconstruct one reflectance spectrum from Fourier coefficients.

        Args:
            coeff: Coefficient vector shaped `(K,)`.
        """

        return np.asarray(coeff, float).reshape(self.k) @ self.basis

    def reconstruct_reflectances(self, data: SpectralData, reflectances: np.ndarray | None = None) -> np.ndarray:
        """Encode and reconstruct a batch of reflectances for diagnostics.

        Args:
            data: Spectral context used when `reflectances` is omitted.
            reflectances: Optional reflectance batch shaped `(N,M)`.
        """

        if reflectances is None:
            reflectances = np.asarray(data.reflectances, float)
        else:
            reflectances = np.asarray(reflectances, float)
        if reflectances.ndim != 2 or reflectances.shape[1] != self.m:
            raise ValueError(f"reflectances must be (N,{self.m}), got {reflectances.shape}")
        coeffs = np.stack([self._encode(reflectances[i]) for i in range(reflectances.shape[0])], axis=0)
        return coeffs @ self.basis

    def integrate_xyz(self, data: SpectralData, ill_indices, path) -> np.ndarray:
        ill = _as_list(ill_indices, data.illuminants.shape[0])
        coeff = self._path_coeff(data, path)
        xyz = np.einsum("ijk,k->ij", self.a_all[ill], coeff)
        return xyz
