"""LuxPy spectral loading and spectral-context helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from presets import DEFAULT_EXCLUDED_ILLUMINANT_NAMES


@dataclass
class SpectralContext:
    """All LuxPy-derived spectral data and conversion helpers for the repro.

    Arrays use the shared wavelength grid `wl` in nanometers. `illuminants`
    stores the interpolated CIE illuminants after Y=100 normalization, and
    `k_ill` records the scale factors applied to the raw LuxPy SPDs. `xyz_cmf_d`
    is the XYZ CMF multiplied by the wavelength step. `pools` groups reflectance
    indices by the paper datasets: CIE 224, Macbeth, and Munsell.
    """

    wl: np.ndarray
    illuminants: np.ndarray
    illuminant_names: list[str]
    reflectances: np.ndarray
    reflectance_names: list[str]
    xyz_cmf: np.ndarray
    xyz_coeff: np.ndarray
    xyz_norm: np.ndarray
    xyz_cmf_d: np.ndarray
    k_ill: np.ndarray
    xyzw: np.ndarray
    xyzw_d65: np.ndarray
    pools: list[np.ndarray]
    pool_titles: list[str]
    d_lambda: float = 1.0
    cieobs: str = "2015_2"

    def _as_index(self, idx):
        if idx is None:
            return slice(None)
        if isinstance(idx, slice):
            return idx
        if np.isscalar(idx):
            return int(idx)
        return np.asarray(idx, dtype=int)

    def _idx_list(self, idx, n: int) -> list[int]:
        idx = self._as_index(idx)
        if isinstance(idx, slice):
            return list(range(*idx.indices(n)))
        if np.isscalar(idx):
            return [int(idx)]
        return list(np.asarray(idx, dtype=int))

    def reflectance_names_of(self, rfl_idx) -> list[str]:
        return [self.reflectance_names[i] for i in self._idx_list(rfl_idx, len(self.reflectance_names))]

    def illuminant_names_of(self, ill_idx) -> list[str]:
        return [self.illuminant_names[i] for i in self._idx_list(ill_idx, len(self.illuminant_names))]

    def _as_2d_3(self, x) -> tuple[np.ndarray, bool]:
        arr = np.asarray(x, dtype=float)
        if arr.shape == (3,):
            return arr[None, :], True
        if arr.ndim == 2 and arr.shape[1] == 3:
            return arr, False
        raise ValueError(f"Expected shape (3,) or (N,3), got {arr.shape}")

    @staticmethod
    def _restore_shape(arr: np.ndarray, single: bool) -> np.ndarray:
        return arr[0] if single else arr

    def lab_distance(self, lab1, lab2, method: str = "CIE76"):
        lab1_2d, single1 = self._as_2d_3(lab1)
        lab2_2d, single2 = self._as_2d_3(lab2)

        if lab1_2d.shape[0] == 1 and lab2_2d.shape[0] > 1:
            lab1_2d = np.repeat(lab1_2d, lab2_2d.shape[0], axis=0)
        if lab2_2d.shape[0] == 1 and lab1_2d.shape[0] > 1:
            lab2_2d = np.repeat(lab2_2d, lab1_2d.shape[0], axis=0)
        if lab1_2d.shape != lab2_2d.shape:
            raise ValueError(f"Lab shapes not compatible: {lab1_2d.shape} vs {lab2_2d.shape}")

        if method.upper() == "CIE76":
            delta_e = np.linalg.norm(lab1_2d - lab2_2d, axis=1)
        else:
            raise ValueError(f"Unknown Delta E method: {method}")

        if single1 and single2:
            return float(delta_e[0])
        return delta_e

    def xyz_to_lab(self, xyz, xyzw):
        import luxpy as lx

        xyz_2d, single = self._as_2d_3(xyz)
        white, _ = self._as_2d_3(xyzw)
        if white.shape[0] == 1 and xyz_2d.shape[0] > 1:
            white = np.repeat(white, xyz_2d.shape[0], axis=0)
        lab = lx.xyz_to_lab(xyz_2d, white, cieobs=self.cieobs)
        return self._restore_shape(lab, single)

    def lab_to_xyz(self, lab, xyzw):
        import luxpy as lx

        lab_2d, single = self._as_2d_3(lab)
        white, _ = self._as_2d_3(xyzw)
        if white.shape[0] == 1 and lab_2d.shape[0] > 1:
            white = np.repeat(white, lab_2d.shape[0], axis=0)
        xyz = lx.lab_to_xyz(lab_2d, white, cieobs=self.cieobs)
        return self._restore_shape(xyz, single)

    def xyz_to_srgb(self, xyz, xyzw=None, adapt_to_d65: bool = True, out_uint8: bool = True):
        import luxpy as lx

        xyz_2d, single = self._as_2d_3(xyz)
        if adapt_to_d65:
            if xyzw is None:
                raise ValueError("adapt_to_d65=True requires xyzw.")
            white1, _ = self._as_2d_3(xyzw)
            if white1.shape[0] == 1 and xyz_2d.shape[0] > 1:
                white1 = np.repeat(white1, xyz_2d.shape[0], axis=0)
            if white1.shape[0] != xyz_2d.shape[0]:
                raise ValueError("xyzw rows must match XYZ rows or be a single whitepoint.")
            white2 = np.repeat(np.atleast_2d(self.xyzw_d65).astype(float), xyz_2d.shape[0], axis=0)
            xyz_2d = lx.cat.apply(
                xyz_2d,
                xyzw1=white1,
                xyzw2=white2,
                cattype="vonkries",
                catmode="1>0>2",
            )
        rgb = lx.xyz_to_srgb(xyz_2d)
        if out_uint8:
            rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
        return self._restore_shape(rgb, single)

    def lab_to_srgb(self, lab, xyzw, adapt_to_d65: bool = True, out_uint8: bool = True):
        return self.xyz_to_srgb(
            self.lab_to_xyz(lab, xyzw),
            xyzw=xyzw,
            adapt_to_d65=adapt_to_d65,
            out_uint8=out_uint8,
        )


SpectralData = SpectralContext


def _as_list(idx, n: int) -> list[int]:
    if idx is None:
        return list(range(n))
    if isinstance(idx, slice):
        return list(range(*idx.indices(n)))
    if np.isscalar(idx):
        return [int(idx)]
    return [int(x) for x in np.asarray(idx).ravel().tolist()]


def _path_list(path) -> list[int]:
    if np.isscalar(path):
        return [int(path)]
    return [int(x) for x in np.asarray(path).ravel().tolist()]


def load_luxpy_spectral_data(cieobs: str = "2015_2") -> SpectralData:
    """Load all starting data for full recomputation from LuxPy.

    Args:
        cieobs: LuxPy observer key, for example `"2015_2"`, used for CMFs and
            the XYZ<->LMS observer matrix.

    Returns:
        A `SpectralContext` on the 360-830 nm, 1 nm grid. No paper cache,
        table CSV, or saved learned parameter file is read here.
    """

    import luxpy as lx

    wl = np.arange(360, 831, 1, dtype=float)
    d_lambda = float(wl[1] - wl[0])

    illuminants = []
    illuminant_names = []
    for name in lx._CIE_ILLUMINANTS["types"]:
        data = lx._CIE_ILLUMINANTS[name]
        if isinstance(data, np.ndarray):
            illuminants.append(lx.cie_interp(data, wl, datatype="spd")[1])
            illuminant_names.append(name)
    if len(illuminant_names) > 44:
        illuminant_names[44] = "LED RGB01"
    illuminants = np.asarray(illuminants, float)

    cie224 = lx.cie_interp(lx._RFL["cri"]["cie-224-2017"]["99"]["1nm"], wl, datatype="rfl")[1:]
    macbeth = lx.cie_interp(lx._RFL["macbeth"]["CC"]["R"], wl, datatype="rfl")[1:]
    munsell = lx.cie_interp(lx._RFL["munsell"]["R"], wl, datatype="rfl")[1:]

    reflectances = np.vstack([cie224, macbeth, munsell])
    reflectance_names = (
        [f"CIE224-{i + 1:02d}" for i in range(cie224.shape[0])]
        + [f"Macbeth-{i + 1:02d}" for i in range(macbeth.shape[0])]
        + [f"Munsell-{i + 1:04d}" for i in range(munsell.shape[0])]
    )

    xyz_cmf = lx.cie_interp(lx._CMF[cieobs]["bar"], wl, datatype="cmf")[1:].T
    xyz_coeff = np.sum(xyz_cmf, axis=0)
    xyz_norm = xyz_cmf / xyz_coeff[None, :]

    xyz_cmf_d = xyz_cmf * d_lambda
    xyzw_raw = illuminants @ xyz_cmf_d
    k_ill = 100.0 / xyzw_raw[:, 1]
    illuminants = illuminants * k_ill[:, None]
    xyzw = illuminants @ xyz_cmf_d
    xyzw_d65 = lx.spd_to_xyz(lx._CIE_ILLUMINANTS["D65"], cieobs=cieobs, relative=True)

    n_cie = cie224.shape[0]
    n_macbeth = macbeth.shape[0]
    pools = [
        np.arange(0, n_cie, dtype=int),
        np.arange(n_cie, n_cie + n_macbeth, dtype=int),
        np.arange(n_cie + n_macbeth, reflectances.shape[0], dtype=int),
    ]
    pool_titles = ["CIE 224:2017 Rf", "Macbeth", "Munsell"]

    return SpectralContext(
        wl=wl,
        illuminants=illuminants,
        illuminant_names=illuminant_names,
        reflectances=reflectances,
        reflectance_names=reflectance_names,
        xyz_cmf=xyz_cmf,
        xyz_coeff=xyz_coeff,
        xyz_norm=xyz_norm,
        xyz_cmf_d=xyz_cmf_d,
        k_ill=k_ill,
        xyzw=xyzw,
        xyzw_d65=xyzw_d65,
        pools=pools,
        pool_titles=pool_titles,
        d_lambda=d_lambda,
        cieobs=cieobs,
    )


def paper_illuminant_indices(data: SpectralData) -> tuple[int, ...]:
    """Return the illuminant set described in the manuscript.

    LuxPy exposes 47 CIE illuminant entries for this observer/data bundle,
    including `L41`. The manuscript list contains 46 illuminants: E, D65,
    A, B, C, F1-F12, F3.1-F3.15, HP1-HP5, and the listed LEDs. This helper
    therefore includes every LuxPy entry except names in
    `DEFAULT_EXCLUDED_ILLUMINANT_NAMES`.

    Args:
        data: Spectral context with LuxPy illuminant names.
    """

    return tuple(
        i
        for i, name in enumerate(data.illuminant_names)
        if str(name) not in DEFAULT_EXCLUDED_ILLUMINANT_NAMES
    )
