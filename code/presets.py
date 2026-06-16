"""Constants and numerical presets for the renderer paper reproduction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FIGURE_MODELS_FIXED = ["RGB", "Darling", "PI map, RGB", "PI map, Darling"]
FIGURE_MODELS_LEARNED = ["Darling", "PI map, K=5", "PI map, K=4", "PI map, K=3"]
IQR_FIXED_ILLUMINANTS = ("L41", "LED_B1", "HP3")
IQR_LEARNED_ILLUMINANTS = ("L41", "LED_B1", "HP3")
MACBETH_MOSAIC_ILLUMINANTS = ("LED_V1", "LED_B2", "HP2")
PAPER_NPZ = "paper_data_de2000_jab.npz"
FIXED_TABLE = "fixed_basis_macbeth_munsell_percentiles.csv"
LEARNED_TABLE = "learned_vs_wandell_macbeth_munsell_percentiles.csv"
ALL_METHODS_TABLE = "all_methods_macbeth_munsell_percentiles.csv"
TRAINING_POOL_INDEX = 0
TABLE_POOL_INDICES = (1, 2)
TABLE_POOL_TITLE = "Macbeth + Munsell"
DEFAULT_EXCLUDED_ILLUMINANT_NAMES = {"E"}
PERCENTILES = np.arange(0, 101, 1, dtype=int)
COMPARISON_TABLE_PERCENTILES = [70, 80, 90, 95, 99]
DEFAULT_GAUSSIAN_BASIS_EPOCHS = {3: 25000, 4: 25000, 5: 25000, 6: 25000}

OKABE_ITO = [
    "#E69F00",
    "#D55E00",
    "#56B4E9",
    "#009E73",
    "#0072B2",
    "#000000",
    "#CC79A7",
    "#F0E442",
]


@dataclass(frozen=True)
class FullRecomputeConfig:
    """Numerical settings for the full LuxPy recomputation path.

    Attributes:
        max_combos: Number of sampled reflectance paths per bounce count and pool.
        max_bounces: Largest bounce count evaluated in the paper figures.
        table_max_bounces: Largest bounce count evaluated in the paper tables.
        covariance_paths: Number of random reflectance paths used to estimate C_R.
        seed: Base random seed for covariance, training samples, and evaluation paths.
        darling_fit_pairs: Maximum illuminant/reflectance pairs for the Darling least-squares fit.
        cieobs: LuxPy CIE observer key used for CMFs and XYZ/LMS conversion.
        gaussian_basis_epochs: Optional epoch override for every learned Gaussian K.
        gaussian_basis_lr: Adam learning rate for learned Gaussian centers and widths.
        gaussian_basis_reg: Weight on the basis regularizer during Gaussian learning.
        gaussian_basis_verbose_every: Print Gaussian learning diagnostics every N epochs; 0 disables it.
    """

    max_combos: int = 4000
    max_bounces: int = 6
    table_max_bounces: int = 3
    covariance_paths: int = 100000
    seed: int = 0
    darling_fit_pairs: int = 10000
    cieobs: str = "2015_2"
    gaussian_basis_epochs: int | None = None
    gaussian_basis_lr: float = 5e-3
    gaussian_basis_reg: float = 1.0
    gaussian_basis_verbose_every: int = 0
