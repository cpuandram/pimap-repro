"""Filesystem paths and compact NPZ I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from presets import PAPER_NPZ


def data_dir(project_root: Path) -> Path:
    return project_root / "data"


def results_dir(project_root: Path) -> Path:
    return project_root / "results"


def figures_dir(project_root: Path) -> Path:
    return results_dir(project_root) / "Figures"


def tables_dir(project_root: Path) -> Path:
    return results_dir(project_root) / "tables"


def ensure_dirs(project_root: Path) -> None:
    for folder in [
        data_dir(project_root) / "cache",
        figures_dir(project_root),
        tables_dir(project_root),
    ]:
        folder.mkdir(parents=True, exist_ok=True)


def load_paper_npz(path: Path) -> dict:
    """Load a compact precomputed NPZ artifact.

    Args:
        path: Path to `paper_data_de2000_jab.npz`.
    """

    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def write_paper_npz(path: Path, paper_data: dict) -> None:
    """Write the compact paper-data cache.

    Args:
        path: Output NPZ path.
        paper_data: Full paper-data dictionary from recomputation.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **paper_data)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
