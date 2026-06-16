"""Figure generation helpers."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from basis import gaussian_basis
from evaluation import percentile_index
from paths import figures_dir
from presets import FIGURE_MODELS_FIXED, FIGURE_MODELS_LEARNED, MACBETH_MOSAIC_ILLUMINANTS, OKABE_ITO


# Exact values printed by the Optica LaTeX template with the FIGMATCH helper.
TEX_PT_PER_INCH = 72.27
LATEX_LINEWIDTH_PT = 379.41753
LATEX_NORMALSIZE_PT = 10.0
LATEX_SMALL_PT = 9.0
LATEX_FOOTNOTESIZE_PT = 8.0

PAPER_TEXT_WIDTH_IN = LATEX_LINEWIDTH_PT / TEX_PT_PER_INCH
PAPER_FONT_SIZE_PT = LATEX_SMALL_PT
PAPER_TICK_SIZE_PT = LATEX_FOOTNOTESIZE_PT
PAPER_LEGEND_SIZE_PT = LATEX_FOOTNOTESIZE_PT
LATEX_PREAMBLE = r"\usepackage[T1]{fontenc}\usepackage{newtxtext,newtxmath}"


def configure_matplotlib(project_root: Path):
    """Configure Matplotlib for deterministic paper artifact rendering.

    Args:
        project_root: Repository root. Also used for
            the local Matplotlib cache directory.
    """

    os.environ.setdefault("MPLCONFIGDIR", str(project_root / ".matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": LATEX_PREAMBLE,
            "font.family": "serif",
            "font.size": PAPER_FONT_SIZE_PT,
            "axes.labelsize": PAPER_FONT_SIZE_PT,
            "axes.titlesize": PAPER_FONT_SIZE_PT,
            "legend.fontsize": PAPER_LEGEND_SIZE_PT,
            "xtick.labelsize": PAPER_TICK_SIZE_PT,
            "ytick.labelsize": PAPER_TICK_SIZE_PT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.format": "pdf",
            "savefig.bbox": "standard",
            "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
        }
    )
    return plt


def save_pdf(fig, output: Path, *, transparent: bool = True) -> None:
    """Save a Matplotlib figure as a PDF.

    Args:
        fig: Matplotlib figure to save.
        output: Destination PDF path.
        transparent: Whether to preserve a transparent figure background.
    """

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, transparent=transparent, bbox_inches=None)


def paper_legend(ax, handles=None, labels=None, *, loc: str = "upper right"):
    """Draw legends with one shared paper style.

    Args:
        ax: Matplotlib axes that owns the legend.
        handles: Optional explicit legend handles.
        labels: Optional explicit legend labels.
        loc: Matplotlib legend location string.
    """

    legend_kwargs = {
        "loc": loc,
        "frameon": True,
        "fancybox": False,
        "framealpha": 0.95,
        "facecolor": "white",
        "edgecolor": "black",
        "handlelength": 0.8,
        "handletextpad": 0.45,
        "borderpad": 0.25,
        "labelspacing": 0.25,
        "borderaxespad": 0.35,
    }
    if handles is None or labels is None:
        legend = ax.legend(**legend_kwargs)
    else:
        legend = ax.legend(handles, labels, **legend_kwargs)
    if legend is not None:
        legend.get_frame().set_linewidth(0.5)
    return legend


def plot_gaussian_bases(project_root: Path, output: Path, paper_data: dict | None = None) -> None:
    """Plot learned Gaussian basis functions from computed paper data.

    Args:
        project_root: Repository root.
        output: Destination PDF path.
        paper_data: Full or NPZ-loaded paper data containing learned Gaussian
            centers and widths.
    """

    plt = configure_matplotlib(project_root)
    wl = np.arange(360, 831, 1.0)
    fig, axes = plt.subplots(2, 2, figsize=(PAPER_TEXT_WIDTH_IN, 3.25), sharex=True)

    if paper_data is None:
        raise ValueError("Gaussian basis plotting requires computed paper_data.")
    else:
        basis_specs = []
        for row, k in enumerate(np.asarray(paper_data["gaussian_k"], int)):
            n = int(np.asarray(paper_data["gaussian_counts"], int)[row])
            mu = np.asarray(paper_data["gaussian_mu"], float)[row, :n]
            sigma = np.asarray(paper_data["gaussian_sigma"], float)[row, :n]
            basis_specs.append((int(k), mu, sigma))

    for ax, (k, mus, sigmas) in zip(axes.ravel(), basis_specs):
        y_max = 0.0
        for mu, sigma in zip(mus, sigmas):
            y = gaussian_basis(wl, mu, sigma, 1.0)
            y_max = max(y_max, float(np.max(y)))
            ax.plot(wl, y, lw=1.2, label=rf"$\mu={mu:.1f},\ \sigma={sigma:.1f}$")
        ax.set_title(rf"$K={k}$")
        ax.set_xlim(360, 830)
        ax.set_ylim(0.0, y_max * 1.08 if y_max > 0.0 else 1.0)
        ax.grid(alpha=0.15)
        paper_legend(ax)

    axes[1, 0].set_xlabel("Wavelength (nm)")
    axes[1, 1].set_xlabel("Wavelength (nm)")
    axes[0, 0].set_ylabel("Basis value")
    axes[1, 0].set_ylabel("Basis value")
    fig.tight_layout(pad=0.35)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, transparent=True)
    plt.close(fig)


def plot_dataset_reflectance_covariances(project_root: Path, output: Path, data) -> None:
    """Plot measured-reflectance covariance for each paper dataset.

    Args:
        project_root: Repository root.
        output: Destination PDF path.
        data: Spectral context containing reflectances, wavelength grid, pools,
            and pool titles.
    """

    plt = configure_matplotlib(project_root)
    wl = np.asarray(data.wl, float)
    wl_mask = (wl >= 400.0) & (wl <= 730.0)
    wl_plot = wl[wl_mask]

    covariances = []
    for pool in data.pools:
        reflectances = np.asarray(data.reflectances[np.asarray(pool, int)], float)
        reflectances = reflectances[:, wl_mask]
        centered = reflectances - reflectances.mean(axis=0, keepdims=True)
        cov = (centered.T @ centered) / max(reflectances.shape[0], 1)
        covariances.append(cov)

    fig, axes = plt.subplots(
        1,
        len(covariances),
        figsize=(PAPER_TEXT_WIDTH_IN, 2.0),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    axes = np.atleast_1d(axes)
    ims = []
    for ax, cov, title, pool in zip(axes, covariances, data.pool_titles, data.pools):
        lo = float(np.nanmin(cov))
        hi = float(np.nanmax(cov))
        denom = hi - lo
        if not np.isfinite(denom) or denom <= 0.0:
            denom = 1.0
        cov_plot = np.clip((cov - lo) / denom, 0.0, 1.0)
        im = ax.imshow(
            cov_plot,
            origin="lower",
            extent=[wl_plot[0], wl_plot[-1], wl_plot[0], wl_plot[-1]],
            cmap="cividis",
            vmin=0.0,
            vmax=1.0,
            aspect="equal",
        )
        ims.append(im)
        ax.set_title(f"{title}\nN={len(pool)}")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_xlim(400, 730)
        ax.set_ylim(400, 730)
        ax.set_xticks([400, 500, 600, 700])
        ax.set_yticks([400, 500, 600, 700])

    axes[0].set_ylabel("Wavelength (nm)")
    cbar = fig.colorbar(ims[-1], ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Min-max normalized covariance")
    fig.subplots_adjust(left=0.09, right=0.86, bottom=0.22, top=0.78, wspace=0.08)

    save_pdf(fig, output)
    plt.close(fig)


def warn_missing_models(cache: dict, models: list[str], output: Path) -> None:
    """Print a compact warning when a figure asks for models absent from cache.

    Args:
        cache: Nested plotting cache from `cache_from_npz`.
        models: Requested model labels.
        output: Figure path, used only to identify the affected artifact.
    """

    available = set(cache.get("models", []))
    missing = [label for label in models if label not in available]
    if missing:
        joined = ", ".join(missing)
        print(f"[plot] {output.name}: missing cached model(s): {joined}")


def plot_iqr_pooled(project_root: Path, cache: dict, output: Path, models: list[str]) -> None:
    """Plot pooled interquartile error curves for selected models.

    Args:
        project_root: Repository root.
        cache: Nested plotting cache from `cache_from_npz`.
        output: Destination PDF path.
        models: Model labels to plot if present in each reflectance pool.
    """

    plt = configure_matplotlib(project_root)
    warn_missing_models(cache, models, output)
    i25 = percentile_index(cache, 25)
    i50 = percentile_index(cache, 50)
    i75 = percentile_index(cache, 75)

    titles = cache.get("pool_titles", ["CIE 224:2017 Rf", "Macbeth", "Munsell"])
    max_bounces = int(cache.get("max_bounces", 6))
    fig, axes = plt.subplots(1, 3, figsize=(PAPER_TEXT_WIDTH_IN, 1.65), sharey=True)

    for col, ax in enumerate(axes):
        ax.set_title(titles[col])
        ax.set_xlim(1, max_bounces)
        ax.set_ylim(0, 4.0)
        ax.grid(alpha=0.12)
        ax.set_xticks(np.arange(1, max_bounces + 1))

        pool_models = cache["pools"][col]["models"]
        for label in models:
            if label not in pool_models:
                continue
            stats = pool_models[label]
            b = np.asarray(stats["bounces"], int)
            p = np.asarray(stats["P"], float)
            line = ax.plot(b, p[:, i50], lw=1.1, label=label)[0]
            ax.fill_between(b, p[:, i25], p[:, i75], color=line.get_color(), alpha=0.18, linewidth=0)

        if col == 1:
            ax.set_xlabel("Number of bounces")
        if col == 2:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(r"$\Delta E_{00}$")
            ax.tick_params(axis="y", right=True, labelright=True, left=False, labelleft=False)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)

    handles, labels = axes[-1].get_legend_handles_labels()
    paper_legend(axes[-1], handles, labels)
    fig.tight_layout(w_pad=0.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, transparent=True)
    plt.close(fig)


def build_cie_illuminants():
    """Reload CIE illuminants from LuxPy for legacy cache compatibility."""

    import luxpy as lx

    wl = np.arange(360, 831, 1.0)
    spds = []
    names = []
    for name in lx._CIE_ILLUMINANTS["types"]:
        data = lx._CIE_ILLUMINANTS[name]
        if isinstance(data, np.ndarray):
            spds.append(lx.cie_interp(data, wl, datatype="spd")[1])
            names.append(name)
    if len(names) > 44:
        names[44] = "LED RGB01"
    return wl, np.asarray(spds, float), names


def plot_paper_spectrum_colors(
    ax,
    spd: np.ndarray,
    *,
    wavelength_lightness: float = 1.0,
    opacity_gamma: float = 1.5,
    opacity_min: float = 1.0,
    opacity_max: float = 1.0,
    contour_lw: float = 1.0,
    contour: bool = True,
) -> None:
    """Draw the LuxPy-style SPD color bars used for the paper figures."""

    import luxpy as lx
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.patches import Polygon

    spd = np.asarray(spd, float)
    cmfs = lx._CMF["1931_2"]["bar"]
    cmfs = cmfs[:, cmfs[1:].sum(axis=0) > 0]
    cmfs = cmfs[:, ~np.isnan(cmfs.sum(axis=0))]
    wavs = np.asarray(cmfs[0], float)
    xyz_locus = cmfs[1:4].T
    srgb = np.clip(lx.xyz_to_srgb(wavelength_lightness * 100.0 * xyz_locus) / 255.0, 0.0, 1.0)

    x_min = float(np.nanmin(spd[0]))
    x_max = float(np.nanmax(spd[0]))
    spdmax = float(np.nanmax(spd[1:]))
    y_max = spdmax * 1.05
    clip_poly = np.vstack([(x_min, 0.0), spd.T, (x_max, 0.0)])

    spd_w = np.asarray(spd[0], float)
    spd_v = np.asarray(spd[1:], float)
    spd_1 = np.nanmax(spd_v, axis=0)
    alpha = np.interp(wavs, spd_w, spd_1, left=0.0, right=0.0)
    alpha = alpha / (np.nanmax(alpha) + 1e-12)
    alpha = np.clip(alpha, float(opacity_min), float(opacity_max))
    if opacity_gamma != 1.0:
        alpha = np.power(alpha, float(opacity_gamma))

    rgba = np.concatenate(
        [(1.0 - alpha[:, None]) + alpha[:, None] * srgb, np.ones((len(wavs), 1))],
        axis=1,
    )

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([0.0, 1.0])
    polygon = Polygon(clip_poly, facecolor="none", edgecolor="none", linewidth=0)
    ax.add_patch(polygon)
    ax.bar(
        x=wavs - 0.1,
        height=y_max,
        width=1.1,
        color=rgba,
        align="edge",
        linewidth=0,
        clip_path=polygon,
        zorder=1,
    )

    if contour:
        y_top = np.interp(wavs, spd_w, spd_1, left=0.0, right=0.0)
        pts = np.column_stack([wavs, y_top])
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap=ListedColormap(srgb), norm=Normalize(wavs.min(), wavs.max()))
        lc.set_array(0.5 * (wavs[:-1] + wavs[1:]))
        lc.set_linewidth(float(contour_lw))
        lc.set_alpha(1.0)
        lc.set_clip_path(polygon)
        lc.set_zorder(10)
        ax.add_collection(lc)

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("")


def draw_normalized_spd(
    ax,
    wl: np.ndarray,
    spd: np.ndarray,
    *,
    opacity_gamma: float = 1.5,
    opacity_min: float = 1.0,
    opacity_max: float = 1.0,
    contour_lw: float = 1.0,
) -> None:
    """Draw a compact normalized illuminant SPD using the paper LuxPy style."""

    wl = np.asarray(wl, float)
    spd = np.asarray(spd, float)
    finite = np.isfinite(wl) & np.isfinite(spd)
    plot_wl = wl[finite]
    plot_spd = spd[finite].copy()
    if plot_spd.size == 0:
        raise ValueError("Cannot draw an empty SPD.")
    peak = float(np.max(plot_spd))
    if peak > 0.0:
        plot_spd = plot_spd / peak

    ax.set_facecolor((1.0, 1.0, 1.0, 1.0))
    plot_paper_spectrum_colors(
        ax,
        np.vstack([plot_wl, plot_spd]),
        opacity_gamma=opacity_gamma,
        opacity_min=opacity_min,
        opacity_max=opacity_max,
        contour_lw=contour_lw,
        contour=True,
    )
    ax.set_xlim(float(plot_wl[0]), float(plot_wl[-1]))
    ax.set_ylim(0, 1.05)
    ax.set_yticks([])
    ax.set_xticks([400, 700])
    ax.margins(y=0.0)


def plot_iqr_three_illuminants(
    project_root: Path,
    cache: dict,
    output: Path,
    models: list[str],
    ill_ids: tuple[int, ...],
) -> None:
    """Plot per-illuminant interquartile error curves with SPD thumbnails.

    Args:
        project_root: Repository root.
        cache: Nested plotting cache from `cache_from_npz`.
        output: Destination PDF path.
        models: Model labels to plot if present in each reflectance pool.
        ill_ids: Illuminant indices whose per-illuminant errors are shown.
    """

    plt = configure_matplotlib(project_root)
    warn_missing_models(cache, models, output)

    if cache.get("illuminant_wl") is not None and cache.get("illuminant_spds") is not None:
        wl = np.asarray(cache["illuminant_wl"], float)
        spds = np.asarray(cache["illuminant_spds"], float)
        names = list(cache.get("illuminant_names") or [])
    else:
        wl, spds, names = build_cie_illuminants()
    ill_ids = tuple(int(ill_id) for ill_id in ill_ids)
    if len(ill_ids) == 0:
        raise ValueError("ill_ids must contain at least one illuminant index.")
    for ill_id in ill_ids:
        if ill_id < 0 or ill_id >= spds.shape[0]:
            raise ValueError(f"Illuminant index {ill_id} is outside the available range 0..{spds.shape[0] - 1}.")

    i25 = percentile_index(cache, 25)
    i50 = percentile_index(cache, 50)
    i75 = percentile_index(cache, 75)
    titles = cache.get("pool_titles", ["CIE 224:2017 Rf", "Macbeth", "Munsell"])
    max_bounces = int(cache.get("max_bounces", 6))
    bounce_ticks = np.arange(1, max_bounces + 1, dtype=int)
    bounce_tick_labels = [str(tick) if tick % 2 == 0 else "" for tick in bounce_ticks]
    error_ticks = np.arange(0, 5, 1, dtype=int)
    n_rows = len(ill_ids)
    bottom_row = n_rows - 1
    ylabel_row = n_rows // 2

    fig, axes = plt.subplots(
        n_rows,
        4,
        figsize=(PAPER_TEXT_WIDTH_IN, 0.95 * n_rows + 0.7),
        gridspec_kw={"width_ratios": [0.9, 1, 1, 1], "wspace": 0.05, "hspace": 0.12},
    )
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes[None, :]

    for row, ill_id in enumerate(ill_ids):
        ax_spd = axes[row, 0]
        draw_normalized_spd(ax_spd, wl, spds[ill_id])
        ax_spd.set_ylabel(names[ill_id], rotation=0, ha="right", va="center")
        if row == 0:
            ax_spd.set_title("Illuminant")
        if row == bottom_row:
            ax_spd.set_xlabel("Wavelength (nm)")
        else:
            ax_spd.set_xlabel("")
            ax_spd.set_xticklabels([])

        for col in range(3):
            ax = axes[row, col + 1]
            ax.set_xlim(1, max_bounces)
            ax.set_ylim(0, 4.0)
            ax.set_xticks(bounce_ticks)
            ax.set_xticklabels(bounce_tick_labels)
            ax.set_yticks(error_ticks)
            ax.grid(alpha=0.12)
            if row == 0:
                ax.set_title(titles[col])
            if row == bottom_row and col == 1:
                ax.set_xlabel("Number of bounces")
            if row < bottom_row:
                ax.tick_params(axis="x", labelbottom=False)
            if col < 2:
                ax.tick_params(axis="y", left=False, labelleft=False)
            else:
                ax.yaxis.tick_right()
                y_labels = [str(tick) for tick in error_ticks]
                if row > 0:
                    y_labels[-1] = ""
                ax.set_yticklabels(y_labels)
                if row == ylabel_row:
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel(r"$\Delta E_{00}$")

            pool_models = cache["pools"][col]["models"]
            for label in models:
                if label not in pool_models:
                    continue
                stats = pool_models[label]
                ill_indices = np.asarray(stats["ill_indices"], int)
                hit = np.where(ill_indices == int(ill_id))[0]
                if len(hit) == 0:
                    continue
                j = int(hit[0])
                b = np.asarray(stats["bounces"], int)
                p_ill = np.asarray(stats["P_ill"], float)
                line = ax.plot(
                    b,
                    p_ill[:, j, i50],
                    lw=1.0,
                    label=label if (row == 0 and col == 2) else "_nolegend_",
                )[0]
                ax.fill_between(
                    b,
                    p_ill[:, j, i25],
                    p_ill[:, j, i75],
                    color=line.get_color(),
                    alpha=0.18,
                    linewidth=0,
                )

    handles, labels = axes[0, 3].get_legend_handles_labels()
    paper_legend(axes[0, 3], handles, labels)
    fig.subplots_adjust(left=0.096, right=0.904, top=0.90, bottom=0.15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, transparent=True)
    plt.close(fig)


def _tex_label(text: str) -> str:
    """Escape plain illuminant names for Matplotlib's TeX renderer."""

    return text.replace("_", r"\_")


def _resolve_illuminant_selector(data, selector) -> tuple[int, str]:
    """Resolve an illuminant name or numeric index to `(index, display_name)`."""

    if isinstance(selector, (int, np.integer)):
        index = int(selector)
    else:
        text = str(selector).strip()
        index = int(text) if text.isdigit() else None
    if index is not None:
        if index < 0 or index >= len(data.illuminant_names):
            raise ValueError(f"Illuminant index {index} is outside 0..{len(data.illuminant_names) - 1}.")
        return index, str(data.illuminant_names[index])

    name = str(selector)
    try:
        return data.illuminant_names.index(name), name
    except ValueError as exc:
        raise ValueError(f"Unknown illuminant {name!r}.") from exc


def _learned_gaussian_basis_from_cache(data, paper_data: dict, k: int) -> np.ndarray:
    """Reconstruct one learned Gaussian basis from cached paper data."""

    gaussian_k = np.asarray(paper_data["gaussian_k"], int)
    rows = np.where(gaussian_k == int(k))[0]
    if rows.size != 1:
        raise ValueError(f"Expected one cached Gaussian basis for K={k}, found {rows.size}.")
    row = int(rows[0])
    count = int(np.asarray(paper_data["gaussian_counts"], int)[row])
    mu = np.asarray(paper_data["gaussian_mu"], float)[row, :count]
    sigma = np.asarray(paper_data["gaussian_sigma"], float)[row, :count]
    return np.vstack([gaussian_basis(data.wl, center, width, data.d_lambda) for center, width in zip(mu, sigma)])


def _render_macbeth_xyz(data, renderer, illuminant_index: int) -> np.ndarray:
    macbeth_indices = np.asarray(data.pools[1], int)
    return np.vstack(
        [renderer.integrate_xyz(data, illuminant_index, int(reflectance_index))[0] for reflectance_index in macbeth_indices]
    )


def _macbeth_xyz_to_rgb(data, xyz: np.ndarray, illuminant_index: int) -> np.ndarray:
    return data.xyz_to_srgb(
        xyz,
        xyzw=data.xyzw[int(illuminant_index)],
        adapt_to_d65=True,
        out_uint8=True,
    )


def _macbeth_de00(data, truth_xyz: np.ndarray, test_xyz: np.ndarray, illuminant_index: int) -> np.ndarray:
    import luxpy as lx

    white = np.repeat(data.xyzw[int(illuminant_index)][None, :], truth_xyz.shape[0], axis=0)
    return np.asarray(
        lx.deltaE.DE2000(test_xyz, truth_xyz, xyzwt=white, xyzwr=white, DEtype="jab"),
        float,
    ).reshape(-1)


def _error_text_color(ground_truth_rgb: np.ndarray) -> str:
    luminance = (
        0.2126 * float(ground_truth_rgb[0])
        + 0.7152 * float(ground_truth_rgb[1])
        + 0.0722 * float(ground_truth_rgb[2])
    )
    return "white" if luminance < 0.5 else "black"


def _add_error_text(ax, x: float, y: float, value: float, ground_truth_rgb: np.ndarray) -> None:
    ax.text(
        x,
        y,
        f"{value:.1f}",
        ha="center",
        va="center",
        fontsize=2.5,
        color=_error_text_color(ground_truth_rgb),
    )


def _add_corner_error_text(
    ax,
    x: float,
    y: float,
    value: float,
    ground_truth_rgb: np.ndarray,
    *,
    ha: str,
    va: str,
) -> None:
    ax.text(
        x,
        y,
        f"{value:.1f}",
        ha=ha,
        va=va,
        fontsize=2.5,
        color=_error_text_color(ground_truth_rgb),
    )


def _draw_split_macbeth_mosaic(
    ax,
    *,
    ground_truth: np.ndarray,
    pi_map: np.ndarray,
    pi_map_k3: np.ndarray,
    rgb: np.ndarray,
    darling: np.ndarray,
    pi_errors: np.ndarray,
    pi_k3_errors: np.ndarray,
    rgb_errors: np.ndarray,
    darling_errors: np.ndarray,
) -> None:
    import matplotlib.patches as patches

    gt = np.asarray(ground_truth, dtype=float).reshape(4, 6, 3) / 255.0
    pi = np.asarray(pi_map, dtype=float).reshape(4, 6, 3) / 255.0
    pi3 = np.asarray(pi_map_k3, dtype=float).reshape(4, 6, 3) / 255.0
    rg = np.asarray(rgb, dtype=float).reshape(4, 6, 3) / 255.0
    da = np.asarray(darling, dtype=float).reshape(4, 6, 3) / 255.0
    pi_err = np.asarray(pi_errors, dtype=float).reshape(4, 6)
    pi3_err = np.asarray(pi_k3_errors, dtype=float).reshape(4, 6)
    rg_err = np.asarray(rgb_errors, dtype=float).reshape(4, 6)
    da_err = np.asarray(darling_errors, dtype=float).reshape(4, 6)
    seam_overlap = 0.004
    method_width = 0.25
    patch_fill = {"edgecolor": "none", "linewidth": 0, "antialiased": False}

    for row in range(4):
        for col in range(6):
            x = float(col)
            y = float(row)
            ax.add_patch(patches.Rectangle((x, y), 1.0, 0.5, facecolor=gt[row, col], **patch_fill))
            ax.add_patch(
                patches.Rectangle(
                    (x, y + 0.5 - seam_overlap),
                    method_width,
                    0.5 + seam_overlap,
                    facecolor=pi3[row, col],
                    **patch_fill,
                )
            )
            ax.add_patch(
                patches.Rectangle(
                    (x + method_width, y + 0.5 - seam_overlap),
                    method_width,
                    0.5 + seam_overlap,
                    facecolor=rg[row, col],
                    **patch_fill,
                )
            )
            ax.add_patch(
                patches.Rectangle(
                    (x + 2.0 * method_width, y + 0.5 - seam_overlap),
                    method_width,
                    0.5 + seam_overlap,
                    facecolor=pi[row, col],
                    **patch_fill,
                )
            )
            ax.add_patch(
                patches.Rectangle(
                    (x + 3.0 * method_width, y + 0.5 - seam_overlap),
                    method_width,
                    0.5 + seam_overlap,
                    facecolor=da[row, col],
                    **patch_fill,
                )
            )
            ax.add_patch(
                patches.Rectangle(
                    (x, y),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor=(0, 0, 0, 0.55),
                    linewidth=0.35,
                )
            )
            ax.plot([x + method_width, x + method_width], [y + 0.5, y + 1.0], color=(0, 0, 0, 0.45), linewidth=0.25)
            ax.plot([x + 2.0 * method_width, x + 2.0 * method_width], [y + 0.5, y + 1.0], color=(0, 0, 0, 0.45), linewidth=0.25)
            ax.plot([x + 3.0 * method_width, x + 3.0 * method_width], [y + 0.5, y + 1.0], color=(0, 0, 0, 0.45), linewidth=0.25)
            boundary_y = y + 0.5
            boundary_color = (0, 0, 0, 0.45)
            boundary_width = 0.28
            marker_half = 0.036
            ax.plot([x, x + marker_half], [boundary_y, boundary_y], color=boundary_color, linewidth=boundary_width)
            ax.plot([x + 1.0 - marker_half, x + 1.0], [boundary_y, boundary_y], color=boundary_color, linewidth=boundary_width)
            for boundary_x in (x + method_width, x + 2.0 * method_width, x + 3.0 * method_width):
                ax.plot(
                    [boundary_x - marker_half, boundary_x + marker_half],
                    [boundary_y, boundary_y],
                    color=boundary_color,
                    linewidth=boundary_width,
                )
            _add_error_text(ax, x + 0.5 * method_width, y + 0.92, pi3_err[row, col], gt[row, col])
            _add_error_text(ax, x + 1.5 * method_width, y + 0.92, rg_err[row, col], gt[row, col])
            _add_error_text(ax, x + 2.5 * method_width, y + 0.92, pi_err[row, col], gt[row, col])
            _add_error_text(ax, x + 3.5 * method_width, y + 0.92, da_err[row, col], gt[row, col])

    ax.set_xlim(0, 6)
    ax.set_ylim(4, 0)
    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_corner_cross_macbeth_mosaic(
    ax,
    *,
    ground_truth: np.ndarray,
    pi_map: np.ndarray,
    pi_map_k3: np.ndarray,
    rgb: np.ndarray,
    darling: np.ndarray,
    pi_errors: np.ndarray,
    pi_k3_errors: np.ndarray,
    rgb_errors: np.ndarray,
    darling_errors: np.ndarray,
) -> None:
    import matplotlib.patches as patches

    gt = np.asarray(ground_truth, dtype=float).reshape(4, 6, 3) / 255.0
    pi = np.asarray(pi_map, dtype=float).reshape(4, 6, 3) / 255.0
    pi3 = np.asarray(pi_map_k3, dtype=float).reshape(4, 6, 3) / 255.0
    rg = np.asarray(rgb, dtype=float).reshape(4, 6, 3) / 255.0
    da = np.asarray(darling, dtype=float).reshape(4, 6, 3) / 255.0
    pi_err = np.asarray(pi_errors, dtype=float).reshape(4, 6)
    pi3_err = np.asarray(pi_k3_errors, dtype=float).reshape(4, 6)
    rg_err = np.asarray(rgb_errors, dtype=float).reshape(4, 6)
    da_err = np.asarray(darling_errors, dtype=float).reshape(4, 6)
    corner = 1.0 / 3.0
    boundary_color = (0, 0, 0, 0.45)
    boundary_width = 0.28
    marker_len = 0.056
    text_pad = 0.026
    bottom_text_pad = -0.005
    patch_fill = {"edgecolor": "none", "linewidth": 0, "antialiased": False}

    def add_corner_markers(x0: float, y0: float) -> None:
        inner_points = (
            (x0 + corner, y0 + corner, -1.0, -1.0),
            (x0 + 1.0 - corner, y0 + corner, 1.0, -1.0),
            (x0 + corner, y0 + 1.0 - corner, -1.0, 1.0),
            (x0 + 1.0 - corner, y0 + 1.0 - corner, 1.0, 1.0),
        )
        for ix, iy, sx, sy in inner_points:
            ax.plot([ix, ix + sx * marker_len], [iy, iy], color=boundary_color, linewidth=boundary_width)
            ax.plot([ix, ix], [iy, iy + sy * marker_len], color=boundary_color, linewidth=boundary_width)
            outer_x = ix + sx * corner
            outer_y = iy + sy * corner
            ax.plot([outer_x, outer_x - sx * marker_len], [iy, iy], color=boundary_color, linewidth=boundary_width)
            ax.plot([ix, ix], [outer_y, outer_y - sy * marker_len], color=boundary_color, linewidth=boundary_width)

    for row in range(4):
        for col in range(6):
            x = float(col)
            y = float(row)
            ax.add_patch(patches.Rectangle((x, y), 1.0, 1.0, facecolor=gt[row, col], **patch_fill))
            ax.add_patch(patches.Rectangle((x, y), corner, corner, facecolor=rg[row, col], **patch_fill))
            ax.add_patch(
                patches.Rectangle((x + 1.0 - corner, y), corner, corner, facecolor=pi3[row, col], **patch_fill)
            )
            ax.add_patch(
                patches.Rectangle((x, y + 1.0 - corner), corner, corner, facecolor=da[row, col], **patch_fill)
            )
            ax.add_patch(
                patches.Rectangle(
                    (x + 1.0 - corner, y + 1.0 - corner),
                    corner,
                    corner,
                    facecolor=pi[row, col],
                    **patch_fill,
                )
            )
            ax.add_patch(
                patches.Rectangle(
                    (x, y),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor=(0, 0, 0, 0.55),
                    linewidth=0.35,
                )
            )
            add_corner_markers(x, y)
            _add_corner_error_text(
                ax, x + text_pad, y + text_pad, rg_err[row, col], gt[row, col], ha="left", va="top"
            )
            _add_corner_error_text(
                ax, x + 1.0 - text_pad, y + text_pad, pi3_err[row, col], gt[row, col], ha="right", va="top"
            )
            _add_corner_error_text(
                ax, x + text_pad, y + 1.0 - bottom_text_pad, da_err[row, col], gt[row, col], ha="left", va="bottom"
            )
            _add_corner_error_text(
                ax,
                x + 1.0 - text_pad,
                y + 1.0 - bottom_text_pad,
                pi_err[row, col],
                gt[row, col],
                ha="right",
                va="bottom",
            )

    ax.set_xlim(0, 6)
    ax.set_ylim(4, 0)
    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_split_patch_legend(ax, *, k: int) -> None:
    import matplotlib.patches as patches

    ax.set_xlim(0, 4.6)
    ax.set_ylim(-0.35, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    x0, y0, w, h = 0.60, 0.16, 0.78, 0.78
    method_w = w / 3.0
    ax.add_patch(patches.Rectangle((x0, y0 + h / 2), w, h / 2, facecolor=(0.78, 0.78, 0.78), edgecolor="black", linewidth=0.6))
    ax.add_patch(patches.Rectangle((x0, y0), method_w, h / 2, facecolor=(0.54, 0.74, 0.93), edgecolor="black", linewidth=0.6))
    ax.add_patch(patches.Rectangle((x0 + method_w, y0), method_w, h / 2, facecolor=(0.94, 0.72, 0.45), edgecolor="black", linewidth=0.6))
    ax.add_patch(patches.Rectangle((x0 + 2.0 * method_w, y0), method_w, h / 2, facecolor=(0.68, 0.84, 0.55), edgecolor="black", linewidth=0.6))

    ax.text(
        x0 + w + 0.25,
        y0 + h,
        "top: Ground truth\n"
        f"bottom left: PI map, $K={k}$\n"
        "bottom middle: Darling\n"
        "bottom right: PI map, $K=3$\n"
        r"numbers: $\Delta E_{00}$",
        va="top",
        fontsize=PAPER_LEGEND_SIZE_PT,
        linespacing=0.98,
    )


def _split_key_entries(k: int) -> list[tuple[str, str]]:
    return [
        ("Ground\ntruth", "#BDBDBD"),
        ("RGB\nK=3", OKABE_ITO[0]),
        ("PI map\nK=3", OKABE_ITO[3]),
        ("Darling\nK=6", OKABE_ITO[1]),
        (f"PI map\nK={k}", OKABE_ITO[2]),
    ]


def _draw_split_key_tile(ax, *, k: int) -> None:
    import matplotlib.patches as patches

    colors = [color for _, color in _split_key_entries(k)]
    method_width = 0.25
    boundary_color = (0, 0, 0, 0.45)
    marker_half = 0.036

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    ax.add_patch(patches.Rectangle((0, 0), 1.0, 0.5, facecolor=colors[0], edgecolor="none"))
    for index, color in enumerate(colors[1:]):
        ax.add_patch(
            patches.Rectangle(
                (index * method_width, 0.5),
                method_width,
                0.5,
                facecolor=color,
                edgecolor="none",
            )
        )

    ax.add_patch(
        patches.Rectangle(
            (0, 0),
            1.0,
            1.0,
            fill=False,
            edgecolor=(0, 0, 0, 0.55),
            linewidth=0.35,
        )
    )
    for boundary_x in (method_width, 2.0 * method_width, 3.0 * method_width):
        ax.plot([boundary_x, boundary_x], [0.5, 1.0], color=boundary_color, linewidth=0.25)
    ax.plot([0, marker_half], [0.5, 0.5], color=boundary_color, linewidth=0.28)
    ax.plot([1.0 - marker_half, 1.0], [0.5, 0.5], color=boundary_color, linewidth=0.28)
    for boundary_x in (method_width, 2.0 * method_width, 3.0 * method_width):
        ax.plot(
            [boundary_x - marker_half, boundary_x + marker_half],
            [0.5, 0.5],
            color=boundary_color,
            linewidth=0.28,
        )


def _draw_corner_key_tile(ax, *, k: int) -> None:
    import matplotlib.patches as patches

    colors = [color for _, color in _split_key_entries(k)]
    corner = 1.0 / 3.0
    boundary_color = (0, 0, 0, 0.45)
    marker_len = 0.056

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    ax.add_patch(patches.Rectangle((0, 0), 1.0, 1.0, facecolor=colors[0], edgecolor="none"))
    ax.add_patch(patches.Rectangle((0, 0), corner, corner, facecolor=colors[1], edgecolor="none"))
    ax.add_patch(patches.Rectangle((1.0 - corner, 0), corner, corner, facecolor=colors[2], edgecolor="none"))
    ax.add_patch(patches.Rectangle((0, 1.0 - corner), corner, corner, facecolor=colors[3], edgecolor="none"))
    ax.add_patch(
        patches.Rectangle((1.0 - corner, 1.0 - corner), corner, corner, facecolor=colors[4], edgecolor="none")
    )
    ax.add_patch(
        patches.Rectangle(
            (0, 0),
            1.0,
            1.0,
            fill=False,
            edgecolor=(0, 0, 0, 0.55),
            linewidth=0.35,
        )
    )
    for ix, iy, sx, sy in (
        (corner, corner, -1.0, -1.0),
        (1.0 - corner, corner, 1.0, -1.0),
        (corner, 1.0 - corner, -1.0, 1.0),
        (1.0 - corner, 1.0 - corner, 1.0, 1.0),
    ):
        ax.plot([ix, ix + sx * marker_len], [iy, iy], color=boundary_color, linewidth=0.28)
        ax.plot([ix, ix], [iy, iy + sy * marker_len], color=boundary_color, linewidth=0.28)
        outer_x = ix + sx * corner
        outer_y = iy + sy * corner
        ax.plot([outer_x, outer_x - sx * marker_len], [iy, iy], color=boundary_color, linewidth=0.28)
        ax.plot([ix, ix], [outer_y, outer_y - sy * marker_len], color=boundary_color, linewidth=0.28)


def _draw_split_key_labels(ax, *, k: int) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    entries = _split_key_entries(k)
    y_positions = np.linspace(0.90, 0.10, len(entries))
    for (label, color), y in zip(entries, y_positions):
        ax.plot(
            0.11,
            y,
            marker="s",
            markersize=7.0,
            markerfacecolor=color,
            markeredgecolor=(0, 0, 0, 0.65),
            markeredgewidth=0.35,
            linestyle="None",
        )
        ax.text(
            0.24,
            y,
            label,
            ha="left",
            va="center",
            fontsize=PAPER_LEGEND_SIZE_PT,
            linespacing=0.92,
        )


def _draw_method_key_column(fig, slot_ax, reference_ax, *, k: int, tile_drawer) -> None:
    import matplotlib.patches as patches

    fig.canvas.draw()
    slot = slot_ax.get_position()
    data_tile = reference_ax.transData.transform([(0.0, 0.0), (1.0, 1.0)])
    figure_tile = fig.transFigure.inverted().transform(data_tile)
    tile_width = abs(figure_tile[1, 0] - figure_tile[0, 0])
    tile_height = abs(figure_tile[1, 1] - figure_tile[0, 1])
    if tile_width > slot.width:
        scale = slot.width / tile_width
        tile_width *= scale
        tile_height *= scale

    label_gap = 0.018
    available_label_height = max(0.01, slot.height - tile_height - label_gap)
    label_height = min(available_label_height, max(tile_height * 5.25, slot.height * 0.50))
    group_height = tile_height + label_gap + label_height
    group_bottom = 0.5 - 0.5 * group_height
    group_bottom = max(slot.y0, min(group_bottom, slot.y1 - group_height))

    key_bottom = group_bottom + label_height + label_gap
    label_bottom = group_bottom
    label_ax = fig.add_axes([slot.x0, label_bottom, slot.width, label_height])
    _draw_split_key_labels(label_ax, k=k)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_bboxes = [line.get_window_extent(renderer) for line in label_ax.lines]
    label_bboxes.extend(text.get_window_extent(renderer) for text in label_ax.texts)
    label_x0 = min(bbox.x0 for bbox in label_bboxes)
    label_x1 = max(bbox.x1 for bbox in label_bboxes)
    label_center = 0.5 * (label_x0 + label_x1)
    label_center_fig = fig.transFigure.inverted().transform((label_center, 0.0))[0]
    key_left = label_center_fig - tile_width / 2.0
    key_left = max(slot.x0, min(key_left, slot.x1 - tile_width))
    key_ax = fig.add_axes([key_left, key_bottom, tile_width, tile_height])
    tile_drawer(key_ax, k=k)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    content_bboxes = [key_ax.get_window_extent(renderer)]
    content_bboxes.extend(line.get_window_extent(renderer) for line in label_ax.lines)
    content_bboxes.extend(text.get_window_extent(renderer) for text in label_ax.texts)
    x0 = min(bbox.x0 for bbox in content_bboxes)
    y0 = min(bbox.y0 for bbox in content_bboxes)
    x1 = max(bbox.x1 for bbox in content_bboxes)
    y1 = max(bbox.y1 for bbox in content_bboxes)
    (box_left, box_bottom), (box_right, box_top) = fig.transFigure.inverted().transform(
        [(x0, y0), (x1, y1)]
    )
    pad_x = 0.004
    pad_y = 0.010
    box_left = max(0.0, box_left - pad_x)
    box_bottom = max(0.0, box_bottom - pad_y)
    box_right = min(1.0, box_right + pad_x)
    box_top = min(1.0, box_top + pad_y)
    fig.add_artist(
        patches.Rectangle(
            (box_left, box_bottom),
            box_right - box_left,
            box_top - box_bottom,
            transform=fig.transFigure,
            fill=False,
            edgecolor=(0, 0, 0, 0.65),
            linewidth=0.45,
            zorder=20,
            clip_on=False,
        )
    )
    slot_ax.axis("off")


def _draw_split_key_column(fig, slot_ax, reference_ax, *, k: int) -> None:
    _draw_method_key_column(fig, slot_ax, reference_ax, k=k, tile_drawer=_draw_split_key_tile)


def _draw_corner_key_column(fig, slot_ax, reference_ax, *, k: int) -> None:
    _draw_method_key_column(fig, slot_ax, reference_ax, k=k, tile_drawer=_draw_corner_key_tile)


def _draw_compact_spd(ax, wl: np.ndarray, spd: np.ndarray) -> None:
    draw_normalized_spd(ax, wl, spd)
    ax.set_xlabel("Wavelength (nm)")
    ax.tick_params(axis="x", pad=1)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def plot_macbeth_multi_illuminant_split_mosaic(
    project_root: Path,
    paper_data: dict,
    output: Path,
    *,
    illuminant_names: tuple[str, str, str] = MACBETH_MOSAIC_ILLUMINANTS,
    k: int = 6,
) -> None:
    """Plot split-patch Macbeth mosaics for three illuminants.

    Each Macbeth patch is divided into ground truth (top), and four bottom
    methods: PI map K=3, RGB, PI map K=k, and Darling.
    The lower method regions include per-patch CIEDE2000 errors computed
    with the same LuxPy settings as the paper.
    """

    from renderers import DarlingSharedBasisRenderer, ExactRenderer, KernelDecoderBasisRenderer, RGBAlbedoRenderer
    from spectral import load_luxpy_spectral_data

    if len(illuminant_names) != 3:
        raise ValueError("The Macbeth mosaic is formatted for exactly three illuminants.")

    plt = configure_matplotlib(project_root)
    data = load_luxpy_spectral_data()
    resolved = [_resolve_illuminant_selector(data, selector) for selector in illuminant_names]
    illuminant_indices = [index for index, _ in resolved]
    display_names = [name for _, name in resolved]
    if paper_data.get("illuminant_wl") is not None and paper_data.get("illuminant_spds") is not None:
        spd_wl = np.asarray(paper_data["illuminant_wl"], float)
        illuminant_spds = np.asarray(paper_data["illuminant_spds"], float)
    else:
        spd_wl = np.asarray(data.wl, float)
        illuminant_spds = np.asarray(data.illuminants, float)

    covariance = np.asarray(paper_data["covariance_CR"], float)
    exact = ExactRenderer()
    rgb = RGBAlbedoRenderer(data)

    pi_basis = _learned_gaussian_basis_from_cache(data, paper_data, k)
    pi_map = KernelDecoderBasisRenderer(f"PI map, K={k}", pi_basis, covariance)
    pi_map.fit(data)
    pi_k3_basis = _learned_gaussian_basis_from_cache(data, paper_data, 3)
    pi_map_k3 = KernelDecoderBasisRenderer("PI map, K=3", pi_k3_basis, covariance)
    pi_map_k3.fit(data)

    darling = DarlingSharedBasisRenderer(data)
    darling.fit(data, None, data.pools[0], max_pairs=10000, seed=1)

    fig = plt.figure(figsize=(PAPER_TEXT_WIDTH_IN, 2.05))
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.0, 1.0, 1.0, 0.48],
        height_ratios=[0.62, 2.20],
        hspace=0.58,
        wspace=0.12,
    )
    key_slot_ax = fig.add_subplot(grid[:, 3])
    key_slot_ax.axis("off")
    first_mosaic_ax = None

    for col, (name, illuminant_index) in enumerate(zip(display_names, illuminant_indices)):
        spd_ax = fig.add_subplot(grid[0, col])
        mosaic_ax = fig.add_subplot(grid[1, col])
        if first_mosaic_ax is None:
            first_mosaic_ax = mosaic_ax

        ground_truth_xyz = _render_macbeth_xyz(data, exact, illuminant_index)
        pi_map_xyz = _render_macbeth_xyz(data, pi_map, illuminant_index)
        pi_map_k3_xyz = _render_macbeth_xyz(data, pi_map_k3, illuminant_index)
        rgb_xyz = _render_macbeth_xyz(data, rgb, illuminant_index)
        darling_xyz = _render_macbeth_xyz(data, darling, illuminant_index)

        _draw_split_macbeth_mosaic(
            mosaic_ax,
            ground_truth=_macbeth_xyz_to_rgb(data, ground_truth_xyz, illuminant_index),
            pi_map=_macbeth_xyz_to_rgb(data, pi_map_xyz, illuminant_index),
            pi_map_k3=_macbeth_xyz_to_rgb(data, pi_map_k3_xyz, illuminant_index),
            rgb=_macbeth_xyz_to_rgb(data, rgb_xyz, illuminant_index),
            darling=_macbeth_xyz_to_rgb(data, darling_xyz, illuminant_index),
            pi_errors=_macbeth_de00(data, ground_truth_xyz, pi_map_xyz, illuminant_index),
            pi_k3_errors=_macbeth_de00(data, ground_truth_xyz, pi_map_k3_xyz, illuminant_index),
            rgb_errors=_macbeth_de00(data, ground_truth_xyz, rgb_xyz, illuminant_index),
            darling_errors=_macbeth_de00(data, ground_truth_xyz, darling_xyz, illuminant_index),
        )
        _draw_compact_spd(spd_ax, spd_wl, illuminant_spds[int(illuminant_index)])
        spd_ax.set_title(_tex_label(name), pad=1)

    fig.subplots_adjust(left=0.035, right=0.985, bottom=-0.02, top=0.895)
    if first_mosaic_ax is not None:
        _draw_split_key_column(fig, key_slot_ax, first_mosaic_ax, k=k)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, transparent=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, transparent=False)
    plt.close(fig)


def plot_macbeth_multi_illuminant_corner_cross_mosaic(
    project_root: Path,
    paper_data: dict,
    output: Path,
    *,
    illuminant_names: tuple[str, str, str] = MACBETH_MOSAIC_ILLUMINANTS,
    k: int = 6,
) -> None:
    """Plot corner-method Macbeth mosaics for three illuminants.

    Each Macbeth patch is first drawn as ground truth, then four corner
    method patches are overlaid as RGB, PI map K=3, Darling, and PI map K=k
    from top-left clockwise. The remaining central cross stays ground truth.
    """

    from renderers import DarlingSharedBasisRenderer, ExactRenderer, KernelDecoderBasisRenderer, RGBAlbedoRenderer
    from spectral import load_luxpy_spectral_data

    if len(illuminant_names) != 3:
        raise ValueError("The Macbeth mosaic is formatted for exactly three illuminants.")

    plt = configure_matplotlib(project_root)
    data = load_luxpy_spectral_data()
    resolved = [_resolve_illuminant_selector(data, selector) for selector in illuminant_names]
    illuminant_indices = [index for index, _ in resolved]
    display_names = [name for _, name in resolved]
    if paper_data.get("illuminant_wl") is not None and paper_data.get("illuminant_spds") is not None:
        spd_wl = np.asarray(paper_data["illuminant_wl"], float)
        illuminant_spds = np.asarray(paper_data["illuminant_spds"], float)
    else:
        spd_wl = np.asarray(data.wl, float)
        illuminant_spds = np.asarray(data.illuminants, float)

    covariance = np.asarray(paper_data["covariance_CR"], float)
    exact = ExactRenderer()
    rgb = RGBAlbedoRenderer(data)

    pi_basis = _learned_gaussian_basis_from_cache(data, paper_data, k)
    pi_map = KernelDecoderBasisRenderer(f"PI map, K={k}", pi_basis, covariance)
    pi_map.fit(data)
    pi_k3_basis = _learned_gaussian_basis_from_cache(data, paper_data, 3)
    pi_map_k3 = KernelDecoderBasisRenderer("PI map, K=3", pi_k3_basis, covariance)
    pi_map_k3.fit(data)

    darling = DarlingSharedBasisRenderer(data)
    darling.fit(data, None, data.pools[0], max_pairs=10000, seed=1)

    fig = plt.figure(figsize=(PAPER_TEXT_WIDTH_IN, 2.05))
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=[1.0, 1.0, 1.0, 0.48],
        height_ratios=[0.62, 2.20],
        hspace=0.58,
        wspace=0.12,
    )
    key_slot_ax = fig.add_subplot(grid[:, 3])
    key_slot_ax.axis("off")
    first_mosaic_ax = None

    for col, (name, illuminant_index) in enumerate(zip(display_names, illuminant_indices)):
        spd_ax = fig.add_subplot(grid[0, col])
        mosaic_ax = fig.add_subplot(grid[1, col])
        if first_mosaic_ax is None:
            first_mosaic_ax = mosaic_ax

        ground_truth_xyz = _render_macbeth_xyz(data, exact, illuminant_index)
        pi_map_xyz = _render_macbeth_xyz(data, pi_map, illuminant_index)
        pi_map_k3_xyz = _render_macbeth_xyz(data, pi_map_k3, illuminant_index)
        rgb_xyz = _render_macbeth_xyz(data, rgb, illuminant_index)
        darling_xyz = _render_macbeth_xyz(data, darling, illuminant_index)

        _draw_corner_cross_macbeth_mosaic(
            mosaic_ax,
            ground_truth=_macbeth_xyz_to_rgb(data, ground_truth_xyz, illuminant_index),
            pi_map=_macbeth_xyz_to_rgb(data, pi_map_xyz, illuminant_index),
            pi_map_k3=_macbeth_xyz_to_rgb(data, pi_map_k3_xyz, illuminant_index),
            rgb=_macbeth_xyz_to_rgb(data, rgb_xyz, illuminant_index),
            darling=_macbeth_xyz_to_rgb(data, darling_xyz, illuminant_index),
            pi_errors=_macbeth_de00(data, ground_truth_xyz, pi_map_xyz, illuminant_index),
            pi_k3_errors=_macbeth_de00(data, ground_truth_xyz, pi_map_k3_xyz, illuminant_index),
            rgb_errors=_macbeth_de00(data, ground_truth_xyz, rgb_xyz, illuminant_index),
            darling_errors=_macbeth_de00(data, ground_truth_xyz, darling_xyz, illuminant_index),
        )
        _draw_compact_spd(spd_ax, spd_wl, illuminant_spds[int(illuminant_index)])
        spd_ax.set_title(_tex_label(name), pad=1)

    fig.subplots_adjust(left=0.035, right=0.985, bottom=-0.02, top=0.895)
    if first_mosaic_ax is not None:
        _draw_corner_key_column(fig, key_slot_ax, first_mosaic_ax, k=k)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, transparent=True)
    fig.savefig(output.with_suffix(".png"), dpi=300, transparent=False)
    plt.close(fig)
