"""High-level recomputation, artifact generation, and reporting pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from basis import darling_basis_matrix, xyz_reflectance_basis
from evaluation import (
    combined_percentile_table_from_stats,
    compute_error_percentiles,
    evaluate_model_grid,
)
from learning import build_reflectance_covariance, learn_gaussian_bases
from paths import data_dir, ensure_dirs, figures_dir, load_paper_npz, tables_dir, write_paper_npz
from plotting import (
    plot_dataset_reflectance_covariances,
    plot_gaussian_bases,
    plot_iqr_pooled,
    plot_iqr_three_illuminants,
    plot_macbeth_multi_illuminant_corner_cross_mosaic,
)
from presets import (
    COMPARISON_TABLE_PERCENTILES,
    FIGURE_MODELS_FIXED,
    FIGURE_MODELS_LEARNED,
    IQR_FIXED_ILLUMINANTS,
    IQR_LEARNED_ILLUMINANTS,
    MACBETH_MOSAIC_ILLUMINANTS,
    PAPER_NPZ,
    PERCENTILES,
    TABLE_POOL_INDICES,
    TABLE_POOL_TITLE,
    TRAINING_POOL_INDEX,
    FullRecomputeConfig,
)
from renderers import (
    DarlingSharedBasisRenderer,
    ExactRenderer,
    KernelDecoderBasisRenderer,
    RGBAlbedoRenderer,
    WandellFourierRenderer,
    WandellWPcaRenderer,
)
from spectral import SpectralData, load_luxpy_spectral_data, paper_illuminant_indices
from tables import generate_tables_from_data


def resolve_illuminant_selectors(data: SpectralData, selectors) -> tuple[int, ...]:
    """Resolve illuminant names or integer-like selectors to LuxPy row indices."""

    indices = []
    for selector in selectors:
        if isinstance(selector, str) and not selector.strip().lstrip("-").isdigit():
            try:
                indices.append(data.illuminant_names.index(selector))
            except ValueError as exc:
                raise ValueError(f"Unknown illuminant {selector!r}.") from exc
        else:
            index = int(selector)
            if index < 0 or index >= len(data.illuminant_names):
                raise ValueError(f"Illuminant index {index} is outside 0..{len(data.illuminant_names) - 1}.")
            indices.append(index)
    return tuple(indices)


def cache_from_npz(data: dict) -> dict:
    """Convert flat NPZ/full-recompute arrays into the plotting cache shape.

    Args:
        data: Dictionary returned by `load_paper_npz` or full recomputation.

    Returns:
        Nested cache with pool/model entries consumed by plotting functions.
    """

    model_labels = [str(x) for x in data["model_labels"]]
    pool_titles = [str(x) for x in data["pool_titles"]]
    bounces = np.asarray(data["bounces"], int)
    percentiles = np.asarray(data["percentiles"], int)
    p_all = np.asarray(data["P"], float)
    p_ill_all = np.asarray(data["P_ill"], float)
    ill_indices = np.asarray(data["ill_indices"], int)

    pools = []
    for pool_index, title in enumerate(pool_titles):
        models = {}
        for model_index, label in enumerate(model_labels):
            models[label] = {
                "bounces": bounces.copy(),
                "percentiles": percentiles.copy(),
                "P": p_all[pool_index, model_index].copy(),
                "P_ill": p_ill_all[pool_index, model_index].copy(),
                "ill_indices": ill_indices.copy(),
            }
        pools.append({"title": title, "models": models})

    return {
        "models": model_labels,
        "pool_titles": pool_titles,
        "pools": pools,
        "percentiles": percentiles,
        "max_bounces": int(bounces[-1]),
        "error_metric": str(data["error_metric"][0]),
        "DEtype": str(data["DEtype"][0]),
        "illuminant_wl": np.asarray(data["illuminant_wl"], float) if "illuminant_wl" in data else None,
        "illuminant_spds": np.asarray(data["illuminant_spds"], float) if "illuminant_spds" in data else None,
        "illuminant_names": [str(x) for x in data["illuminant_names"]] if "illuminant_names" in data else None,
    }


def build_renderers(data: SpectralData, covariance: np.ndarray, config: FullRecomputeConfig):
    """Construct and fit every renderer needed by the full recompute.

    Args:
        data: Spectral context loaded from LuxPy.
        covariance: Reflectance-path covariance matrix used by kernel decoders
            and learned Gaussian basis fitting.
        config: Full recomputation settings controlling sample counts, seeds,
            Darling fitting, and Gaussian learning.

    Returns:
        Dictionary containing truth/baseline renderers, learned Gaussian renderers,
        Wandell WPCA/Fourier renderers, and the learned Gaussian parameter payload.
    """

    gaussian_params = learn_gaussian_bases(data, covariance, config)
    ill_indices = None
    training_reflectances = data.pools[TRAINING_POOL_INDEX]

    truth = ExactRenderer()
    rgb = RGBAlbedoRenderer(data)

    darling = DarlingSharedBasisRenderer(data)
    darling.fit(
        data,
        ill_indices,
        training_reflectances,
        max_pairs=config.darling_fit_pairs,
        seed=config.seed + 1,
    )

    pimap = {}
    for k in [3, 4, 5, 6]:
        basis = np.asarray(gaussian_params[k]["basis"], float)
        renderer = KernelDecoderBasisRenderer(f"PI map, K={k}", basis, covariance)
        renderer.fit(data, ill_indices)
        pimap[k] = renderer

    rgb_decoder = KernelDecoderBasisRenderer("PI map, RGB", xyz_reflectance_basis(data), covariance)
    rgb_decoder.fit(data, ill_indices)

    darling_basis = darling_basis_matrix(data)
    darling_decoder = KernelDecoderBasisRenderer("PI map, Darling", darling_basis, covariance)
    darling_decoder.fit(data, ill_indices)

    y_weight = np.sqrt(np.maximum(data.xyz_cmf[:, 1], 0.0))
    wandell = {}
    wandell_fourier = {}
    for k in [3, 4, 5, 6]:
        wandell[k] = WandellWPcaRenderer(
            rf"Wandell $K={k}$",
            data,
            k=k,
            weights=y_weight,
            fit_indices=training_reflectances,
        )
        wandell_fourier[k] = WandellFourierRenderer(rf"Wandell Fourier $K={k}$", data, k=k)

    return {
        "truth": truth,
        "rgb": rgb,
        "darling": darling,
        "pimap": pimap,
        "rgb_decoder": rgb_decoder,
        "darling_decoder": darling_decoder,
        "wandell": wandell,
        "wandell_fourier": wandell_fourier,
        "gaussian_params": gaussian_params,
    }


def _gaussian_param_arrays(gaussian_params: dict[int, dict]):
    """Pack variable-length learned Gaussian parameters into rectangular arrays.

    Args:
        gaussian_params: Mapping from K to `learn_gaussian_basis` result dict.
    """

    gaussian_k = np.array(sorted(gaussian_params), dtype=int)
    max_channels = max(gaussian_k)
    gaussian_mu = np.full((len(gaussian_k), max_channels), np.nan, dtype=float)
    gaussian_sigma = np.full_like(gaussian_mu, np.nan)
    gaussian_epochs = np.zeros(len(gaussian_k), dtype=int)
    gaussian_loss = np.full(len(gaussian_k), np.nan, dtype=float)
    gaussian_counts = np.zeros(len(gaussian_k), dtype=int)
    for row, k in enumerate(gaussian_k):
        mu = np.asarray(gaussian_params[int(k)]["mu"], float)
        sigma = np.asarray(gaussian_params[int(k)]["sigma"], float)
        gaussian_counts[row] = len(mu)
        gaussian_mu[row, : len(mu)] = mu
        gaussian_sigma[row, : len(sigma)] = sigma
        gaussian_epochs[row] = int(gaussian_params[int(k)].get("epochs", 0))
        gaussian_loss[row] = float(gaussian_params[int(k)].get("loss_total", np.nan))
    return gaussian_k, gaussian_counts, gaussian_mu, gaussian_sigma, gaussian_epochs, gaussian_loss


def _gaussian_loss_component_arrays(gaussian_params: dict[int, dict]) -> tuple[np.ndarray, np.ndarray]:
    """Pack kernel and basis-regularizer loss components by K.

    Args:
        gaussian_params: Mapping from K to `learn_gaussian_basis` result dict.
    """

    gaussian_k = np.array(sorted(gaussian_params), dtype=int)
    gaussian_loss_kernel = np.full(len(gaussian_k), np.nan, dtype=float)
    gaussian_loss_basis_reg = np.full(len(gaussian_k), np.nan, dtype=float)
    for row, k in enumerate(gaussian_k):
        gaussian_loss_kernel[row] = float(gaussian_params[int(k)].get("loss_kernel", np.nan))
        gaussian_loss_basis_reg[row] = float(gaussian_params[int(k)].get("loss_basis_reg", np.nan))
    return gaussian_loss_kernel, gaussian_loss_basis_reg


def build_paper_data_from_full_recompute(project_root: Path, config: FullRecomputeConfig) -> dict:
    """Recompute every paper data array from LuxPy and renderer code.

    Args:
        project_root: Repository root; only used for
            output directory creation and Matplotlib/cache paths.
        config: Numerical settings for covariance estimation, renderer fitting,
            Gaussian learning, and evaluation sample counts.

    Returns:
        In-memory paper-data dictionary. This function does not read the bundled
        NPZ, legacy table CSVs, or saved Gaussian parameters.
    """

    ensure_dirs(project_root)

    data = load_luxpy_spectral_data(config.cieobs)
    ill_indices = paper_illuminant_indices(data)
    training_pool = data.pools[TRAINING_POOL_INDEX]
    covariance = build_reflectance_covariance(
        data.reflectances,
        training_pool,
        n_paths=config.covariance_paths,
        seed=config.seed,
    )
    renderers = build_renderers(data, covariance, config)

    figure_renderers = [
        renderers["rgb"],
        renderers["darling"],
        renderers["rgb_decoder"],
        renderers["darling_decoder"],
        renderers["pimap"][3],
        renderers["pimap"][4],
        renderers["pimap"][5],
        renderers["pimap"][6],
    ]
    p, p_ill = evaluate_model_grid(
        data,
        renderers["truth"],
        figure_renderers,
        ill_indices,
        max_bounces=config.max_bounces,
        max_combos=config.max_combos,
        seed=config.seed,
    )

    table_pool = np.concatenate([data.pools[int(index)] for index in TABLE_POOL_INDICES]).astype(int)
    table_pool_title = TABLE_POOL_TITLE
    training_pool_title = data.pool_titles[TRAINING_POOL_INDEX]

    fixed_rows = [
        ("RGB", renderers["rgb"]),
        ("Darling", renderers["darling"]),
        ("PI map, RGB", renderers["rgb_decoder"]),
        ("PI map, Darling", renderers["darling_decoder"]),
    ]
    fixed_stats = []
    for method_index, (method, renderer) in enumerate(fixed_rows):
        fixed_stats.append(
            (
                method,
                compute_error_percentiles(
                    data,
                    renderers["truth"],
                    renderer,
                    ill_indices,
                    table_pool,
                    max_bounces=config.table_max_bounces,
                    max_combos=config.max_combos,
                    seed=config.seed + 10000 + 500 * method_index,
                    percentiles=np.arange(0, 101, 1, dtype=int),
                ),
            )
        )
    fixed_bounces, fixed_methods, fixed_percentiles, fixed_values = combined_percentile_table_from_stats(
        fixed_stats,
        percentiles=[70, 80, 90, 95, 99, 100],
    )
    learned_rows = [
        ("pimap_k6", "PI map, K=6", renderers["pimap"][6]),
        ("pimap_k5", "PI map, K=5", renderers["pimap"][5]),
        ("pimap_k4", "PI map, K=4", renderers["pimap"][4]),
        ("pimap_k3", "PI map, K=3", renderers["pimap"][3]),
        ("wandell_k6", "Wandell K=6", renderers["wandell"][6]),
        ("wandell_k5", "Wandell K=5", renderers["wandell"][5]),
        ("wandell_k4", "Wandell K=4", renderers["wandell"][4]),
        ("wandell_k3", "Wandell K=3", renderers["wandell"][3]),
    ]
    learned_stats = []
    for column_index, (_, display_name, renderer) in enumerate(learned_rows):
        learned_stats.append(
            (
                display_name,
                compute_error_percentiles(
                    data,
                    renderers["truth"],
                    renderer,
                    ill_indices,
                    table_pool,
                    max_bounces=config.table_max_bounces,
                    max_combos=config.max_combos,
                    seed=config.seed + 20000 + 1000 * column_index,
                    percentiles=np.arange(0, 101, 1, dtype=int),
                ),
            )
        )
    learned_bounces, learned_methods, learned_percentiles, learned_values = combined_percentile_table_from_stats(
        learned_stats,
        percentiles=[70, 80, 90, 95, 99, 100],
    )

    fourier_stats = []
    for k in [6, 5, 4, 3]:
        method_index = len(fourier_stats)
        fourier_stats.append(
            (
                f"Wandell Fourier K={k}",
                compute_error_percentiles(
                    data,
                    renderers["truth"],
                    renderers["wandell_fourier"][k],
                    ill_indices,
                    table_pool,
                    max_bounces=config.table_max_bounces,
                    max_combos=config.max_combos,
                    seed=config.seed + 30000 + 1000 * method_index,
                    percentiles=np.arange(0, 101, 1, dtype=int),
                ),
            )
        )
    all_bounces, all_methods, all_percentiles, all_values = combined_percentile_table_from_stats(
        fixed_stats + learned_stats + fourier_stats,
        percentiles=COMPARISON_TABLE_PERCENTILES,
    )

    (
        gaussian_k,
        gaussian_counts,
        gaussian_mu,
        gaussian_sigma,
        gaussian_epochs,
        gaussian_loss,
    ) = _gaussian_param_arrays(renderers["gaussian_params"])
    gaussian_loss_kernel, gaussian_loss_basis_reg = _gaussian_loss_component_arrays(renderers["gaussian_params"])

    return {
        "model_labels": np.asarray([r.label for r in figure_renderers]),
        "pool_titles": np.asarray(data.pool_titles),
        "table_pool_title": np.asarray([table_pool_title]),
        "training_pool_title": np.asarray([training_pool_title]),
        "percentiles": PERCENTILES,
        "bounces": np.arange(1, config.max_bounces + 1, dtype=int),
        "ill_indices": np.asarray(ill_indices, int),
        "P": p,
        "P_ill": p_ill,
        "error_metric": np.array(["de2000"]),
        "DEtype": np.array(["jab"]),
        "covariance_CR": np.asarray(covariance, float),
        "covariance_wl": np.asarray(data.wl, float),
        "gaussian_k": gaussian_k,
        "gaussian_counts": gaussian_counts,
        "gaussian_mu": gaussian_mu,
        "gaussian_sigma": gaussian_sigma,
        "gaussian_epochs": gaussian_epochs,
        "gaussian_loss": gaussian_loss,
        "gaussian_loss_kernel": gaussian_loss_kernel,
        "gaussian_loss_basis_reg": gaussian_loss_basis_reg,
        "illuminant_wl": np.asarray(data.wl, float),
        "illuminant_spds": np.asarray(data.illuminants, float),
        "illuminant_names": np.asarray(data.illuminant_names),
        "fixed_bounces": fixed_bounces,
        "fixed_methods": fixed_methods,
        "fixed_percentiles": fixed_percentiles,
        "fixed_values": fixed_values,
        "learned_bounces": np.asarray(learned_bounces, int),
        "learned_methods": np.asarray(learned_methods),
        "learned_percentiles": np.asarray(learned_percentiles),
        "learned_values": np.asarray(learned_values, float),
        "all_methods_bounces": all_bounces,
        "all_methods_methods": all_methods,
        "all_methods_percentiles": all_percentiles,
        "all_methods_values": all_values,
    }


def build_npz_from_full_recompute(project_root: Path, config: FullRecomputeConfig) -> Path:
    """Run full recomputation and write the resulting compact NPZ.

    Args:
        project_root: Repository root.
        config: Full recomputation settings.
    """

    output = data_dir(project_root) / "cache" / PAPER_NPZ
    paper_data = build_paper_data_from_full_recompute(project_root, config)
    write_paper_npz(output, paper_data)
    print(f"Wrote full recomputation NPZ to {output}")
    return output


def load_paper_data(
    project_root: Path,
    mode: str = "npz",
    full_config: FullRecomputeConfig | None = None,
) -> dict:
    """Load or recompute the paper-data dictionary.

    Args:
        project_root: Repository root.
        mode: `"npz"` reads the compact NPZ; `"full"` recomputes from LuxPy.
        full_config: Optional settings used only when `mode == "full"`.
    """

    if mode == "full":
        return build_paper_data_from_full_recompute(project_root, full_config or FullRecomputeConfig())
    elif mode == "npz":
        npz_path = data_dir(project_root) / "cache" / PAPER_NPZ
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Missing compact NPZ: {npz_path}. Run `python main.py --mode full pack-npz` first."
            )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return load_paper_npz(npz_path)


def generate_tables(
    project_root: Path,
    mode: str = "npz",
    full_config: FullRecomputeConfig | None = None,
) -> None:
    """Load/recompute paper data and generate all table artifacts.

    Args:
        project_root: Repository root.
        mode: `"npz"` reads the compact data cache; `"full"` recomputes first.
        full_config: Optional full recomputation settings used only in full mode.
    """

    paper_data = load_paper_data(project_root, mode=mode, full_config=full_config)
    generate_tables_from_data(project_root, paper_data)


def generate_figures_from_data(
    project_root: Path,
    paper_data: dict,
    illuminant_ids: tuple[int, ...] | None = None,
) -> None:
    """Generate all figure artifacts from an already-loaded paper-data dict.

    Args:
        project_root: Repository root.
        paper_data: Full or NPZ-loaded paper data.
        illuminant_ids: Optional illuminant indices for the per-illuminant
            SPD/error figures. Defaults to the paper preset.
    """

    ensure_dirs(project_root)
    cache = cache_from_npz(paper_data)

    spectral_data = load_luxpy_spectral_data()
    fixed_illuminant_ids = (
        tuple(int(illuminant_id) for illuminant_id in illuminant_ids)
        if illuminant_ids is not None
        else resolve_illuminant_selectors(spectral_data, IQR_FIXED_ILLUMINANTS)
    )
    learned_illuminant_ids = (
        tuple(int(illuminant_id) for illuminant_id in illuminant_ids)
        if illuminant_ids is not None
        else resolve_illuminant_selectors(spectral_data, IQR_LEARNED_ILLUMINANTS)
    )
    plot_dataset_reflectance_covariances(
        project_root,
        figures_dir(project_root) / "reflectance_covariance_3datasets.pdf",
        spectral_data,
    )
    plot_gaussian_bases(project_root, figures_dir(project_root) / "gaussian_bases_2x2.pdf", paper_data)
    plot_iqr_pooled(project_root, cache, figures_dir(project_root) / "iqr_3pools.pdf", FIGURE_MODELS_FIXED)
    plot_iqr_three_illuminants(
        project_root,
        cache,
        figures_dir(project_root) / "iqr_3ills_spd.pdf",
        FIGURE_MODELS_FIXED,
        ill_ids=fixed_illuminant_ids,
    )
    plot_iqr_pooled(
        project_root,
        cache,
        figures_dir(project_root) / "iqr_3pools_darling_torch.pdf",
        FIGURE_MODELS_LEARNED,
    )
    plot_iqr_three_illuminants(
        project_root,
        cache,
        figures_dir(project_root) / "iqr_3ills_spd_darling_torch.pdf",
        FIGURE_MODELS_LEARNED,
        ill_ids=learned_illuminant_ids,
    )
    mosaic_safe_names = "_".join(name.lower().replace(".", "p") for name in MACBETH_MOSAIC_ILLUMINANTS)
    plot_macbeth_multi_illuminant_corner_cross_mosaic(
        project_root,
        paper_data,
        figures_dir(project_root) / f"macbeth_multi_{mosaic_safe_names}_corner_cross_pi_map_k6_darling.pdf",
        illuminant_names=MACBETH_MOSAIC_ILLUMINANTS,
        k=6,
    )
    print(f"Wrote figures to {figures_dir(project_root)}")


def generate_figures(
    project_root: Path,
    mode: str = "npz",
    full_config: FullRecomputeConfig | None = None,
    illuminant_ids: tuple[int, ...] | None = None,
) -> None:
    """Load/recompute paper data and generate all figure artifacts.

    Args:
        project_root: Repository root.
        mode: `"npz"` reads the compact data cache; `"full"` recomputes first.
        full_config: Optional full recomputation settings used only in full mode.
        illuminant_ids: Optional illuminant indices for the per-illuminant
            SPD/error figures.
    """

    paper_data = load_paper_data(project_root, mode=mode, full_config=full_config)
    generate_figures_from_data(project_root, paper_data, illuminant_ids=illuminant_ids)


def generate_all(
    project_root: Path,
    mode: str = "npz",
    full_config: FullRecomputeConfig | None = None,
    illuminant_ids: tuple[int, ...] | None = None,
) -> None:
    """Load/recompute paper data once, then generate figures and tables.

    Args:
        project_root: Repository root.
        mode: `"npz"` reads the compact data cache; `"full"` recomputes first.
        full_config: Optional full recomputation settings used only in full mode.
    """

    paper_data = load_paper_data(project_root, mode=mode, full_config=full_config)
    generate_figures_from_data(project_root, paper_data, illuminant_ids=illuminant_ids)
    generate_tables_from_data(project_root, paper_data)


def print_gaussian_basis(
    project_root: Path,
    mode: str = "npz",
    full_config: FullRecomputeConfig | None = None,
) -> None:
    """Print learned Gaussian basis parameters from NPZ or full recomputation.

    Args:
        project_root: Repository root.
        mode: `"npz"` reads the compact data cache; `"full"` recomputes first.
        full_config: Optional full recomputation settings used only in full mode.
    """

    if mode == "full":
        config = full_config or FullRecomputeConfig()
        data = load_luxpy_spectral_data(config.cieobs)
        covariance = build_reflectance_covariance(
            data.reflectances,
            data.pools[0],
            n_paths=config.covariance_paths,
            seed=config.seed,
        )
        gaussian_params = learn_gaussian_bases(data, covariance, config)
        (
            gaussian_k,
            gaussian_counts,
            gaussian_mu,
            gaussian_sigma,
            gaussian_epochs,
            gaussian_loss,
        ) = _gaussian_param_arrays(gaussian_params)
        gaussian_loss_kernel, gaussian_loss_basis_reg = _gaussian_loss_component_arrays(gaussian_params)
        paper_data = {
            "gaussian_k": gaussian_k,
            "gaussian_counts": gaussian_counts,
            "gaussian_mu": gaussian_mu,
            "gaussian_sigma": gaussian_sigma,
            "gaussian_epochs": gaussian_epochs,
            "gaussian_loss": gaussian_loss,
            "gaussian_loss_kernel": gaussian_loss_kernel,
            "gaussian_loss_basis_reg": gaussian_loss_basis_reg,
        }
    else:
        paper_data = load_paper_data(project_root, mode=mode, full_config=full_config)

    required = ["gaussian_k", "gaussian_counts", "gaussian_mu", "gaussian_sigma"]
    missing = [key for key in required if key not in paper_data]
    if missing:
        raise KeyError(f"Missing Gaussian basis arrays in paper data: {missing}")

    gaussian_k = np.asarray(paper_data["gaussian_k"], int)
    counts = np.asarray(paper_data["gaussian_counts"], int)
    mu_all = np.asarray(paper_data["gaussian_mu"], float)
    sigma_all = np.asarray(paper_data["gaussian_sigma"], float)
    epochs_all = np.asarray(paper_data.get("gaussian_epochs", np.zeros_like(gaussian_k)), int)
    loss_all = np.asarray(paper_data.get("gaussian_loss", np.full_like(gaussian_k, np.nan, dtype=float)), float)
    loss_kernel_all = np.asarray(
        paper_data.get("gaussian_loss_kernel", np.full_like(gaussian_k, np.nan, dtype=float)),
        float,
    )
    loss_basis_reg_all = np.asarray(
        paper_data.get("gaussian_loss_basis_reg", np.full_like(gaussian_k, np.nan, dtype=float)),
        float,
    )

    print("Learned Gaussian basis parameters")
    print("Rows are normalized Gaussian channels sampled on the paper wavelength grid.")
    for row, k in enumerate(gaussian_k):
        n = int(counts[row])
        mus = mu_all[row, :n]
        sigmas = sigma_all[row, :n]
        print()
        print(
            f"K={int(k)}  epochs={int(epochs_all[row])}  "
            f"loss={loss_all[row]:.12g}  "
            f"kernel={loss_kernel_all[row]:.12g}  "
            f"basis_reg={loss_basis_reg_all[row]:.12g}"
        )
        print("channel,mu_nm,sigma_nm")
        for channel, (mu, sigma) in enumerate(zip(mus, sigmas), start=1):
            print(f"{channel},{mu:.6f},{sigma:.6f}")


def print_inventory(project_root: Path) -> None:
    """Print the key repro folders and example commands.

    Args:
        project_root: Repository root.
    """

    print("PI-map paper reproduction")
    print(f"  root:    {project_root}")
    print(f"  code:    {project_root / 'code'}")
    print(f"  data:    {data_dir(project_root)}")
    print(f"  figures: {figures_dir(project_root)}")
    print(f"  tables:  {tables_dir(project_root)}")
    print()
    print("Main command:")
    print("  python main.py --mode npz all")
    print("  python main.py --mode full all")
