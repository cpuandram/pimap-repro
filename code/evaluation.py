"""Error evaluation and percentile-table assembly helpers."""

from __future__ import annotations

import numpy as np

from learning import sample_paths
from presets import COMPARISON_TABLE_PERCENTILES, PERCENTILES, TABLE_POOL_INDICES
from renderers import Renderer
from spectral import SpectralData, _as_list, paper_illuminant_indices


def compute_error_percentiles(
    data: SpectralData,
    truth: Renderer,
    renderer: Renderer,
    ill_indices,
    reflectance_pool: np.ndarray,
    *,
    max_bounces: int,
    max_combos: int,
    seed: int,
    percentiles: np.ndarray = PERCENTILES,
    de_type: str = "jab",
) -> dict:
    """Evaluate DE2000 percentiles for one renderer over sampled paths.

    Args:
        data: Spectral context containing spectra, whitepoints, and reflectance pools.
        truth: Ground-truth renderer, normally `ExactRenderer`.
        renderer: Approximate renderer to compare against `truth`.
        ill_indices: Illuminants to evaluate; index, list, slice, or `None`.
        reflectance_pool: Candidate reflectance indices for path sampling.
        max_bounces: Largest bounce count to evaluate.
        max_combos: Number of random paths per bounce count.
        seed: Random seed for path sampling.
        percentiles: Percentiles to store in the output arrays.
        de_type: LuxPy DE2000 type, `"jab"` for the paper plots.

    Returns:
        Dict with bounce counts, percentiles, pooled errors `P`, and per-illuminant
        errors `P_ill`.
    """

    import luxpy as lx

    rng = np.random.default_rng(seed)
    ill = np.asarray(_as_list(ill_indices, data.illuminants.shape[0]), int)
    bounces = np.arange(1, int(max_bounces) + 1, dtype=int)
    pct = np.asarray(percentiles, int)
    pooled = np.zeros((len(bounces), len(pct)), float)
    per_ill = np.zeros((len(bounces), len(ill), len(pct)), float)
    white = data.xyzw[ill]

    for bi, bounce in enumerate(bounces):
        errors = np.zeros((int(max_combos), len(ill)), float)
        paths = sample_paths(rng, reflectance_pool, int(bounce), int(max_combos))
        for pi, path in enumerate(paths):
            xyz_true = truth.integrate_xyz(data, ill, path)
            xyz_test = renderer.integrate_xyz(data, ill, path)
            de = lx.deltaE.DE2000(xyz_test, xyz_true, xyzwt=white, xyzwr=white, DEtype=de_type)
            errors[pi] = np.asarray(de, float).reshape(-1)

        pooled[bi] = np.percentile(errors.reshape(-1), pct)
        per_ill[bi] = np.percentile(errors, pct, axis=0).T

    return {
        "bounces": bounces,
        "percentiles": pct,
        "P": pooled,
        "P_ill": per_ill,
        "ill_indices": ill,
    }


def evaluate_model_grid(
    data: SpectralData,
    truth: Renderer,
    renderers: list[Renderer],
    ill_indices,
    *,
    max_bounces: int,
    max_combos: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate all figure renderers over all reflectance pools.

    Args:
        data: Spectral context with the paper reflectance pools.
        truth: Ground-truth renderer.
        renderers: Approximate renderers to evaluate in figure order.
        ill_indices: Illuminants included in the evaluation.
        max_bounces: Largest bounce count for figures.
        max_combos: Number of sampled paths per pool/model/bounce.
        seed: Base seed; pool and model offsets are added internally.

    Returns:
        Tuple `(P, P_ill)` where `P` is pooled over illuminants and paths, and
        `P_ill` keeps illuminants separate.
    """

    p_all = np.empty(
        (len(data.pools), len(renderers), int(max_bounces), len(PERCENTILES)),
        dtype=float,
    )
    p_ill_all = np.empty(
        (
            len(data.pools),
            len(renderers),
            int(max_bounces),
            len(_as_list(ill_indices, data.illuminants.shape[0])),
            len(PERCENTILES),
        ),
        dtype=float,
    )

    for pool_index, pool in enumerate(data.pools):
        for model_index, renderer in enumerate(renderers):
            print(f"[full] {data.pool_titles[pool_index]} | {renderer.label}")
            stats = compute_error_percentiles(
                data,
                truth,
                renderer,
                ill_indices,
                pool,
                max_bounces=max_bounces,
                max_combos=max_combos,
                seed=seed + 1000 * pool_index + 100 * model_index,
            )
            p_all[pool_index, model_index] = stats["P"]
            p_ill_all[pool_index, model_index] = stats["P_ill"]
    return p_all, p_ill_all


def percentile_table_from_renderer(
    data: SpectralData,
    truth: Renderer,
    rows: list[tuple[str, Renderer]],
    *,
    max_bounces: int,
    max_combos: int,
    seed: int,
    percentiles: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a percentile table directly from live renderers.

    Args:
        data: Spectral context.
        truth: Ground-truth renderer used as the reference.
        rows: `(method_name, renderer)` pairs in the desired table order.
        max_bounces: Largest bounce count in the table.
        max_combos: Number of random held-out paths per method and bounce count.
        seed: Base seed; method offsets are added internally.
        percentiles: Percentiles to extract into the final table.

    Returns:
        Arrays for bounce numbers, method labels, percentile labels, and values.
    """

    pct = np.asarray(percentiles, int)
    out_bounces = []
    out_methods = []
    out_values = []
    table_pool = np.concatenate([data.pools[int(index)] for index in TABLE_POOL_INDICES]).astype(int)
    ill_indices = paper_illuminant_indices(data)

    method_stats = []
    for method_index, (method, renderer) in enumerate(rows):
        stats = compute_error_percentiles(
            data,
            truth,
            renderer,
            ill_indices,
            table_pool,
            max_bounces=max_bounces,
            max_combos=max_combos,
            seed=seed + 500 * method_index,
            percentiles=np.arange(0, 101, 1, dtype=int),
        )
        method_stats.append((method, stats))

    idx = [percentile_index(method_stats[0][1], p) for p in pct]
    for bounce_index, bounce in enumerate(method_stats[0][1]["bounces"]):
        for method, stats in method_stats:
            out_bounces.append(int(bounce))
            out_methods.append(method)
            out_values.append(stats["P"][bounce_index, idx])

    return (
        np.asarray(out_bounces, int),
        np.asarray(out_methods),
        np.asarray([f"p{p}" for p in pct]),
        np.asarray(out_values, float),
    )


def combined_percentile_table_from_stats(
    method_stats: list[tuple[str, dict]],
    *,
    percentiles: list[int] = COMPARISON_TABLE_PERCENTILES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build one row-style percentile table from precomputed method stats.

    Args:
        method_stats: `(method_name, stats)` pairs, where each `stats` is from
            `compute_error_percentiles`.
        percentiles: Percentiles to keep as table columns.

    Returns:
        Arrays for bounce numbers, method labels, percentile labels, and values.
    """

    pct = list(map(int, percentiles))
    out_bounces = []
    out_methods = []
    out_values = []

    bounces = np.asarray(method_stats[0][1]["bounces"], int)
    for bounce_index, bounce in enumerate(bounces):
        for method, stats in method_stats:
            idx = [percentile_index(stats, p) for p in pct]
            out_bounces.append(int(bounce))
            out_methods.append(method)
            out_values.append(np.asarray(stats["P"], float)[bounce_index, idx])

    return (
        np.asarray(out_bounces, int),
        np.asarray(out_methods),
        np.asarray([f"p{p}" for p in pct]),
        np.asarray(out_values, float),
    )


def percentile_index(cache: dict, value: int) -> int:
    """Return the column index for one stored percentile value.

    Args:
        cache: Error cache containing a `percentiles` array.
        value: Requested percentile, for example 50 or 95.
    """

    pct = np.asarray(cache["percentiles"], int)
    hits = np.where(pct == int(value))[0]
    if len(hits) != 1:
        raise ValueError(f"Percentile {value} not found in cache.")
    return int(hits[0])
