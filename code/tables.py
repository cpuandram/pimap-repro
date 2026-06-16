"""Table formatting and writing helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from paths import ensure_dirs, tables_dir, write_text
from presets import ALL_METHODS_TABLE, FIXED_TABLE, LEARNED_TABLE, TABLE_POOL_TITLE


LEARNED_WANDELL_TABLE_METHODS = (
    "PI map, K=6",
    "PI map, K=5",
    "PI map, K=4",
    "PI map, K=3",
    "Wandell K=6",
)


def fixed_table_rows_from_npz(paper_data: dict) -> list[dict[str, str]]:
    """Format fixed-basis table rows from paper-data arrays.

    Args:
        paper_data: Full or NPZ-loaded paper data containing `fixed_*` arrays.
    """

    bounces = np.asarray(paper_data["fixed_bounces"], int)
    methods = [str(x) for x in paper_data["fixed_methods"]]
    percentiles = [str(x) for x in paper_data["fixed_percentiles"]]
    values = np.asarray(paper_data["fixed_values"], float)
    rows = []
    for i, method in enumerate(methods):
        row = {"bounce": str(int(bounces[i])), "method": method}
        for j, percentile in enumerate(percentiles):
            row[percentile] = f"{values[i, j]:.3f}"
        rows.append(row)
    return rows


def learned_table_rows_from_npz(paper_data: dict) -> list[dict[str, str]]:
    """Format learned-vs-Wandell row-style table rows from paper-data arrays.

    Args:
        paper_data: Full or NPZ-loaded paper data containing `learned_*` arrays.
    """

    bounces = np.asarray(paper_data["learned_bounces"], int)
    methods = [str(x) for x in paper_data["learned_methods"]]
    percentiles = [str(x) for x in paper_data["learned_percentiles"]]
    values = np.asarray(paper_data["learned_values"], float)
    rows = []
    for i, method in enumerate(methods):
        row = {"bounce": str(int(bounces[i])), "method": method}
        for j, percentile in enumerate(percentiles):
            row[percentile] = f"{values[i, j]:.3f}"
        rows.append(row)
    return rows


def all_methods_table_rows_from_npz(paper_data: dict) -> list[dict[str, str]]:
    """Format the combined row-style held-out comparison table.

    Args:
        paper_data: Full or NPZ-loaded paper data containing `all_methods_*` arrays.
    """

    required = [
        "all_methods_bounces",
        "all_methods_methods",
        "all_methods_percentiles",
        "all_methods_values",
    ]
    missing = [key for key in required if key not in paper_data]
    if missing:
        raise KeyError(
            "Combined all-method table is missing from this paper data. "
            "Regenerate with `python main.py --mode full tables` or repack the NPZ."
        )

    bounces = np.asarray(paper_data["all_methods_bounces"], int)
    methods = [str(x) for x in paper_data["all_methods_methods"]]
    percentiles = [str(x) for x in paper_data["all_methods_percentiles"]]
    values = np.asarray(paper_data["all_methods_values"], float)
    rows = []
    for i, method in enumerate(methods):
        row = {"bounce": str(int(bounces[i])), "method": method}
        for j, percentile in enumerate(percentiles):
            row[percentile] = f"{values[i, j]:.3f}"
        rows.append(row)
    return rows


def write_rows_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write homogeneous dictionary rows to CSV.

    Args:
        path: Destination CSV path.
        rows: Row dictionaries; the first row defines the field order.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def table_dataset_name(paper_data: dict) -> str:
    """Return and validate the dataset used for table evaluation.

    Args:
        paper_data: Full or NPZ-loaded paper data.
    """

    if "table_pool_title" not in paper_data:
        raise KeyError(
            "This paper data cache predates the held-out table switch. "
            "Regenerate with `python main.py --mode full pack-npz` or run full-mode tables."
        )
    name = str(np.asarray(paper_data["table_pool_title"]).reshape(-1)[0])
    if name != TABLE_POOL_TITLE:
        raise ValueError(f"Expected tables to be evaluated on {TABLE_POOL_TITLE}, but cache says {name!r}.")
    return name


def latex_value(value: float, *, best: bool = False, second: bool = False) -> str:
    """Format one table value with optional ranking markup."""

    text = f"{float(value):.3f}"
    if best:
        return rf"\textbf{{{text}}}"
    if second:
        return rf"\underline{{{text}}}"
    return text


def latex_method_name(method: str, row_best: np.ndarray, row_second: np.ndarray) -> str:
    """Format a method label using the first percentile column ranking."""

    if bool(row_best[0]):
        return rf"\textbf{{{method}}}"
    if bool(row_second[0]):
        return rf"\underline{{{method}}}"
    return method


def ranking_masks(bounces: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean masks for best and second-best entries per bounce/column."""

    best = np.zeros_like(values, dtype=bool)
    second = np.zeros_like(values, dtype=bool)
    for bounce in np.unique(bounces):
        rows = np.where(bounces == bounce)[0]
        if rows.size == 0:
            continue
        for col in range(values.shape[1]):
            order = rows[np.argsort(values[rows, col])]
            best[order[0], col] = True
            if order.size > 1:
                second[order[1], col] = True
    return best, second


def write_percentile_latex_table(
    path: Path,
    *,
    bounces: np.ndarray,
    methods: list[str],
    percentiles: list[str],
    values: np.ndarray,
    caption: str,
    label: str,
) -> None:
    """Write a manuscript-style percentile comparison table.

    Args:
        path: Destination `.tex` path.
        bounces: Row bounce numbers.
        methods: Row method labels.
        percentiles: Percentile column labels, e.g. `p70`.
        values: Numeric table values with shape `(rows, percentiles)`.
        caption: LaTeX caption body.
        label: LaTeX label name.
    """

    bounces = np.asarray(bounces, int)
    values = np.asarray(values, float)
    best, second = ranking_masks(bounces, values)
    colspec = "|c|l|" + "c|" * len(percentiles)

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\small",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\hline",
        "Bounce & Method & " + " & ".join(percentiles) + r" \\",
        r"\hline",
    ]

    for bounce in np.unique(bounces):
        rows = np.where(bounces == bounce)[0]
        for local_row, row_index in enumerate(rows):
            bounce_cell = rf"\multirow{{{len(rows)}}}{{*}}{{{int(bounce)}}}" if local_row == 0 else ""
            method = latex_method_name(methods[row_index], best[row_index], second[row_index])
            cells = [
                latex_value(values[row_index, col], best=best[row_index, col], second=second[row_index, col])
                for col in range(values.shape[1])
            ]
            lines.append(f"{bounce_cell} & {method} & " + " & ".join(cells) + r" \\")
        lines.append(r"\hline")

    lines.extend(
        [
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
            "",
        ]
    )
    write_text(path, "\n".join(lines))


def table_fixed_basis(project_root: Path, paper_data: dict) -> None:
    """Write CSV and TeX for the fixed-basis held-out percentile table.

    Args:
        project_root: Repository root.
        paper_data: Full or NPZ-loaded paper data containing fixed-table arrays.
    """

    dataset_name = table_dataset_name(paper_data)
    rows = fixed_table_rows_from_npz(paper_data)
    write_rows_csv(tables_dir(project_root) / FIXED_TABLE, rows)
    write_percentile_latex_table(
        tables_dir(project_root) / "fixed_basis_macbeth_munsell_percentiles.tex",
        bounces=np.asarray(paper_data["fixed_bounces"], int),
        methods=[str(x) for x in paper_data["fixed_methods"]],
        percentiles=[str(x) for x in paper_data["fixed_percentiles"]],
        values=np.asarray(paper_data["fixed_values"], float),
        caption=(
            rf"$\Delta E_{{00}}$ percentile errors on the {dataset_name} reflectance dataset "
            r"for different bounces, pooled over the CIE illuminants (Sec.~\ref{sec:impl_datasets}). "
            r"Best in \textbf{bold}, second-best \underline{underlined}."
        ),
        label="tab:macbeth_munsell_fixed_p70_p80_p90_p95_p99_p100",
    )


def table_learned_wandell(project_root: Path, paper_data: dict) -> None:
    """Write CSV and TeX for learned Gaussian vs Wandell K=6 held-out percentiles.

    Args:
        project_root: Repository root.
        paper_data: Full or NPZ-loaded paper data containing learned-table arrays.
    """

    dataset_name = table_dataset_name(paper_data)
    methods = np.asarray([str(x) for x in paper_data["learned_methods"]])
    keep = np.isin(methods, LEARNED_WANDELL_TABLE_METHODS)
    bounces = np.asarray(paper_data["learned_bounces"], int)
    for bounce in np.unique(bounces):
        selected = methods[(bounces == bounce) & keep].tolist()
        if selected != list(LEARNED_WANDELL_TABLE_METHODS):
            raise ValueError(
                f"Expected learned-vs-Wandell rows {LEARNED_WANDELL_TABLE_METHODS} "
                f"for bounce {bounce}, but cache contains {selected}."
            )
    rows = [
        row
        for row in learned_table_rows_from_npz(paper_data)
        if row["method"] in LEARNED_WANDELL_TABLE_METHODS
    ]
    write_rows_csv(tables_dir(project_root) / LEARNED_TABLE, rows)
    write_percentile_latex_table(
        tables_dir(project_root) / "learned_vs_wandell_macbeth_munsell_percentiles.tex",
        bounces=bounces[keep],
        methods=methods[keep].tolist(),
        percentiles=[str(x) for x in paper_data["learned_percentiles"]],
        values=np.asarray(paper_data["learned_values"], float)[keep],
        caption=(
            rf"$\Delta E_{{00}}$ percentile errors on the {dataset_name} reflectance dataset "
            r"for learned Gaussian bases and the Wandell $K=6$ reconstruction baseline, "
            r"pooled over the CIE illuminants "
            r"(Sec.~\ref{sec:impl_datasets}). Best in \textbf{bold}, second-best "
            r"\underline{underlined}."
        ),
        label="tab:macbeth_munsell_learned_wandell_p70_p80_p90_p95_p99_p100",
    )


def table_all_methods(project_root: Path, paper_data: dict) -> None:
    """Write the combined held-out percentile table for all compared methods.

    Args:
        project_root: Repository root.
        paper_data: Full or NPZ-loaded paper data containing all-method arrays.
    """

    rows = all_methods_table_rows_from_npz(paper_data)
    write_rows_csv(tables_dir(project_root) / ALL_METHODS_TABLE, rows)

    lines = [
        r"\begin{tabular}{clrrrrr}",
        r"\hline",
        r"Bounce & Method & p70 & p80 & p90 & p95 & p99 \\",
        r"\hline",
    ]
    previous_bounce = None
    for row in rows:
        if previous_bounce is not None and row["bounce"] != previous_bounce:
            lines.append(r"\hline")
        previous_bounce = row["bounce"]
        lines.append(
            f"{row['bounce']} & {row['method']} & {row['p70']} & {row['p80']} & "
            f"{row['p90']} & {row['p95']} & {row['p99']} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", ""])
    write_text(tables_dir(project_root) / "all_methods_macbeth_munsell_percentiles.tex", "\n".join(lines))


def table_cost(project_root: Path) -> None:
    """Write the analytic FMA-cost comparison table.

    Args:
        project_root: Repository root.
    """

    rows = []
    for lights in range(1, 6):
        row = {"L": lights}
        for k in range(3, 7):
            row[f"shared_k{k}"] = 2 * lights * k + 3 * k
            row[f"pimap_k{k}"] = 3 * lights * k + 3 * lights
        rows.append(row)

    csv_path = tables_dir(project_root) / "fma_cost_table.csv"
    with csv_path.open("w", newline="") as handle:
        fieldnames = ["L"] + [f"{kind}_k{k}" for k in range(3, 7) for kind in ("shared", "pimap")]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        r"\begin{tabular}{crrrrrrrr}",
        r"\hline",
        r"$L$ & Shared K=3 & PI map K=3 & Shared K=4 & PI map K=4 & Shared K=5 & PI map K=5 & Shared K=6 & PI map K=6 \\",
        r"\hline",
    ]
    for row in rows:
        vals = [
            row["L"],
            row["shared_k3"],
            row["pimap_k3"],
            row["shared_k4"],
            row["pimap_k4"],
            row["shared_k5"],
            row["pimap_k5"],
            row["shared_k6"],
            row["pimap_k6"],
        ]
        lines.append(" & ".join(map(str, vals)) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", ""])
    write_text(tables_dir(project_root) / "fma_cost_table.tex", "\n".join(lines))


def generate_tables_from_data(project_root: Path, paper_data: dict) -> None:
    """Generate all table artifacts from an already-loaded paper-data dict.

    Args:
        project_root: Repository root.
        paper_data: Full or NPZ-loaded paper data.
    """

    ensure_dirs(project_root)
    table_fixed_basis(project_root, paper_data)
    table_learned_wandell(project_root, paper_data)
    if "all_methods_values" in paper_data:
        table_all_methods(project_root, paper_data)
    else:
        print("Skipping combined all-method table: paper data does not contain all_methods_* arrays.")
    table_cost(project_root)
    print(f"Wrote tables to {tables_dir(project_root)}")
