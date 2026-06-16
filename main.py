#!/usr/bin/env python3
"""Single entry point for the renderer paper reproduction capsule."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
CODE_DIR = PROJECT_ROOT / "code"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))


def add_code_dir() -> None:
    code_path = str(CODE_DIR)
    if code_path not in sys.path:
        sys.path.insert(0, code_path)


def parse_figure_illuminants(value: str) -> tuple[int, ...]:
    """Parse a comma-separated list of illuminant indices.

    Args:
        value: Text such as `"8,16,35"` passed on the command line.
    """

    try:
        ids = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integer illuminant indices.") from exc
    if not ids:
        raise argparse.ArgumentTypeError("Expected at least one illuminant index.")
    return ids


def parse_illuminant_selectors(value: str) -> tuple[str, ...]:
    """Parse three comma-separated CIE illuminant names or indices."""

    selectors = tuple(part.strip() for part in value.split(",") if part.strip())
    if len(selectors) != 3:
        raise argparse.ArgumentTypeError("Expected exactly three comma-separated illuminant names or indices.")
    return selectors


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the repro capsule.

    The `--full-*` options only affect `--mode full`; `--mode npz` reads the
    compact bundled paper data and ignores recomputation settings.
    """

    add_code_dir()
    from presets import MACBETH_MOSAIC_ILLUMINANTS

    parser = argparse.ArgumentParser(
        description="Reproduce paper figures and tables from the clean capsule."
    )
    parser.add_argument(
        "--mode",
        choices=("npz", "full"),
        default="npz",
        help=(
            "npz loads the compact bundled data file; full recomputes the "
            "paper data from LuxPy spectra and clean renderer implementations."
        ),
    )
    parser.add_argument("--full-max-combos", type=int, default=4000)
    parser.add_argument("--full-cov-paths", type=int, default=100000)
    parser.add_argument("--full-seed", type=int, default=0)
    parser.add_argument(
        "--full-basis-epochs",
        type=int,
        default=None,
        help=(
            "Override Gaussian-basis training epochs for each K. By default, "
            "full mode uses 25000 epochs for every K."
        ),
    )
    parser.add_argument("--full-basis-lr", type=float, default=5e-3)
    parser.add_argument("--full-basis-reg", type=float, default=1.0)
    parser.add_argument("--full-basis-verbose-every", type=int, default=0)
    parser.add_argument(
        "--figure-illuminants",
        type=parse_figure_illuminants,
        default=None,
        help=(
            "Comma-separated illuminant indices for the per-illuminant SPD/error "
            "figures, for example `8,16` or `8,16,35,42`. Defaults to the paper preset."
        ),
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("all", help="Generate all figures and tables.")
    p.add_argument(
        "--figure-illuminants",
        type=parse_figure_illuminants,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    p.set_defaults(command="all")

    p = sub.add_parser("figures", help="Generate all paper figures.")
    p.add_argument(
        "--figure-illuminants",
        type=parse_figure_illuminants,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    p.set_defaults(command="figures")

    p = sub.add_parser("tables", help="Generate all paper tables.")
    p.set_defaults(command="tables")

    p = sub.add_parser("inventory", help="Print folder and artifact inventory.")
    p.set_defaults(command="inventory")

    p = sub.add_parser("pack-npz", help="Recompute and pack the compact NPZ data file.")
    p.set_defaults(command="pack-npz")

    p = sub.add_parser("basis", help="Print learned Gaussian basis parameters.")
    p.set_defaults(command="basis")

    p = sub.add_parser("macbeth-mosaic", help="Generate the split Macbeth multi-illuminant comparison figure.")
    p.add_argument(
        "--illuminants",
        type=parse_illuminant_selectors,
        default=MACBETH_MOSAIC_ILLUMINANTS,
        help="Exactly three comma-separated illuminant names or indices, for example LED_V1,LED_B2,HP2 or 45,39,34.",
    )
    p.add_argument("--k", type=int, default=6, help="Learned PI-map basis dimension.")
    p.set_defaults(command="macbeth-mosaic")

    p = sub.add_parser("macbeth-corner-mosaic", help="Generate the corner/cross Macbeth comparison figure.")
    p.add_argument(
        "--illuminants",
        type=parse_illuminant_selectors,
        default=MACBETH_MOSAIC_ILLUMINANTS,
        help="Exactly three comma-separated illuminant names or indices, for example LED_V1,LED_B2,HP2 or 45,39,34.",
    )
    p.add_argument("--k", type=int, default=6, help="Learned PI-map basis dimension.")
    p.set_defaults(command="macbeth-corner-mosaic")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch CLI commands.

    Args:
        argv: Optional argument list for tests or programmatic use. `None`
            uses the process command line.
    """

    add_code_dir()
    from pipeline import (
        build_npz_from_full_recompute,
        generate_all,
        generate_figures,
        generate_tables,
        load_paper_data,
        print_gaussian_basis,
        print_inventory,
    )
    from plotting import (
        plot_macbeth_multi_illuminant_corner_cross_mosaic,
        plot_macbeth_multi_illuminant_split_mosaic,
    )
    from presets import FullRecomputeConfig

    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "all"
    full_config = FullRecomputeConfig(
        max_combos=args.full_max_combos,
        covariance_paths=args.full_cov_paths,
        seed=args.full_seed,
        gaussian_basis_epochs=args.full_basis_epochs,
        gaussian_basis_lr=args.full_basis_lr,
        gaussian_basis_reg=args.full_basis_reg,
        gaussian_basis_verbose_every=args.full_basis_verbose_every,
    )

    try:
        if command == "all":
            generate_all(
                PROJECT_ROOT,
                mode=args.mode,
                full_config=full_config,
                illuminant_ids=args.figure_illuminants,
            )
        elif command == "figures":
            generate_figures(
                PROJECT_ROOT,
                mode=args.mode,
                full_config=full_config,
                illuminant_ids=args.figure_illuminants,
            )
        elif command == "tables":
            generate_tables(PROJECT_ROOT, mode=args.mode, full_config=full_config)
        elif command == "inventory":
            print_inventory(PROJECT_ROOT)
        elif command == "pack-npz":
            build_npz_from_full_recompute(PROJECT_ROOT, full_config)
        elif command == "basis":
            print_gaussian_basis(PROJECT_ROOT, mode=args.mode, full_config=full_config)
        elif command == "macbeth-mosaic":
            paper_data = load_paper_data(PROJECT_ROOT, mode=args.mode, full_config=full_config)
            safe_names = "_".join(str(name).lower().replace(".", "p") for name in args.illuminants)
            output = (
                PROJECT_ROOT
                / "results"
                / "Figures"
                / f"macbeth_multi_{safe_names}_split_gt_pi_k{int(args.k)}_darling.pdf"
            )
            plot_macbeth_multi_illuminant_split_mosaic(
                PROJECT_ROOT,
                paper_data,
                output,
                illuminant_names=args.illuminants,
                k=int(args.k),
            )
            print(f"Wrote Macbeth mosaic to {output}")
        elif command == "macbeth-corner-mosaic":
            paper_data = load_paper_data(PROJECT_ROOT, mode=args.mode, full_config=full_config)
            safe_names = "_".join(str(name).lower().replace(".", "p") for name in args.illuminants)
            output = (
                PROJECT_ROOT
                / "results"
                / "Figures"
                / f"macbeth_multi_{safe_names}_corner_cross_gt_pi_k{int(args.k)}_darling.pdf"
            )
            plot_macbeth_multi_illuminant_corner_cross_mosaic(
                PROJECT_ROOT,
                paper_data,
                output,
                illuminant_names=args.illuminants,
                k=int(args.k),
            )
            print(f"Wrote Macbeth corner mosaic to {output}")
        else:
            parser.error(f"Unknown command: {command}")
    except (KeyError, ValueError) as exc:
        message = exc.args[0] if exc.args else str(exc)
        parser.exit(1, f"error: {message}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
