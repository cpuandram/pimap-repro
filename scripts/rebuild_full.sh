#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.matplotlib-cache}"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$MPLCONFIGDIR"

python main.py --mode full pack-npz
python main.py --mode npz all
