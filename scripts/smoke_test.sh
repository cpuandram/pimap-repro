#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.matplotlib-cache}"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$MPLCONFIGDIR"

python - <<'PY'
import matplotlib
import numpy
import luxpy

print("numpy", numpy.__version__)
print("matplotlib", matplotlib.__version__)
print("luxpy", getattr(luxpy, "__version__", "unknown"))
PY

python main.py inventory
python main.py --mode npz tables
python main.py --mode npz figures
python main.py --mode npz basis
