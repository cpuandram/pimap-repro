# Code Modules

- `presets.py`: numerical constants and `FullRecomputeConfig`
- `spectral.py`: LuxPy spectral loading and the spectral context
- `basis.py`: fixed XYZ, Darling, and Gaussian basis helpers
- `renderers.py`: exact, RGB, Darling, kernel-decoder, and Wandell renderers
- `learning.py`: reflectance covariance and learned Gaussian basis optimization
- `evaluation.py`: Delta E evaluation and percentile arrays
- `plotting.py`: manuscript figure generation
- `tables.py`: CSV/TeX table generation
- `paths.py`: cache/result folders and NPZ I/O
- `pipeline.py`: high-level recomputation and artifact generation

The public entry point is `../main.py`.
