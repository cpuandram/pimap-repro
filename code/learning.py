"""Covariance estimation and learned Gaussian basis optimization."""

from __future__ import annotations

import numpy as np

from presets import DEFAULT_GAUSSIAN_BASIS_EPOCHS, FullRecomputeConfig
from spectral import SpectralData


def sample_paths(rng: np.random.Generator, pool: np.ndarray, bounces: int, count: int) -> list[list[int]]:
    """Sample random reflectance paths for evaluation.

    Args:
        rng: NumPy random generator.
        pool: Reflectance indices eligible for sampling.
        bounces: Number of reflectance events in each path.
        count: Number of paths to sample.
    """

    return [list(map(int, rng.choice(pool, size=bounces, replace=False))) for _ in range(int(count))]


def build_reflectance_covariance(
    reflectances: np.ndarray,
    pool: np.ndarray,
    *,
    n_paths: int,
    seed: int,
    bounce_counts=(1, 2, 3),
    bounce_probs=(0.34, 0.33, 0.33),
) -> np.ndarray:
    """Estimate the wavelength covariance of random reflectance path products.

    Args:
        reflectances: Reflectance spectra shaped `(Nref,M)`.
        pool: Reflectance indices from which paths are sampled.
        n_paths: Number of random products used in the streaming covariance estimate.
        seed: Random seed for path sampling.
        bounce_counts: Candidate bounce counts for each sampled path.
        bounce_probs: Sampling probabilities corresponding to `bounce_counts`.

    Returns:
        Covariance matrix `C_R` shaped `(M,M)`.
    """

    rng = np.random.default_rng(seed)
    counts = np.asarray(bounce_counts, int)
    probs = np.asarray(bounce_probs, float)
    probs = probs / probs.sum()

    mean = np.zeros(reflectances.shape[1], float)
    m2 = np.zeros((reflectances.shape[1], reflectances.shape[1]), float)
    for i in range(1, int(n_paths) + 1):
        b = int(rng.choice(counts, p=probs))
        ids = rng.choice(pool, size=b, replace=True)
        spec = np.prod(reflectances[ids], axis=0)
        delta = spec - mean
        mean += delta / i
        m2 += np.outer(delta, spec - mean)
    return m2 / max(int(n_paths), 1)


def _gaussian_basis_epochs(config: FullRecomputeConfig, k: int) -> int:
    if config.gaussian_basis_epochs is not None:
        return int(config.gaussian_basis_epochs)
    return int(DEFAULT_GAUSSIAN_BASIS_EPOCHS[int(k)])


def learn_gaussian_basis(
    data: SpectralData,
    covariance: np.ndarray,
    *,
    k: int,
    epochs: int,
    lr: float,
    basis_reg_weight: float,
    seed: int,
    verbose_every: int,
    jitter_s: float = 1e-8,
    sigma_min: float = 2.0,
    sigma_max: float = 80.0,
) -> dict:
    """Learn the shared Gaussian basis used by the paper renderer.

    This is the clean-capsule version of the dirty file's
    JointGaussianBasisGLS_BasisRegD/train_gaussian_basis_no_reflectances path:
    Gaussian centers and widths are optimized, while the per-illuminant
    decoders are recomputed in closed form at every step.

    Args:
        data: Spectral context containing LuxPy spectra, CMFs, and illuminants.
        covariance: Reflectance-path covariance `C_R` shaped `(M,M)`.
        k: Number of Gaussian basis channels.
        epochs: Number of Adam optimization steps. `0` evaluates the initialization.
        lr: Adam learning rate.
        basis_reg_weight: Weight applied to the channel-overlap/coverage regularizer.
        seed: Torch random seed.
        verbose_every: Print optimization status every N epochs; 0 disables printing.
        jitter_s: Diagonal regularization for the GLS system.
        sigma_min: Lower bound for Gaussian sigma in nanometers.
        sigma_max: Upper bound for Gaussian sigma in nanometers.

    Returns:
        Dictionary containing learned `mu`, `sigma`, `basis`, GLS `decoders_xyz`,
        epoch count, and final loss diagnostics.
    """

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Full Gaussian-basis recomputation requires PyTorch.") from exc

    torch.manual_seed(int(seed))
    dtype = torch.float64
    device = torch.device("cpu")

    wl = torch.as_tensor(data.wl, device=device, dtype=dtype)
    illuminants = torch.as_tensor(data.illuminants, device=device, dtype=dtype)
    cmf_d = torch.as_tensor(data.xyz_cmf_d, device=device, dtype=dtype)
    covariance_t = torch.as_tensor(covariance, device=device, dtype=dtype)
    d_lambda = torch.tensor(float(data.d_lambda), device=device, dtype=dtype)
    d_weight = torch.as_tensor(np.sqrt(np.maximum(data.xyz_cmf[:, 1], 0.0)), device=device, dtype=dtype)
    d_weight = d_weight.clamp_min(0)
    d_weight = d_weight / d_weight.mean().clamp_min(1e-12)

    g_rows = []
    for ill in range(illuminants.shape[0]):
        g_m = illuminants[ill, :, None] * cmf_d
        g_rows.append(g_m.T)
    g_target = torch.cat(g_rows, dim=0)
    gc_target = g_target @ covariance_t

    init_centers = torch.linspace(447.0, 622.4, steps=int(k), device=device, dtype=dtype)
    init_sigmas = torch.full((int(k),), 12.0, device=device, dtype=dtype)

    def inv_softplus(y):
        return torch.log(torch.expm1(y).clamp_min(1e-12))

    mu_min = float(data.wl[0])
    mu_max = float(data.wl[-1])
    mu0_frac = ((init_centers[0] - mu_min) / (mu_max - mu_min)).clamp(1e-4, 1 - 1e-4)
    mu0_raw = torch.nn.Parameter(torch.log(mu0_frac) - torch.log1p(-mu0_frac))
    if int(k) > 1:
        init_gaps = torch.diff(init_centers).clamp_min(1e-3)
        gap_raw = torch.nn.Parameter(inv_softplus(init_gaps))
        params = [mu0_raw, gap_raw]
    else:
        gap_raw = None
        params = [mu0_raw]
    sigma_raw = torch.nn.Parameter(inv_softplus(init_sigmas - float(sigma_min)).clamp(min=-20, max=20))
    params.append(sigma_raw)

    def centers_sigmas():
        mu0 = mu_min + torch.sigmoid(mu0_raw) * (mu_max - mu_min)
        if int(k) > 1:
            gaps = torch.nn.functional.softplus(gap_raw)
            mu = torch.cat([mu0.reshape(1), mu0 + torch.cumsum(gaps, dim=0)])
        else:
            mu = mu0.reshape(1)
        mu = mu.clamp(mu_min, mu_max)
        sigma = float(sigma_min) + torch.nn.functional.softplus(sigma_raw)
        sigma = torch.minimum(sigma, torch.tensor(float(sigma_max), device=device, dtype=dtype))
        return mu, sigma

    def basis_x():
        mu, sigma = centers_sigmas()
        gaussian = torch.exp(-0.5 * ((wl[None, :] - mu[:, None]) / sigma[:, None].clamp_min(1e-6)) ** 2)
        gaussian_d = gaussian * d_lambda
        return gaussian_d / gaussian_d.sum(dim=1, keepdim=True).clamp_min(1e-30)

    def compute_w_star(x_basis):
        xc = x_basis @ covariance_t
        system = xc @ x_basis.T
        system = system + float(jitter_s) * torch.eye(int(k), device=device, dtype=dtype)
        rhs = gc_target @ x_basis.T
        return torch.linalg.solve(system, rhs.T).T

    def basis_reg_loss(x_basis):
        d_norm = d_weight / d_weight.sum().clamp_min(1e-12)
        num = x_basis * d_weight[None, :]
        den = num.sum(dim=0, keepdim=True).clamp_min(1e-12)
        responsibilities = num / den
        overlap = (d_norm * (1.0 - (responsibilities * responsibilities).sum(dim=0))).sum()
        mass_k = (responsibilities * d_norm[None, :]).sum(dim=1).clamp_min(1e-12)
        coverage = (-torch.log(mass_k)).mean()
        return 2.0 * overlap + coverage

    optimizer = torch.optim.Adam(params, lr=float(lr))
    final_info = {}
    for epoch in range(1, int(epochs) + 1):
        optimizer.zero_grad(set_to_none=True)
        x_basis = basis_x()
        w_flat = compute_w_star(x_basis)
        diff = (w_flat @ x_basis) - g_target
        kernel_loss = ((diff @ covariance_t) * diff).sum()
        reg_loss = basis_reg_loss(x_basis)
        total_loss = kernel_loss + float(basis_reg_weight) * reg_loss
        total_loss.backward()
        optimizer.step()

        if verbose_every and (epoch == 1 or epoch % int(verbose_every) == 0):
            mu, sigma = centers_sigmas()
            print(
                f"[basis K={k}] epoch={epoch:5d} total={float(total_loss.detach()):.6g} "
                f"kernel={float(kernel_loss.detach()):.6g} reg={float(reg_loss.detach()):.6g} "
                f"mu={np.array2string(mu.detach().cpu().numpy(), precision=2)}"
            )

        final_info = {
            "loss_total": float(total_loss.detach()),
            "loss_kernel": float(kernel_loss.detach()),
            "loss_basis_reg": float(reg_loss.detach()),
        }

    with torch.no_grad():
        x_basis = basis_x()
        w_flat = compute_w_star(x_basis)
        mu, sigma = centers_sigmas()
        diff = (w_flat @ x_basis) - g_target
        kernel_loss = ((diff @ covariance_t) * diff).sum()
        reg_loss = basis_reg_loss(x_basis)
        final_info = {
            "loss_total": float(kernel_loss + float(basis_reg_weight) * reg_loss),
            "loss_kernel": float(kernel_loss),
            "loss_basis_reg": float(reg_loss),
        }

    return {
        "mu": mu.detach().cpu().numpy(),
        "sigma": sigma.detach().cpu().numpy(),
        "basis": x_basis.detach().cpu().numpy(),
        "decoders_xyz": w_flat.detach().cpu().numpy().reshape(data.illuminants.shape[0], 3, int(k)),
        "epochs": int(epochs),
        **final_info,
    }


def learn_gaussian_bases(data: SpectralData, covariance: np.ndarray, config: FullRecomputeConfig) -> dict[int, dict]:
    """Learn every Gaussian basis size used by the paper figures/tables.

    Args:
        data: Spectral context from LuxPy.
        covariance: Reflectance-path covariance matrix `C_R`.
        config: Full recomputation settings, including epoch schedule overrides.

    Returns:
        Mapping from K in `{3,4,5,6}` to the result of `learn_gaussian_basis`.
    """

    learned = {}
    for k in [3, 4, 5, 6]:
        epochs = _gaussian_basis_epochs(config, k)
        print(f"[full] learning Gaussian basis K={k} ({epochs} epochs)")
        learned[k] = learn_gaussian_basis(
            data,
            covariance,
            k=k,
            epochs=epochs,
            lr=config.gaussian_basis_lr,
            basis_reg_weight=config.gaussian_basis_reg,
            seed=config.seed,
            verbose_every=config.gaussian_basis_verbose_every,
        )
    return learned
