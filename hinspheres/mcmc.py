"""
MCMC posterior sampling using emcee for HINSA model parameters.

Provides Bayesian posterior estimation after CMA-ES optimization,
revealing parameter degeneracies and true uncertainties beyond
the Gaussian approximation of CMA-ES covariance.
"""

import numpy as np


def _integrated_autocorr_time(chain_2d, c=5, quiet=False):
    """Estimate integrated autocorrelation time using initial positive sequence.

    Parameters
    ----------
    chain_2d : ndarray, shape (n_samples, n_dim)
    c : int — truncation parameter
    quiet : bool — suppress warnings

    Returns
    -------
    tau : ndarray, shape (n_dim,) — autocorrelation time per parameter
    """
    n_samples, n_dim = chain_2d.shape
    tau = np.full(n_dim, np.nan)

    for d in range(n_dim):
        x = chain_2d[:, d]
        x = x - np.mean(x)
        n = len(x)

        # Compute autocovariance via FFT
        f = np.fft.fft(x, n=2*n)
        acf = np.real(np.fft.ifft(f * np.conj(f)))[:n]
        acf = acf / acf[0]

        # Initial positive sequence estimator
        tau_d = 1.0
        for i in range(1, n // 2):
            if acf[i] < 0.0:
                break
            tau_d += 2.0 * acf[i] * (1.0 - i / (c * tau_d)) if i < c * tau_d else 0.0
            if i >= c * tau_d:
                break
        tau[d] = max(tau_d, 1.0)

    return tau


def _gelman_rubin(chains_list):
    """Compute Gelman-Rubin R-hat statistic for multiple chains.

    Parameters
    ----------
    chains_list : list of ndarray, each shape (n_samples, n_dim)

    Returns
    -------
    rhat : ndarray, shape (n_dim,)
    """
    m = len(chains_list)
    n = min(c.shape[0] for c in chains_list)
    p = chains_list[0].shape[1]

    # Chain means and variances
    chain_means = np.array([c[:n].mean(axis=0) for c in chains_list])  # (m, p)
    chain_vars = np.array([c[:n].var(axis=0, ddof=1) for c in chains_list])  # (m, p)

    # Between-chain variance
    grand_mean = chain_means.mean(axis=0)  # (p,)
    B = n * chain_vars.mean(axis=0)  # Actually: n / (m-1) * sum((mean_j - grand_mean)^2)
    B = n / (m - 1) * np.sum((chain_means - grand_mean) ** 2, axis=0)

    # Within-chain variance
    W = chain_vars.mean(axis=0)

    # Marginal posterior variance
    var_plus = ((n - 1) / n) * W + (1.0 / n) * B

    rhat = np.sqrt(var_plus / W) if W.min() > 0 else np.full(p, np.nan)
    return rhat


def _check_convergence(sampler, burnin, verbose=True):
    """Check MCMC convergence and print diagnostics.

    Returns
    -------
    converged : bool
    diagnostics : dict
    """
    try:
        chain_raw = sampler.get_chain()  # (nsteps, nwalkers, n_dim)
        nsteps, nwalkers, n_dim = chain_raw.shape

        # Discard burn-in
        post_burn = chain_raw[burnin:]  # (nsteps-burnin, nwalkers, n_dim)
        n_post = post_burn.shape[0]

        # 1. Integrated autocorrelation time (flatten across walkers)
        flat = post_burn.reshape(-1, n_dim)
        tau = _integrated_autocorr_time(flat, quiet=True)
        ess = flat.shape[0] / tau  # effective sample size per parameter

        # 2. Gelman-Rubin R-hat (each walker as a separate chain)
        chains = [post_burn[:, i, :] for i in range(nwalkers)]
        rhat = _gelman_rubin(chains)

        # 3. Heidelberg-Welch-style: check if mean has stabilized
        half = n_post // 2
        first_half_mean = post_burn[:half].mean(axis=(0, 1))
        second_half_mean = post_burn[half:].mean(axis=(0, 1))
        mean_shift = np.abs(second_half_mean - first_half_mean)
        mean_shift_rel = mean_shift / (flat.std(axis=0) + 1e-30)

        converged = bool(np.all(rhat < 1.1) and np.all(ess > 100))

        diagnostics = {
            'tau': tau,
            'ess': ess,
            'rhat': rhat,
            'mean_shift_rel': mean_shift_rel,
            'converged': converged,
        }

        if verbose:
            print(f"\n  === Convergence Diagnostics ===")
            print(f"  R-hat max:       {np.nanmax(rhat):.4f}  {'OK' if np.all(rhat < 1.1) else 'WARN (>1.1)'}")
            print(f"  ESS min:         {np.nanmin(ess):.0f}  {'OK' if np.all(ess > 100) else 'WARN (<100)'}")
            print(f"  tau max:         {np.nanmax(tau):.1f}")
            print(f"  mean shift max:  {np.nanmax(mean_shift_rel):.4f}")
            print(f"  Converged:       {'YES' if converged else 'NO'}")
            if not converged:
                reasons = []
                if np.any(rhat >= 1.1):
                    reasons.append(f"R-hat>1.1 (max={np.nanmax(rhat):.3f})")
                if np.any(ess < 100):
                    reasons.append(f"ESS<100 (min={np.nanmin(ess):.0f})")
                print(f"  Reasons: {', '.join(reasons)}")
                print(f"  Consider increasing nsteps.")

        return converged, diagnostics

    except Exception as e:
        if verbose:
            print(f"  Convergence check failed: {e}")
        return False, {'converged': False, 'error': str(e)}


def _compute_log_likelihood(params, cfg, obs_map, bg_map, center_yx,
                            pixel_scale_pc, weight_map, galactic_b_deg,
                            R_out_pc, velo_kms_arr, velo_mask, mode, n_jobs):
    """Compute log-likelihood given a parameter dict.

    Parameters
    ----------
    params : dict — physical-space parameter dict
    cfg : Config
    obs_map : 3D array
    bg_map : 3D array or None
    center_yx : tuple
    pixel_scale_pc : float
    weight_map : 2D array
    galactic_b_deg : float
    R_out_pc : float
    velo_kms_arr : 1D array
    velo_mask : 1D bool array or None
    mode : str
    n_jobs : int

    Returns
    -------
    chi2 : float
    """
    from .models import build_synthetic_hinsa, residual_map

    try:
        if mode == 'second_derivative':
            from .models import inverse_build_hinsa_cube
            T_bg_reconstructed = inverse_build_hinsa_cube(
                cfg, params, obs_map, center_yx, pixel_scale_pc,
                galactic_b_deg=galactic_b_deg,
                R_out_pc=R_out_pc, n_jobs=n_jobs)
            dv = abs(cfg.v_max_kms - cfg.v_min_kms) / (cfg.n_v_channels - 1)
            d2 = np.zeros_like(T_bg_reconstructed)
            d2[1:-1] = (T_bg_reconstructed[2:] + T_bg_reconstructed[:-2]
                        - 2.0 * T_bg_reconstructed[1:-1]) / (dv**2)
            weight_map_3d = np.broadcast_to(
                weight_map[np.newaxis, :, :], T_bg_reconstructed.shape).copy()
            if velo_mask is not None:
                weight_map_3d[~velo_mask] = 0.0
            r_value = np.sum(d2**2 * weight_map_3d) * dv / np.sum(weight_map_3d)
            chi2 = -0.5 * r_value
        else:
            model = build_synthetic_hinsa(
                cfg, params, bg_map,
                center_yx=center_yx,
                pixel_scale_pc=pixel_scale_pc,
                galactic_b_deg=galactic_b_deg,
                R_out_pc=R_out_pc, n_jobs=n_jobs,
                velo_bg_kms=velo_kms_arr)
            if velo_mask is not None:
                w_3d = np.broadcast_to(
                    weight_map[np.newaxis, :, :], model.shape).copy()
                w_3d[~velo_mask] = 0.0
                res = residual_map(obs_map, model, weights=w_3d)
            else:
                res = residual_map(obs_map, model, weights=weight_map)
            n_eff = np.sum(weight_map) if weight_map is not None else obs_map.size
            chi2 = -0.5 * res**2 * n_eff

        if not np.isfinite(chi2):
            return -np.inf
        return chi2

    except Exception:
        return -np.inf


def run_mcmc(best_params, cfg, param_keys, phys_lows, phys_highs,
             obs_map, bg_map, center_yx, pixel_scale_pc,
             weight_map, galactic_b_deg, R_out_pc,
             velo_kms_arr, velo_mask, mode='forward',
             n_jobs=1, nwalkers=32, nsteps=2000, burnin=500,
             seed=None, verbose=True, progress=True):
    """Run emcee MCMC sampling starting from CMA-ES best-fit.

    Parameters
    ----------
    best_params : dict
        Best-fit parameters from CMA-ES.
    cfg : Config
    param_keys : list
        Flat parameter key list (same order as CMA-ES).
    phys_lows, phys_highs : 1D arrays
        Physical bounds for each parameter.
    obs_map : 3D array
    bg_map : 3D array or None
    center_yx : tuple
    pixel_scale_pc : float
    weight_map : 2D array
    galactic_b_deg : float
    R_out_pc : float
    velo_kms_arr : 1D array
    velo_mask : 1D bool array or None
    mode : str
    n_jobs : int
    nwalkers : int
    nsteps : int
    burnin : int
    seed : int or None
    verbose : bool
    progress : bool

    Returns
    -------
    result : dict
    """
    try:
        import emcee
    except ImportError:
        raise ImportError("MCMC requires emcee: pip install emcee")

    from .fitters import _params_to_flat, _flat_to_params

    _, x0_full, _, _ = _params_to_flat(best_params, cfg.bounds_pc, cfg)

    phys_ranges = phys_highs - phys_lows
    free_idx = [i for i in range(len(param_keys)) if phys_ranges[i] > 0]
    n_dim = len(free_idx)

    if nwalkers < 2 * n_dim:
        nwalkers = max(32, 2 * n_dim + 2)
        if verbose:
            print(f"  nwalkers adjusted to {nwalkers} (>= 2 x {n_dim})")

    x0_free = x0_full[free_idx]

    if seed is not None:
        np.random.seed(seed)
    p0 = x0_free + 1e-3 * np.random.randn(nwalkers, n_dim)
    p0 = np.clip(p0, 0.0, 1.0)

    def _log_prob_free(x_free):
        x_full = np.copy(x0_full)
        for fi, fv in zip(free_idx, x_free):
            x_full[fi] = fv
        params = _flat_to_params(x_full, param_keys, cfg,
                                 phys_lows=phys_lows, phys_highs=phys_highs)
        lp = log_prior(x_full, phys_lows, phys_highs, free_idx)
        if not np.isfinite(lp):
            return -np.inf
        ll = _compute_log_likelihood(
            params, cfg, obs_map, bg_map, center_yx, pixel_scale_pc,
            weight_map, galactic_b_deg, R_out_pc, velo_kms_arr,
            velo_mask, mode, n_jobs)
        return lp + ll

    # Set up parallel pool for walker-level parallelism
    # Use ThreadPool (not ProcessPool) because _log_prob_free is a closure
    # and can't be pickled for inter-process communication.
    # Inner joblib parallelism in build_synthetic_hinsa handles CPU-level parallelism.
    pool = None
    if n_jobs != 1:
        try:
            from multiprocessing.pool import ThreadPool as _ThreadPool
            _n_workers = None if n_jobs == -1 else n_jobs
            pool = _ThreadPool(processes=_n_workers)
            if verbose:
                _actual = _n_workers or 'all CPUs'
                print(f"  MCMC pool: {_actual} threads for walker parallelism")
        except Exception:
            pool = None
            if verbose:
                print("  MCMC pool: fallback to serial (thread pool unavailable)")

    try:
        sampler = emcee.EnsembleSampler(nwalkers, n_dim, _log_prob_free, pool=pool)
    except Exception:
        sampler = emcee.EnsembleSampler(nwalkers, n_dim, _log_prob_free)

    if verbose:
        print(f"  Running emcee: {nwalkers} walkers x {nsteps} steps "
              f"({n_dim} free parameters)")

    sampler.run_mcmc(p0, nsteps, progress=progress)

    # --- Convergence check and auto-extend ---
    max_rounds = 5
    for rnd in range(max_rounds):
        conv, diag = _check_convergence(sampler, burnin, verbose=verbose)
        if conv:
            break
        if rnd < max_rounds - 1:
            extra = nsteps
            if verbose:
                print(f"\n  Not converged — extending MCMC by {extra} steps (round {rnd+2}/{max_rounds})")
            try:
                sampler.run_mcmc(None, extra, progress=progress)
            except Exception:
                last_pos = sampler.get_last_sample()
                sampler.run_mcmc(last_pos, extra, progress=progress)
            nsteps = sampler.get_chain().shape[0]
            burnin = min(burnin, nsteps // 4)

    # Clean up pool after all sampling is done
    if pool is not None:
        try:
            pool.close()
            pool.join()
        except Exception:
            pass

    chain = sampler.get_chain(discard=burnin, flat=True)
    lnprob = sampler.get_log_prob(discard=burnin, flat=True)

    chain_phys = chain * (phys_highs[free_idx] - phys_lows[free_idx]) + phys_lows[free_idx]

    median_params = {}
    std_params = {}
    ci_95 = {}

    for k in param_keys:
        median_params[k] = best_params.get(k, 0.0)
        std_params[k] = 0.0
        ci_95[k] = (best_params.get(k, 0.0), best_params.get(k, 0.0))

    for i, idx in enumerate(free_idx):
        pname = param_keys[idx]
        samples = chain_phys[:, i]
        median_params[pname] = float(np.median(samples))
        std_params[pname] = float(np.std(samples))
        ci_95[pname] = (float(np.percentile(samples, 2.5)),
                        float(np.percentile(samples, 97.5)))

    for arr_name in ('f_HI', 'multipliers'):
        n = cfg.n_shells if arr_name == 'f_HI' else max(1, cfg.n_shells - 1)
        flat_prefix = 'f_HI' if arr_name == 'f_HI' else 'mult'
        arr_median = np.zeros(n)
        arr_std = np.zeros(n)
        arr_ci = np.zeros((n, 2))
        for j in range(n):
            key = f'{flat_prefix}_{j}'
            if key in [param_keys[fi] for fi in free_idx]:
                i_free = [i for i, fi in enumerate(free_idx)
                          if param_keys[fi] == key][0]
                samples = chain_phys[:, i_free]
                arr_median[j] = np.median(samples)
                arr_std[j] = np.std(samples)
                arr_ci[j] = [np.percentile(samples, 2.5),
                             np.percentile(samples, 97.5)]
            else:
                arr_median[j] = best_params.get(arr_name, np.zeros(n))[j] if arr_name in best_params else 0.0
        median_params[arr_name] = arr_median
        std_params[arr_name] = arr_std
        ci_95[arr_name] = arr_ci

    if verbose:
        print(f"  MCMC complete: {chain.shape[0]} effective samples")
        print(f"  Best log-probability: {lnprob.max():.4f}")

    return {
        'sampler': sampler,
        'chain': chain,
        'chain_phys': chain_phys,
        'lnprob': lnprob,
        'median_params': median_params,
        'std_params': std_params,
        'ci_95': ci_95,
        'param_keys': param_keys,
        'free_idx': free_idx,
        'phys_lows': phys_lows,
        'phys_highs': phys_highs,
        'n_dim': n_dim,
        'nwalkers': nwalkers,
        'nsteps': nsteps,
        'burnin': burnin,
    }


def log_prior(x01, phys_lows, phys_highs, free_idx):
    """Uniform prior within physical bounds (operates on full [0,1] vector)."""
    for fi in free_idx:
        if x01[fi] < 0.0 or x01[fi] > 1.0:
            return -np.inf
    return 0.0


def save_corner_plot(mcmc_result, save_path, param_names=None,
                     truths=None, figsize=None, dpi=150):
    """Generate and save corner plot from MCMC chain."""
    try:
        import corner
    except ImportError:
        raise ImportError("corner plot requires corner: pip install corner")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    chain = mcmc_result['chain_phys']
    free_idx = mcmc_result['free_idx']
    param_keys = mcmc_result['param_keys']

    if param_names is None:
        param_names = [param_keys[i] for i in free_idx]

    if truths is not None:
        truth_values = []
        for i in free_idx:
            pk = param_keys[i]
            if pk in truths:
                truth_values.append(truths[pk])
            elif pk.startswith('f_HI_'):
                idx = int(pk.split('_')[-1])
                if 'f_HI' in truths:
                    truth_values.append(np.asarray(truths['f_HI']).flat[idx])
                else:
                    truth_values = None
                    break
            elif pk.startswith('mult_'):
                idx = int(pk.split('_')[-1])
                if 'multipliers' in truths:
                    truth_values.append(truths['multipliers'][idx])
                else:
                    truth_values = None
                    break
            else:
                truth_values = None
                break
    else:
        truth_values = None

    if figsize is None:
        n = len(free_idx)
        figsize = (2.5 * n, 2.5 * n)

    fig = corner.corner(
        chain,
        labels=param_names,
        truths=truth_values,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 10},
        label_kwargs={"fontsize": 11},
        figsize=figsize,
    )

    fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)

    return save_path


def save_chain_plot(mcmc_result, save_path, param_names=None,
                    figsize=(12, 8), dpi=150):
    """Generate and save MCMC chain trace plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sampler = mcmc_result['sampler']
    free_idx = mcmc_result['free_idx']
    param_keys = mcmc_result['param_keys']

    if param_names is None:
        param_names = [param_keys[i] for i in free_idx]

    n_dim = mcmc_result['n_dim']
    chain = sampler.get_chain()

    fig, axes = plt.subplots(n_dim, 1, figsize=figsize, sharex=True)
    if n_dim == 1:
        axes = [axes]

    for i in range(n_dim):
        ax = axes[i]
        for j in range(chain.shape[1]):
            ax.plot(chain[:, j, i], alpha=0.3, linewidth=0.5)
        ax.set_ylabel(param_names[i], fontsize=10)
        ax.axvline(mcmc_result['burnin'], color='red', ls='--',
                   alpha=0.5, label='burn-in' if i == 0 else None)
        if i == 0:
            ax.legend(fontsize=8)

    axes[-1].set_xlabel('Step', fontsize=11)
    fig.suptitle('MCMC Chain Trace', fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)

    return save_path
