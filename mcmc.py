"""
MCMC posterior sampling using emcee for HINSA model parameters.

Provides Bayesian posterior estimation after CMA-ES optimization,
revealing parameter degeneracies and true uncertainties beyond
the Gaussian approximation of CMA-ES covariance.
"""

import os
# Must be set BEFORE numba is imported (by any transitive import).
# Parent process uses 1 numba thread; multiprocessing workers override
# via _worker_init + numba.set_num_threads(1).
os.environ.setdefault('NUMBA_NUM_THREADS', '1')

import multiprocessing
import numpy as np

from .config import get_cfg_value


def _worker_init():
    """Per-worker init: pin each subprocess to 1 numba thread to avoid
    oversubscription when using a multiprocessing pool."""
    os.environ['NUMBA_NUM_THREADS'] = '1'
    try:
        import numba
        numba.set_num_threads(1)
    except Exception:
        pass


# Counter for _LogProbFree failures (first 5 are logged with details).
_logprob_fail_count = multiprocessing.Array('i', [0])

# Module-level shared arrays (set by run_mcmc, accessed by _LogProbFree).
# With fork-based multiprocessing, child processes inherit these via
# copy-on-write — no pickling of large arrays.
_shared_obs_map = None
_shared_bg_map = None
_shared_w_3d_pre = None
_shared_x0_full = None


def _gelman_rubin(chains_list):
    """Split-R-hat (Vehtari et al. 2021), treating each walker as a chain.

    Each chain is split in half and the second half reversed, giving 2M
    sub-chains. This is more sensitive to non-stationarity in the tails
    than the plain within/between-chain R-hat and is robust to the
    correlation between emcee walkers.

    Parameters
    ----------
    chains_list : list of ndarray, each shape (n_samples, n_dim)

    Returns
    -------
    rhat : ndarray, shape (n_dim,)
    """
    m = len(chains_list)
    p = chains_list[0].shape[1]

    n_half = min(c.shape[0] for c in chains_list) // 2
    if n_half < 2:
        return np.full(p, np.nan)

    sub_chains = []
    for c in chains_list:
        c = c[: 2 * n_half]
        sub_chains.append(c[:n_half])
        sub_chains.append(c[n_half:][::-1])

    sub_means = np.array([sc.mean(axis=0) for sc in sub_chains])  # (2M, p)
    sub_vars = np.array([sc.var(axis=0, ddof=1) for sc in sub_chains])  # (2M, p)

    grand = sub_means.mean(axis=0)
    B = n_half / (len(sub_chains) - 1) * np.sum((sub_means - grand) ** 2, axis=0)
    W = sub_vars.mean(axis=0)

    var_plus = ((n_half - 1) / n_half) * W + B / n_half
    with np.errstate(divide='ignore', invalid='ignore'):
        rhat = np.sqrt(var_plus / np.maximum(W, 1e-300))
    rhat[~np.isfinite(rhat)] = np.nan
    return rhat


def _check_convergence(sampler, burnin, verbose=True, rhat_thresh=1.05,
                       ess_thresh=100):
    """Check MCMC convergence and print diagnostics.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
    burnin : int
    verbose : bool
    rhat_thresh : float
        R-hat threshold below which a parameter counts as converged
        (default 1.05, the emcee-recommended value for slow-mixing
        high-dimensional problems).
    ess_thresh : float
        Minimum effective sample size per parameter.

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

        # 1. Integrated autocorrelation time.
        #    Estimate per-walker and average: concatenating walkers into one
        #    long chain biases the autocorrelation estimate (this was the
        #    source of the bogus/negative tau values seen earlier).
        import emcee
        taus = np.zeros((nwalkers, n_dim))
        for w in range(nwalkers):
            try:
                taus[w] = np.atleast_1d(emcee.autocorr.integrated_time(
                    post_burn[:, w, :], tol=0, quiet=True))
            except Exception:
                taus[w] = np.nan
        with np.errstate(invalid='ignore'):
            tau = np.nanmedian(taus, axis=0)
        tau = np.atleast_1d(tau)
        # n_eff = total samples / tau (per-walker tau, so total = n_post*nwalkers)
        ess = n_post * nwalkers / tau  # effective sample size per parameter

        # 2. Split-Gelman-Rubin R-hat (each walker as a separate chain)
        chains = [post_burn[:, i, :] for i in range(nwalkers)]
        rhat = _gelman_rubin(chains)

        # 3. Heidelberg-Welch-style: check if mean has stabilized
        half = n_post // 2
        first_half_mean = post_burn[:half].mean(axis=(0, 1))
        second_half_mean = post_burn[half:].mean(axis=(0, 1))
        mean_shift = np.abs(second_half_mean - first_half_mean)
        mean_shift_rel = mean_shift / (np.nanstd(post_burn, axis=(0, 1)) + 1e-30)

        # 4. Detect parameters pinned at a prior boundary (truncated posterior).
        #    A param whose marginal piles up on a bound has ~zero within-chain
        #    variance there, so split-R-hat is ill-defined/inflated and the
        #    gate would never pass no matter how long we run. We detect a
        #    *genuine* truncation (every walker sits at the same bound for
        #    >90% of samples) and exclude such params from the R-hat/ESS gate,
        #    but report them explicitly so callers quote them as limits rather
        #    than Gaussian intervals. If some walkers explore off the bound,
        #    the param is NOT pegged and still gates on R-hat.
        _eps = 1e-4
        pin_low = post_burn < _eps
        pin_high = post_burn > (1.0 - _eps)
        frac_low = pin_low.mean(axis=(0, 1))
        frac_high = pin_high.mean(axis=(0, 1))
        walker_low = pin_low.mean(axis=0)  # (nwalkers, n_dim)
        walker_high = pin_high.mean(axis=0)
        pegged_low = (frac_low > 0.99) & (walker_low.min(axis=0) > 0.9)
        pegged_high = (frac_high > 0.99) & (walker_high.min(axis=0) > 0.9)
        pegged = pegged_low | pegged_high
        interior = ~pegged

        if interior.any():
            rhat_ok = bool(np.all(rhat[interior] < rhat_thresh))
            ess_ok = bool(np.all(ess[interior] > ess_thresh))
        else:
            rhat_ok = True
            ess_ok = True
        converged = bool(rhat_ok and ess_ok)

        diagnostics = {
            'tau': tau,
            'ess': ess,
            'rhat': rhat,
            'mean_shift_rel': mean_shift_rel,
            'rhat_thresh': rhat_thresh,
            'ess_thresh': ess_thresh,
            'converged': converged,
            'pegged_low': pegged_low,
            'pegged_high': pegged_high,
        }

        if verbose:
            rhat_ok = np.all(rhat[interior] < rhat_thresh)
            ess_ok = np.all(ess[interior] > ess_thresh)
            print(f"\n  === Convergence Diagnostics ===")
            print(f"  R-hat max:       {np.nanmax(rhat):.4f}  "
                  f"{'OK' if rhat_ok else f'WARN (>={rhat_thresh})'}")
            print(f"  ESS min:         {np.nanmin(ess):.0f}  "
                  f"{'OK' if ess_ok else f'WARN (<={ess_thresh})'}")
            print(f"  tau max:         {np.nanmax(tau):.1f}")
            print(f"  mean shift max:  {np.nanmax(mean_shift_rel):.4f}")
            n_peg = int(np.sum(pegged))
            if n_peg:
                lo_idx = np.nonzero(pegged_low)[0].tolist()
                hi_idx = np.nonzero(pegged_high)[0].tolist()
                print(f"  pegged at bounds: {n_peg} param(s) "
                      f"(excluded from gate; "
                      f"lo:{len(lo_idx)} hi:{len(hi_idx)})")
            print(f"  Converged:       {'YES' if converged else 'NO'}")
            if not converged:
                reasons = []
                if np.any(~np.isfinite(rhat)):
                    reasons.append("R-hat NaN (chains too short?)")
                elif not rhat_ok:
                    reasons.append(f"R-hat>={rhat_thresh} (max={np.nanmax(rhat):.3f})")
                if not ess_ok:
                    reasons.append(f"ESS<{ess_thresh} (min={np.nanmin(ess):.0f})")
                print(f"  Reasons: {', '.join(reasons)}")
                print(f"  Consider increasing nsteps.")

        return converged, diagnostics

    except Exception as e:
        if verbose:
            print(f"  Convergence check failed: {e}")
        return False, {'converged': False, 'error': str(e)}


def _compute_log_likelihood(params, cfg, obs_map, bg_map, center_yx,
                            pixel_scale_pc, weight_map, galactic_b_deg,
                            R_out_pc, velo_kms_arr, velo_mask, mode, n_jobs,
                            noise_sigma=None, spatial_res_arcmin=None,
                            vel_res_kms=None):
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
    noise_sigma : float
        RMS noise per pixel (K). The likelihood is calibrated so that
        chi2 = -0.5 * sum((obs-model)^2 * w / sigma^2), i.e. the radial
        weights w only set the relative weighting while the absolute
        posterior width is set by the real survey noise sigma
        (CRAFTS 0.17 K, Zhang+2019).
    spatial_res_arcmin : float or None
        If set, the forward model is smoothed to this beam before
        comparison, exactly as in the CMA-ES objective (the synthetic
        observations were generated with this smoothing, so the model
        MUST be smoothed the same way or the likelihood is biased
        against the true solution).
    vel_res_kms : float or None
        Velocity resolution (FWHM, km/s) for smoothing along the
        velocity axis, same convention as in the CMA-ES objective.

    Returns
    -------
    chi2 : float
    """
    from .models import build_synthetic_hinsa, residual_map

    if noise_sigma is None:
        noise_sigma = get_cfg_value(cfg, 'noise_sigma_K')

    try:
        if mode == 'second_derivative':
            from .models import inverse_build_hinsa_cube
            T_bg_reconstructed = inverse_build_hinsa_cube(
                cfg, params, obs_map, center_yx, pixel_scale_pc,
                galactic_b_deg=galactic_b_deg,
                R_out_pc=R_out_pc, n_jobs=n_jobs,
                velo_kms=velo_kms_arr)
            dv = abs(velo_kms_arr[1] - velo_kms_arr[0]) if len(velo_kms_arr) > 1 else 1.0
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
                velo_bg_kms=velo_kms_arr,
                velo_kms=velo_kms_arr)
            # Apply beam/velocity smoothing to the forward model, exactly as
            # in the CMA-ES objective (obs was generated with this smoothing).
            if (spatial_res_arcmin is not None or vel_res_kms is not None) \
                    and model.ndim == 3:
                from scipy.ndimage import gaussian_filter
                _sf = 2.0 * np.sqrt(2.0 * np.log(2.0))
                _s_v = 0.0
                _s_xy = 0.0
                if vel_res_kms is not None:
                    dv = abs(velo_kms_arr[1] - velo_kms_arr[0]) \
                        if len(velo_kms_arr) > 1 else 1.0
                    _s_v = (vel_res_kms / dv) / _sf
                if spatial_res_arcmin is not None:
                    pix_arcmin = pixel_scale_pc / get_cfg_value(cfg, 'distance_pc') \
                        * (180.0 / np.pi) * 60.0
                    _s_xy = (spatial_res_arcmin / pix_arcmin) / _sf
                if _s_v > 0 or _s_xy > 0:
                    model = gaussian_filter(
                        model, sigma=(_s_v, _s_xy, _s_xy), mode='reflect')
            if velo_mask is not None:
                w_3d = np.broadcast_to(
                    weight_map[np.newaxis, :, :], model.shape).copy()
                w_3d[~velo_mask] = 0.0
            else:
                w_3d = np.broadcast_to(
                    weight_map[np.newaxis, :, :], model.shape).copy()
            res = residual_map(obs_map, model, weights=w_3d)
            # residual_map returns sum((obs-model)^2 * w) / sum(w) over the
            # (possibly velocity-masked) 3D mask, so the weighted sum is
            # res * sum(w_3d). n_eff MUST be the 3D masked weight sum, not
            # the 2D spatial sum (which would drop the channel factor and
            # wrongly narrow/widen the posterior).
            n_eff = float(np.sum(w_3d))
            # Calibrate against the real per-channel noise sigma (K):
            # chi2 = -0.5 * sum((obs-model)^2 * w / sigma^2)
            chi2 = -0.5 * res * n_eff / noise_sigma**2

        if not np.isfinite(chi2):
            return -np.inf
        return chi2

    except Exception:
        return -np.inf


class _LogProbFree:
    """Picklable callable for emcee likelihood evaluation.

    Large arrays (obs_map, bg_map, w_3d_pre, x0_full) are stored as
    module-level globals (_shared_*) and accessed by reference. With
    fork-based multiprocessing, child processes inherit them via
    copy-on-write — no pickling overhead for large data.
    """

    def __init__(self, free_idx, param_keys, cfg,
                 phys_lows, phys_highs, log_prior,
                 noise_sigma, mode, velo_kms_arr,
                 galactic_b_deg, R_out_pc, center_yx,
                 pixel_scale_pc, n_jobs, _beam_sigma):
        # Only store small/immutable config — no large numpy arrays
        self.free_idx = free_idx
        self.param_keys = param_keys
        self.cfg = cfg
        self.phys_lows = phys_lows
        self.phys_highs = phys_highs
        self.log_prior = log_prior
        self.noise_sigma = noise_sigma
        self.mode = mode
        self.velo_kms_arr = velo_kms_arr
        self.galactic_b_deg = galactic_b_deg
        self.R_out_pc = R_out_pc
        self.center_yx = center_yx
        self.pixel_scale_pc = pixel_scale_pc
        self.n_jobs = n_jobs
        self._beam_sigma = _beam_sigma

    def __call__(self, x_free):
        from .models import (build_synthetic_hinsa, _njit_chi2_3d,
                             compute_foreground_params)
        from .fitters import _flat_to_params
        from scipy.ndimage import gaussian_filter as _gf

        # Access shared arrays from module globals (fork-inherited, no pickle)
        x0_full = _shared_x0_full
        obs_map = _shared_obs_map
        bg_map = _shared_bg_map
        w_3d_pre = _shared_w_3d_pre

        x_full = np.copy(x0_full)
        for fi, fv in zip(self.free_idx, x_free):
            x_full[fi] = fv
        params = _flat_to_params(x_full, self.param_keys, self.cfg,
                                 phys_lows=self.phys_lows,
                                 phys_highs=self.phys_highs)
        lp = self.log_prior(x_full, self.phys_lows, self.phys_highs,
                            self.free_idx)
        if not np.isfinite(lp):
            return -np.inf
        # Physical constraint: all per-shell f_HI must be <= 1
        fHI_arr = params.get('f_HI', None)
        if fHI_arr is None:
            # Parametric mode: compute from peak_shell + multipliers
            from .profiles import abundance_profile_111n
            ps = params.get('peak_shell', self.cfg.default_params['peak_shell'])
            fhp = params.get('f_HI_peak', self.cfg.default_params['f_HI_peak'])
            mlts = params.get('multipliers', None)
            n_shells = self.cfg.n_shells if mlts is None \
                else len(mlts) + 1
            if mlts is None:
                mlts = (np.ones(max(1, n_shells - 1))
                        * self.cfg.default_params['multipliers_value'])
            fHI_arr = abundance_profile_111n(n_shells, ps, mlts,
                                              f_HI_peak=fhp)
        if np.any(np.asarray(fHI_arr) > 1.0):
            return -np.inf
        try:
            if self.mode == 'second_derivative':
                from .models import inverse_build_hinsa_cube
                T_bg_reconstructed = inverse_build_hinsa_cube(
                    self.cfg, params, obs_map, self.center_yx,
                    self.pixel_scale_pc,
                    galactic_b_deg=self.galactic_b_deg,
                    R_out_pc=self.R_out_pc, n_jobs=self.n_jobs,
                    velo_kms=self.velo_kms_arr)
                dv = abs(self.velo_kms_arr[1] - self.velo_kms_arr[0]) \
                    if len(self.velo_kms_arr) > 1 else 1.0
                d2 = np.zeros_like(T_bg_reconstructed)
                d2[1:-1] = (T_bg_reconstructed[2:] + T_bg_reconstructed[:-2]
                            - 2.0 * T_bg_reconstructed[1:-1]) / (dv**2)
                n_eff = float(np.sum(w_3d_pre))
                r_value = np.sum(d2**2 * w_3d_pre) * dv / n_eff
                chi2 = -0.5 * r_value
            else:
                model = build_synthetic_hinsa(
                    self.cfg, params, bg_map,
                    center_yx=self.center_yx,
                    pixel_scale_pc=self.pixel_scale_pc,
                    galactic_b_deg=self.galactic_b_deg,
                    R_out_pc=self.R_out_pc, n_jobs=self.n_jobs,
                    velo_bg_kms=self.velo_kms_arr,
                    velo_kms=self.velo_kms_arr)
                if self._beam_sigma is not None and model.ndim == 3:
                    model = _gf(model, sigma=self._beam_sigma, mode='reflect')
                chi2 = _njit_chi2_3d(obs_map, model,
                                     w_3d_pre, self.noise_sigma)
            if not np.isfinite(chi2):
                import traceback as _tb
                _logprob_fail_count[0] += 1
                if _logprob_fail_count[0] <= 5:
                    print(f"  [LOGPROB WARN] chi2 non-finite: {chi2}, "
                          f"model range=[{np.nanmin(model):.4f}, {np.nanmax(model):.4f}], "
                          f"model NaN count={np.isnan(model).sum()}, "
                          f"params={ {{k: f'{v:.6g}' for k, v in params.items()}} }",
                          flush=True)
                return -np.inf
            return chi2
        except Exception as exc:
            _logprob_fail_count[0] += 1
            if _logprob_fail_count[0] <= 5:
                import traceback as _tb
                def _fmt(v):
                    if isinstance(v, np.ndarray):
                        return repr(v)
                    return f'{v:.6g}'
                print(f"  [LOGPROB ERROR] {type(exc).__name__}: {exc}\n"
                      f"    params={{{', '.join(f'{k}={_fmt(v)}' for k, v in params.items())}}}\n"
                      f"    {_tb.format_exc()}", flush=True)
            return -np.inf


def run_mcmc(best_params, cfg, param_keys, phys_lows, phys_highs,
             obs_map, bg_map, center_yx, pixel_scale_pc,
             weight_map, galactic_b_deg, R_out_pc,
             velo_kms_arr, velo_mask, mode='forward',
             n_jobs=1, nwalkers=32, nsteps=2000, burnin=500,
             seed=None, verbose=True, progress=True, noise_sigma=None,
             init_std=None, spatial_res_arcmin=None, vel_res_kms=None,
             n_pool=0):
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
    noise_sigma : float or None
        RMS noise per pixel (K) for likelihood calibration. If None,
        uses ``Config.noise_sigma_K`` (default CRAFTS 0.17 K; Zhang+2019).
    init_std : dict or None
        Physical-space spread (std) per flat parameter key, used to
        scatter the walkers around the CMA-ES best-fit. CMA-ES std is
        inflated by 10x (it underestimates posterior width) and floored
        at 1% of the physical range. Missing keys use the 1% floor
        (normalized space).
    spatial_res_arcmin : float or None
        Smooth the forward model to this beam before computing the
        residual, exactly as in the CMA-ES objective. CRITICAL: the
        synthetic observations were generated with this smoothing, so
        without it the likelihood is biased and the posterior will not
        converge near the CMA-ES solution.
    vel_res_kms : float or None
        Velocity resolution (FWHM, km/s) for velocity-axis smoothing.
    n_pool : int
        Number of worker processes for walker-level parallelism.
        Each worker runs with NUMBA_NUM_THREADS=1 to avoid oversubscription.
        0 (default) = serial walkers + numba-internal parallelism.

    Returns
    -------
    result : dict
    """
    try:
        import emcee
    except ImportError:
        raise ImportError("MCMC requires emcee: pip install emcee")

    from .fitters import _params_to_flat, _flat_to_params

    if noise_sigma is None:
        noise_sigma = get_cfg_value(cfg, 'noise_sigma_K')

    # Build x0 in the SAME normalized [0,1] space as the likelihood
    # (phys_lows/phys_highs as passed by the caller).  Using cfg.bounds_pc
    # here is WRONG when the caller bounds differ from the config defaults:
    # walkers would start at the same physical optimum but at different
    # [0,1] coordinates than the likelihood actually evaluates, i.e. offset
    # from the posterior peak → R-hat never converges.
    phys_ranges = phys_highs - phys_lows
    free_idx = [i for i in range(len(param_keys)) if phys_ranges[i] > 0]
    n_dim = len(free_idx)
    x0_full = np.zeros(len(param_keys))
    for i, key in enumerate(param_keys):
        if phys_ranges[i] <= 0 or key.startswith('_fixed_'):
            x0_full[i] = 0.0
            continue
        if key.startswith('f_HI_') and key != 'f_HI_peak':
            val = float(np.asarray(best_params['f_HI'])[int(key.split('_')[-1])])
            # Same log10 box as _params_to_flat / _flat_to_params
            if phys_lows[i] > 0:
                lo10, hi10 = np.log10(phys_lows[i]), np.log10(phys_highs[i])
                x0_full[i] = (np.log10(val) - lo10) / (hi10 - lo10)
                continue
        elif key == 'peak_shell':
            val = float(best_params.get('peak_shell', cfg.default_params['peak_shell']))
        elif key == 'f_HI_peak':
            val = float(best_params.get('f_HI_peak', cfg.default_params['f_HI_peak']))
            if phys_lows[i] > 0:
                lo10, hi10 = np.log10(phys_lows[i]), np.log10(phys_highs[i])
                x0_full[i] = (np.log10(val) - lo10) / (hi10 - lo10)
                continue
        elif key.startswith('mult_'):
            idx = int(key.split('_')[-1])
            val = float(best_params.get('multipliers', [0.0])[idx]) \
                if idx < len(best_params.get('multipliers', [])) \
                else cfg.default_params['multipliers_value']
        else:
            val = float(best_params[key])
        x0_full[i] = (val - phys_lows[i]) / phys_ranges[i]
    x0_full = np.clip(x0_full, 0.0, 1.0)

    if nwalkers < 2 * n_dim:
        nwalkers = max(32, 2 * n_dim + 2)
        if verbose:
            print(f"  nwalkers adjusted to {nwalkers} (>= 2 x {n_dim})")

    x0_free = x0_full[free_idx]

    if seed is not None:
        np.random.seed(seed)

    # Walker scatter: initialize in a narrow Gaussian around the CMA-ES
    # optimum (x0_free) with σ=0.03 in normalized [0,1] space. This
    # ensures walkers start near the correct posterior mode instead of
    # drifting to wrong modes in the prior volume.
    p0 = np.random.normal(loc=x0_free, scale=0.03, size=(nwalkers, n_dim))
    p0 = np.clip(p0, 0.0, 1.0)

    # --- Pre-compute constant arrays for the likelihood kernel ---
    # These depend only on cfg/bg_map/weight_map (fixed during MCMC).
    # Large arrays are stored as module-level globals (_shared_*) so
    # fork-based multiprocessing inherits them via copy-on-write
    # without pickling overhead.
    from .models import (build_synthetic_hinsa, _njit_chi2_3d,
                         compute_foreground_params)

    # Pre-compute 3D weight map (broadcast + velo_mask applied once)
    _obs_shape = obs_map.shape
    if velo_mask is not None:
        w_3d_pre = np.broadcast_to(
            weight_map[np.newaxis, :, :], _obs_shape).copy()
        w_3d_pre[~velo_mask] = 0.0
    else:
        w_3d_pre = np.broadcast_to(
            weight_map[np.newaxis, :, :], _obs_shape).copy()

    # Store large arrays as module globals for fork-shared access
    global _shared_obs_map, _shared_bg_map, _shared_w_3d_pre, _shared_x0_full
    _shared_obs_map = obs_map
    _shared_bg_map = bg_map
    _shared_w_3d_pre = w_3d_pre
    _shared_x0_full = x0_full

    # Pre-compute beam smoothing sigma (constant during MCMC)
    _beam_sigma = None
    if spatial_res_arcmin is not None or vel_res_kms is not None:
        from scipy.ndimage import gaussian_filter as _gf
        _sf = 2.0 * np.sqrt(2.0 * np.log(2.0))
        _s_v = 0.0
        _s_xy = 0.0
        if vel_res_kms is not None:
            _dv = abs(velo_kms_arr[1] - velo_kms_arr[0]) \
                if len(velo_kms_arr) > 1 else 1.0
            _s_v = (vel_res_kms / _dv) / _sf
        if spatial_res_arcmin is not None:
            _pix_arcmin = pixel_scale_pc / get_cfg_value(cfg, 'distance_pc') \
                * (180.0 / np.pi) * 60.0
            _s_xy = (spatial_res_arcmin / _pix_arcmin) / _sf
        _beam_sigma = (_s_v, _s_xy, _s_xy)

    log_prob_fn = _LogProbFree(
        free_idx=free_idx, param_keys=param_keys,
        cfg=cfg, phys_lows=phys_lows, phys_highs=phys_highs,
        log_prior=log_prior,
        noise_sigma=noise_sigma, mode=mode, velo_kms_arr=velo_kms_arr,
        galactic_b_deg=galactic_b_deg, R_out_pc=R_out_pc,
        center_yx=center_yx, pixel_scale_pc=pixel_scale_pc,
        n_jobs=n_jobs, _beam_sigma=_beam_sigma)

    # --- Pool-based walker parallelism ---
    # When n_pool > 0, walkers are evaluated in parallel across worker
    # processes (each pinned to 1 numba thread via _worker_init).
    # _LogProbFree is a picklable callable (not a closure) so it works
    # with multiprocessing. When n_pool=0, emcee runs walkers serially
    # and numba prange provides internal parallelism.
    pool = None
    if n_pool > 0:
        pool = multiprocessing.Pool(processes=n_pool, initializer=_worker_init)
        if verbose:
            print(f"  Using multiprocessing pool: {n_pool} workers, "
                  f"NUMBA_NUM_THREADS=1 each")

    sampler = emcee.EnsembleSampler(nwalkers, n_dim, log_prob_fn,
                                     pool=pool)

    if verbose:
        print(f"  Running emcee: {nwalkers} walkers x {nsteps} steps "
              f"({n_dim} free parameters)")

    sampler.run_mcmc(p0, nsteps, progress=progress)

    # --- Convergence check and auto-extend ---
    # Adaptive burn-in: raise the discarded portion based on the estimated
    # autocorrelation time (burn-in ~ 10 x tau is the emcee rule of thumb)
    # instead of trusting a fixed small burnin.
    # max_rounds limits how far the chain may extend. Each round doubles
    # the total length (extra = nsteps), so with 8 rounds it may reach
    # ~64000 steps. Eval cost is ~1ms per likelihood, so extending is cheap;
    # giving up too early is what leaves R-hat pinned just above 1.05.
    max_rounds = 8
    max_total_steps = 50000  # hard cap to prevent exponential blowup
    converged = False
    diag_final = {}
    for rnd in range(max_rounds):
        conv, diag = _check_convergence(sampler, burnin, verbose=verbose)

        # Adaptive burn-in: recompute with tau-based burnin if it is larger.
        tau_max = float(np.nanmax(np.atleast_1d(diag.get('tau', 0.0)))) \
            if np.any(np.isfinite(diag.get('tau', np.array([0.0])))) else 0.0
        burnin_adaptive = int(np.ceil(10.0 * tau_max)) if np.isfinite(tau_max) else burnin
        if burnin_adaptive > burnin:
            burnin = min(burnin_adaptive, nsteps // 2)
            if verbose:
                print(f"  Adaptive burn-in raised to {burnin} "
                      f"(10 x tau={tau_max:.1f})")
            conv, diag = _check_convergence(sampler, burnin, verbose=verbose)

        if conv:
            converged = True
            diag_final = diag
            break
        if rnd < max_rounds - 1:
            extra = nsteps
            # Hard cap: stop extending if total would exceed max_total_steps
            if nsteps + extra > max_total_steps:
                extra = max(0, max_total_steps - nsteps)
                if extra == 0:
                    if verbose:
                        print(f"\n  Reached max_total_steps={max_total_steps}, stopping extension.")
                    break
            if verbose:
                print(f"\n  Not converged — extending MCMC by {extra} steps (round {rnd+2}/{max_rounds})")
            try:
                sampler.run_mcmc(None, extra, progress=progress)
            except Exception:
                last_pos = sampler.get_last_sample()
                # Add broad perturbation to break degeneracy among walkers.
                # 0.01 was too small — walkers remain collapsed and R-hat
                # explodes to 1e13+. 0.10 scatters them across 10% of
                # the [0,1] range, enough to break linear dependence.
                coords = last_pos.coords.copy()
                noise = 0.10 * np.random.randn(*coords.shape)
                coords = np.clip(coords + noise, 0.0, 1.0)
                last_pos_perturbed = last_pos
                last_pos_perturbed.coords = coords
                sampler.run_mcmc(last_pos_perturbed, extra, progress=progress)
            nsteps = sampler.get_chain().shape[0]
            burnin = min(burnin, nsteps // 4)
        else:
            diag_final = diag

    if not converged:
        n_peg = int(np.sum(np.asarray(diag_final.get('pegged_low', []))) +
                    np.sum(np.asarray(diag_final.get('pegged_high', []))))
        peg_note = f" ({n_peg} param(s) pinned at prior bounds, excluded from gate)" if n_peg else ""
        if verbose:
            print(f"\n  *** WARNING: MCMC did NOT converge after {max_rounds} "
                  f"rounds ({nsteps} steps, R-hat max "
                  f"{np.nanmax(np.atleast_1d(diag_final.get('rhat', np.array([np.nan])))):.3f}{peg_note}). "
                  f"Any reported posterior intervals are unreliable. ***")
        import warnings
        warnings.warn(
            f"MCMC chain did not converge after {max_rounds} rounds "
            f"({nsteps} steps). Reported CI may be unreliable.")

    chain = sampler.get_chain(discard=burnin, flat=True)
    lnprob = sampler.get_log_prob(discard=burnin, flat=True)

    # Physical mapping must mirror _flat_to_params: f_HI_* dims use a log10
    # box, everything else is linear.  (Linear chain*(hi-lo)+lo here would
    # silently mis-scale the f_HI columns and corrupt the corner plot.)
    free_keys = [pk for i in range(len(param_keys)) for pk in [param_keys[i]]
                 if phys_highs[i] != phys_lows[i]]
    chain_phys = np.empty_like(chain)
    for c, key in enumerate(free_keys):
        i = list(param_keys).index(key)
        lo, hi = phys_lows[i], phys_highs[i]
        if key.startswith('f_HI_') and lo > 0:
            lo10, hi10 = np.log10(lo), np.log10(hi)
            chain_phys[:, c] = 10.0 ** (chain[:, c] * (hi10 - lo10) + lo10)
        else:
            chain_phys[:, c] = chain[:, c] * (hi - lo) + lo

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

    if pool is not None:
        pool.close()
        pool.join()

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
        'converged': converged,
        'rhat': np.atleast_1d(diag_final.get('rhat', np.array([np.nan]))),
        'ess': np.atleast_1d(diag_final.get('ess', np.array([np.nan]))),
        'tau': np.atleast_1d(diag_final.get('tau', np.array([np.nan]))),
        'mean_shift_rel': np.atleast_1d(
            diag_final.get('mean_shift_rel', np.array([np.nan]))),
        'pegged_low': np.atleast_1d(diag_final.get('pegged_low', np.array([False]))),
        'pegged_high': np.atleast_1d(diag_final.get('pegged_high', np.array([False]))),
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
