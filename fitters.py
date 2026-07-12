"""
Optimization engine: profile parameters → best fit to observed HINSA map.
"""

import numpy as np


def fit_hinspheres(cfg, obs_hinsa_map, T_HI_true_map=None,
                    method='CMA-ES', maxiter=500, n_jobs=4,
                    params_init=None, bounds=None,
                    max_gen=None, popsize=None, seed=None, verbose=False,
                    center_yx=None, pixel_scale_pc=None, R_out_pc=None,
                    mode='forward', obs_hdr=None, distance_pc=None,
                    fit_velocity_radius_kms=None,
                    spatial_res_pc=None, vel_res_kms=None):
    """Fit spherical HINSA model to observed absorption map.

    Parameters
    ----------
    cfg : Config
    obs_hinsa_map : 3D array
        Observed HI cube (n_v, ny, nx) in K.
    T_HI_true_map : 3D array or None
        Recovered unabsorbed HI brightness cube (K) — used as RT background.
        Required for mode='forward'. Ignored for mode='second_derivative'.
    method : str
        'CMA-ES'.
    maxiter : int
    n_jobs : int
    params_init : dict or None
        Initial parameter guess. If None, uses defaults.
        Can contain either 'peak_shell'+'multipliers' or 'f_HI' for abundance.
    bounds : dict or None
        Parameter bounds. If None, uses cfg.bounds_pc.

    Returns
    -------
    best_params : dict
    result_obj : cma.CMAEvolutionStrategyResult
    """
    from .models import build_synthetic_hinsa, residual_map

    # --- Build parameter vector ---
    param_keys = ['rho0', 'r0', 'alpha', 'T0', 'T1', 'rT',
                  'abundance', 'f_ff', 'turb_kms', 'v_offset',
                  'v_rot_kms', 'rot_pa_deg']

    if bounds is None:
        bounds = cfg.bounds_pc

    if params_init is None:
        dp = cfg.default_params
        params_init = {
            'rho0': dp['rho0'],
            'r0': dp['r0'],
            'alpha': dp['alpha'],
            'T0': dp['T0'],
            'T1': dp['T1'],
            'rT': dp['rT'],
            'peak_shell': dp['peak_shell'],
            'f_HI_peak': dp['f_HI_peak'],
            'multipliers': np.ones(cfg.n_shells - 1) * dp['multipliers_value'],
            'f_ff': dp['f_ff'],
            'turb_kms': dp['turb_kms'],
            'v_offset': dp['v_offset'],
            'v_rot_kms': dp['v_rot_kms'],
            'rot_pa_deg': dp['rot_pa_deg'],
        }

    # CMA-ES optimization
    if method == 'CMA-ES':
        if max_gen is not None:
            maxiter = max_gen
        best_params, history, param_stds = _optimize_cmaes(
            cfg, obs_hinsa_map, T_HI_true_map,
            params_init, bounds, maxiter, n_jobs,
            popsize=popsize, seed=seed, verbose=verbose,
            center_yx=center_yx, pixel_scale_pc=pixel_scale_pc,
            R_out_pc=R_out_pc, mode=mode,
            obs_hdr=obs_hdr, distance_pc=distance_pc,
            fit_velocity_radius_kms=fit_velocity_radius_kms,
            spatial_res_pc=spatial_res_pc, vel_res_kms=vel_res_kms)
        return best_params, history, param_stds
    else:
        raise ValueError(f"Unknown method: {method}")


def _optimize_cmaes(cfg, obs_map, T_HI_true, params_init,
                     bounds, maxiter, n_jobs, popsize=None,
                     seed=None, verbose=False,
                     center_yx=None, pixel_scale_pc=None, R_out_pc=None,
                     mode='forward', obs_hdr=None, distance_pc=None,
                     fit_velocity_radius_kms=None,
                     spatial_res_pc=None, vel_res_kms=None):
    """CMA-ES optimizer with ask/tell + joblib parallel evaluation."""
    try:
        import cma
    except ImportError:
        raise ImportError("CMA-ES requires the `cma` package: pip install cma")

    from .models import build_synthetic_hinsa, residual_map
    from joblib import Parallel, delayed

    # --- Derive defaults for center_yx and pixel_scale_pc ---
    if center_yx is None:
        if obs_map.ndim == 2:
            center_yx = (obs_map.shape[0] // 2, obs_map.shape[1] // 2)
        else:
            center_yx = (obs_map.shape[1] // 2, obs_map.shape[2] // 2)
    if pixel_scale_pc is None:
        if obs_hdr is not None and distance_pc is not None:
            cd1 = abs(obs_hdr.get('CDELT1', 0.0))
            cd2 = abs(obs_hdr.get('CDELT2', 0.0))
            pixscale_deg = (cd1 + cd2) / 2.0
            pixel_scale_pc = pixscale_deg * np.pi / 180.0 * distance_pc
        elif hasattr(cfg, 'pixel_scale_pc'):
            pixel_scale_pc = cfg.pixel_scale_pc
        else:
            raise ValueError(
                "pixel_scale_pc is None and cannot be computed: "
                "provide obs_hdr + distance_pc, or set pixel_scale_pc explicitly.")

    # --- Radial weight map: weight = 1/r^weight_index ---
    if obs_map.ndim == 2:
        ny, nx = obs_map.shape
    else:
        ny, nx = obs_map.shape[1], obs_map.shape[2]
    yc, xc = center_yx
    yy, xx = np.mgrid[0:ny, 0:nx]
    r_map = np.sqrt(((yy - yc).astype(float))**2 + ((xx - xc).astype(float))**2)
    r_map_pc = r_map * pixel_scale_pc
    r_map_pc[r_map_pc == 0] = pixel_scale_pc  # center → same weight as r=1 pixel
    weight_map = 1.0 / r_map_pc ** cfg.weight_index

    # --- Compute velocity axis (always needed for build_synthetic_hinsa) ---
    crval3 = obs_hdr.get('CRVAL3', 0.0) if obs_hdr is not None else 0.0
    cdelt3_val = obs_hdr.get('CDELT3', 0.0) if obs_hdr is not None else 200.0
    crpix3 = obs_hdr.get('CRPIX3', 1.0) if obs_hdr is not None else 1.0
    n_v_ch = obs_map.shape[0] if obs_map.ndim == 3 else cfg.n_v_channels
    velo_kms_arr = (crval3 + cdelt3_val * (np.arange(n_v_ch) - (crpix3 - 1))) / 1000.0
    if velo_kms_arr[-1] < velo_kms_arr[0]:
        velo_kms_arr = velo_kms_arr[::-1]

    # --- Velocity mask: restrict fitting to ±fit_velocity_radius_kms ---
    velo_mask = None
    if fit_velocity_radius_kms is not None and obs_hdr is not None:
        v_center = cfg.vlsr_kms
        velo_mask = np.abs(velo_kms_arr - v_center) <= fit_velocity_radius_kms
        if verbose:
            v_lo = float(velo_kms_arr[velo_mask].min()) if np.any(velo_mask) else v_center
            v_hi = float(velo_kms_arr[velo_mask].max()) if np.any(velo_mask) else v_center
            n_ch = int(np.sum(velo_mask))
            print(f'  Velocity mask: {v_lo:.2f} ~ {v_hi:.2f} km/s ({n_ch}/{n_v_ch} channels)')

    # 3D weight map for second_derivative mode: broadcast 2D weights to all velocity channels
    if mode == 'second_derivative':
        n_v = obs_map.shape[0] if obs_map.ndim == 3 else cfg.n_v_channels
        weight_map_3d = np.broadcast_to(weight_map[np.newaxis, :, :],
                                         (n_v, ny, nx)).copy()
        if velo_mask is not None:
            weight_map_3d[~velo_mask] = 0.0
    else:
        weight_map_3d = None

    # --- Pre-compute smoothing sigmas for the optimization loop ---
    _sigma_fwhm2sig = 2.0 * np.sqrt(2.0 * np.log(2.0))
    _sigma_v = 0.0
    _sigma_xy = 0.0
    if vel_res_kms is not None and obs_map.ndim == 3:
        dv = abs(obs_hdr.get('CDELT3', 200.0)) / 1000.0 if obs_hdr else 1.0
        _sigma_v = (vel_res_kms / dv) / _sigma_fwhm2sig
    if spatial_res_pc is not None:
        _sigma_xy = (spatial_res_pc / pixel_scale_pc) / _sigma_fwhm2sig
    _need_smooth = (_sigma_v > 0) or (_sigma_xy > 0)

    param_keys, x0, phys_lows, phys_highs = _params_to_flat(params_init, bounds, cfg)

    # Separate fixed params (phys_low == phys_high, zero range) from free params
    phys_ranges = phys_highs - phys_lows
    fixed_idx = [i for i in range(len(param_keys)) if phys_ranges[i] == 0]
    free_idx = [i for i in range(len(param_keys)) if phys_ranges[i] > 0]

    fixed_keys = [param_keys[i] for i in fixed_idx]

    x0_free = x0[free_idx]
    # CMA-ES bounds: all [0,1] in normalized space
    low_free = np.zeros(len(free_idx))
    high_free = np.ones(len(free_idx))
    free_keys = [param_keys[i] for i in free_idx]

    n_dim = len(x0_free)
    if popsize is None:
        popsize = max(6, 4 + int(3 * np.log(n_dim)))

    options = {
        'bounds': [low_free, high_free],
        'maxiter': maxiter,
        'popsize': popsize,
        'verb_disp': int(verbose),
        'verb_log': 0,
        'tolx': 1e-6,
        'tolfun': 1e-6,
    }
    if seed is not None:
        options['seed'] = seed
    if verbose:
        options['verbose'] = -2

    es = cma.CMAEvolutionStrategy(x0_free, 0.2, options)

    best_fun = np.inf
    best_x = None
    generation = 0

    def _reconstruct(x_free):
        """Merge free CMA-ES vector with fixed params → full param dict."""
        # Build full [0,1] vector: insert fixed values back
        x_full = np.copy(x0)
        for fi, fv in zip(free_idx, x_free):
            x_full[fi] = fv
        return _flat_to_params(x_full, param_keys, cfg,
                               phys_lows=phys_lows, phys_highs=phys_highs)

    while not es.stop():
        solutions = es.ask()

        def _eval_one(x):
            params = _reconstruct(x)
            try:
                if mode == 'second_derivative':
                    from .models import inverse_build_hinsa_cube
                    # obs_map is 3D cube in this mode
                    T_bg_reconstructed = inverse_build_hinsa_cube(
                        cfg, params, obs_map, center_yx, pixel_scale_pc,
                        galactic_b_deg=getattr(cfg, 'galactic_b_deg', None),
                        R_out_pc=R_out_pc, n_jobs=1)
                    # Apply beam/velocity smoothing to reconstructed T_bg
                    # (obs_map was smoothed during generation; T_bg must match)
                    if _need_smooth and T_bg_reconstructed.ndim == 3:
                        from scipy.ndimage import gaussian_filter
                        T_bg_reconstructed = gaussian_filter(
                            T_bg_reconstructed,
                            sigma=(_sigma_v, _sigma_xy, _sigma_xy),
                            mode='reflect')
                    # Compute R-value: integrated squared 2nd derivative
                    dv = abs(cfg.v_max_kms - cfg.v_min_kms) / (cfg.n_v_channels - 1)
                    d2 = np.zeros_like(T_bg_reconstructed)
                    d2[1:-1] = (T_bg_reconstructed[2:] + T_bg_reconstructed[:-2]
                                - 2.0 * T_bg_reconstructed[1:-1]) / (dv**2)
                    # Sum R over all pixels (weighted by 1/r), normalized by Σw
                    r_smooth = np.sum(d2**2 * weight_map_3d) * dv
                    w_sum = np.sum(weight_map_3d)
                    r_smooth = r_smooth / w_sum if w_sum > 0 else r_smooth

                    return float(r_smooth)
                else:
                    m = build_synthetic_hinsa(cfg, params, T_HI_true,
                                               center_yx=center_yx,
                                               pixel_scale_pc=pixel_scale_pc,
                                               galactic_b_deg=getattr(cfg, 'galactic_b_deg', None),
                                               R_out_pc=R_out_pc, n_jobs=1,
                                               velo_bg_kms=velo_kms_arr)
                    # Apply beam/velocity smoothing during optimization
                    if _need_smooth and m.ndim == 3:
                        from scipy.ndimage import gaussian_filter
                        m = gaussian_filter(m, sigma=(_sigma_v, _sigma_xy, _sigma_xy),
                                            mode='reflect')
                    if velo_mask is not None:
                        # Apply velocity mask: set weights to 0 outside range
                        w_3d = np.broadcast_to(weight_map[np.newaxis, :, :], m.shape).copy()
                        w_3d[~velo_mask] = 0.0
                        return float(residual_map(obs_map, m, weights=w_3d))
                    return float(residual_map(obs_map, m, weights=weight_map))
            except Exception:
                return 1e10

        fitness = Parallel(n_jobs=n_jobs)(
            delayed(_eval_one)(x) for x in solutions
        )
        es.tell(solutions, fitness)

        gen_best = min(fitness)
        if gen_best < best_fun:
            best_fun = gen_best
            best_x = solutions[fitness.index(gen_best)]

        generation += 1
        if verbose:
            print(f"  gen {generation:4d} | best={best_fun:.4e} | "
                  f"current={gen_best:.4e} | sigma={es.sigma:.4e}")

    if best_x is None or not np.all(np.isfinite(best_x)):
        best_x = es.result.xbest
    if best_x is None or not np.all(np.isfinite(best_x)):
        # All evaluations failed — return default params with infinite residual
        params_best = _flat_to_params(x0, param_keys, cfg,
                                       phys_lows=phys_lows, phys_highs=phys_highs)
        param_stds = {}
        for k in param_keys:
            param_stds[k] = 0.0
        return params_best, es.result, param_stds
    params_best = _reconstruct(best_x)

    # Extract 1-sigma uncertainties from CMA-ES covariance
    # stds_free are in [0,1] normalized space.
    # Convert to physical space: σ_phys = σ_01 * (phys_high - phys_low)
    stds_free = es.result.stds
    param_stds = {}

    for fi in fixed_idx:
        param_stds[param_keys[fi]] = 0.0  # fixed params have 0 uncertainty
    for i, idx in enumerate(free_idx):
        pname = param_keys[idx]
        phys_range = phys_highs[idx] - phys_lows[idx]
        param_stds[pname] = float(stds_free[i] * phys_range)

    # Aggregate array-element stds (f_HI_0, f_HI_1, ..., mult_0, ...) into arrays
    for arr_name in ('f_HI', 'multipliers'):
        n = cfg.n_shells if arr_name == 'f_HI' else max(1, cfg.n_shells - 1)
        flat_prefix = 'f_HI' if arr_name == 'f_HI' else 'mult'
        arr_std = np.zeros(n)
        for i in range(n):
            key = f'{flat_prefix}_{i}'
            if key in param_stds:
                arr_std[i] = param_stds.pop(key)
        # CMA-ES stds are in log-space for log-transformed params.
        # Convert to physical space: σ_phys ≈ value * σ_log
        arr_vals = params_best.get(arr_name)
        if arr_vals is not None and len(arr_vals) == n:
            arr_std = np.abs(np.asarray(arr_vals, dtype=float)) * arr_std
        param_stds[arr_name] = arr_std

    return params_best, es.result, param_stds


def _params_to_flat(params_init, bounds, cfg):
    """Convert parameter dict to flat [0,1] vector for CMA-ES.

    All parameters are linearly mapped to [0,1]:
        x_01 = (val - phys_low) / (phys_high - phys_low)

    Returns keys, x_01, phys_lows, phys_highs so _flat_to_params can
    map back to physical space.

    Supports two abundance modes:
      - 'f_HI' key present: direct per-shell abundance (n_shells values)
      - 'peak_shell' + 'multipliers': parametric abundance (1 + n_shells-1 values)
    """
    keys = []
    phys_lows = []
    phys_highs = []

    def _add(key, val, lo, hi):
        keys.append(key)
        phys_lows.append(float(lo))
        phys_highs.append(float(hi))

    # --- Scalar parameters ---
    rho0_bounds = bounds.get('rho0', cfg.bounds_pc['rho0'])
    _add('rho0', params_init['rho0'], rho0_bounds[0], rho0_bounds[1])
    r0_bounds = bounds.get('r0', cfg.bounds_pc['r0'])
    _add('r0', params_init['r0'], r0_bounds[0], r0_bounds[1])
    alpha_bounds = bounds.get('alpha', cfg.bounds_pc['alpha'])
    _add('alpha', params_init['alpha'], alpha_bounds[0], alpha_bounds[1])
    T0_bounds = bounds.get('T0', cfg.bounds_pc['T0'])
    _add('T0', params_init['T0'], T0_bounds[0], T0_bounds[1])
    T1_bounds = bounds.get('T1', cfg.bounds_pc['T1'])
    _add('T1', params_init['T1'], T1_bounds[0], T1_bounds[1])
    rT_bounds = bounds.get('rT', cfg.bounds_pc['rT'])
    _add('rT', params_init['rT'], rT_bounds[0], rT_bounds[1])

    # --- Abundance: either f_HI (direct) or peak_shell + multipliers ---
    if 'f_HI' in params_init:
        f_HI = np.asarray(params_init['f_HI'], dtype=float)
        f_hi_bounds = bounds.get('f_HI', cfg.bounds_pc['f_HI'])
        for i in range(cfg.n_shells):
            if isinstance(f_hi_bounds, (list, tuple)) and len(f_hi_bounds) == cfg.n_shells:
                lo, hi = f_hi_bounds[i]
            else:
                lo, hi = f_hi_bounds
            _add(f'f_HI_{i}', float(f_HI[i]), float(lo), float(hi))
    else:
        ps_lo, ps_hi = float(bounds['peak_shell'][0]), float(bounds['peak_shell'][1])
        if ps_lo == ps_hi:
            keys.append('_fixed_peak_shell')
            phys_lows.append(ps_lo)
            phys_highs.append(ps_hi)
        else:
            _add('peak_shell', float(params_init['peak_shell']), ps_lo, ps_hi)

        f_hp_lo = float(bounds.get('f_HI_peak', cfg.bounds_pc['f_HI_peak'])[0])
        f_hp_hi = float(bounds.get('f_HI_peak', cfg.bounds_pc['f_HI_peak'])[1])
        if f_hp_lo == f_hp_hi:
            keys.append('_fixed_f_HI_peak')
            phys_lows.append(f_hp_lo)
            phys_highs.append(f_hp_hi)
        else:
            _add('f_HI_peak',
                 float(params_init.get('f_HI_peak', cfg.default_params['f_HI_peak'])),
                 f_hp_lo, f_hp_hi)

        for i in range(cfg.n_shells - 1):
            m = params_init['multipliers'][i] if i < len(params_init['multipliers']) else cfg.default_params['multipliers_value']
            ml_bounds = bounds.get('multipliers', cfg.bounds_pc['multipliers'])
            _add(f'mult_{i}', float(m),
                 float(ml_bounds[0]), float(ml_bounds[1]))

    # --- Kinematics ---
    f_ff_bounds = bounds.get('f_ff', cfg.bounds_pc['f_ff'])
    _add('f_ff', params_init['f_ff'], f_ff_bounds[0], f_ff_bounds[1])
    turb_kms_bounds = bounds.get('turb_kms', cfg.bounds_pc['turb_kms'])
    _add('turb_kms', params_init['turb_kms'], turb_kms_bounds[0], turb_kms_bounds[1])
    v_offset_bounds = bounds.get('v_offset', cfg.bounds_pc['v_offset'])
    _add('v_offset', params_init['v_offset'], v_offset_bounds[0], v_offset_bounds[1])
    v_rot_kms_bounds = bounds.get('v_rot_kms', cfg.bounds_pc['v_rot_kms'])
    _add('v_rot_kms', params_init['v_rot_kms'], v_rot_kms_bounds[0], v_rot_kms_bounds[1])
    rot_pa_deg_bounds = bounds.get('rot_pa_deg', cfg.bounds_pc['rot_pa_deg'])
    _add('rot_pa_deg', params_init['rot_pa_deg'], rot_pa_deg_bounds[0], rot_pa_deg_bounds[1])

    # Normalize to [0,1]
    phys_lows = np.array(phys_lows)
    phys_highs = np.array(phys_highs)
    phys_ranges = phys_highs - phys_lows
    phys_ranges[phys_ranges == 0] = 1.0  # fixed params: range=0, keep at 0

    x_raw = []
    for i, key in enumerate(keys):
        if key.startswith('_fixed_'):
            x_raw.append(0.0)  # fixed params are always at 0
        else:
            # Reconstruct physical value from params_init
            if key == 'rho0':
                val = params_init['rho0']
            elif key == 'r0':
                val = params_init['r0']
            elif key == 'alpha':
                val = params_init['alpha']
            elif key == 'T0':
                val = params_init['T0']
            elif key == 'T1':
                val = params_init['T1']
            elif key == 'rT':
                val = params_init['rT']
            elif key == 'f_HI_peak':
                val = float(params_init.get('f_HI_peak', cfg.default_params['f_HI_peak']))
            elif key.startswith('f_HI_'):
                idx = int(key.split('_')[-1])
                val = float(np.asarray(params_init['f_HI'], dtype=float)[idx])
            elif key == 'peak_shell':
                val = float(params_init['peak_shell'])
            elif key.startswith('mult_'):
                idx = int(key.split('_')[-1])
                val = float(params_init['multipliers'][idx]) if idx < len(params_init['multipliers']) else cfg.default_params['multipliers_value']
            elif key == 'f_ff':
                val = params_init['f_ff']
            elif key == 'turb_kms':
                val = params_init['turb_kms']
            elif key == 'v_offset':
                val = params_init['v_offset']
            elif key == 'v_rot_kms':
                val = params_init['v_rot_kms']
            elif key == 'rot_pa_deg':
                val = params_init['rot_pa_deg']
            else:
                val = phys_lows[i]
            x_raw.append((val - phys_lows[i]) / phys_ranges[i])

    return keys, np.clip(np.array(x_raw), 0.0, 1.0), phys_lows, phys_highs


def _flat_to_params(x, keys, cfg, phys_lows=None, phys_highs=None):
    """Convert flat [0,1] vector back to parameter dict.

    Maps each dimension from [0,1] to physical space:
        val = x_01 * (phys_high - phys_low) + phys_low
    """
    p = {}
    idx = 0

    def _next():
        nonlocal idx
        v = x[idx]; idx += 1
        return v

    def _to_phys(v01, i):
        """Map [0,1] → physical space."""
        if phys_lows is not None and phys_highs is not None:
            return v01 * (phys_highs[i] - phys_lows[i]) + phys_lows[i]
        return v01

    p['rho0'] = _to_phys(_next(), idx - 1); 
    p['r0'] = _to_phys(_next(), idx - 1)
    p['alpha'] = _to_phys(_next(), idx - 1)
    p['T0'] = _to_phys(_next(), idx - 1)
    p['T1'] = _to_phys(_next(), idx - 1)
    p['rT'] = _to_phys(_next(), idx - 1)

    # Abundance: direct f_HI or peak_shell + multipliers
    if keys[idx].startswith('f_HI_'):
        f_HI = np.zeros(cfg.n_shells)
        for i in range(cfg.n_shells):
            f_HI[i] = np.clip(_to_phys(_next(), idx - 1), 1e-10, 0.5)
        p['f_HI'] = f_HI
    else:
        if keys[idx] == '_fixed_peak_shell':
            p['peak_shell'] = int(phys_lows[idx]) if phys_lows is not None else cfg.default_params['peak_shell']
            idx += 1
        else:
            p['peak_shell'] = int(round(_to_phys(_next(), idx - 1)))

        if keys[idx] == '_fixed_f_HI_peak':
            p['f_HI_peak'] = float(phys_lows[idx]) if phys_lows is not None else cfg.default_params['f_HI_peak']
            idx += 1
        else:
            p['f_HI_peak'] = float(np.clip(_to_phys(_next(), idx - 1), 1e-10, 0.5))

        n_mult = cfg.n_shells - 1
        p['multipliers'] = np.clip(
            np.array([_to_phys(_next(), idx - 1) for _ in range(n_mult)]),
            0.01, 1.0)
        # Note: idx is already advanced by the loop above

    p['f_ff'] = np.clip(_to_phys(_next(), idx - 1), 1e-10, 0.5)
    p['turb_kms'] = np.clip(_to_phys(_next(), idx - 1), 1e-10, 5.0)
    p['v_offset'] = _to_phys(_next(), idx - 1)
    p['v_rot_kms'] = np.clip(_to_phys(_next(), idx - 1), 0.0, 10.0)
    p['rot_pa_deg'] = _to_phys(_next(), idx - 1)

    return p


# ============================================================
# Top-level fitting wrapper
# ============================================================

def fit_hinsa_model(obs_hinsa_fits, obs_background_fits=None,
                    R_out_pc=None, distance_pc=None, vlsr_kms=0.0,
                    n_shells=None, spatial_res_pc=None, vel_res_kms=None,
                    params_init=None, bounds=None,
                    max_gen=500, popsize=None,
                    n_jobs=4, seed=None, verbose=True,
                    output_dir=None, mode='forward',
                    fit_velocity_radius_kms=3.0):
    """Fit a spherical HINSA model to an observed absorption map using CMA-ES.

    One-call wrapper: provide FITS files + geometry, get best fit.
    Pixel scale, n_v_channels, v_min/max_kms are read from FITS headers
    automatically (from prepare_hinspheres_input output).

    Parameters
    ----------
    obs_hinsa_fits : str
        Path to 3D FITS: observed HI cube (K).
    obs_background_fits : str or None
        Path to 3D FITS: observed HI brightness **without the cold cloud**
        (K), i.e. the sum of galactic foreground HI emission and background HI.
        The foreground is stripped internally before cloud RT and re-applied
        afterwards.  Required for mode='forward'. Ignored for
        mode='second_derivative'.
    R_out_pc : float
        Cloud outer radius (pc).
    distance_pc : float
        Cloud distance (pc).
    vlsr_kms : float
        Cloud systemic velocity (km/s).
    n_shells : int
        Number of concentric shells.
    spatial_res_pc : float or None
        If set, Gaussian-smooth model map to this FWHM (pc).
    vel_res_kms : float or None
        If set, Gaussian-smooth model cube along velocity axis (FWHM km/s).
    params_init : dict or None
        Initial parameter guess for CMA-ES. If None, uses defaults.
        Keys: rho0, r0, alpha, T0, T1, rT, f_ff, turb_kms, v_offset,
              v_rot_kms, rot_pa_deg, and either f_HI or peak_shell+multipliers.
    bounds : dict or None
        Parameter bounds for CMA-ES. If None, uses Config.bounds_pc defaults.
    max_gen : int
        CMA-ES maximum generations.
    popsize : int or None
        Population size. If None, auto-determined.
    n_jobs : int
        Parallel jobs (-1 = all CPUs).
    seed : int or None
        Random seed.
    verbose : bool
        Print progress.
    output_dir : str or None
        Directory for output files. If None, uses directory of obs_hinsa_fits.
        Generates: {name}_bestfit.fits (model cube) and {name}_bestfit.png (diagnostic).
    mode : str
        'forward' — standard forward modeling with background FITS (default).
        'second_derivative' — Liu Method 2: no background needed, minimizes
            integrated squared 2nd derivative of reconstructed T_bg.

    Returns
    -------
    result : dict with keys:
        'best_params' : dict — best-fit parameter values
        'model_map' : 2D array — synthetic HINSA map at best fit
        'history' : cma result object
        'cfg' : Config — the configuration used
        'residual' : float — final residual value
    """
    from astropy.io import fits as pyfits
    from .config import Config
    from .models import build_synthetic_hinsa, residual_map

    # Apply Config defaults for None values
    if R_out_pc is None:
        R_out_pc = Config.R_out_pc
    if distance_pc is None:
        distance_pc = Config.distance_pc
    if n_shells is None:
        n_shells = Config.n_shells

    # --------------------------------------------------------
    # Read FITS headers to extract pixel scale & velocity info
    # --------------------------------------------------------
    with pyfits.open(obs_hinsa_fits) as hdul:
        obs_hinsa_cube = hdul[0].data.astype(np.float64)
        h_obs = hdul[0].header

    # pixel scale (abs, arcmin → pc)
    cdelt1 = abs(h_obs.get('CDELT1', 0.0))  # degrees
    cdelt2 = abs(h_obs.get('CDELT2', 0.0))
    pixscale_deg = (cdelt1 + cdelt2) / 2.0
    pixscale_arcmin = pixscale_deg * 60.0
    pixel_scale_pc = pixscale_arcmin / 60.0 * np.pi / 180.0 * distance_pc
    if verbose:
        print(f'Pixel scale: {pixscale_arcmin:.3f} arcmin = {pixel_scale_pc:.3f} pc')

    # velocity axis from obs FITS header (3D cube)
    v_min_kms = -20.0
    v_max_kms = 20.0
    n_v_channels = 201

    # Extract velocity header keywords (always, so they're in scope for npz save)
    crval3 = h_obs.get('CRVAL3', 0.0)
    cdelt3 = h_obs.get('CDELT3', 0.0)
    # Ascending velocity axis for build_synthetic_hinsa (always ascending v_grid)
    _crpix3 = h_obs.get('CRPIX3', 1.0)
    _naxis3 = h_obs.get('NAXIS3', 201)
    _v_raw = (crval3 + cdelt3 * (np.arange(_naxis3) - (_crpix3 - 1))) / 1000.0
    velo_kms_asc = _v_raw[::-1] if _v_raw[-1] < _v_raw[0] else _v_raw.copy()

    if obs_hinsa_cube.ndim == 3:
        crpix3 = h_obs.get('CRPIX3', 1.0)
        naxis3 = h_obs.get('NAXIS3', obs_hinsa_cube.shape[0])
        v_axis_m = crval3 + cdelt3 * (np.arange(naxis3) - (crpix3 - 1))
        v_axis_kms = v_axis_m / 1000.0
        v_min_kms = float(np.min(v_axis_kms))
        v_max_kms = float(np.max(v_axis_kms))
        n_v_channels = int(naxis3)
        # Flip cube if velocity axis is descending (CDELT3 < 0)
        if cdelt3 < 0:
            obs_hinsa_cube = obs_hinsa_cube[::-1, :, :].copy()
            if verbose:
                print(f'  Flipped cube along velocity axis (CDELT3={cdelt3:.4f} < 0)')
        if verbose:
            print(f'Obs cube: {obs_hinsa_cube.shape}, '
                  f'{n_v_channels} ch, {v_min_kms:.2f} to {v_max_kms:.2f} km/s')

    # Load background FITS (3D cube used as RT background)
    obs_background_map = None
    if mode == 'second_derivative':
        # Second-derivative mode: no background FITS needed
        if obs_background_fits is not None:
            if verbose:
                print('Warning: obs_background_fits ignored in second_derivative mode')
        if verbose:
            print('Second-derivative mode: no background FITS needed')
    else:
        # Forward mode: background FITS required
        if obs_background_fits is None:
            raise ValueError("obs_background_fits is required for mode='forward'. "
                           "For mode='second_derivative', no background FITS is needed.")
        with pyfits.open(obs_background_fits) as hdul:
            obs_background_map = hdul[0].data.astype(np.float64)
        # Flip background cube if velocity axis is descending
        if cdelt3 < 0 and obs_background_map.ndim == 3:
            obs_background_map = obs_background_map[::-1, :, :].copy()
        if verbose:
            print(f'Background FITS (fg+bg, no cloud): {obs_background_map.shape}, '
                  f'median={np.nanmedian(obs_background_map):.1f} K')

    if verbose:
        print(f'Observed HINSA: {obs_hinsa_cube.shape}, '
              f'peak={np.nanmax(obs_hinsa_cube):.2f} K')

    # Cloud center (default: cube center)
    center_yx = (obs_hinsa_cube.shape[1] // 2, obs_hinsa_cube.shape[2] // 2)

    # --- Extract galactic latitude from FITS header ---
    from .models import _compute_galactic_b
    galactic_b_deg = _compute_galactic_b(h_obs, center_yx=center_yx)
    if verbose:
        print(f'Galactic b = {galactic_b_deg:.2f} deg (from FITS header)')

    # --------------------------------------------------------
    # Build Config (geometry + velocity grid from FITS)
    # --------------------------------------------------------
    cfg = Config(
        n_shells=n_shells,
        R_out_pc=R_out_pc,
        distance_pc=distance_pc,
        vlsr_kms=vlsr_kms,
        v_min_kms=v_min_kms,
        v_max_kms=v_max_kms,
        n_v_channels=n_v_channels,
    )
    cfg.galactic_b_deg = galactic_b_deg

    # --- Default initial params ---
    dp = cfg.default_params
    # Determine abundance mode from user input
    _use_peak_shell_mode = (params_init is not None and
                            ('peak_shell' in params_init or 'multipliers' in params_init))
    default_params_init = {
        'rho0': dp['rho0'],
        'r0': dp['r0'],
        'alpha': dp['alpha'],
        'T0': dp['T0'],
        'T1': dp['T1'],
        'rT': dp['rT'],
        'f_ff': dp['f_ff'],
        'turb_kms': dp['turb_kms'],
        'v_offset': dp['v_offset'],
        'v_rot_kms': dp['v_rot_kms'],
        'rot_pa_deg': dp['rot_pa_deg'],
    }
    if _use_peak_shell_mode:
        default_params_init['peak_shell'] = dp['peak_shell']
        default_params_init['f_HI_peak'] = dp['f_HI_peak']
        default_params_init['multipliers'] = np.ones(n_shells - 1) * dp['multipliers_value']
    else:
        default_params_init['f_HI'] = np.ones(n_shells) * dp['f_HI_peak']

    if params_init is None:
        params_init = default_params_init
    else:
        # Merge user params with defaults (user overrides defaults)
        for k, v in default_params_init.items():
            if k not in params_init:
                params_init[k] = v

    # --- Build model cube from params (for residual computation) ---
    def _build_model(best_p):
        if mode == 'second_derivative':
            from .models import inverse_build_hinsa_cube
            return inverse_build_hinsa_cube(
                cfg, best_p, obs_hinsa_cube, center_yx, pixel_scale_pc,
                galactic_b_deg=galactic_b_deg,
                R_out_pc=R_out_pc, n_jobs=n_jobs)
        else:
            return build_synthetic_hinsa(cfg, best_p,
                                        bg_cube=obs_background_map,
                                        center_yx=center_yx,
                                        pixel_scale_pc=pixel_scale_pc,
                                        galactic_b_deg=galactic_b_deg,
                                        R_out_pc=R_out_pc,
                                        n_jobs=n_jobs,
                                        velo_bg_kms=velo_kms_asc)

    # --- Compute final residual from model cube ---
    def _compute_residual(mc):
        ny_f, nx_f = obs_hinsa_cube.shape[1], obs_hinsa_cube.shape[2]
        yy_f, xx_f = np.mgrid[0:ny_f, 0:nx_f]
        if mode == 'second_derivative':
            _s_fwhm2sig = 2.0 * np.sqrt(2.0 * np.log(2.0))
            _s_v = 0.0
            _s_xy = 0.0
            if vel_res_kms is not None and mc.ndim == 3:
                _dv = abs(h_obs.get('CDELT3', 200.0)) / 1000.0
                _s_v = (vel_res_kms / _dv) / _s_fwhm2sig
            if spatial_res_pc is not None:
                _s_xy = (spatial_res_pc / pixel_scale_pc) / _s_fwhm2sig
            if (_s_v > 0 or _s_xy > 0) and mc.ndim == 3:
                from scipy.ndimage import gaussian_filter as _gf
                mc = _gf(mc, sigma=(_s_v, _s_xy, _s_xy), mode='reflect')
            dv = abs(cfg.v_max_kms - cfg.v_min_kms) / (cfg.n_v_channels - 1)
            d2 = np.zeros_like(mc)
            d2[1:-1] = (mc[2:] + mc[:-2] - 2.0 * mc[1:-1]) / (dv**2)
            r_map_f = np.sqrt(((yy_f - center_yx[0]).astype(float))**2 +
                              ((xx_f - center_yx[1]).astype(float))**2)
            r_map_pc_f = r_map_f * pixel_scale_pc
            r_map_pc_f[r_map_pc_f == 0] = pixel_scale_pc
            wm3 = (1.0 / r_map_pc_f)[np.newaxis, :, :]
            wm3 = np.broadcast_to(wm3, mc.shape).copy()
            if fit_velocity_radius_kms is not None and h_obs is not None:
                _crval3 = h_obs.get('CRVAL3', 0.0)
                _cdelt3 = h_obs.get('CDELT3', 0.0)
                _crpix3 = h_obs.get('CRPIX3', 1.0)
                _nv = mc.shape[0]
                _v_arr = (_crval3 + _cdelt3 * (np.arange(_nv) - (_crpix3 - 1))) / 1000.0
                if _v_arr[-1] < _v_arr[0]:
                    _v_arr = _v_arr[::-1]
                _vmask = np.abs(_v_arr - cfg.vlsr_kms) <= fit_velocity_radius_kms
                wm3[~_vmask] = 0.0
            return np.sum(d2**2 * wm3) * dv / np.sum(wm3)
        else:
            r_map_f = np.sqrt(((yy_f - center_yx[0]).astype(float))**2 +
                              ((xx_f - center_yx[1]).astype(float))**2)
            r_map_pc_f = r_map_f * pixel_scale_pc
            r_map_pc_f[r_map_pc_f == 0] = pixel_scale_pc
            weight_map_f = 1.0 / r_map_pc_f
            # Apply velocity mask to 3D cubes for forward mode
            mc_use = mc
            obs_use = obs_hinsa_cube
            if fit_velocity_radius_kms is not None and h_obs is not None:
                _crval3 = h_obs.get('CRVAL3', 0.0)
                _cdelt3 = h_obs.get('CDELT3', 0.0)
                _crpix3 = h_obs.get('CRPIX3', 1.0)
                _nv = mc.shape[0]
                _v_arr = (_crval3 + _cdelt3 * (np.arange(_nv) - (_crpix3 - 1))) / 1000.0
                if _v_arr[-1] < _v_arr[0]:
                    _v_arr = _v_arr[::-1]
                _vmask = np.abs(_v_arr - cfg.vlsr_kms) > fit_velocity_radius_kms
                mc_use = mc.copy(); mc_use[_vmask] = np.nan
                obs_use = obs_hinsa_cube.copy(); obs_use[_vmask] = np.nan
            return residual_map(obs_use, mc_use, weights=weight_map_f)

    # --- Determine if peak_shell scanning is needed ---
    model_cube = None
    final_res = None
    history = None
    all_runs = None
    _ps_bounds = bounds.get('peak_shell', (1, n_shells))
    _ps_lo, _ps_hi = int(_ps_bounds[0]), int(_ps_bounds[1])
    _peak_shell_scan = (_ps_lo != _ps_hi) and ('peak_shell' in params_init)

    if _peak_shell_scan:
        # --- Multi-run peak_shell scanning ---
        if verbose:
            print(f'\n=== peak_shell scanning: testing integer values {_ps_lo}..{_ps_hi} ===')
        all_runs = []
        for _ps_val in range(_ps_lo, _ps_hi + 1):
            if verbose:
                print(f'\n--- peak_shell = {_ps_val}/{_ps_hi} ---')
            bounds_ps = dict(bounds)
            bounds_ps['peak_shell'] = (float(_ps_val), float(_ps_val))
            params_init_ps = dict(params_init)
            params_init_ps['peak_shell'] = _ps_val
            bp, hist, pst = fit_hinspheres(
                cfg, obs_hinsa_cube,
                T_HI_true_map=obs_background_map,
                params_init=params_init_ps,
                bounds=bounds_ps,
                max_gen=max_gen, popsize=popsize,
                n_jobs=n_jobs, seed=seed, verbose=verbose,
                center_yx=center_yx, pixel_scale_pc=pixel_scale_pc,
                R_out_pc=R_out_pc, mode=mode, obs_hdr=h_obs,
                distance_pc=distance_pc,
                fit_velocity_radius_kms=fit_velocity_radius_kms,
                spatial_res_pc=spatial_res_pc, vel_res_kms=vel_res_kms)
            mc = _build_model(bp)
            res = _compute_residual(mc)
            all_runs.append({'peak_shell': _ps_val, 'params': bp,
                             'param_stds': pst, 'residual': res,
                             'model_cube': mc, 'history': hist})
            if verbose:
                print(f'  peak_shell={_ps_val}  residual={res:.4e}')
        all_runs.sort(key=lambda r: r['residual'])
        best_run = all_runs[0]
        best_params = best_run['params']
        param_stds = best_run['param_stds']
        final_res = best_run['residual']
        model_cube = best_run['model_cube']
        history = best_run['history']
        if verbose:
            print(f'\n=== Best peak_shell = {best_run["peak_shell"]} (residual={final_res:.4e}) ===')
            print(f'=== Top {min(3, len(all_runs))} results ===')
            for i, r in enumerate(all_runs[:3]):
                print(f'  #{i+1}: peak_shell={r["peak_shell"]}  residual={r["residual"]:.4e}')
    else:
        # --- Single-run CMA-ES (existing behavior) ---
        best_params, history, param_stds = fit_hinspheres(
            cfg, obs_hinsa_cube,
            T_HI_true_map=obs_background_map,
            params_init=params_init,
            bounds=bounds,
            max_gen=max_gen,
            popsize=popsize,
            n_jobs=n_jobs,
            seed=seed,
            verbose=verbose,
            center_yx=center_yx,
            pixel_scale_pc=pixel_scale_pc,
            R_out_pc=R_out_pc,
            mode=mode,
            obs_hdr=h_obs,
            distance_pc=distance_pc,
            fit_velocity_radius_kms=fit_velocity_radius_kms,
            spatial_res_pc=spatial_res_pc,
            vel_res_kms=vel_res_kms)

    # --- Build best-fit model cube (if not already built by scanning) ---
    if model_cube is None:
        model_cube = _build_model(best_params)

    # --- Compute relative errors for simulated clouds ---
    relative_errors = None
    _basename_check = obs_hinsa_fits.rsplit('/', 1)[-1].rsplit('\\', 1)[-1]
    if 'generate' in _basename_check:
        # Extract true parameters from FITS MOD_* header keywords
        _mod_map = {
            'rho0': 'MOD_RHOC', 'r0': 'MOD_R0', 'alpha': 'MOD_ALPH',
            'T0': 'MOD_T0', 'T1': 'MOD_T1', 'rT': 'MOD_RT',
            'f_ff': 'MOD_FFF', 'turb_kms': 'MOD_TURB',
            'v_offset': 'MOD_VOFS', 'v_rot_kms': 'MOD_VROT',
            'rot_pa_deg': 'MOD_RPA',
        }
        true_params = {}
        for pkey, hdr_key in _mod_map.items():
            if hdr_key in h_obs:
                true_params[pkey] = float(h_obs[hdr_key])
        # f_HI: MOD_FHI1 .. MOD_FHI{n_shells}
        f_hi_true = []
        for i in range(n_shells):
            hk = f'MOD_FHI{i+1}'
            if hk in h_obs:
                f_hi_true.append(float(h_obs[hk]))
        if f_hi_true:
            true_params['f_HI'] = np.array(f_hi_true)

        # Compute relative errors: (fitted - true) / true
        relative_errors = {}
        for pk, tv in true_params.items():
            fv = best_params.get(pk)
            if fv is None:
                continue
            if isinstance(tv, np.ndarray):
                # Avoid division by zero: use absolute difference for near-zero values
                with np.errstate(divide='ignore', invalid='ignore'):
                    re = np.where(np.abs(tv) > 1e-10,
                                  (np.asarray(fv) - tv) / tv,
                                  np.asarray(fv) - tv)
                relative_errors[pk] = re
            else:
                if abs(tv) > 1e-10:
                    relative_errors[pk] = (fv - tv) / tv
                else:
                    relative_errors[pk] = fv - tv

        if verbose and relative_errors:
            print('\n===== Relative errors (fitted - true) / true =====')
            for pk, re_val in relative_errors.items():
                if isinstance(re_val, np.ndarray):
                    re_str = np.array2string(re_val, precision=4, suppress_small=True)
                    print(f'  {pk:15s} = {re_str}')
                else:
                    print(f'  {pk:15s} = {re_val:+.4f}')

    # --- Apply beam convolution if requested ---
    # In second_derivative mode, skip spatial smoothing: the R-value is computed
    # from the reconstructed T_bg via inverse RT (per-pixel), and smoothing
    # would artificially reduce the 2nd derivative.
    if mode != 'second_derivative' and (spatial_res_pc is not None or vel_res_kms is not None):
        from scipy.ndimage import gaussian_filter
        sigma_fwhm2sig = 2.0 * np.sqrt(2.0 * np.log(2.0))
        sigma_v = 0.0
        sigma_xy = 0.0
        if vel_res_kms is not None and model_cube.ndim == 3:
            dv = abs(h_obs.get('CDELT3', 200.0)) / 1000.0  # m/s → km/s
            sigma_v = (vel_res_kms / dv) / sigma_fwhm2sig
        if spatial_res_pc is not None:
            sigma_xy = (spatial_res_pc / pixel_scale_pc) / sigma_fwhm2sig
        if model_cube.ndim == 3:
            model_cube = gaussian_filter(model_cube,
                                          sigma=(sigma_v, sigma_xy, sigma_xy),
                                          mode='reflect')
        else:
            model_cube = gaussian_filter(model_cube, sigma_xy,
                                          mode='reflect')
        if verbose:
            parts = []
            if spatial_res_pc is not None:
                parts.append(f'spatial={spatial_res_pc:.3f} pc ({sigma_xy * sigma_fwhm2sig:.1f} pix)')
            if vel_res_kms is not None:
                parts.append(f'velocity={vel_res_kms:.3f} km/s ({sigma_v * sigma_fwhm2sig:.1f} ch)')
            print(f'Smoothing: {", ".join(parts)}')

    # --- Compute final residual (skip if already computed by peak_shell scan) ---
    if final_res is None:
        ny_f, nx_f = obs_hinsa_cube.shape[1], obs_hinsa_cube.shape[2]
        yy_f, xx_f = np.mgrid[0:ny_f, 0:nx_f]
        r_map_f = np.sqrt(((yy_f - center_yx[0]).astype(float))**2 +
                           ((xx_f - center_yx[1]).astype(float))**2)
        r_map_pc_f = r_map_f * pixel_scale_pc
        r_map_pc_f[r_map_pc_f == 0] = pixel_scale_pc
        weight_map_f = 1.0 / r_map_pc_f

        if mode == 'second_derivative':
            _s_fwhm2sig = 2.0 * np.sqrt(2.0 * np.log(2.0))
            _s_v = 0.0
            _s_xy = 0.0
            if vel_res_kms is not None and model_cube.ndim == 3:
                _dv = abs(h_obs.get('CDELT3', 200.0)) / 1000.0
                _s_v = (vel_res_kms / _dv) / _s_fwhm2sig
            if spatial_res_pc is not None:
                _s_xy = (spatial_res_pc / pixel_scale_pc) / _s_fwhm2sig
            if (_s_v > 0 or _s_xy > 0) and model_cube.ndim == 3:
                from scipy.ndimage import gaussian_filter as _gf
                model_cube = _gf(model_cube,
                                 sigma=(_s_v, _s_xy, _s_xy),
                                 mode='reflect')
            dv = abs(cfg.v_max_kms - cfg.v_min_kms) / (cfg.n_v_channels - 1)
            d2 = np.zeros_like(model_cube)
            d2[1:-1] = (model_cube[2:] + model_cube[:-2]
                        - 2.0 * model_cube[1:-1]) / (dv**2)
            weight_map_3d_f = np.broadcast_to(weight_map_f[np.newaxis, :, :],
                                               model_cube.shape).copy()
            if fit_velocity_radius_kms is not None and h_obs is not None:
                _crval3 = h_obs.get('CRVAL3', 0.0)
                _cdelt3 = h_obs.get('CDELT3', 0.0)
                _crpix3 = h_obs.get('CRPIX3', 1.0)
                _nv = model_cube.shape[0]
                _v_arr = (_crval3 + _cdelt3 * (np.arange(_nv) - (_crpix3 - 1))) / 1000.0
                if _v_arr[-1] < _v_arr[0]:
                    _v_arr = _v_arr[::-1]
                _vmask = np.abs(_v_arr - cfg.vlsr_kms) <= fit_velocity_radius_kms
                weight_map_3d_f[~_vmask] = 0.0
            final_res = np.sum(d2**2 * weight_map_3d_f) * dv / np.sum(weight_map_3d_f)
        else:
            # Forward mode: apply velocity mask via NaN before residual_map
            mc_use = model_cube
            obs_use = obs_hinsa_cube
            if fit_velocity_radius_kms is not None and h_obs is not None:
                _crval3 = h_obs.get('CRVAL3', 0.0)
                _cdelt3 = h_obs.get('CDELT3', 0.0)
                _crpix3 = h_obs.get('CRPIX3', 1.0)
                _nv = model_cube.shape[0]
                _v_arr = (_crval3 + _cdelt3 * (np.arange(_nv) - (_crpix3 - 1))) / 1000.0
                if _v_arr[-1] < _v_arr[0]:
                    _v_arr = _v_arr[::-1]
                _vmask = np.abs(_v_arr - cfg.vlsr_kms) > fit_velocity_radius_kms
                mc_use = model_cube.copy(); mc_use[_vmask] = np.nan
                obs_use = obs_hinsa_cube.copy(); obs_use[_vmask] = np.nan
            final_res = residual_map(obs_use, mc_use, weights=weight_map_f)

    # --- Print results (always show peak_shell scan results) ---
    if _peak_shell_scan and all_runs is not None:
        print(f'\n=== peak_shell scan results (sorted by residual) ===')
        for i, r in enumerate(all_runs):
            ps = r['peak_shell']
            res = r['residual']
            marker = ' <-- best' if i == 0 else ''
            print(f'  #{i+1}: peak_shell={ps}  residual={res:.4e}{marker}')

    if verbose:
        print(f'\nFinal residual: {final_res:.4e}')
        if 'peak_shell' in best_params:
            print(f'  peak_shell  = {best_params["peak_shell"]}')
        if 'f_HI_peak' in best_params:
            print(f'  f_HI_peak   = {best_params["f_HI_peak"]:.4f}')
        print('Best-fit params (±1σ):')
        for k, v in best_params.items():
            if k in ('peak_shell', 'f_HI_peak'):
                continue
            std = param_stds.get(k, 0.0)
            if isinstance(v, np.ndarray):
                std_arr = param_stds.get(k, 0.0)
                if isinstance(std_arr, np.ndarray):
                    print(f'  {k:15s} = {np.array2string(v, precision=4)} ± {np.array2string(std_arr, precision=4)}')
                else:
                    print(f'  {k:15s} = {np.array2string(v, precision=4)}')
            else:
                print(f'  {k:15s} = {v:.4f} ± {std:.4f}')

    # --- Save best-fit model FITS ---
    import os as _os
    if output_dir is None:
        output_dir = _os.path.dirname(obs_hinsa_fits) or '.'
    _os.makedirs(output_dir, exist_ok=True)
    basename = _os.path.splitext(_os.path.basename(obs_hinsa_fits))[0]
    fits_path = _os.path.join(output_dir, basename + '_bestfit.fits')

    out_hdr = pyfits.Header()
    out_hdr['SIMPLE'] = True
    out_hdr['BITPIX'] = -32
    out_hdr['NAXIS'] = 3
    out_hdr['NAXIS1'] = model_cube.shape[2]
    out_hdr['NAXIS2'] = model_cube.shape[1]
    out_hdr['NAXIS3'] = model_cube.shape[0]

    # Spatial headers from obs
    out_hdr['CTYPE1'] = h_obs.get('CTYPE1', 'RA--CAR')
    out_hdr['CTYPE2'] = h_obs.get('CTYPE2', 'DEC-CAR')
    out_hdr['CRPIX1'] = 1
    out_hdr['CRPIX2'] = 1
    out_hdr['CRVAL1'] = h_obs.get('CRVAL1', 0.0)
    out_hdr['CRVAL2'] = h_obs.get('CRVAL2', 0.0)
    out_hdr['CDELT1'] = h_obs.get('CDELT1', -0.025)
    out_hdr['CDELT2'] = h_obs.get('CDELT2', 0.025)
    out_hdr['EPOCH'] = 2000.0

    # Velocity headers from obs
    out_hdr['CTYPE3'] = 'VELO-LSR'
    out_hdr['CRPIX3'] = 1
    out_hdr['CRVAL3'] = h_obs.get('CRVAL3', 0.0)
    out_hdr['CDELT3'] = h_obs.get('CDELT3', 200.0)

    # Embed best-fit model parameters (FITS keyword ≤8 chars)
    _fits_key = {
        'rho0': 'MOD_RHO', 'r0': 'MOD_R0', 'alpha': 'MOD_ALP',
        'T0': 'MOD_T0', 'T1': 'MOD_T1', 'rT': 'MOD_RT',
        'f_HI_peak': 'MOD_FHI', 'peak_shell': 'MOD_PSH',
        'f_ff': 'MOD_FFF', 'turb_kms': 'MOD_TURB',
        'v_offset': 'MOD_VOFF', 'v_rot_kms': 'MOD_VROT',
        'rot_pa_deg': 'MOD_RPA', 'multipliers': 'MOD_MLT',
        'f_HI': 'MOD_FHI',
    }
    for k, v in best_params.items():
        if isinstance(v, np.ndarray):
            base = _fits_key.get(k, f'MOD_{k[:5].upper()}')
            for i, val in enumerate(v):
                out_hdr[f'{base}{i+1}'] = (float(val), f'{k}[{i+1}]')
        else:
            key = _fits_key.get(k, f'MOD_{k[:5].upper()}')
            out_hdr[key] = (float(v), k)

    pyfits.writeto(fits_path, model_cube.astype(np.float32), out_hdr, overwrite=True)
    if verbose:
        print(f'\nSaved model FITS: {fits_path}')

    # --- Save diagnostic PNG ---
    png_path = _os.path.join(output_dir, basename + '_bestfit.png')
    _save_fit_diagnostic_png(
        png_path, cfg, best_params, model_cube,
        obs_hinsa_cube, obs_background_map, center_yx,
        pixel_scale_pc, vlsr_kms, h_obs,
        mode=mode,
        fit_velocity_radius_kms=fit_velocity_radius_kms,
    )
    if verbose:
        print(f'Saved diagnostic: {png_path}')

    # --- Save grid spectrum PNG ---
    grid_png_path = _os.path.join(output_dir, basename + '_grid_spectra.png')
    _save_grid_spectrum_png(
        grid_png_path, obs_hinsa_cube, model_cube, obs_background_map,
        center_yx, vlsr_kms, h_obs, verbose=verbose, mode=mode,
    )
    if verbose:
        print(f'Saved grid spectra: {grid_png_path}')

    # --- Save comprehensive fit_result.npz ---
    npz_path = _os.path.join(output_dir, basename + '_fit_result.npz')
    npz_save_dict = {
        **{k: v for k, v in best_params.items()},
        **{f'std_{k}': v for k, v in param_stds.items()},
        'residual': final_res,
        'model_cube': model_cube,
        'obs_cube': obs_hinsa_cube,
        'bg_cube': obs_background_map if obs_background_map is not None else np.array([]),
        'velo_kms': v_axis_kms if obs_hinsa_cube.ndim == 3 else np.sort((crval3 + np.arange(obs_hinsa_cube.shape[0]) * cdelt3) / 1000.0),
        'center_yx': np.array(center_yx),
        'pixel_scale_pc': pixel_scale_pc,
        'vlsr_kms': vlsr_kms,
        'R_out_pc': R_out_pc,
        'distance_pc': distance_pc,
        'n_shells': n_shells,
        'spatial_res_pc': spatial_res_pc if spatial_res_pc is not None else 0.0,
        'vel_res_kms': vel_res_kms if vel_res_kms is not None else 0.0,
        'obs_hinsa_fits': obs_hinsa_fits,
        'obs_background_fits': obs_background_fits if obs_background_fits else '',
        'mode': mode,
    }
    if all_runs is not None:
        scan_peaks = np.array([r['peak_shell'] for r in all_runs])
        scan_residuals = np.array([r['residual'] for r in all_runs])
        npz_save_dict['peak_shell_scan_peaks'] = scan_peaks
        npz_save_dict['peak_shell_scan_residuals'] = scan_residuals
    if relative_errors is not None:
        for pk, re_val in relative_errors.items():
            npz_save_dict[f'relative_error_{pk}'] = re_val
        # Also save true params for convenience
        for pk, tv in true_params.items():
            npz_save_dict[f'true_{pk}'] = tv
    np.savez_compressed(npz_path, **npz_save_dict)
    if verbose:
        print(f'Saved fit result:  {npz_path}')

    return {
        'best_params': best_params,
        'param_stds': param_stds,
        'relative_errors': relative_errors,
        'model_cube': model_cube,
        'history': history,
        'cfg': cfg,
        'residual': final_res,
        'fits_path': fits_path,
        'png_path': png_path,
        'grid_png_path': grid_png_path,
        'npz_path': npz_path,
        'all_runs': all_runs,
    }


def _save_fit_diagnostic_png(png_path, cfg, params, model_cube,
                              obs_cube, bg_cube, center_yx,
                              pixel_scale_pc, vlsr_kms, obs_hdr,
                              mode='forward',
                              fit_velocity_radius_kms=None):
    """6-panel diagnostic: obs vs best-fit model."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .profiles import density_plummer, temperature_plummer, abundance_profile_111n
    from .models import compute_radial_profiles

    yc, xc = center_yx
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Velocity axis (ensure ascending)
    crval3 = obs_hdr.get('CRVAL3', 0.0)
    cdelt3 = obs_hdr.get('CDELT3', 200.0)
    crpix3 = obs_hdr.get('CRPIX3', 1.0)
    n_v = obs_cube.shape[0]
    velo_kms = (crval3 + cdelt3 * (np.arange(n_v) - (crpix3 - 1))) / 1000.0
    # Data cubes are already pre-flipped to ascending by fit_hinsa_model;
    # only fix the velocity axis order if needed.
    if velo_kms[-1] < velo_kms[0]:
        velo_kms = velo_kms[::-1]

    r_mid = cfg.r_mid

    # 1. Density profile
    n_H = density_plummer(r_mid, params['rho0'], params['r0'], params['alpha'])
    axes[0, 0].loglog(r_mid, n_H, 'o-', color='C0')
    axes[0, 0].set_xlabel('r (pc)')
    axes[0, 0].set_ylabel('n_H (cm$^{-3}$)')
    axes[0, 0].set_title('Density')

    # 2. Temperature profile
    T = temperature_plummer(r_mid, params['T0'], params['T1'], params['rT'])
    axes[0, 1].plot(r_mid, T, 'o-', color='C1')
    axes[0, 1].set_xlabel('r (pc)')
    axes[0, 1].set_ylabel('T_spin (K)')
    axes[0, 1].set_title('Temperature')

    # 3. Abundance profile
    if 'f_HI' in params:
        f_HI = np.asarray(params['f_HI'], dtype=float)
    else:
        f_HI = abundance_profile_111n(cfg.n_shells, params.get('peak_shell', 1),
                                       params.get('multipliers', np.ones(cfg.n_shells-1)),
                                       f_HI_peak=params.get('f_HI_peak', 1.0))
    axes[0, 2].plot(r_mid, f_HI, 'o-', color='C2')
    axes[0, 2].set_xlabel('r (pc)')
    axes[0, 2].set_ylabel('f_HI')
    axes[0, 2].set_title('HI Abundance')

    # 4. Moment 0: obs vs model — color range from data min/max
    v_center = vlsr_kms + params.get('v_offset', 0.0)
    dv = np.abs(velo_kms - v_center)
    vmask = dv <= 1.0
    if np.any(vmask):
        mom0_obs = np.sum(obs_cube[vmask], axis=0) * np.abs(cdelt3 / 1000.0)
        mom0_mod = np.sum(model_cube[vmask], axis=0) * np.abs(cdelt3 / 1000.0)
    else:
        mom0_obs = np.zeros(obs_cube.shape[1:])
        mom0_mod = np.zeros(model_cube.shape[1:])
    vmin_m0 = min(np.nanmin(mom0_obs), np.nanmin(mom0_mod))
    vmax_m0 = max(np.nanmax(mom0_obs), np.nanmax(mom0_mod), 0.01)
    im = axes[1, 0].imshow(mom0_obs, origin='lower', cmap='RdYlBu_r',
                            vmin=vmin_m0, vmax=vmax_m0)
    axes[1, 0].plot(xc, yc, 'w+', ms=10, mew=1.5)
    axes[1, 0].set_title(f'Moment 0 obs ({v_center:.1f}±1 km/s)')
    fig.colorbar(im, ax=axes[1, 0], label='K km/s')

    # 5. Cold cloud HI column density map (circularly symmetric)
    n_H = density_plummer(r_mid, params['rho0'], params['r0'], params['alpha'])
    if 'f_HI' in params:
        f_HI = np.asarray(params['f_HI'], dtype=float)
    else:
        f_HI = abundance_profile_111n(cfg.n_shells, params.get('peak_shell', 1),
                                       params.get('multipliers', np.ones(cfg.n_shells-1)),
                                       f_HI_peak=params.get('f_HI_peak', 1.0))
    pc_to_cm = 3.086e18
    dr = cfg.r_outer - cfg.r_inner  # shell thickness in pc
    nHI_shell = n_H * f_HI * dr * pc_to_cm  # column density per shell (cm^-2)

    # Project onto 2D: NHI(R) = 2 * integral from R to R_out of nHI(r)/sqrt(r^2 - R^2) dr
    r_s = r_mid
    n_fine = 500
    R_grid_fine = np.linspace(0, r_s[-1] * 0.999, n_fine)
    NHI_1d = np.zeros_like(R_grid_fine)
    for iR, R in enumerate(R_grid_fine):
        mask = r_s > R
        r_use = r_s[mask]
        n_use = nHI_shell[mask]
        sqrt_arg = r_use**2 - R**2
        sqrt_arg[sqrt_arg <= 0] = 1e-30
        integrand = n_use / np.sqrt(sqrt_arg)
        NHI_1d[iR] = 2.0 * np.trapz(integrand, r_use)

    # Build 2D map: compute on fine pixel grid then downsample
    ny, nx = obs_cube.shape[1], obs_cube.shape[2]
    # Use 10x oversampled grid for smooth rendering
    oversample = 10
    yy_f, xx_f = np.mgrid[0:ny*oversample, 0:nx*oversample]
    xc_f = xc * oversample + oversample // 2
    yc_f = yc * oversample + oversample // 2
    R_pix_f = np.sqrt((xx_f - xc_f)**2 + (yy_f - yc_f)**2) * pixel_scale_pc / oversample
    NHI_fine = np.interp(R_pix_f.ravel(), R_grid_fine, NHI_1d).reshape(ny*oversample, nx*oversample)
    # Downsample by averaging
    NHI_map = NHI_fine.reshape(ny, oversample, nx, oversample).mean(axis=(1, 3))

    from matplotlib.colors import LogNorm
    NHI_pos = NHI_map[NHI_map > 0]
    vmin_nhi = np.nanmin(NHI_pos) * 0.5 if len(NHI_pos) > 0 else 1e18
    vmax_nhi = np.nanmax(NHI_pos) * 1.5 if len(NHI_pos) > 0 else 1e22
    im = axes[1, 1].imshow(NHI_map, origin='lower', cmap='YlOrRd',
                            norm=LogNorm(vmin=vmin_nhi, vmax=vmax_nhi))
    axes[1, 1].plot(xc, yc, 'b+', ms=10, mew=1.5)
    axes[1, 1].set_title('Cold Cloud N(HI) (cm$^{-2}$)')
    fig.colorbar(im, ax=axes[1, 1], label='cm$^{-2}$')

    # 6. Spectrum at center: obs vs model
    spec_obs = obs_cube[:, yc, xc].astype(float)
    if mode == 'second_derivative':
        spec_mod = model_cube[:, yc, xc].astype(float)
        spec_bg = np.full(n_v, 30.0)  # No background in second_derivative mode
        axes[1, 2].plot(velo_kms, spec_obs, 'k', lw=1.5, label='Obs')
        axes[1, 2].plot(velo_kms, spec_mod, 'r--', lw=1.5, label='Reconstructed T_bg')
    else:
        spec_mod = model_cube[:, yc, xc].astype(float)
        spec_bg = bg_cube[:, yc, xc].astype(float) if bg_cube is not None else np.full(n_v, 30.0)
        axes[1, 2].plot(velo_kms, spec_bg, 'gray', alpha=0.4, label='Background')
        axes[1, 2].plot(velo_kms, spec_obs, 'k', lw=1.5, label='Obs')
        axes[1, 2].plot(velo_kms, spec_mod, 'r--', lw=1.5, label='Best-fit model')
    axes[1, 2].axvline(vlsr_kms, color='blue', ls=':', alpha=0.5)
    if fit_velocity_radius_kms is not None:
        v_lo = vlsr_kms - fit_velocity_radius_kms
        v_hi = vlsr_kms + fit_velocity_radius_kms
        axes[1, 2].axvspan(v_lo, v_hi, alpha=0.15, color='blue',
                           label=f'Fit range (±{fit_velocity_radius_kms:.1f})')
    axes[1, 2].set_xlabel('v (km/s)')
    axes[1, 2].set_ylabel('T_B (K)')
    axes[1, 2].set_title(f'Spectrum ({yc},{xc})')
    axes[1, 2].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _save_grid_spectrum_png(png_path, obs_cube, model_cube, bg_cube,
                             center_yx, vlsr_kms, obs_hdr, verbose=False,
                             mode='forward'):
    """Single Moment 0 background with grid of spectra overlaid at spatial positions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    yc, xc = center_yx
    n_v, ny, nx = obs_cube.shape

    # Velocity axis (ensure ascending)
    crval3 = obs_hdr.get('CRVAL3', 0.0)
    cdelt3 = obs_hdr.get('CDELT3', 200.0)
    crpix3 = obs_hdr.get('CRPIX3', 1.0)
    velo_kms = (crval3 + cdelt3 * (np.arange(n_v) - (crpix3 - 1))) / 1000.0
    # Data cubes are already pre-flipped to ascending by fit_hinsa_model;
    # only fix the velocity axis order if needed.
    if velo_kms[-1] < velo_kms[0]:
        velo_kms = velo_kms[::-1]

    # Moment 0 map (±1 km/s of vlsr) as single background
    dv = np.abs(velo_kms - vlsr_kms)
    vmask_m0 = dv <= 1.0
    if np.any(vmask_m0):
        moment0 = np.sum(obs_cube[vmask_m0], axis=0) * np.abs(cdelt3 / 1000.0)
    else:
        moment0 = np.mean(obs_cube, axis=0)
    vmin_m0, vmax_m0 = np.nanpercentile(moment0, [5, 95])

    # Grid spacing (every 3 pixels)
    grid_step = 3
    gy_start = (yc % grid_step) + grid_step // 2
    gx_start = (xc % grid_step) + grid_step // 2
    gy_positions = list(range(gy_start, ny, grid_step))
    gx_positions = list(range(gx_start, nx, grid_step))

    # Velocity window: center on data range midpoint or vlsr, ±5 km/s
    v_mid = vlsr_kms
    # If vlsr is near data edge, center on data midpoint instead
    if vlsr_kms < velo_kms[0] + 3 or vlsr_kms > velo_kms[-1] - 3:
        v_mid = (velo_kms[0] + velo_kms[-1]) / 2.0
    v_lo = v_mid - 5.0
    v_hi = v_mid + 5.0
    vmask_win = (velo_kms >= v_lo) & (velo_kms <= v_hi)
    if np.sum(vmask_win) < 3:
        vmask_win = np.ones(n_v, dtype=bool)
        v_lo, v_hi = velo_kms[0], velo_kms[-1]
    v_win = velo_kms[vmask_win]

    # Global y-range: use percentile within vlsr ± 3 km/s (tighter than x-axis)
    vmask_y = (velo_kms >= vlsr_kms - 3.0) & (velo_kms <= vlsr_kms + 3.0)
    if np.sum(vmask_y) < 3:
        vmask_y = vmask_win
    all_obs = obs_cube[vmask_y].ravel()
    all_mod = model_cube[vmask_y].ravel()
    all_spec = np.concatenate([all_obs, all_mod])
    ymin_global = np.nanpercentile(all_spec, 1)
    ymax_global = np.nanpercentile(all_spec, 99)
    y_margin = (ymax_global - ymin_global) * 0.1
    ymin_global -= y_margin
    ymax_global += y_margin

    # --- Single axes, full background + overlaid spectra ---
    fig, ax = plt.subplots(figsize=(12, 12))

    # Background: Moment 0 map covering full spatial extent
    ax.imshow(moment0, origin='lower', cmap='RdYlBu_r',
              vmin=vmin_m0, vmax=vmax_m0, alpha=0.3,
              extent=[0, nx, 0, ny], aspect='auto')

    # Spectral spatial extents
    spec_x_span = grid_step * 0.8
    spec_y_height = grid_step * 0.6

    for gy in gy_positions:
        for gx in gx_positions:
            # Average over 3x3 pixels
            j_lo = max(0, gy - 1)
            j_hi = min(ny, gy + 2)
            i_lo = max(0, gx - 1)
            i_hi = min(nx, gx + 2)
            spec_obs = np.mean(obs_cube[:, j_lo:j_hi, i_lo:i_hi], axis=(1, 2))
            spec_mod = np.mean(model_cube[:, j_lo:j_hi, i_lo:i_hi], axis=(1, 2))
            s_obs = spec_obs[vmask_win]
            s_mod = spec_mod[vmask_win]

            # Map velocity → spatial x (left=small v, right=large v)
            x_coords = gx - spec_x_span / 2 + (v_win - v_win[0]) / (v_win[-1] - v_win[0]) * spec_x_span
            # Map intensity → spatial y
            y_base = gy - spec_y_height / 2
            y_obs = y_base + (s_obs - ymin_global) / (ymax_global - ymin_global) * spec_y_height
            y_mod = y_base + (s_mod - ymin_global) / (ymax_global - ymin_global) * spec_y_height

            ax.plot(x_coords, y_obs, '-', color='#3366cc', lw=1.2, alpha=0.9)
            ax.plot(x_coords, y_mod, '-', color='#cc3333', lw=1.2, alpha=0.9)

            # White dot at grid center
            ax.plot(gx, gy, 'o', color='white', ms=3, alpha=0.7, zorder=5)

    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_xlabel('Pixel X', fontsize=12)
    ax.set_ylabel('Pixel Y', fontsize=12)
    if mode == 'second_derivative':
        ax.set_title(f'Observed (blue) vs Reconstructed T_bg (red) — {vlsr_kms:.1f} km/s ±5 km/s, Moment 0 bg',
                     fontsize=13, fontweight='bold')
    else:
        ax.set_title(f'Observed (blue) vs Model (red) — {vlsr_kms:.1f} km/s ±5 km/s, Moment 0 bg',
                     fontsize=13, fontweight='bold')
    ax.set_aspect('auto')

    plt.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def reload_fit_result(npz_path, output_dir=None, verbose=True, mode='forward'):
    """Load a fit_result.npz and regenerate diagnostic PNGs + FITS.

    Supports two npz formats:
      - New (from fit_hinsa_model): has model_cube, obs_cube, bg_cube, etc.
      - Old (manual save): has params + residual + model_map; FITS files
        searched from the same directory.

    Parameters
    ----------
    npz_path : str
        Path to the .npz file saved by fit_hinsa_model.
    output_dir : str or None
        Where to save regenerated files. If None, same directory as npz.
    verbose : bool
    mode : str
        'forward' — standard forward modeling (default).
        'second_derivative' — Liu Method 2: no background needed.

    Returns
    -------
    result : dict with all loaded data and paths.
    """
    import os as _os
    from astropy.io import fits as pyfits

    data = np.load(npz_path, allow_pickle=True)
    npz_dir = _os.path.dirname(_os.path.abspath(npz_path))

    # --- Extract best-fit params ---
    param_keys = ['rho0', 'r0', 'alpha', 'T0', 'T1', 'rT',
                  'peak_shell', 'f_HI_peak', 'f_ff', 'turb_kms',
                  'v_offset', 'v_rot_kms', 'rot_pa_deg']
    best_params = {}
    param_stds = {}
    for k in param_keys:
        if k in data:
            val = float(data[k])
            if k == 'peak_shell':
                val = int(round(val))
            best_params[k] = val
            std_key = f'std_{k}'
            param_stds[k] = float(data[std_key]) if std_key in data else 0.0
    if 'f_HI' in data:
        best_params['f_HI'] = data['f_HI']
        param_stds['f_HI'] = data['std_f_HI'] if 'std_f_HI' in data else np.zeros_like(data['f_HI'])
    if 'multipliers' in data:
        best_params['multipliers'] = data['multipliers']
        param_stds['multipliers'] = data['std_multipliers'] if 'std_multipliers' in data else np.zeros_like(data['multipliers'])

    residual = float(data['residual'])

    # --- Detect npz format ---
    has_cubes = 'model_cube' in data and 'obs_cube' in data

    if has_cubes:
        # New format: everything in npz
        model_cube = data['model_cube']
        obs_cube = data['obs_cube']
        bg_cube = data['bg_cube'] if data['bg_cube'].size > 0 else None
        velo_kms = data['velo_kms']
        center_yx = tuple(data['center_yx'].astype(int))
        pixel_scale_pc = float(data['pixel_scale_pc'])
        vlsr_kms = float(data['vlsr_kms'])
        R_out_pc = float(data['R_out_pc'])
        distance_pc = float(data['distance_pc'])
        n_shells = int(data['n_shells'])
        spatial_res_pc = float(data['spatial_res_pc']) if float(data['spatial_res_pc']) > 0 else None
        vel_res_kms = float(data['vel_res_kms']) if float(data['vel_res_kms']) > 0 else None
        obs_hinsa_fits = str(data['obs_hinsa_fits'])
        obs_background_fits = str(data['obs_background_fits']) if data['obs_background_fits'] else None
        # Read mode from npz if saved (default to 'forward' for backward compatibility)
        if 'mode' in data:
            mode = str(data['mode'])
    else:
        # Old format: params + residual + model_map, possibly with metadata keys
        model_cube = data['model_map'] if 'model_map' in data else None

        # Read metadata from npz if present (user may have saved them)
        if 'pixel_scale_pc' in data:
            pixel_scale_pc = float(data['pixel_scale_pc'])
        else:
            pixel_scale_pc = 0.0
        if 'R_out_pc' in data:
            R_out_pc = float(data['R_out_pc'])
        else:
            R_out_pc = 0.0
        if 'distance_pc' in data:
            distance_pc = float(data['distance_pc'])
        else:
            distance_pc = 0.0
        if 'n_shells' in data:
            n_shells = int(data['n_shells'])
        else:
            n_shells = 9
        if 'spatial_res_pc' in data:
            spatial_res_pc = float(data['spatial_res_pc']) if float(data['spatial_res_pc']) > 0 else None
        else:
            spatial_res_pc = None
        if 'vel_res_kms' in data:
            vel_res_kms = float(data['vel_res_kms']) if float(data['vel_res_kms']) > 0 else None
        else:
            vel_res_kms = None
        if 'velo_kms' in data:
            velo_kms = data['velo_kms']
        else:
            velo_kms = None
        if 'center_yx' in data:
            center_yx = tuple(data['center_yx'].astype(int))
        else:
            center_yx = None

        # Search FITS in npz directory — find 3D HI cube and background
        obs_cube, bg_cube, obs_hinsa_fits, obs_background_fits = None, None, None, None
        candidates_obs = []
        candidates_bg = []
        for f in sorted(_os.listdir(npz_dir)):
            fp = _os.path.join(npz_dir, f)
            if not f.endswith('.fits'):
                continue
            try:
                shape = pyfits.getdata(fp).shape
            except Exception:
                continue
            if len(shape) != 3:
                continue
            if '_HI_background' in f:
                candidates_bg.append((fp, shape))
            elif '_HI_cube' in f and '_bestfit' not in f:
                candidates_obs.append((fp, shape))

        # Prefer cube matching model_cube dimensions
        if model_cube is not None:
            target_shape = model_cube.shape
            for fp, shape in candidates_obs:
                if shape == target_shape:
                    obs_cube = pyfits.getdata(fp)
                    obs_hinsa_fits = fp
                    break
            for fp, shape in candidates_bg:
                if shape == target_shape:
                    bg_cube = pyfits.getdata(fp)
                    obs_background_fits = fp
                    break

        # Fallback: just take first match
        if obs_cube is None and candidates_obs:
            fp, shape = candidates_obs[0]
            obs_cube = pyfits.getdata(fp)
            obs_hinsa_fits = fp
        if bg_cube is None and candidates_bg:
            fp, shape = candidates_bg[0]
            bg_cube = pyfits.getdata(fp)
            obs_background_fits = fp

        if obs_hinsa_fits is None or obs_cube is None:
            raise FileNotFoundError(
                f'No 3D *_HI_cube.fits found in {npz_dir}. '
                'Cannot regenerate obs/model plots without the observed cube.')

        h_obs = pyfits.getheader(obs_hinsa_fits)
        crval3 = h_obs.get('CRVAL3', 0.0)
        cdelt3 = h_obs.get('CDELT3', 200.0)
        crpix3 = h_obs.get('CRPIX3', 1.0)
        if velo_kms is None:
            velo_kms = (crval3 + cdelt3 * (np.arange(obs_cube.shape[0]) - (crpix3 - 1))) / 1000.0
            if velo_kms[-1] < velo_kms[0]:
                velo_kms = velo_kms[::-1]
        if 'vlsr_kms' in data:
            vlsr_kms = float(data['vlsr_kms'])
        else:
            vlsr_kms = (velo_kms[0] + velo_kms[-1]) / 2.0
        if center_yx is None:
            center_yx = (obs_cube.shape[1] // 2, obs_cube.shape[2] // 2)

    if output_dir is None:
        output_dir = _os.path.dirname(npz_path) or '.'
    _os.makedirs(output_dir, exist_ok=True)
    basename = _os.path.splitext(_os.path.basename(npz_path))[0].replace('_fit_result', '')

    h_obs = pyfits.getheader(obs_hinsa_fits)

    if verbose:
        print(f'Loaded: {npz_path}')
        print(f'  Format: {"new" if has_cubes else "old (reconstructed from FITS)"}')
        print(f'  Best-fit params:')
        for k, v in best_params.items():
            std = param_stds.get(k, 0)
            v_str = np.array2string(v, precision=4) if isinstance(v, np.ndarray) else f'{v:.4f}'
            std_str = np.array2string(std, precision=4) if isinstance(std, np.ndarray) else f'{std:.4f}'
            print(f'    {k:15s} = {v_str} ± {std_str}')
        print(f'  Residual: {residual:.4e}')
        if model_cube is not None:
            print(f'  Model cube: {model_cube.shape}')

    # --- Regenerate FITS ---
    fits_path = _os.path.join(output_dir, basename + '_bestfit.fits')
    if model_cube is not None:
        out_hdr = pyfits.Header()
        out_hdr['SIMPLE'] = True
        out_hdr['BITPIX'] = -32
        out_hdr['NAXIS'] = 3
        out_hdr['NAXIS1'] = model_cube.shape[2]
        out_hdr['NAXIS2'] = model_cube.shape[1]
        out_hdr['NAXIS3'] = model_cube.shape[0]
        out_hdr['CTYPE1'] = h_obs.get('CTYPE1', 'RA--CAR')
        out_hdr['CTYPE2'] = h_obs.get('CTYPE2', 'DEC-CAR')
        out_hdr['CRPIX1'] = 1; out_hdr['CRPIX2'] = 1
        out_hdr['CRVAL1'] = h_obs.get('CRVAL1', 0.0)
        out_hdr['CRVAL2'] = h_obs.get('CRVAL2', 0.0)
        out_hdr['CDELT1'] = h_obs.get('CDELT1', -0.025)
        out_hdr['CDELT2'] = h_obs.get('CDELT2', 0.025)
        out_hdr['EPOCH'] = 2000.0
        out_hdr['CTYPE3'] = 'VELO-LSR'
        out_hdr['CRPIX3'] = 1
        out_hdr['CRVAL3'] = h_obs.get('CRVAL3', 0.0)
        out_hdr['CDELT3'] = h_obs.get('CDELT3', 200.0)

        _fits_key = {
            'rho0': 'MOD_RHO', 'r0': 'MOD_R0', 'alpha': 'MOD_ALP',
            'T0': 'MOD_T0', 'T1': 'MOD_T1', 'rT': 'MOD_RT',
            'f_HI_peak': 'MOD_FHI', 'peak_shell': 'MOD_PSH',
            'f_ff': 'MOD_FFF', 'turb_kms': 'MOD_TURB',
            'v_offset': 'MOD_VOFF', 'v_rot_kms': 'MOD_VROT',
            'rot_pa_deg': 'MOD_RPA', 'multipliers': 'MOD_MLT',
            'f_HI': 'MOD_FHI',
        }
        for k, v in best_params.items():
            if isinstance(v, np.ndarray):
                base = _fits_key.get(k, f'MOD_{k[:5].upper()}')
                for i, val in enumerate(v):
                    out_hdr[f'{base}{i+1}'] = (float(val), f'{k}[{i+1}]')
            else:
                key = _fits_key.get(k, f'MOD_{k[:5].upper()}')
                out_hdr[key] = (float(v), k)

        pyfits.writeto(fits_path, model_cube.astype(np.float32), out_hdr, overwrite=True)
        if verbose:
            print(f'\nSaved model FITS: {fits_path}')
    else:
        if verbose:
            print(f'\nNo model_cube in npz — skipping FITS output')

    # --- Regenerate 6-panel diagnostic PNG ---
    png_path = _os.path.join(output_dir, basename + '_bestfit.png')
    if obs_cube is not None and model_cube is not None:
        from .config import Config
        cfg = Config(n_shells=n_shells, R_out_pc=R_out_pc, vlsr_kms=vlsr_kms,
                     v_min_kms=float(velo_kms[0]), v_max_kms=float(velo_kms[-1]),
                     n_v_channels=len(velo_kms), distance_pc=distance_pc)
        _save_fit_diagnostic_png(
            png_path, cfg, best_params, model_cube,
            obs_cube, bg_cube, center_yx,
            pixel_scale_pc, vlsr_kms, h_obs,
            mode=mode,
        )
        if verbose:
            print(f'Saved diagnostic: {png_path}')
    else:
        if verbose:
            print(f'No obs_cube — skipping diagnostic PNG')

    # --- Regenerate grid spectrum PNG ---
    grid_png_path = _os.path.join(output_dir, basename + '_grid_spectra.png')
    if obs_cube is not None and model_cube is not None:
        _save_grid_spectrum_png(
            grid_png_path, obs_cube, model_cube, bg_cube,
            center_yx, vlsr_kms, h_obs, verbose=verbose, mode=mode,
        )
        if verbose:
            print(f'Saved grid spectra: {grid_png_path}')
    else:
        grid_png_path = None

    return {
        'best_params': best_params,
        'param_stds': param_stds,
        'model_cube': model_cube,
        'obs_cube': obs_cube,
        'bg_cube': bg_cube,
        'velo_kms': velo_kms,
        'center_yx': center_yx,
        'pixel_scale_pc': pixel_scale_pc,
        'vlsr_kms': vlsr_kms,
        'R_out_pc': R_out_pc,
        'distance_pc': distance_pc,
        'residual': residual,
        'fits_path': fits_path if model_cube is not None else None,
        'png_path': png_path if obs_cube is not None else None,
        'grid_png_path': grid_png_path,
    }
