"""
Optimization engine: profile parameters → best fit to observed HINSA map.
"""

import numpy as np


def fit_hinspheres(cfg, obs_hinsa_map, T_HI_true_map=None,
                    method='CMA-ES', maxiter=5000, n_jobs=4,
                    params_init=None, bounds=None,
                    max_gen=None, popsize=None, seed=None, verbose=False):
    """Fit spherical HINSA model to observed absorption map.

    Parameters
    ----------
    cfg : Config
    obs_hinsa_map : 2D array
        Observed HINSA peak absorption map (K).
    T_HI_true_map : 2D array or None
        Recovered unabsorbed HI brightness (K) — used as RT background.
    method : str
        'CMA-ES', 'DE', 'Powell', or 'MCMC'.
    maxiter : int
    n_jobs : int
    params_init : dict or None
    bounds : dict or None

    Returns
    -------
    best_params : dict
    result_obj : OptimizeResult-like
    """
    from .models import build_synthetic_hinsa, residual_map

    # --- Build parameter vector ---
    param_keys = ['rho0', 'r0', 'alpha', 'T0', 'T1', 'rT',
                  'peak_shell', 'multipliers', 'f_ff', 'turb_kms', 'v_offset',
                  'v_rot_kms', 'rot_pa_deg']

    if bounds is None:
        bounds = cfg.bounds_pc

    if params_init is None:
        params_init = {
            'rho0': 5000.0,
            'r0': 0.04,
            'alpha': 2.0,
            'T0': 10.0,
            'T1': 40.0,
            'rT': 0.05,
            'peak_shell': 5,
            'multipliers': np.ones(cfg.n_shells - 1) * 0.7,
            'f_ff': 0.1,
            'turb_kms': 0.2,
            'v_offset': 0.0,
            'v_rot_kms': 0.0,
            'rot_pa_deg': 0.0,
        }

    # CMA-ES optimization
    if method == 'CMA-ES':
        if max_gen is not None:
            maxiter = max_gen
        return _optimize_cmaes(cfg, obs_hinsa_map, T_HI_true_map,
                                params_init, bounds, maxiter, n_jobs,
                                popsize=popsize, seed=seed, verbose=verbose)
    else:
        raise ValueError(f"Unknown method: {method}")


def _optimize_cmaes(cfg, obs_map, T_HI_true, params_init,
                     bounds, maxiter, n_jobs, popsize=None,
                     seed=None, verbose=False):
    """CMA-ES optimizer with ask/tell + joblib parallel evaluation."""
    try:
        import cma
    except ImportError:
        raise ImportError("CMA-ES requires the `cma` package: pip install cma")

    from .models import build_synthetic_hinsa, residual_map
    from joblib import Parallel, delayed

    param_keys, x0, low, high = _params_to_flat(params_init, bounds, cfg)
    n_dim = len(x0)
    if popsize is None:
        popsize = max(6, 4 + int(3 * np.log(n_dim)))

    options = {
        'bounds': [low, high],
        'maxiter': maxiter,
        'popsize': popsize,
        'verb_disp': int(verbose),
        'verb_log': 0,
        'tolx': 1e-4,
        'tolfun': 1e-4,
    }
    if seed is not None:
        options['seed'] = seed
    if verbose:
        options['verbose'] = -2

    es = cma.CMAEvolutionStrategy(x0, 0.2, options)

    best_fun = np.inf
    best_x = None
    generation = 0

    while not es.stop():
        solutions = es.ask()  # candidate solutions
        # Parallel evaluation across candidates (outer) × pixels (inner via n_jobs)
        def _eval_one(x):
            params = _flat_to_params(x, param_keys, cfg)
            try:
                m = build_synthetic_hinsa(cfg, params, T_HI_true, n_jobs=1)
                return float(residual_map(obs_map, m))
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

    if best_x is None:
        best_x = es.result.xbest
    params_best = _flat_to_params(best_x, param_keys, cfg)
    return params_best, es.result


def _params_to_flat(params_init, bounds, cfg):
    """Convert parameter dict to flat normalized vector."""
    keys = []
    vals = []
    lows = []
    highs = []

    # rho0 — log scale
    keys.append('rho0')
    vals.append(np.log10(params_init['rho0']))
    lows.append(2.0)    # 100 cm^-3
    highs.append(5.0)   # 100000 cm^-3

    # r0
    keys.append('r0')
    vals.append(params_init['r0'])
    lows.append(bounds['r0'][0])
    highs.append(bounds['r0'][1])

    # alpha
    keys.append('alpha')
    vals.append(params_init['alpha'])
    lows.append(bounds['alpha'][0])
    highs.append(bounds['alpha'][1])

    # T0
    keys.append('T0')
    vals.append(params_init['T0'])
    lows.append(bounds['T0'][0])
    highs.append(bounds['T0'][1])

    # T1
    keys.append('T1')
    vals.append(params_init['T1'])
    lows.append(bounds['T1'][0])
    highs.append(bounds['T1'][1])

    # rT
    keys.append('rT')
    vals.append(params_init['rT'])
    lows.append(bounds['rT'][0])
    highs.append(bounds['rT'][1])

    # peak_shell (integer, 1-indexed)
    keys.append('peak_shell')
    vals.append(float(params_init['peak_shell']))
    lows.append(float(bounds['peak_shell'][0]))
    highs.append(float(bounds['peak_shell'][1]))

    # multipliers (n_shells - 1)
    for i in range(cfg.n_shells - 1):
        keys.append(f'mult_{i}')
        m = params_init['multipliers'][i] if i < len(params_init['multipliers']) else 0.7
        vals.append(m)
        lows.append(bounds['multipliers'][0])
        highs.append(bounds['multipliers'][1])

    # f_ff
    keys.append('f_ff')
    vals.append(params_init['f_ff'])
    lows.append(bounds['f_ff'][0])
    highs.append(bounds['f_ff'][1])

    # turb_kms
    keys.append('turb_kms')
    vals.append(params_init['turb_kms'])
    lows.append(bounds['turb_kms'][0])
    highs.append(bounds['turb_kms'][1])

    # v_offset
    keys.append('v_offset')
    vals.append(params_init['v_offset'])
    lows.append(bounds['v_offset'][0])
    highs.append(bounds['v_offset'][1])

    # v_rot_kms
    keys.append('v_rot_kms')
    vals.append(params_init['v_rot_kms'])
    lows.append(bounds['v_rot_kms'][0])
    highs.append(bounds['v_rot_kms'][1])

    # rot_pa_deg
    keys.append('rot_pa_deg')
    vals.append(params_init['rot_pa_deg'])
    lows.append(bounds['rot_pa_deg'][0])
    highs.append(bounds['rot_pa_deg'][1])

    return keys, np.array(vals), np.array(lows), np.array(highs)


def _flat_to_params(x, keys, cfg):
    """Convert flat vector back to parameter dict."""
    p = {}
    idx = 0

    p['rho0'] = 10.0 ** x[idx]; idx += 1
    p['r0'] = x[idx]; idx += 1
    p['alpha'] = x[idx]; idx += 1
    p['T0'] = x[idx]; idx += 1
    p['T1'] = x[idx]; idx += 1
    p['rT'] = x[idx]; idx += 1
    p['peak_shell'] = int(round(x[idx])); idx += 1

    n_mult = cfg.n_shells - 1
    p['multipliers'] = np.clip(x[idx:idx + n_mult], 0.1, 1.0)
    idx += n_mult

    p['f_ff'] = x[idx]; idx += 1
    p['turb_kms'] = x[idx]; idx += 1
    p['v_offset'] = x[idx]; idx += 1
    p['v_rot_kms'] = x[idx]; idx += 1
    p['rot_pa_deg'] = x[idx]; idx += 1

    return p
