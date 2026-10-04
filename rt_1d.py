"""
1D four-layer radiative transfer model for HINSA along the cloud diameter.

At impact parameter b = 0 (cloud centre), the line of sight pierces the
full diameter, producing eight layers (far -> near):

  Background HI
    -> back outer   (R_outer -> R_env,  z < 0)
    -> back env     (R_env   -> R_mid,  z < 0)
    -> back mid     (R_mid   -> R_core, z < 0)
    -> back core    (R_core  -> 0,      z < 0)
    -> front core   (0       -> R_core, z > 0)
    -> front mid    (R_core  -> R_mid,  z > 0)
    -> front env    (R_mid   -> R_env,  z > 0)
    -> front outer  (R_env   -> R_outer,z > 0)
  Observer

The transition zone (mid) properties are linearly interpolated between
core and envelope, adding no extra free parameters.

This module is the 1D simplification of the spherically symmetric RT
in :mod:`hinspheres.rt`, suitable for single-beam observations where
the line of sight passes through the cloud centre.
"""

import numpy as np
from scipy.optimize import curve_fit

from .config import Config as _Cfg, get_cfg_value

# ---- Physical constants (CGS, matching hinspheres.config) ----
_cfg = _Cfg()
K_B = _cfg.k_B
M_H = _cfg.m_H
MU_H = _cfg.mu_H
G_ = _cfg.G
PC_CM = _cfg.pc_cm
M_SUN_G = _cfg.M_sun_g
C_LIGHT = _cfg.c_light
NU_21CM = _cfg.nu_21cm
A_10 = _cfg.A_10
H_PLANCK = _cfg.h_planck
T_CMB = _cfg.T_cmb


# ======================================================================
# Helper functions
# ======================================================================

def _enclosed_mass_profile(n_core, n_mid, n_env, R_core, R_mid, R_env):
    """Compute enclosed solar mass at any radius using the three-layer model.

    Returns a callable ``M_at_r(r_pc) -> M_enc(M_sun)``.
    """
    def M_at_r(r):
        r = float(r)
        if r <= R_core:
            vol = (4.0 / 3.0) * np.pi * r ** 3
            return vol * n_core * PC_CM ** 3 * MU_H * M_H / M_SUN_G
        elif r <= R_mid:
            vol_core = (4.0 / 3.0) * np.pi * R_core ** 3
            vol_shell = (4.0 / 3.0) * np.pi * (r ** 3 - R_core ** 3)
            return (vol_core * n_core + vol_shell * n_mid) * PC_CM ** 3 * MU_H * M_H / M_SUN_G
        else:
            vol_core = (4.0 / 3.0) * np.pi * R_core ** 3
            vol_mid = (4.0 / 3.0) * np.pi * (R_mid ** 3 - R_core ** 3)
            vol_shell = (4.0 / 3.0) * np.pi * (r ** 3 - R_mid ** 3)
            return (vol_core * n_core + vol_mid * n_mid + vol_shell * n_env) * PC_CM ** 3 * MU_H * M_H / M_SUN_G

    return M_at_r


def _infall_velocity(r_pc, M_enc_Msun):
    """Free-fall velocity (km/s) at radius r given enclosed mass."""
    r_cm = r_pc * PC_CM
    M_cgs = M_enc_Msun * M_SUN_G
    v_ff_cgs = np.sqrt(2.0 * G_ * M_cgs / np.maximum(r_cm, 1e-10 * PC_CM))
    return v_ff_cgs / 1e5


def compute_tau0(n_HI, T_spin, dl_pc, sigma_kms):
    """Peak line-centre optical depth.

    .. math::

        \\tau_0 = \\frac{3 h c^3 A_{10}}{32\\pi k_B \\nu_{21}^2}
                  \\frac{n_\\mathrm{HI}\\, dl}{T_\\mathrm{spin}\\,
                  \\sigma_v \\sqrt{2\\pi}}

    Parameters
    ----------
    n_HI : float
        HI number density (cm^-3).
    T_spin : float
        Spin temperature (K).
    dl_pc : float
        Path length through the layer (pc).
    sigma_kms : float
        Velocity dispersion (km/s).

    Returns
    -------
    float
        Dimensionless peak optical depth.
    """
    const = 3.0 * H_PLANCK * C_LIGHT ** 3 * A_10 / \
            (32.0 * np.pi * K_B * NU_21CM ** 2)
    sigma_cms = sigma_kms * 1e5
    if sigma_cms < 1.0:
        return 0.0
    return const * n_HI * (dl_pc * PC_CM) / (T_spin * sigma_cms * np.sqrt(2.0 * np.pi))


def compute_sigma_v(T_K, turb_kms):
    """Total velocity dispersion.

    .. math::

        \\sigma_v = \\sqrt{\\sigma_\\mathrm{th}^2 + \\sigma_\\mathrm{turb}^2}

    where :math:`\\sigma_\\mathrm{th} = \\sqrt{k_B T / m_H}`.

    Parameters
    ----------
    T_K : float
        Gas temperature (K).
    turb_kms : float
        Turbulent velocity dispersion (km/s).

    Returns
    -------
    float
        Total velocity dispersion (km/s).
    """
    sigma_th = np.sqrt(K_B * T_K / M_H) / 1e5
    return np.sqrt(sigma_th ** 2 + turb_kms ** 2)


# ======================================================================
# Plummer extrapolation for outer envelope
# ======================================================================

def _plummer_extrapolate_n_outer(n_core, n_mid, n_env, R_env_pc,
                                  r_core_ratio=None, r_mid_ratio=None,
                                  r_outer_factor=None):
    """Extrapolate n_outer using a generalized Plummer fit to inner 3 layers.

    Fits :math:`n(r) = n_0 / (1 + (r/a)^2)^{p/2}` to the three inner layers
    (core, mid, env) at their average radii, then evaluates at
    R_outer = r_outer_factor * R_env.

    Parameters
    ----------
    n_core, n_mid, n_env : float
        Densities of inner 3 layers (cm^-3).
    R_env_pc : float
        Envelope radius (pc).
    r_core_ratio, r_mid_ratio : float or None
        Ratio of core/mid radius to R_env. If None, read from Config.
    r_outer_factor : float or None
        Multiplier for outer envelope radius. If None, read from Config.

    Returns
    -------
    n_outer : float
        Extrapolated outer envelope density (cm^-3).
    """
    if r_core_ratio is None:
        r_core_ratio = get_cfg_value(_cfg, 'r_core_ratio')
    if r_mid_ratio is None:
        r_mid_ratio = get_cfg_value(_cfg, 'r_mid_ratio')
    if r_outer_factor is None:
        r_outer_factor = get_cfg_value(_cfg, 'r_outer_factor')
    R_core = R_env_pc * r_core_ratio
    R_mid = R_env_pc * r_mid_ratio
    R_outer = R_env_pc * r_outer_factor

    # Average radii within each layer
    r_avg = np.array([R_core * 0.5,
                      0.5 * (R_core + R_mid),
                      0.5 * (R_mid + R_env_pc)])
    n_vals = np.array([n_core, n_mid, n_env])

    def plummer(r, n0, a, p):
        return n0 / (1.0 + (r / a) ** 2) ** (p / 2.0)

    try:
        p0 = [n_core, R_env_pc * 0.5, 2.0]
        popt, _ = curve_fit(plummer, r_avg, n_vals, p0=p0,
                            bounds=([0, 0.01, 0.5], [n_core * 10, R_env_pc * 5, 10.0]),
                            maxfev=5000)
        n_outer = float(plummer(R_outer, *popt))
        n_outer = max(n_outer, n_env * 0.01)
    except Exception:
        n_outer = n_env * (R_env_pc / R_outer) ** 2

    return n_outer


def _extrapolate_f_HI_outer(f_core, f_mid, f_env, R_core, R_mid, R_env,
                             r_outer_factor=None):
    """Extrapolate f_HI for the outer envelope via log-log linear fit."""
    if r_outer_factor is None:
        r_outer_factor = get_cfg_value(_cfg, 'r_outer_factor')
    R_outer = R_env * r_outer_factor
    r_avg = np.array([R_core * 0.5, 0.5 * (R_core + R_mid), 0.5 * (R_mid + R_env)])
    f_vals = np.array([f_core, f_mid, f_env])
    r_outer_avg = 0.5 * (R_env + R_outer)

    if np.all(f_vals > 0) and np.all(r_avg > 0):
        coeffs = np.polyfit(np.log(r_avg), np.log(f_vals), 1)
        f_outer = float(np.exp(np.polyval(coeffs, np.log(r_outer_avg))))
        f_outer = max(f_outer, f_env * 0.01)
    else:
        f_outer = f_env

    return f_outer


# ======================================================================
# Main RT functions
# ======================================================================

def four_layer_absorption_spectrum(v_grid_kms, params, T_bg_spec, cfg=None):
    """Compute HINSA absorption spectrum for the 1D four-layer model.

    .. math::

        \\Delta T(v) = T_\\mathrm{bg}(v) - T_\\mathrm{model}(v)

    where :math:`T_\\mathrm{model}` is the brightness temperature after
    radiative transfer through eight layers (four on each side of the
    cloud centre).

    Parameters
    ----------
    v_grid_kms : 1D array
        Velocity axis (km/s).
    params : dict
        Required keys:

        - ``R_env_pc``      : cloud inner-envelope outer radius (pc)
        - ``n_H_core_cm3``  : core H-nucleus density (cm^-3)
        - ``n_H_env_cm3``   : inner-envelope H-nucleus density (cm^-3)
        - ``T_core_K``      : core spin temperature (K)
        - ``T_env_K``       : inner-envelope spin temperature (K)
        - ``f_HI_core``     : core HI abundance
        - ``f_HI_env``      : inner-envelope HI abundance
        - ``vlsr_kms``      : systemic velocity (km/s)
        - ``turb_kms``      : turbulent velocity dispersion (km/s)
        - ``f_ff``          : free-fall fraction (default 0.1)

        Optional keys:

        - ``R_core_pc``     : core radius (pc, default ``R_env * r_core_ratio``)
        - ``R_mid_pc``      : transition radius (pc, default ``R_env * r_mid_ratio``)
        - ``R_outer_pc``    : outer envelope radius (pc, default ``R_env * r_outer_factor``)
        - ``n_H_mid_cm3``   : transition density (cm^-3, default average)
        - ``n_H_outer_cm3`` : outer envelope density (cm^-3, extrapolated)
        - ``T_mid_K``       : transition temperature (K, default average)
        - ``T_outer_K``     : outer envelope temperature (K, default from Config)
        - ``f_HI_mid``      : transition HI abundance (default average)
        - ``f_HI_outer``    : outer HI abundance (extrapolated)
        - ``v_offset``      : velocity offset (km/s, default 0)
    cfg : Config or None
        If None, uses module-level default Config instance.

    T_bg_spec : 1D array or float
        Background brightness temperature (K).

    Returns
    -------
    abs_spec : 1D array
        Absorption spectrum (positive = absorption).
    """
    if cfg is None:
        cfg = _cfg
    # --- Geometry ---
    r_core_ratio = get_cfg_value(cfg, 'r_core_ratio')
    r_mid_ratio = get_cfg_value(cfg, 'r_mid_ratio')
    r_outer_factor = get_cfg_value(cfg, 'r_outer_factor')
    R_env = params['R_env_pc']
    R_core = params.get('R_core_pc', R_env * r_core_ratio)
    R_mid = params.get('R_mid_pc', R_env * r_mid_ratio)
    R_outer = params.get('R_outer_pc', R_env * r_outer_factor)
    R_core = min(R_core, R_mid)
    R_mid = min(R_mid, R_env)
    R_outer = max(R_outer, R_env)

    # --- Densities ---
    n_core = params['n_H_core_cm3']
    n_env = params['n_H_env_cm3']
    n_mid = params.get('n_H_mid_cm3', 0.5 * (n_core + n_env))

    n_outer = params.get('n_H_outer_cm3', None)
    if n_outer is None:
        n_outer = _plummer_extrapolate_n_outer(
            n_core, n_mid, n_env, R_env,
            r_core_ratio=r_core_ratio, r_mid_ratio=r_mid_ratio,
            r_outer_factor=r_outer_factor)

    # --- Temperatures ---
    T_core = max(params['T_core_K'], T_CMB)
    T_env = max(params['T_env_K'], T_CMB)
    T_mid = max(params.get('T_mid_K', 0.5 * (T_core + T_env)), T_CMB)
    _T_outer_default = get_cfg_value(cfg, 'T_outer_default')
    T_outer = max(params.get('T_outer_K', _T_outer_default), T_CMB)

    # --- HI abundances ---
    f_core = params['f_HI_core']
    f_env = params['f_HI_env']
    f_mid = params.get('f_HI_mid', 0.5 * (f_core + f_env))

    f_outer = params.get('f_HI_outer', None)
    if f_outer is None:
        f_outer = _extrapolate_f_HI_outer(
            f_core, f_mid, f_env, R_core, R_mid, R_env,
            r_outer_factor=r_outer_factor)

    n_HI_core = n_core * f_core
    n_HI_mid = n_mid * f_mid
    n_HI_env = n_env * f_env
    n_HI_outer = n_outer * f_outer

    # --- Velocity dispersions ---
    turb_kms = params['turb_kms']
    sigma_c = compute_sigma_v(T_core, turb_kms)
    sigma_m = compute_sigma_v(T_mid, turb_kms)
    sigma_e = compute_sigma_v(T_env, turb_kms)
    sigma_o = compute_sigma_v(T_outer, turb_kms)

    # --- Systemic velocity ---
    f_ff = params.get('f_ff', 0.1)
    v_offset = params.get('v_offset', 0.0)
    v_sys = params['vlsr_kms'] + v_offset

    # --- Enclosed mass for infall ---
    M_at_r = _enclosed_mass_profile(n_core, n_mid, n_env, R_core, R_mid, R_env)

    r_char_core = R_core * 0.5
    r_char_mid = 0.5 * (R_core + R_mid)
    r_char_env = 0.5 * (R_mid + R_env)
    r_char_outer = 0.5 * (R_env + R_outer)
    v_ff_core = _infall_velocity(r_char_core, M_at_r(r_char_core)) * f_ff
    v_ff_mid = _infall_velocity(r_char_mid, M_at_r(r_char_mid)) * f_ff
    v_ff_env = _infall_velocity(r_char_env, M_at_r(r_char_env)) * f_ff
    v_ff_outer = _infall_velocity(r_char_outer, M_at_r(r_char_outer)) * f_ff

    # --- Path lengths along diameter (one direction) ---
    dl_core = R_core
    dl_mid = R_mid - R_core
    dl_env = R_env - R_mid
    dl_outer = R_outer - R_env

    # --- Layer definitions: background -> observer ---
    layers = [
        ('back_outer',  dl_outer,  n_HI_outer, T_outer, sigma_o, -v_ff_outer + v_sys),
        ('back_env',    dl_env,    n_HI_env,   T_env,   sigma_e, -v_ff_env + v_sys),
        ('back_mid',    dl_mid,    n_HI_mid,   T_mid,   sigma_m, -v_ff_mid + v_sys),
        ('back_core',   dl_core,   n_HI_core,  T_core,  sigma_c, -v_ff_core + v_sys),
        ('front_core',  dl_core,   n_HI_core,  T_core,  sigma_c, +v_ff_core + v_sys),
        ('front_mid',   dl_mid,    n_HI_mid,   T_mid,   sigma_m, +v_ff_mid + v_sys),
        ('front_env',   dl_env,    n_HI_env,   T_env,   sigma_e, +v_ff_env + v_sys),
        ('front_outer', dl_outer,  n_HI_outer, T_outer, sigma_o, +v_ff_outer + v_sys),
    ]

    # --- Forward RT: background -> layers -> observer ---
    n_v = len(v_grid_kms)
    if np.ndim(T_bg_spec) == 0:
        T = np.full(n_v, float(T_bg_spec))
    else:
        T = np.asarray(T_bg_spec, dtype=float).copy()

    for _, dl, n_HI, T_spin, sigma, v_center in layers:
        tau0 = compute_tau0(n_HI, T_spin, dl, sigma)
        dv = v_grid_kms - v_center
        tau_v = tau0 * np.exp(-0.5 * (dv / sigma) ** 2)
        exp_neg_tau = np.exp(-tau_v)
        T = T * exp_neg_tau + T_spin * (1.0 - exp_neg_tau)

    if np.ndim(T_bg_spec) == 0:
        T_bg_arr = np.full(n_v, float(T_bg_spec))
    else:
        T_bg_arr = np.asarray(T_bg_spec, dtype=float)

    return T_bg_arr - T  # positive = absorption


def four_layer_forward_rt(v_grid_kms, params, T_bg_spec, cfg=None):
    """Forward RT: return the full brightness-temperature spectrum.

    Same as :func:`four_layer_absorption_spectrum` but returns
    :math:`T_\\mathrm{model}(v)` instead of the absorption
    :math:`\\Delta T`.

    Parameters
    ----------
    v_grid_kms : 1D array
        Velocity axis (km/s).
    params : dict
        Same as :func:`four_layer_absorption_spectrum`.
    T_bg_spec : 1D array or float
        Background brightness temperature (K).
    cfg : Config or None
        If None, uses module-level default Config instance.

    Returns
    -------
    T_model : 1D array
        Model brightness temperature spectrum (K).
    """
    abs_spec = four_layer_absorption_spectrum(v_grid_kms, params, T_bg_spec, cfg=cfg)
    if np.ndim(T_bg_spec) == 0:
        T_bg_arr = np.full_like(abs_spec, float(T_bg_spec))
    else:
        T_bg_arr = np.asarray(T_bg_spec, dtype=float)
    return T_bg_arr - abs_spec
