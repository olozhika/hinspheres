"""
Physical constants and default configuration for hinspheres.
"""

import numpy as np


class Config:
    """Global configuration for sphere geometry, physics, and fitting bounds."""

    # ---- Spherical shells ----
    n_shells: int = 7
    R_out_pc: float = 3.        # total cloud radius in pc

    # ---- Fitting weight ----
    weight_index: float = 1.   # exponent for radial weight: w = 1/r^weight_index
                                # 0.0 = uniform, 0.5 = 1/sqrt(r), 1.0 = 1/r

    # ---- Physical constants (CGS) ----
    k_B = 1.380649e-16            # erg/K
    m_H = 1.6735e-24              # g
    m_H2 = 2.0 * 1.6735e-24       # g
    mu_H = 1.40                   # mean gas mass per H nucleus (incl. He), in
                                  # units of m_H; X=0.70, Y=0.28, Z=0.02
                                  # (McKee & Ostriker 2007, ARAA 45, 565).
                                  # Used for enclosed mass: rho = mu_H * m_H * n_H,
                                  # where n_H is total H-nucleus density.
    G = 6.67430e-8                # cm^3 g^-1 s^-2
    pc_cm = 3.085677581e18        # cm/pc
    M_sun_g = 1.989e33            # g, solar mass

    # ---- HI 21cm ----
    nu_21cm = 1.420405751e9       # Hz
    c_light = 2.99792458e10       # cm/s
    A_10 = 2.884e-15              # s^-1, Einstein A coefficient for 21cm
    h_planck = 6.62607015e-27     # erg·s
    T_cmb = 2.725                 # K, cosmic microwave background

    # ---- Galactic HI foreground (Li & Goldsmith 2003) ----
    #[SET tau_h_total TO ZERO TO 禁用 FOREGROUND Section]
    tau_h_total: float = 0.0       # total galactic HI optical depth along LOS 
    T_HI_galactic: float = 100.0   # galactic HI spin temperature (K), CNM+WNM effective spin temperature, opacity-weighted. 
    p_min: float = 0.7             # minimum background fraction p (Li 2003 default)
    galactic_HI_disk_fwhm_pc: float = 360.0  # Lockman 1984, used in Li 2003 eq.9

    # ---- Telescope ----
    beam_fwhm_arcmin: float = 3.0  # FAST HI beam
    distance_pc: float = 140.0     # default distance
    pixel_scale_arcmin: float = 1.5   # synthetic-core spatial pixel scale (arcmin)
    delta_v_kms: float = 0.2          # synthetic-core velocity channel width (km/s)

    # ---- Fitting window ----
    fit_velocity_radius_kms: float = 3.0   # fit only ± this around vlsr_kms
    noise_sigma_K: float = 0.17            # CRAFTS RMS noise per 0.2 km/s channel (Zhang+2019)

    # ---- FITS header fallback defaults ----
    default_cdelt_deg: float = 0.025   # fallback pixel scale (degrees) when CDELT1/CDELT2 missing
    default_cdelt3_ms: float = 200.0   # fallback velocity channel width (m/s) when CDELT3 missing

    # ---- Fallback background and grid defaults ----
    default_bg_temp_K: float = 30.0    # fallback uniform background T_B (K) when no FITS provided
    default_npix: int = 21             # fallback spatial grid pixels when no FITS provided

    # ---- 1D four-layer model (rt_1d.py) geometry and defaults ----
    r_core_ratio: float = 0.3          # core radius = R_env * r_core_ratio
    r_mid_ratio: float = 0.6           # transition radius = R_env * r_mid_ratio
    r_outer_factor: float = 2.55       # outer envelope radius = R_env * r_outer_factor
    T_outer_default: float = 58.0      # default outer envelope spin temperature (K)

    # ---- Velocity channels ----
    v_min_kms: float = -20.0
    v_max_kms: float = 20.0
    n_v_channels: int = 201
    vlsr_kms: float = 0.0            # cloud systemic velocity (km/s), set from catalog

    # ---- Fitting bounds for CMA-ES ----
    bounds_pc = {
        'rho0': (5e2, 5e5),   # central density cm^-3
        'r0': (0.01, 1.2),          # Plummer core radius in pc
        'alpha': (0.5, 4.0),         # Plummer power-law index
        'T0': (5.0, 30.0),           # central temperature K
        'T1': (10.0, 100.0),          # ambient temperature K
        'rT': (0.01, 1.2),          # temperature transition radius pc
        
        'peak_shell': (3, 7),        # HI abundance peak shell (1-indexed)
        'f_HI_peak': (0.001, 1.0),   # peak shell HI abundance
        'multipliers': (0.01, 0.999),   # abundance decrease factor per step
        #'f_HI': (0.0001, 1.0),        # per-shell HI abundance (f_HI mode)
        
        'f_ff': (0.0001, 1.0),         # free-fall fraction
        'turb_kms': (0.0001, 5.0),     # turbulence km/s
        'v_offset': (-2.0, 2.0),     # systemic velocity offset relative to Vlsr (km/s)
        'v_rot_kms': (-5.0, 5.0),    # rotation velocity at R_out (km/s)
        'rot_pa_deg': (0.0, 180.0),  # rotation axis position angle (deg, N to E)
    }

    # ---- Default initial parameters for CMA-ES ----
    default_params = {
        'rho0': 1.82e4,
        'r0': 0.138,
        'alpha': 2.07,
        'T0': 7.0,
        'T1': 80.0,
        'rT': 0.1,
        
        #'f_HI': np.array([0.005,0.008,0.015,0.025,0.035,0.040,0.035,0.025,0.015]), #用这个模式就是自由让每层HI丰度自己跑 
        'peak_shell': 7,
        'f_HI_peak': 0.05,
        'multipliers_value': 0.7,    # default multiplier for each shell
        
        'f_ff': 0.1,
        'turb_kms': 0.2,
        'v_offset': 0.0,
        'v_rot_kms': 0.0,
        'rot_pa_deg': 45.0,
    }

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        # Auto-compute derived quantities
        self._build_shell_radii()
        # Update peak_shell upper bound to match n_shells
        self.bounds_pc = dict(self.__class__.bounds_pc)
        _ps = self.bounds_pc['peak_shell']
        self.bounds_pc['peak_shell'] = (_ps[0], max(1, self.n_shells))
        # Cap rT upper bound at R_out: the temperature transition radius
        # must lie inside the modeled cloud, so T1 stays interpretable
        # as the envelope temperature (T(R_out) = (T0+T1)/2 when rT = R_out).
        _rt = self.bounds_pc['rT']
        self.bounds_pc['rT'] = (_rt[0], min(_rt[1], self.R_out_pc))

    def _build_shell_radii(self):
        """Uniformly spaced shells from 0 to R_out."""
        dr = self.R_out_pc / self.n_shells
        self.r_inner = np.linspace(0, self.R_out_pc - dr, self.n_shells)
        self.r_outer = np.linspace(dr, self.R_out_pc, self.n_shells)
        self.r_mid = (self.r_inner + self.r_outer) / 2.0
        self.shell_vol = (4.0 / 3.0) * np.pi * (self.r_outer**3 - self.r_inner**3)

    @property
    def beam_sigma_pc(self):
        """Beam FWHM in pc at the cloud distance."""
        arcmin_rad = self.beam_fwhm_arcmin / 60.0 * np.pi / 180.0
        return (arcmin_rad * self.distance_pc) / (2 * np.sqrt(2 * np.log(2)))

    @property
    def pc_per_pix(self):
        """Simplified pixel scale assuming Nyquist-sampled beam."""
        arcmin_rad = self.beam_fwhm_arcmin / 60.0 * np.pi / 180.0
        return arcmin_rad * self.distance_pc / 3.0  # ~3 pix per beam


def get_cfg_value(cfg, name):
    """Get a config value from a Config instance (or class) or raise.

    Priority: instance attribute → class attribute → error.
    Use this in functions that expect a ``cfg`` instance to make sure
    every numeric parameter is defined in ``config.py``, with no silent
    hardcoded fallback.
    """
    if hasattr(cfg, name):
        return getattr(cfg, name)
    if hasattr(Config, name):
        return getattr(Config, name)
    raise AttributeError(
        f"'{name}' is not defined in Config. "
        f"Add it to hinspheres/config.py."
    )
