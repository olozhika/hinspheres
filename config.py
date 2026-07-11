"""
Physical constants and default configuration for hinspheres.
"""

import numpy as np


class Config:
    """Global configuration for sphere geometry, physics, and fitting bounds."""

    # ---- Spherical shells ----
    n_shells: int = 9
    R_out_pc: float = 5.        # total cloud radius in pc

    # ---- Fitting weight ----
    weight_index: float = 0.5   # exponent for radial weight: w = 1/r^weight_index
                                # 0.0 = uniform, 0.5 = 1/sqrt(r), 1.0 = 1/r

    # ---- Physical constants (CGS) ----
    k_B = 1.380649e-16            # erg/K
    m_H = 1.6735e-24              # g
    m_H2 = 2.0 * 1.6735e-24       # g
    mu = 2.33                     # mean molecular weight
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
    tau_h_total: float = 0.0       # total galactic HI optical depth along LOS [SET IT TO ZERO TO 禁用 FOREGROUND]
    T_HI_galactic: float = 100.0   # galactic HI spin temperature (K), CNM+WNM effective spin temperature, opacity-weighted.
    p_min: float = 0.7             # minimum background fraction p (Li 2003 default)
    galactic_HI_disk_fwhm_pc: float = 360.0  # Lockman 1984, used in Li 2003 eq.9

    # ---- Telescope ----
    beam_fwhm_arcmin: float = 4.0  # FAST HI beam
    distance_pc: float = 140.0     # default distance

    # ---- FITS header fallback defaults ----
    default_cdelt_deg: float = 0.025   # fallback pixel scale (degrees) when CDELT1/CDELT2 missing
    default_cdelt3_ms: float = 200.0   # fallback velocity channel width (m/s) when CDELT3 missing

    # ---- Velocity channels ----
    v_min_kms: float = -20.0
    v_max_kms: float = 20.0
    n_v_channels: int = 201
    vlsr_kms: float = 0.0            # cloud systemic velocity (km/s), set from catalog

    # ---- Fitting bounds for CMA-ES ----
    bounds_pc = {
        'rho0': (100.0, 100000.0),   # central density cm^-3
        'r0': (0.01, 0.3),          # Plummer core radius in pc
        'alpha': (1.0, 5.0),         # Plummer power-law index
        'T0': (5.0, 30.0),           # central temperature K
        'T1': (10.0, 80.0),          # ambient temperature K
        'rT': (0.01, 0.3),          # temperature transition radius pc
        'peak_shell': (1, 9),        # HI abundance peak shell (1-indexed)
        'f_HI_peak': (0.0001, 0.5),   # peak shell HI abundance
        'multipliers': (0.01, 1.0),   # abundance decrease factor per step
        'f_HI': (0.0001, 0.5),        # per-shell HI abundance (f_HI mode)
        'f_ff': (0.01, 0.5),         # free-fall fraction
        'turb_kms': (0.05, 0.5),     # turbulence km/s
        'v_offset': (-3.0, 3.0),     # systemic velocity offset relative to Vlsr (km/s)
        'v_rot_kms': (0.0, 5.0),    # rotation velocity at R_out (km/s)
        'rot_pa_deg': (0.0, 180.0),  # rotation axis position angle (deg, N to E)
    }

    # ---- Default initial parameters for CMA-ES ----
    default_params = {
        'rho0': 5000.0,
        'r0': 0.1,
        'alpha': 2.0,
        'T0': 10.0,
        'T1': 58.0,
        'rT': 0.1,
        'peak_shell': 9,
        'f_HI_peak': 0.05,
        'multipliers_value': 0.7,    # default multiplier for each shell
        'f_ff': 0.1,
        'turb_kms': 0.2,
        'v_offset': 0.0,
        'v_rot_kms': 0.0,
        'rot_pa_deg': 0.0,
    }

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        # Auto-compute derived quantities
        self._build_shell_radii()
        # Update peak_shell upper bound to match n_shells
        self.bounds_pc = dict(self.__class__.bounds_pc)
        self.bounds_pc['peak_shell'] = (1, max(1, self.n_shells))

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
