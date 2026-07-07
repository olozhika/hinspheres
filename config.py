"""
Physical constants and default configuration for hinspheres.
"""

import numpy as np


class Config:
    """Global configuration for sphere geometry, physics, and fitting bounds."""

    # ---- Spherical shells ----
    n_shells: int = 9
    R_out_pc: float = 0.15        # total cloud radius in pc

    # ---- Physical constants (CGS) ----
    k_B = 1.380649e-16            # erg/K
    m_H = 1.6735e-24              # g
    m_H2 = 2.0 * 1.6735e-24       # g
    mu = 2.33                     # mean molecular weight
    G = 6.67430e-8                # cm^3 g^-1 s^-2
    pc_cm = 3.085677581e18        # cm/pc

    # ---- HI 21cm ----
    nu_21cm = 1.420405751e9       # Hz
    c_light = 2.99792458e10       # cm/s
    A_10 = 2.884e-15              # s^-1, Einstein A coefficient for 21cm
    h_planck = 6.62607015e-27     # erg·s
    T_cmb = 2.725                 # K, cosmic microwave background

    # ---- Telescope ----
    beam_fwhm_arcmin: float = 4.0  # FAST HI beam
    distance_pc: float = 140.0     # default distance

    # ---- Velocity channels ----
    v_min_kms: float = -20.0
    v_max_kms: float = 20.0
    n_v_channels: int = 201
    vlsr_kms: float = 0.0            # cloud systemic velocity (km/s), set from catalog

    # ---- Fitting bounds for CMA-ES ----
    bounds_pc = {
        'r0': (0.01, 0.15),        # Plummer core radius in pc
        'alpha': (1.0, 5.0),       # Plummer power-law index
        'T0': (5.0, 30.0),         # central temperature K
        'T1': (10.0, 80.0),        # ambient temperature K
        'rT': (0.01, 0.15),        # temperature transition radius pc
        'peak_shell': (1, 9),      # HI abundance peak shell (1-indexed)
        'multipliers': (0.1, 1.0), # abundance decrease factor per step
        'f_ff': (0.01, 0.5),       # free-fall fraction
        'turb_kms': (0.05, 0.5),   # turbulence km/s
        'v_offset': (-3.0, 3.0),   # systemic velocity offset relative to Vlsr (km/s)
        'v_rot_kms': (0.0, 5.0),  # rotation velocity at R_out (km/s)
        'rot_pa_deg': (0.0, 180.0),  # rotation axis position angle (deg, N to E)
    }

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        # Auto-compute derived quantities
        self._build_shell_radii()

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
