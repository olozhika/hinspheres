"""
Radial profiles for spherically symmetric cloud models.

All profiles return arrays evaluated at shell midpoints.
"""

import numpy as np


def density_plummer(r_mid, rho0, r0, alpha):
    """Plummer density profile.
    
    Parameters
    ----------
    r_mid : 1D array
        Shell mid-point radii in pc.
    rho0 : float
        Central density in cm^-3.
    r0 : float
        Core radius in pc.
    alpha : float
        Power-law index (alpha > 0).

    Returns
    -------
    n_H : 1D array
        H-nucleus density (cm^-3) at each shell.
    """
    return rho0 / (1.0 + (r_mid / r0)**alpha)


def density_from_column(r_mid, r_outer, column_density_profile,
                         r0, alpha, n_shells, pc_cm):
    """Fit the Plummer normalization (rho0) from a column density profile
    (e.g., from Planck Av map). The shape (r0, alpha) is assumed fixed;
    only rho0 is scaled to match the observed column density at each
    impact parameter.

    Parameters
    ----------
    r_mid : 1D array (n_shells,)
    r_outer : 1D array (n_shells,)
    column_density_profile : 1D array (n_b,)
        Observed column density as function of impact parameter.
    r0, alpha : float
        Plummer shape parameters.
    n_shells : int
    pc_cm : float

    Returns
    -------
    rho0 : float
        Best-fit central density (cm^-3).
    """
    # Integrate a normalized Plummer along LOS at each impact parameter
    b_grid = np.linspace(0, r_outer[-1], len(column_density_profile))
    n_H_norm = np.zeros_like(b_grid)

    for ib, b in enumerate(b_grid):
        tau = 0.0
        for k in range(n_shells):
            r = r_mid[k]
            dens = 1.0 / (1.0 + (r / r0)**alpha)  # unit central density
            # LOS path segment
            if b < r_outer[k]:
                r_in = max(r_outer[k-1], b) if k > 0 else b
                r_out = r_outer[k]
                dl = 2.0 * np.sqrt(max(0, r_out**2 - b**2))
                if k > 0 and b < r_outer[k-1]:
                    dl -= 2.0 * np.sqrt(max(0, r_outer[k-1]**2 - b**2))
                tau += dens * dl
        n_H_norm[ib] = tau * pc_cm  # column density for unit central density

    # Scale factor from least-squares
    scale = np.nansum(column_density_profile * n_H_norm) / max(np.nansum(n_H_norm**2), 1e-30)
    return scale


def temperature_plummer(r_mid, T0, T1, rt, T_dex=2.0):
    """Plummer-like temperature profile.
    
    T(r) = T1 + (T0 - T1) / (1 + (r/rt)^T_dex)
    
    Parameters
    ----------
    r_mid : 1D array
    T0 : float
        Central temperature in K.
    T1 : float
        Ambient (outer) temperature in K.
    rt : float
        Transition radius in pc.
    T_dex : float
        Steepness (default 2.0).

    Returns
    -------
    T : 1D array
    """
    return T1 + (T0 - T1) / (1.0 + (r_mid / rt)**T_dex)


def abundance_profile_111n(n_shells, peak_shell, multipliers, f_HI_peak=1.0):
    """Unimodal abundance profile following l1517's 111.n convention.

    The HI abundance peaks at `peak_shell` (1-indexed) and decreases
    monotonically outward via multipliers.

    Parameters
    ----------
    n_shells : int
    peak_shell : int
        1-indexed shell where f_HI peaks.
    multipliers : 1D array
        Len = n_shells - 1. Each entry in (0, 1] gives the ratio
        abundance[m] / abundance[m-1] going outward from the peak.
        The actual values are the exponential of the optimized parameter
        clipped to [0.1, 1.0].
    f_HI_peak : float
        Absolute HI abundance at the peak shell (default 1.0).

    Returns
    -------
    f_HI : 1D array (n_shells,)
        HI fraction f_HI = n_HI / n_H at each shell.
    """
    f_HI = np.ones(n_shells, dtype=float)
    k = peak_shell - 1  # 0-indexed peak

    # Build left side (k → 0)
    for i in range(k, -1, -1):
        if i == k:
            f_HI[i] = f_HI_peak
        else:
            idx = i  # multiplier index
            f_HI[i] = f_HI[i + 1] * multipliers[idx]

    # Build right side (k → n_shells-1)
    for i in range(k + 1, n_shells):
        idx = i - 1
        f_HI[i] = f_HI[i - 1] * multipliers[idx]

    return f_HI


def infall_velocity(r_mid, r_outer, f_ff, M_enc_Msun, G, Msun_cgs):
    """Infall velocity as a fraction of free-fall.

    v_infall(r) = f_ff * sqrt(2 * G * M(<r) / r)

    Parameters
    ----------
    r_mid : 1D array (n_shells,)
        Shell midpoints in cm.
    r_outer : 1D array (n_shells,)
        Shell outer boundaries in cm.
    f_ff : float
        Fraction of free-fall speed (0 < f_ff <= 1).
    M_enc_Msun : float or 1D array (n_shells,)
        Enclosed mass (M_sun) at each shell's outer boundary.
        If scalar, same mass used for all shells (legacy behavior).
    G : float
        Gravitational constant in cgs.
    Msun_cgs : float
        Solar mass in grams.

    Returns
    -------
    v_infall : 1D array (n_shells,)
        Infall speed (positive = inward) in cm/s.
    """
    M_enc_cgs = np.asarray(M_enc_Msun, dtype=float) * Msun_cgs
    if M_enc_cgs.ndim == 0:
        M_enc_cgs = np.full_like(r_mid, M_enc_cgs)
    v_ff = np.sqrt(2.0 * G * M_enc_cgs / np.maximum(r_mid, 1e-10 * r_outer[-1]))
    return f_ff * v_ff


_ABUNDANCE_WIDTH_MIN = 0.3  # prevent unrealistically narrow HI shells


def smooth_abundance_111n(r_mid, peak_shell, width_pc):
    """Smooth Gaussian-like abundance profile as alternative to step-wise 111n.

    f_HI(r) = exp(-0.5 * ((r - r_peak) / width)^2)

    Parameters
    ----------
    r_mid : 1D array
    peak_shell : int
        1-indexed shell number for the peak.
    width_pc : float
        Gaussian width in pc.

    Returns
    -------
    f_HI : 1D array
    """
    width_pc = max(width_pc, _ABUNDANCE_WIDTH_MIN * (r_mid[-1] - r_mid[0]) / len(r_mid))
    r_peak = r_mid[peak_shell - 1] if peak_shell <= len(r_mid) else r_mid[-1]
    return np.exp(-0.5 * ((r_mid - r_peak) / width_pc)**2)
