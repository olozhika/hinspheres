"""
Spherically symmetric ray-tracing and HI 21cm radiative transfer.

Geometry: N concentric shells, each with constant (n_HI, T_spin, v_los, sigma).
The line of sight at impact parameter b intersects a subset of shells,
each traversed twice (front + back hemisphere) except the innermost.

Radiative transfer follows the Li & Goldsmith (2003) three-component
formalism along each LOS, with the cold cloud divided into N shells.
"""

import math
import numpy as np
from numba import njit


# ---- Numba-JITted hot-path functions (used by models.py pixel loop) ----

@njit(cache=True)
def _njit_los_path_lengths(b, r_outer, n_shells):
    max_layers = 2 * n_shells
    out_k = np.zeros(max_layers, dtype=np.int64)
    out_dl = np.zeros(max_layers)
    out_z = np.zeros(max_layers)
    n = 0

    i_center = 0
    while i_center < n_shells and b >= r_outer[i_center]:
        i_center += 1
    if i_center >= n_shells:
        return out_k, out_dl, out_z, 0

    for k in range(n_shells - 1, i_center - 1, -1):
        r_in = r_outer[k - 1] if k > i_center else b
        r_out = r_outer[k]
        dl = math.sqrt(max(0.0, r_out**2 - b**2))
        if k > i_center:
            dl -= math.sqrt(max(0.0, r_outer[k - 1]**2 - b**2))
        if dl > 1e-10:
            z_mid = -0.5 * (math.sqrt(max(0.0, r_out**2 - b**2)) +
                            (math.sqrt(max(0.0, r_in**2 - b**2)) if k > i_center else 0.0))
            out_k[n], out_dl[n], out_z[n] = k, dl, z_mid
            n += 1

    for k in range(i_center, n_shells):
        r_in = r_outer[k - 1] if k > i_center else b
        r_out = r_outer[k]
        dl = math.sqrt(max(0.0, r_out**2 - b**2))
        if k > i_center:
            dl -= math.sqrt(max(0.0, r_outer[k - 1]**2 - b**2))
        if dl > 1e-10:
            z_mid = (math.sqrt(max(0.0, r_out**2 - b**2)) +
                     (math.sqrt(max(0.0, r_in**2 - b**2)) if k > i_center else 0.0)) / 2.0
            out_k[n], out_dl[n], out_z[n] = k, dl, z_mid
            n += 1

    return out_k, out_dl, out_z, n


@njit(cache=True)
def _njit_compute_layer_tau0(c_light, A_10, nu_21cm, pc_cm,
                              h_planck, k_B,
                              n_HI, T_spin, dl_pc, sigma_kms):
    const = 3.0 * h_planck * c_light**3 * A_10 / \
            (32.0 * math.pi * k_B * nu_21cm**2)
    sigma_cms = sigma_kms * 1e5
    if sigma_cms < 1.0:
        return 0.0
    T_spin_safe = T_spin if T_spin > 1.0 else 1.0
    return const * n_HI * (dl_pc * pc_cm) / (T_spin_safe * sigma_cms * math.sqrt(2.0 * math.pi))


@njit(cache=True)
def _njit_los_velocity(b, z, r_mid, v_infall_kms,
                        v_rot_kms, rot_pa_deg, dx, dy, r_cloud):
    r = math.sqrt(b * b + z * z)
    if r < 1e-10:
        return 0.0
    cos_theta = z / r
    idx = np.searchsorted(r_mid, r)
    idx = min(idx, len(v_infall_kms) - 1)
    v_inf = v_infall_kms[idx]
    v_los_inf = v_inf * (1.0 if z >= 0 else -1.0) * abs(cos_theta)

    v_los_rot = 0.0
    if v_rot_kms != 0.0 and (dx != 0.0 or dy != 0.0):
        pa_rad = rot_pa_deg * math.pi / 180.0
        if r_cloud > 0.0:
            omega = v_rot_kms / r_cloud
            v_los_rot = omega * (-math.sin(pa_rad) * dy - math.cos(pa_rad) * dx)

    return v_los_inf + v_los_rot


@njit(cache=True)
def _njit_rt_pixel(v_grid, n_layers, tau0, v_center, sigma, T_s, T_bg):
    T = np.copy(T_bg)
    for k in range(n_layers):
        dv = v_grid - v_center[k]
        tau_v = tau0[k] * np.exp(-0.5 * (dv / sigma[k])**2)
        e_tau = np.exp(-tau_v)
        T = T * e_tau + T_s[k] * (1.0 - e_tau)
    return T


@njit(cache=True)
def _njit_inverse_rt_pixel(v_grid, n_layers, tau0, v_center, sigma, T_s, T_obs):
    T = np.copy(T_obs)
    eps_min = 0.01
    for k in range(n_layers - 1, -1, -1):
        dv = v_grid - v_center[k]
        tau_v = tau0[k] * np.exp(-0.5 * (dv / sigma[k])**2)
        e_tau = np.exp(-tau_v)
        e_tau = np.maximum(e_tau, eps_min)
        T = (T - T_s[k] * (1.0 - e_tau)) / e_tau
    return T


# ---- Original public functions (kept for backward compat) ----

def make_shells(r_mid, r_outer, n_H, T_spin, v_infall, sigma_v, pc_cm):
    """Assemble shell arrays for the ray-tracer.

    Note: This function is not currently used by the package.

    Parameters
    ----------
    r_mid : 1D array (n_shells,) — pc
    r_outer : 1D array (n_shells,) — pc
    n_H : 1D array (n_shells,) — cm^-3, total H density
    T_spin : 1D array (n_shells,) — K
    v_infall : 1D array (n_shells,) — cm/s, positive = inward
    sigma_v : 1D array (n_shells,) — cm/s, 1-sigma line width
    pc_cm : float

    Returns
    -------
    r_out_cm : 1D array — outer radii in cm
    n_H : 1D array — total H density (cm^-3), passed through
    T_spin : 1D array — spin temperature (K), passed through
    v_infall : 1D array — infall velocity (cm/s), passed through
    sigma_v : 1D array — line width (cm/s), passed through
    """
    r_out_cm = r_outer * pc_cm
    return r_out_cm, n_H, T_spin, v_infall, sigma_v


def los_path_lengths(b, r_outer, n_shells):
    """Compute path lengths through each shell for a given impact parameter.

    Parameters
    ----------
    b : float
        Impact parameter in pc.
    r_outer : 1D array (n_shells,)
        Shell outer radii in pc.
    n_shells : int

    Returns
    -------
    layers : list of (shell_idx, dl_pc, z_mid_pc)
        Each layer is a segment along the LOS.
        Front hemisphere: z positive, Back hemisphere: z negative.
    """
    layers = []
    # Find innermost shell intersected
    i_center = 0
    while i_center < n_shells and b >= r_outer[i_center]:
        i_center += 1
    if i_center >= n_shells:
        return layers  # no intersection

    # Order: background → observer.
    # Light path: z=-inf → back outermost → back innermost → front innermost → front outermost → z=+inf
    # Back hemisphere (z < 0): outermost shell first (closest to background)
    for k in range(n_shells - 1, i_center - 1, -1):
        r_in = r_outer[k - 1] if k > i_center else b
        r_out = r_outer[k]
        dl = np.sqrt(max(0.0, r_out**2 - b**2))
        if k > i_center:
            dl -= np.sqrt(max(0.0, r_outer[k - 1]**2 - b**2))
        if dl > 1e-10:
            z_mid = -0.5 * (np.sqrt(max(0.0, r_out**2 - b**2)) +
                            (np.sqrt(max(0.0, r_in**2 - b**2)) if k > i_center else 0.0))
            layers.append((k, dl, z_mid))

    # Front hemisphere (z > 0): innermost shell first, outermost last (closest to observer)
    for k in range(i_center, n_shells):
        r_in = r_outer[k - 1] if k > i_center else b
        r_out = r_outer[k]
        dl = np.sqrt(max(0.0, r_out**2 - b**2))
        if k > i_center:
            dl -= np.sqrt(max(0.0, r_outer[k - 1]**2 - b**2))
        if dl > 1e-10:
            z_mid = (np.sqrt(max(0.0, r_out**2 - b**2)) +
                     (np.sqrt(max(0.0, r_in**2 - b**2)) if k > i_center else 0.0)) / 2.0
            layers.append((k, dl, z_mid))

    return layers


def los_velocity(b, z, r_mid, v_infall_kms, v_rot_kms=0.0, rot_pa_deg=0.0,
                 dx=0.0, dy=0.0, r_cloud=None):
    """Project 3D infall + rotation velocity onto the LOS at position z.

    For spherical infall: v_los_inf = v_infall * cos(theta)
        Front side (z>0): gas falls inward (away from observer) → +v_los (redshift)
        Back side (z<0): gas falls inward (toward observer) → -v_los (blueshift)
    For rotation (rigid-body): v_los_rot = Omega * (-sin(PA)*dy - cos(PA)*dx)
        where Omega = v_rot_kms / r_cloud.
        The rotation axis is on the sky plane at position angle PA (N→E).
        The LOS component depends on sky-plane position (dx, dy), NOT depth z.

    Parameters
    ----------
    b : float — impact parameter (pc)
    z : float — position along LOS (pc), z>0 = front side (toward observer)
    r_mid : 1D array (n_shells,) — shell midpoints (pc)
    v_infall_kms : 1D array (n_shells,) — km/s, positive = inward
    v_rot_kms : float — rotation velocity at cloud radius (km/s), default 0
    rot_pa_deg : float — position angle of rotation axis (deg, N→E), default 0
    dx : float — offset from cloud center along x-axis (pc), default 0
        Note: in FITS pixel coords x increases westward, so dx>0 = west.
    dy : float — offset from cloud center along y-axis (pc), default 0
        Note: in FITS pixel coords y increases northward, so dy>0 = north.
    r_cloud : float or None — cloud outer radius (pc) for rotation Omega.
        If None, uses r_mid[-1].

    Returns
    -------
    v_los : float — km/s (infall + rotation)
    """
    r = np.sqrt(b**2 + z**2)
    if r < 1e-10:
        return 0.0
    cos_theta = z / r
    # Find the enclosing shell
    idx = np.searchsorted(r_mid, r)
    idx = min(idx, len(v_infall_kms) - 1)
    v_inf = v_infall_kms[idx]
    # LOS: front side (z>0) infalling gas moves away from observer → +v_los
    #       back side (z<0) infalling gas moves toward observer → -v_los
    v_los_inf = v_inf * np.sign(z) * abs(cos_theta)

    # Rotation: rigid-body rotation around axis on the sky plane at PA (N→E)
    # Axis direction in FITS coords (x_West, y_North): n = (-sin(PA), cos(PA))
    # v = Omega × (n × r) → v_los = Omega * (-sin(PA)*dy - cos(PA)*dx)
    #   dx>0 is west, dy>0 is north (FITS convention).
    #   The LOS component depends on sky-plane position (dx, dy), NOT depth z.
    v_los_rot = 0.0
    if v_rot_kms != 0.0 and (dx != 0.0 or dy != 0.0):
        pa_rad = rot_pa_deg * np.pi / 180.0
        if r_cloud is None:
            r_cloud = r_mid[-1] if len(r_mid) > 0 else 1.0
        if r_cloud > 1e-10:
            omega = v_rot_kms / r_cloud  # km/s per pc
            v_los_rot = omega * (-np.sin(pa_rad) * dy - np.cos(pa_rad) * dx)

    return v_los_inf + v_los_rot


def hi_opacity_profile(v_grid_kms, v_center_kms, sigma_kms, tau_0):
    """Gaussian line profile:
    tau(v) = tau_0 * exp(-(v - v_center)^2 / (2 * sigma^2))
    """
    dv = v_grid_kms - v_center_kms
    return tau_0 * np.exp(-0.5 * (dv / sigma_kms)**2)


def radiative_transfer_pixel(
    v_grid_kms,
    n_layers,
    layer_tau0,
    layer_v_center_kms,
    layer_sigma_kms,
    layer_T_spin,
    T_bg_kms,
    T_HI_galactic=None,
    tau_fg=None
):
    """Solve radiative transfer along a single LOS for all velocity channels.

    Follows Li & Goldsmith (2003) three-component model:
      bg HI → cold cloud (N shells) → fg HI → observer

    Parameters
    ----------
    v_grid_kms : 1D array — velocity channels (km/s)
    n_layers : int — number of shell intersections on this LOS
    layer_tau0 : 1D array (n_layers,) — peak tau per layer
    layer_v_center_kms : 1D array 
    layer_sigma_kms : 1D array 
    layer_T_spin : 1D array
    T_bg_kms : float or 1D array — pure background brightness temperature (K),
        i.e. after foreground has been stripped.  Cloud RT is applied to this.
    T_HI_galactic : float or None — spin temperature of galactic foreground HI (K).
        If None, foreground RT step is skipped (foreground applied externally).
    tau_fg : float or None — foreground HI optical depth.
        If None, foreground RT step is skipped (foreground applied externally).

    Returns
    -------
    T_B : 1D array — brightness temperature at each velocity channel
    """
    n_v = len(v_grid_kms)

    # Broadcast scalar background to array
    if np.ndim(T_bg_kms) == 0:
        T = np.full(n_v, float(T_bg_kms))
    else:
        T = np.asarray(T_bg_kms, dtype=float).copy()

    # Vectorized: loop over layers, operate on full velocity array
    for k in range(n_layers):
        tau_v = layer_tau0[k] * np.exp(
            -0.5 * ((v_grid_kms - layer_v_center_kms[k]) / layer_sigma_kms[k])**2)
        exp_neg_tau = np.exp(-tau_v)
        T = T * exp_neg_tau + \
                layer_T_spin[k] * (1.0 - exp_neg_tau)

    # Foreground HI (only if not handled externally)
    if tau_fg is not None and T_HI_galactic is not None:
        exp_neg_tau_fg = np.exp(-tau_fg)
        T = T * exp_neg_tau_fg + \
                float(T_HI_galactic) * (1.0 - exp_neg_tau_fg)

    return T


def compute_layer_tau0(cfg, n_HI, T_spin, dl_pc, sigma_kms):
    """Compute peak optical depth for a single layer.

    From the HI 21cm line opacity:
      tau_0 = (3 h c^3 A_10) / (32 pi k_B nu_21cm^2)
              * (n_HI * dl) / (T_spin * sigma_v * sqrt(2*pi))

    Returns dimensionless tau_0.
    """
    const = 3.0 * cfg.h_planck * cfg.c_light**3 * cfg.A_10 / \
            (32.0 * np.pi * cfg.k_B * cfg.nu_21cm**2)
    sigma_cms = sigma_kms * 1e5
    if sigma_cms < 1.0:
        return 0.0
    T_spin_safe = T_spin if T_spin > 1.0 else 1.0
    return const * n_HI * (dl_pc * cfg.pc_cm) / (T_spin_safe * sigma_cms * np.sqrt(2.0 * np.pi))


def inverse_radiative_transfer_pixel(
    v_grid_kms,
    n_layers,
    layer_tau0,
    layer_v_center_kms,
    layer_sigma_kms,
    layer_T_spin,
    T_obs_kms,
):
    """Reverse RT: from observed spectrum, peel back each layer to recover T_bg.

    Works near → far (opposite direction of radiative_transfer_pixel).
    For each layer, inverts:
        T_before = [T_after - T_ex * (1 - exp(-tau))] / exp(-tau)

    Parameters
    ----------
    v_grid_kms : 1D array — velocity channels (km/s)
    n_layers : int — number of shell intersections on this LOS
    layer_tau0 : 1D array (n_layers,) — peak tau per layer
    layer_v_center_kms : 1D array
    layer_sigma_kms : 1D array
    layer_T_spin : 1D array — excitation temperature per layer
    T_obs_kms : 1D array — observed brightness temperature (K)

    Returns
    -------
    T_bg_reconstructed : 1D array — reconstructed unabsorbed background (K)
    """
    T = np.asarray(T_obs_kms, dtype=np.float64).copy()

    # Peel back layers: near → far
    # layers from los_path_lengths are ordered far → near,
    # so we iterate in reverse
    for k in range(n_layers - 1, -1, -1):
        tau_v = layer_tau0[k] * np.exp(
            -0.5 * ((v_grid_kms - layer_v_center_kms[k]) / layer_sigma_kms[k])**2)
        exp_neg_tau = np.exp(-tau_v)

        # Invert: T_before = (T_after - T_ex * (1 - exp(-tau))) / exp(-tau)
        # For saturated channels (tau large), exp(-tau) is tiny and the
        # division amplifies any model/data mismatch.  Rather than skip
        # (which corrupts subsequent layers), cap the amplification factor
        # at 1/eps_min to prevent blow-up while still allowing correction.
        eps_min = 0.01  # corresponds to tau_max ≈ 4.6
        exp_neg_tau = np.maximum(exp_neg_tau, eps_min)
        T_before = (T - layer_T_spin[k] * (1.0 - exp_neg_tau)) / exp_neg_tau
        T = T_before

    return T
