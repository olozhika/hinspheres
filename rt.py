"""
Spherically symmetric ray-tracing and HI 21cm radiative transfer.

Geometry: N concentric shells, each with constant (n_HI, T_spin, v_los, sigma).
The line of sight at impact parameter b intersects a subset of shells,
each traversed twice (front + back hemisphere) except the innermost.

Radiative transfer follows the Li & Goldsmith (2003) three-component
formalism along each LOS, with the cold cloud divided into N shells.
"""

import numpy as np


def make_shells(r_mid, r_outer, n_H, T_spin, v_infall, sigma_v, pc_cm):
    """Assemble shell arrays for the ray-tracer.

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
    r_out_cm : 1D array
    n_H_cm : 1D array
    T_K : 1D array
    v_los_func : callable(impact_param, z)
        Returns LOS velocity at position z along the line of sight.
    sigma_arr : 1D array
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
                 dx=0.0):
    """Project 3D infall + rotation velocity onto the LOS at position z.

    For spherical infall: v_los_inf = -v_infall * cos(theta)
    For rotation (rigid-body): v_los_rot = Omega * (-dx*cos(PA) - z*sin(PA))
        where Omega = v_rot_kms / R_out.
        The rotation axis is on the sky plane at position angle PA (N→E).
        At different depths z along the LOS, the rotation LOS component changes.

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
    v_los_inf = -v_inf * np.sign(z) * abs(cos_theta)

    # Rotation: rigid-body rotation around axis on the sky plane at PA (N→E)
    # v = Omega × (axis × r) → v_los = Omega * (-dx*cos(PA) - z*sin(PA))
    #   dx>0 is west (FITS convention), z>0 is toward observer.
    #   At dx>0, z=0 (east side of cloud on sky): v_los = -Omega*dx*cos(PA)
    #     → PA=0° (axis N): blueshift (gas moves toward observer) ✓
    #   At dx=0, z>0 (front side): v_los = -Omega*z*sin(PA)
    #     → PA=90° (axis E): blueshift (gas moves toward observer) ✓
    v_los_rot = 0.0
    if v_rot_kms != 0.0 and (dx != 0.0 or z != 0.0):
        pa_rad = rot_pa_deg * np.pi / 180.0
        r_cloud = r_mid[-1] if len(r_mid) > 0 else 1.0
        if r_cloud > 1e-10:
            omega = v_rot_kms / r_cloud  # km/s per pc
            v_los_rot = omega * (-dx * np.cos(pa_rad) - z * np.sin(pa_rad))

    return v_los_inf + v_los_rot


def hi_opacity_profile(v_grid_kms, v_center_kms, sigma_kms, tau_0):
    """Gaussian line profile:
    tau(v) = tau_0 * exp(-(v - v_center)^2 / (2 * sigma^2))
    """
    dv = v_grid_kms - v_center_kms
    return tau_0 * np.exp(-0.5 * (dv / sigma_kms)**2)


def hi_tau_0(n_HI, T_spin, dl_pc, pc_cm):
    """Peak optical depth for a HI 21cm layer.

    From: tau_0 = (3 * c^2 * A_10 * n_HI) / (8 * pi * nu_21cm^2 * sigma_v) * dl / T_spin
    Divided by sqrt(2*pi*sigma^2) for the line profile.
    """
    const = 3.0 * (2.99792458e10)**2 * 2.884e-15 / \
            (8.0 * np.pi * (1.420405751e9)**2)
    # tau_0 per unit length per unit n_HI per unit 1/T_spin (cgs)
    # integrated over line profile → divide by sigma_v_cms * sqrt(2*pi)
    sigma_v_cms = 1e5  # placeholder, will be multiplied by actual sigma
    return const * n_HI * dl_pc * pc_cm / T_spin / (sigma_v_cms * np.sqrt(2.0 * np.pi))


def radiative_transfer_pixel(
    v_grid_kms,
    n_layers,
    layer_tau0,
    layer_v_center_kms,
    layer_sigma_kms,
    layer_T_spin,
    T_bg_kms,
    T_fg_kms,
    tau_bg,
    tau_fg
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
    T_bg_kms : float or 1D array — background HI brightness temperature (K).
        If 1D array, must have same length as v_grid_kms (velocity-dependent background).
    T_fg_kms : float — foreground HI brightness temp (K)
    tau_bg : float — background HI optical depth (optically thin → small)
    tau_fg : float — foreground HI optical depth

    Returns
    -------
    T_B : 1D array — brightness temperature at each velocity channel
    """
    n_v = len(v_grid_kms)
    T_B = np.zeros(n_v)

    # Broadcast scalar background/foreground to array
    if np.ndim(T_bg_kms) == 0:
        T_bg_arr = np.full(n_v, float(T_bg_kms))
    else:
        T_bg_arr = np.asarray(T_bg_kms, dtype=float)

    if np.ndim(T_fg_kms) == 0:
        T_fg_arr = np.full(n_v, float(T_fg_kms))
    else:
        T_fg_arr = np.asarray(T_fg_kms, dtype=float)

    for iv in range(n_v):
        v = v_grid_kms[iv]

        T_obs = T_bg_arr[iv]

        # Propagate through each cold cloud layer
        for k in range(n_layers):
            tau_v = layer_tau0[k] * np.exp(
                -0.5 * ((v - layer_v_center_kms[k]) / layer_sigma_kms[k])**2)
            T_obs = T_obs * np.exp(-tau_v) + \
                    layer_T_spin[k] * (1.0 - np.exp(-tau_v))

        # Foreground HI: absorbs T_obs, emits at T_fg
        T_obs = T_obs * np.exp(-tau_fg) + \
                T_fg_arr[iv] * (1.0 - np.exp(-tau_fg))

        T_B[iv] = T_obs

    return T_B


def compute_layer_tau0(n_HI, T_spin, dl_pc, sigma_kms, pc_cm):
    """Compute peak optical depth for a single layer.

    From the HI 21cm line opacity:
      tau_0 = (3 c^2 A_10) / (8 pi nu_21cm^2) * (n_HI * dl) / (T_spin * sigma_v * sqrt(2*pi))
    
    Returns dimensionless tau_0.
    """
    const = 3.0 * (2.99792458e10)**2 * 2.884e-15 / \
            (8.0 * np.pi * (1.420405751e9)**2)
    sigma_cms = sigma_kms * 1e5
    if sigma_cms < 1.0:
        return 0.0
    return const * n_HI * (dl_pc * pc_cm) / (T_spin * sigma_cms * np.sqrt(2.0 * np.pi))


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
        # Guard against exp(-tau) ~ 0 (very optically thick)
        safe = exp_neg_tau > 1e-10
        T_new = np.where(
            safe,
            (T - layer_T_spin[k] * (1.0 - exp_neg_tau)) / np.where(safe, exp_neg_tau, 1.0),
            T  # if tau too large, keep as-is (saturated)
        )
        T = T_new

    return T


def inverse_radiative_transfer_pixel_iter(
    v_grid_kms,
    n_layers,
    layer_tau0,
    layer_v_center_kms,
    layer_sigma_kms,
    layer_T_spin,
    T_obs_kms,
    n_iter=3,
    poly_order=3,
):
    """Liu Method 2: iterative inverse RT with smoothing.

    Starts from T_obs, iteratively refines T_bg using:
        T_bg,(k+1) = T_obs + (1 - exp(-tau)) * smooth(T_bg,(k))

    Parameters
    ----------
    v_grid_kms : 1D array — velocity channels (km/s)
    n_layers : int
    layer_tau0, layer_v_center_kms, layer_sigma_kms, layer_T_spin : 1D arrays
    T_obs_kms : 1D array — observed brightness temperature (K)
    n_iter : int — number of Liu Method 2 iterations
    poly_order : int — polynomial order for smoothing

    Returns
    -------
    T_bg_reconstructed : 1D array
    """
    # Compute total tau(v) across all layers
    tau_total = np.zeros_like(v_grid_kms)
    for k in range(n_layers):
        tau_total += layer_tau0[k] * np.exp(
            -0.5 * ((v_grid_kms - layer_v_center_kms[k]) / layer_sigma_kms[k])**2)

    exp_neg_tau = np.exp(-tau_total)
    one_minus_exp = 1.0 - exp_neg_tau

    # Initial guess: simple inverse (no smoothing)
    T_bg = np.where(
        exp_neg_tau > 1e-10,
        (T_obs_kms - 0.0 * one_minus_exp) / np.where(exp_neg_tau > 1e-10, exp_neg_tau, 1.0),
        T_obs_kms
    )

    # Liu Method 2 iterations
    for _ in range(n_iter):
        # Smooth T_bg with polynomial
        coeffs = np.polyfit(v_grid_kms, T_bg, poly_order)
        T_bg_smooth = np.polyval(coeffs, v_grid_kms)
        # Update: T_bg,(k+1) = T_obs + (1 - exp(-tau)) * smooth(T_bg)
        T_bg = T_obs_kms + one_minus_exp * T_bg_smooth

    return T_bg
