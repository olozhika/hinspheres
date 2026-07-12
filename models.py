"""
Forward model builder: physical parameters → synthetic HINSA map/cube.
"""

import os
import numpy as np
from joblib import Parallel, delayed

from .config import Config
from .profiles import (
    density_plummer, temperature_plummer,
    abundance_profile_111n, infall_velocity
)
from .rt import (
    los_path_lengths, compute_layer_tau0,
    radiative_transfer_pixel, los_velocity,
    inverse_radiative_transfer_pixel,
)


def _compute_galactic_b(fits_header, center_yx=None):
    """Compute galactic latitude of cloud center from FITS header coordinates.

    Parameters
    ----------
    fits_header : astropy.io.fits.Header
        Must contain CRVAL1/CRVAL2 (RA/DEC in degrees) and CRPIX1/CRPIX2.
    center_yx : tuple (yc, xc) or None
        Cloud center pixel. If None, uses the FITS reference pixel position.

    Returns
    -------
    b_deg : float — galactic latitude (degrees)
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    crval1 = fits_header.get('CRVAL1', 0.0)
    crval2 = fits_header.get('CRVAL2', 0.0)
    crpix1 = fits_header.get('CRPIX1', 1.0)
    crpix2 = fits_header.get('CRPIX2', 1.0)
    cdelt1 = fits_header.get('CDELT1', 0.0)
    cdelt2 = fits_header.get('CDELT2', 0.0)

    if center_yx is not None:
        yc, xc = center_yx
        ra = crval1 + cdelt1 * (xc - (crpix1 - 1))
        dec = crval2 + cdelt2 * (yc - (crpix2 - 1))
    else:
        ra, dec = crval1, crval2

    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
    return coord.galactic.b.deg


def compute_foreground_params(cfg, galactic_b_deg=None):
    """Compute foreground optical depth and HI spin temperature from galactic model.

    Uses Li & Goldsmith (2003) eq.9: p = erfc(sqrt(4*ln2) * D*sin(b) / z)
    where z = 360 pc is the galactic HI disk FWHM.

    Parameters
    ----------
    cfg : Config
        Must have tau_h_total, T_HI_galactic, p_min, distance_pc.
    galactic_b_deg : float or None
        Galactic latitude (degrees). If None, raises ValueError.

    Returns
    -------
    tau_fg : float — foreground HI optical depth
    T_HI_galactic : float — foreground HI spin temperature (K)

    Raises
    ------
    ValueError
        If galactic_b_deg is None or cfg.distance_pc is invalid.
    """
    from scipy.special import erfc

    if galactic_b_deg is None:
        raise ValueError(
            "Cannot determine galactic latitude: provide a FITS file with "
            "RA/DEC coordinates, or pass galactic_b_deg explicitly.")
    if cfg.distance_pc is None or cfg.distance_pc <= 0:
        raise ValueError(
            "Cannot determine cloud distance: set distance_pc in Config.")

    D_kpc = cfg.distance_pc / 1000.0
    b_rad = np.abs(galactic_b_deg) * np.pi / 180.0

    # Li 2003 eq.9: galactic HI disk as Gaussian (Lockman 1984)
    z_pc = cfg.galactic_HI_disk_fwhm_pc
    sigma = z_pc / np.sqrt(8.0 * np.log(2.0))
    arg = D_kpc * 1000.0 * np.sin(b_rad) / (sigma * np.sqrt(2.0))
    p = erfc(arg)
    p = max(p, cfg.p_min)

    tau_fg = (1.0 - p) * cfg.tau_h_total
    return tau_fg, cfg.T_HI_galactic


def build_synthetic_hinsa(cfg, params, bg_cube, center_yx, pixel_scale_pc,
                          galactic_b_deg=None,
                          R_out_pc=None, vlsr_kms=None, n_jobs=1):
    """Forward-model a synthetic HI cube with HINSA absorption.

    The output grid always matches the input bg_cube.

    ``bg_cube`` is the **observed** HI brightness-temperature cube *without*
    the cold cloud, i.e. it already contains both galactic foreground HI
    emission and the background HI continuum.  Internally the foreground is
    stripped before the cloud RT and re-applied afterwards, so the cloud
    absorption acts on the *pure* background.

    Parameters
    ----------
    cfg : Config
    params : dict
        Cloud model parameters.
    bg_cube : ndarray, shape (n_v, ny, nx) or (ny, nx)
        Observed HI brightness-temperature (K) **without the cold cloud**,
        i.e. the sum of foreground galactic HI emission and background HI.
        - 3D: velocity-dependent.
        - 2D: broadcast to all velocities.
        - None: constant 30 K (shape derived from cfg).
    center_yx : tuple (yc, xc)
        Pixel position of cloud centre in the bg spatial grid.
    pixel_scale_pc : float
        Spatial pixel scale in pc/pixel.
    galactic_b_deg : float or None
        Galactic latitude (degrees). Required for foreground HI calculation.
        If None, raises ValueError via compute_foreground_params.
    R_out_pc : float or None
        Override cloud outer radius (pc). If None, uses cfg.R_out_pc.
    vlsr_kms : float or None
        Override cloud systemic velocity. If None, uses cfg.vlsr_kms.
    n_jobs : int

    Returns
    -------
    out_cube : 3D array (n_v, ny, nx)
        Synthetic cube: foreground+background modified by cloud RT at each pixel.
    """
    # Temporarily override cfg if requested
    orig_R_out = cfg.R_out_pc
    orig_vlsr = cfg.vlsr_kms
    if R_out_pc is not None and R_out_pc != cfg.R_out_pc:
        cfg.R_out_pc = R_out_pc
        cfg._build_shell_radii()
    if vlsr_kms is not None:
        cfg.vlsr_kms = vlsr_kms

    try:
        ns = cfg.n_shells

        # --- 1. Radial profiles ---
        n_HI, T_spin, sigma_v, v_infall_kms = compute_radial_profiles(cfg, params)

        # --- 2. Determine grid from bg_cube ---
        if bg_cube is not None and np.any(np.isfinite(bg_cube)):
            if bg_cube.ndim == 3:
                n_v, ny, nx = bg_cube.shape
            else:
                ny, nx = bg_cube.shape
                n_v = cfg.n_v_channels
        else:
            # No background: build grid from cfg
            npix = int(2 * cfg.R_out_pc / cfg.pc_per_pix) + 1
            npix = max(npix, 3) | 1
            ny = nx = npix
            n_v = cfg.n_v_channels

        v_grid = np.linspace(cfg.v_min_kms, cfg.v_max_kms, n_v)
        yc, xc = center_yx

        # --- 3. Build background array ---
        # bg_cube contains foreground+background; subtract foreground to get
        # the pure background that the cloud RT should act on.
        bg_is_3d = False
        T_bg_3d = None
        T_bg_2d = None

        if bg_cube is not None and np.any(np.isfinite(bg_cube)):
            if bg_cube.ndim == 3:
                bg_is_3d = True
                T_bg_3d = bg_cube.astype(np.float64)
            else:
                T_bg_2d = bg_cube.astype(np.float64)
        else:
            T_bg_2d = np.full((ny, nx), 30.0)

        # --- Foreground HI from galactic model (Li & Goldsmith 2003) ---
        tau_fg, T_HI_gal = compute_foreground_params(cfg, galactic_b_deg)
        exp_neg_tau_fg = np.exp(-tau_fg)
        T_fg_emission = T_HI_gal * (1.0 - exp_neg_tau_fg)

        # Strip foreground: T_bg_clean = (T_obs - T_fg_emission) / exp(-tau_fg)
        if tau_fg > 0 and exp_neg_tau_fg > 0:
            if bg_is_3d:
                T_bg_3d = (T_bg_3d - T_fg_emission) / exp_neg_tau_fg
            elif T_bg_2d is not None:
                T_bg_2d = (T_bg_2d - T_fg_emission) / exp_neg_tau_fg

        v_offset = params.get('v_offset', 0.0)
        v_rot_kms = params.get('v_rot_kms', 0.0)
        rot_pa_deg = params.get('rot_pa_deg', 0.0)

        r_outer = cfg.r_outer
        r_mid = cfg.r_mid
        pc_cm = cfg.pc_cm

        # --- 4. Impact-parameter map ---
        yy, xx = np.mgrid[0:ny, 0:nx]
        dx_map = (xx - xc).astype(float) * pixel_scale_pc
        dy_map = (yy - yc).astype(float) * pixel_scale_pc
        b_map = np.sqrt(dx_map**2 + dy_map**2)

        # --- 5. Ray-trace per pixel ---
        out_cube = np.empty((n_v, ny, nx), dtype=np.float64)

        def _compute_pixel(j, i):
            b = b_map[j, i]
            dx = dx_map[j, i]
            dy = dy_map[j, i]

            if bg_is_3d:
                T_bg_spec = T_bg_3d[:, j, i]
            else:
                T_bg_spec = T_bg_2d[j, i]

            if b >= r_outer[-1]:
                return T_bg_spec.copy() if bg_is_3d else np.full(n_v, T_bg_spec)

            layers = los_path_lengths(b, r_outer, ns)
            if not layers:
                return T_bg_spec.copy() if bg_is_3d else np.full(n_v, T_bg_spec)

            n_layer = len(layers)
            tau0_arr = np.zeros(n_layer)
            v_center_arr = np.zeros(n_layer)
            sigma_arr = np.zeros(n_layer)
            T_s_arr = np.zeros(n_layer)

            for il, (k, dl, z) in enumerate(layers):
                tau0_arr[il] = compute_layer_tau0(cfg, n_HI[k], T_spin[k], dl,
                                                   sigma_v[k])
                v_los = los_velocity(b, z, r_mid, v_infall_kms,
                                     v_rot_kms=v_rot_kms, rot_pa_deg=rot_pa_deg,
                                     dx=dx, dy=dy, r_cloud=r_outer[-1])
                v_center_arr[il] = v_los + cfg.vlsr_kms + v_offset
                sigma_arr[il] = sigma_v[k]
                T_s_arr[il] = T_spin[k]

            # --- 6. Forward RT per pixel ---
            # radiative_transfer_pixel does cloud RT only (no foreground).
            # Foreground is applied uniformly to the full cube afterwards.
            T_B = radiative_transfer_pixel(
                    v_grid, n_layer, tau0_arr, v_center_arr,
                    sigma_arr, T_s_arr, T_bg_spec
                )
            return T_B

        results = Parallel(n_jobs=n_jobs)(
            delayed(_compute_pixel)(j, i)
            for j in range(ny) for i in range(nx)
        )

        for idx, (j, i) in enumerate(
                [(j, i) for j in range(ny) for i in range(nx)]):
            out_cube[:, j, i] = results[idx]

        # Apply foreground HI uniformly to the entire cube.
        # out_cube currently contains cloud-RT(pure background).
        # Apply: T_obs = T_cloud * exp(-tau_fg) + T_fg * (1 - exp(-tau_fg))
        out_cube *= exp_neg_tau_fg
        out_cube += T_fg_emission

    finally:
        # Restore cfg
        cfg.R_out_pc = orig_R_out
        cfg._build_shell_radii()
        cfg.vlsr_kms = orig_vlsr

    return out_cube


def inverse_build_hinsa_cube(cfg, params, obs_cube, center_yx, pixel_scale_pc,
                              galactic_b_deg=None,
                              R_out_pc=None, vlsr_kms=None, n_jobs=1):
    """Inverse RT: from observed cube, recover reconstructed background T_bg.

    Uses the physical model to compute tau at each pixel, then inverts
    the RT to recover the unabsorbed background. For second-derivative fitting.

    Parameters
    ----------
    cfg : Config
    params : dict — cloud model parameters
    obs_cube : ndarray, shape (n_v, ny, nx) — observed HI cube
    center_yx : tuple (yc, xc)
    pixel_scale_pc : float — pc/pixel
    galactic_b_deg : float or None
        Galactic latitude (degrees). Required for foreground stripping.
    R_out_pc : float or None
    vlsr_kms : float or None
    n_jobs : int

    Returns
    -------
    T_bg_cube : 3D array (n_v, ny, nx) — reconstructed background
    """
    orig_R_out = cfg.R_out_pc
    orig_vlsr = cfg.vlsr_kms
    if R_out_pc is not None and R_out_pc != cfg.R_out_pc:
        cfg.R_out_pc = R_out_pc
        cfg._build_shell_radii()
    if vlsr_kms is not None:
        cfg.vlsr_kms = vlsr_kms

    try:
        ns = cfg.n_shells
        n_v, ny, nx = obs_cube.shape
        v_grid = np.linspace(cfg.v_min_kms, cfg.v_max_kms, n_v)
        yc, xc = center_yx

        # --- Radial profiles ---
        n_HI, T_spin, sigma_v, v_infall_kms = compute_radial_profiles(cfg, params)

        v_offset = params.get('v_offset', 0.0)
        v_rot_kms = params.get('v_rot_kms', 0.0)
        rot_pa_deg = params.get('rot_pa_deg', 0.0)

        # --- Foreground HI from galactic model ---
        tau_fg, T_HI_gal = compute_foreground_params(cfg, galactic_b_deg)
        exp_neg_tau_fg = np.exp(-tau_fg)

        r_outer = cfg.r_outer
        r_mid = cfg.r_mid
        pc_cm = cfg.pc_cm

        # --- Impact-parameter map ---
        yy, xx = np.mgrid[0:ny, 0:nx]
        dx_map = (xx - xc).astype(float) * pixel_scale_pc
        dy_map = (yy - yc).astype(float) * pixel_scale_pc
        b_map = np.sqrt(dx_map**2 + dy_map**2)

        # --- Inverse RT per pixel ---
        T_bg_cube = np.empty((n_v, ny, nx), dtype=np.float64)

        def _compute_pixel(j, i):
            b = b_map[j, i]
            dx = dx_map[j, i]
            dy = dy_map[j, i]
            T_obs_spec = obs_cube[:, j, i].astype(np.float64)

            # [Fix] Step 1: strip foreground HI FIRST (closest to observer)
            T_after_fg = (T_obs_spec - T_HI_gal * (1.0 - exp_neg_tau_fg)) / exp_neg_tau_fg

            if b >= r_outer[-1]:
                return T_after_fg

            layers = los_path_lengths(b, r_outer, ns)
            if not layers:
                return T_after_fg

            n_layer = len(layers)
            tau0_arr = np.zeros(n_layer)
            v_center_arr = np.zeros(n_layer)
            sigma_arr = np.zeros(n_layer)
            T_s_arr = np.zeros(n_layer)

            for il, (k, dl, z) in enumerate(layers):
                tau0_arr[il] = compute_layer_tau0(cfg, n_HI[k], T_spin[k], dl,
                                                   sigma_v[k])
                v_los = los_velocity(b, z, r_mid, v_infall_kms,
                                     v_rot_kms=v_rot_kms, rot_pa_deg=rot_pa_deg,
                                     dx=dx, dy=dy, r_cloud=r_outer[-1])
                v_center_arr[il] = v_los + cfg.vlsr_kms + v_offset
                sigma_arr[il] = sigma_v[k]
                T_s_arr[il] = T_spin[k]

            # [Fix] Step 2: peel back cloud layers using foreground-stripped spectrum
            T_bg = inverse_radiative_transfer_pixel(
                v_grid, n_layer, tau0_arr, v_center_arr,
                sigma_arr, T_s_arr, T_after_fg)
            # Physical guard: clamp reconstructed T_bg to reasonable range.
            # Allow negative values (they indicate model/data mismatch and will
            # be penalized by the R-value metric via sharp edges in d²T_bg).
            # Only cap extreme blow-up from division by tiny exp(-tau).
            T_max = float(np.max(T_obs_spec)) * 3.0 + 50.0
            np.clip(T_bg, -T_max, T_max, out=T_bg)
            return T_bg

        results = Parallel(n_jobs=n_jobs)(
            delayed(_compute_pixel)(j, i)
            for j in range(ny) for i in range(nx)
        )

        for idx, (j, i) in enumerate(
                [(j, i) for j in range(ny) for i in range(nx)]):
            T_bg_cube[:, j, i] = results[idx]

    finally:
        # Restore cfg
        cfg.R_out_pc = orig_R_out
        cfg._build_shell_radii()
        cfg.vlsr_kms = orig_vlsr

    return T_bg_cube


def compute_enclosed_mass(n_H, cfg):
    """Enclosed mass profile (M_sun) at each shell's outer boundary.

    Returns 1D array of length n_shells, where M_enc[k] = mass inside r_outer[k].
    """
    M_enc = np.zeros(cfg.n_shells)
    mass_cum_g = 0.0
    for k in range(cfg.n_shells):
        n_k = n_H[k]
        vol = cfg.shell_vol[k]
        mass_cum_g += n_k * cfg.m_H * (vol * cfg.pc_cm**3)
        M_enc[k] = mass_cum_g / cfg.M_sun_g  # M_sun
    return M_enc


def compute_radial_profiles(cfg, params):
    """Compute shell-by-shell physical profiles from parameters.

    Returns (n_HI, T_spin, sigma_v, v_infall_kms) all as 1-D arrays.
    """
    n_H = density_plummer(cfg.r_mid, params['rho0'], params['r0'], params['alpha'])
    T = temperature_plummer(cfg.r_mid, params['T0'], params['T1'], params['rT'])
    if 'f_HI' in params:
        f_HI = np.asarray(params['f_HI'], dtype=float)
    else:
        f_HI = abundance_profile_111n(cfg.n_shells, params['peak_shell'],
                                       params['multipliers'],
                                       f_HI_peak=params.get('f_HI_peak', 1.0))
    n_HI = n_H * f_HI
    T_spin = np.maximum(T, cfg.T_cmb)

    M_enc = compute_enclosed_mass(n_H, cfg)
    v_infall_kms = infall_velocity(
        cfg.r_mid * cfg.pc_cm, cfg.r_outer * cfg.pc_cm,
        params['f_ff'], M_enc, cfg.G, cfg.M_sun_g
    ) / 1e5

    sigma_thermal = np.sqrt(cfg.k_B * T / cfg.m_H) / 1e5
    sigma_v = np.sqrt(sigma_thermal**2 + params['turb_kms']**2)

    return n_HI, T_spin, sigma_v, v_infall_kms


def synthetic_spectrum_at_pixel(cfg, params, b_pc, v_grid_kms, T_bg,
                                galactic_b_deg=None):
    """Compute the full HI spectrum at a single impact parameter.

    Parameters
    ----------
    cfg : Config
    params : dict
    b_pc : float — impact parameter (pc)
    v_grid_kms : 1-D array — velocity channels (km/s)
    T_bg : float or 1-D array — background HI brightness temperature (K).
        If 1D array, must have same length as v_grid_kms.
    galactic_b_deg : float or None
        Galactic latitude (degrees). Required for foreground calculation.

    Returns
    -------
    T_B : 1-D array — brightness temperature at each velocity channel
    """
    n_HI, T_spin, sigma_v, v_infall_kms = compute_radial_profiles(cfg, params)
    ns = cfg.n_shells
    v_offset = params.get('v_offset', 0.0)
    v_rot_kms = params.get('v_rot_kms', 0.0)
    rot_pa_deg = params.get('rot_pa_deg', 0.0)

    if b_pc > cfg.r_outer[-1]:
        if np.ndim(T_bg) == 0:
            return np.full_like(v_grid_kms, T_bg)
        else:
            return np.asarray(T_bg, dtype=float).copy()

    layers = los_path_lengths(b_pc, cfg.r_outer, ns)
    if not layers:
        if np.ndim(T_bg) == 0:
            return np.full_like(v_grid_kms, T_bg)
        else:
            return np.asarray(T_bg, dtype=float).copy()

    n_layer = len(layers)
    tau0_arr = np.zeros(n_layer)
    v_center_arr = np.zeros(n_layer)
    sigma_arr = np.zeros(n_layer)
    T_s_arr = np.zeros(n_layer)

    for il, (k, dl, z) in enumerate(layers):
        tau0_arr[il] = compute_layer_tau0(cfg, n_HI[k], T_spin[k], dl,
                                           sigma_v[k])
        v_los = los_velocity(b_pc, z, cfg.r_mid, v_infall_kms,
                             v_rot_kms=v_rot_kms, rot_pa_deg=rot_pa_deg,
                             r_cloud=cfg.r_outer[-1])
        v_center_arr[il] = v_los + cfg.vlsr_kms + v_offset  # observed frame
        sigma_arr[il] = sigma_v[k]
        T_s_arr[il] = T_spin[k]

    tau_fg, T_HI_gal = compute_foreground_params(cfg, galactic_b_deg)
    T_B = radiative_transfer_pixel(
        v_grid_kms, n_layer, tau0_arr, v_center_arr,
        sigma_arr, T_s_arr, T_bg, T_HI_gal, tau_fg
    )
    return T_B


def residual_map(obs_map, model_map, weights=None):
    """Normalized residual between observed and modeled HINSA.

    Returns weighted mean of (obs-model)^2 / |obs|, normalized by the sum
    of weights.  Zero for a perfect fit; independent of map size.

    Parameters
    ----------
    obs_map, model_map : ndarray
    weights : 2D ndarray (ny, nx) or None
        Radial weights (1/r). If None, uniform weight=1.
    """
    # Resize model to match observed if spatial shapes differ
    if model_map.shape != obs_map.shape:
        from scipy.ndimage import zoom
        if model_map.ndim == 3 and obs_map.ndim == 3:
            zoom_factors = (1.0,
                            obs_map.shape[1] / model_map.shape[1],
                            obs_map.shape[2] / model_map.shape[2])
        else:
            zoom_factors = (obs_map.shape[0] / model_map.shape[0],
                            obs_map.shape[1] / model_map.shape[1])
        model_map = zoom(model_map, zoom_factors, order=1)

    mask = np.isfinite(obs_map) & np.isfinite(model_map)
    if not np.any(mask):
        return 1e10
    diff = obs_map[mask] - model_map[mask]
    denom = np.maximum(np.abs(obs_map[mask]), 0.01)
    val = diff**2 / denom

    if weights is not None:
        if model_map.ndim == 3 and weights.ndim == 2:
            w = np.broadcast_to(weights, model_map.shape)
        else:
            w = weights
        val = val * w[mask]
        w_sum = np.nansum(w[mask])
    else:
        w_sum = float(np.sum(mask))

    return float(np.nansum(val) / w_sum)


def generate_sim_hinsa(output_path, background=None, center_pixel=None,
                       galactic_b_deg=None,
                       vlsr_kms=None, R_out_pc=None, distance_pc=None,
                       rho0=5000.0, r0=0.1, alpha=2.0,
                       T0=10.0, T1=58.0, rT=0.1,
                       peak_shell=9, f_HI_peak=0.05, multipliers=None,
                       abundance=None,
                       f_ff=0.1, turb_kms=0.2, v_offset=0.0,
                       v_rot_kms=0.0, rot_pa_deg=0.0,
                       spatial_res_pc=None, vel_res_kms=None,
                       n_shells=9, n_jobs=1, verbose=True):
    """Generate a synthetic HINSA datacube with a spherical cloud model.

    Supports two modes:
    1. FITS background: place cloud in front of a real HI brightness-T cube.
       The FITS is the observed HI **without the cold cloud** (foreground +
       background combined).  Foreground is stripped internally before cloud
       RT and re-applied afterwards.
    2. Constant background: generate a uniform-T background cube automatically.

    Parameters
    ----------
    output_path : str
        Path for the output FITS file.
    background : str or None
        Path to a HI brightness-temperature FITS cube (n_v, ny, nx) that
        represents the observed sky **without the cold cloud** — i.e. the
        combined foreground galactic HI emission and background HI.
        The foreground is automatically stripped, cloud RT is applied to the
        pure background, and the foreground is re-applied to the output.
        If None, a constant-background cube is generated automatically.
    center_pixel : tuple (yc, xc) or None
        Cloud center pixel in the background cube. If None, uses cube center.
    galactic_b_deg : float or None
        Galactic latitude (degrees). If None and background FITS is provided,
        extracted from FITS header coordinates. If None and no FITS (constant
        background mode), raises ValueError.
    vlsr_kms : float or None
        Cloud systemic velocity (km/s). If None, defaults to 0.
    R_out_pc : float or None
        Cloud outer radius (pc). If None, auto-estimated from distance and
        background cube angular size.
    distance_pc : float or None
        Cloud distance (pc). If None, defaults to 140 pc.
    rho0 : float
        Central H nucleon density (cm^-3).
    r0 : float
        Plummer core radius (pc).
    alpha : float
        Plummer power-law index.
    T0 : float
        Central spin temperature (K).
    T1 : float
        Ambient spin temperature (K).
    rT : float
        Temperature transition radius (pc).
    peak_shell : int
        HI abundance peak shell (1-indexed).
    multipliers : array or None
        Abundance decrease factors per shell. If None, uses default.
    abundance : array or None
        Direct HI abundance f_HI for each shell (length = n_shells).
        Overrides peak_shell/multipliers if provided.
    f_ff : float
        Free-fall velocity fraction (infall speed = f_ff * v_ff).
    turb_kms : float
        Turbulent velocity (km/s).
    v_offset : float
        Systemic velocity offset from vlsr (km/s).
    v_rot_kms : float
        Rotation velocity at cloud radius (km/s). 0 = no rotation.
    rot_pa_deg : float
        Position angle of rotation axis (deg, N to E). 0 = rotation axis along N.
    spatial_res_pc : float or None
        Target spatial resolution (FWHM in pc). If set, the output cube is
        convolved with a Gaussian beam to this resolution. If None, no smoothing.
    vel_res_kms : float or None
        Target velocity resolution (FWHM in km/s). If set, the output cube is
        convolved with a Gaussian along the velocity axis. If None, no smoothing.
    n_shells : int
        Number of concentric shells.
    n_jobs : int
        Parallel jobs for ray-tracing.
    verbose : bool
        Print progress info.

    Returns
    -------
    out_dict : dict with keys:
        'output_path' : str -- path to saved FITS
        'out_cube' : ndarray -- the synthetic cube (n_v, ny, nx)
        'velo_kms' : ndarray -- velocity axis (km/s)
        'cfg' : Config -- the hinspheres Config used
        'params' : dict -- the model parameters used
        'hdr' : astropy.io.fits.Header -- the FITS header
    """
    from astropy.io import fits as pyfits
    from .config import Config

    if verbose:
        print("=== generate_sim_hinsa ===")

    # --- Load or build background cube ---
    if background is not None:
        with pyfits.open(background) as hdul:
            bg_cube = hdul[0].data.astype(np.float32)
            bg_hdr = hdul[0].header
        n_v, ny, nx = bg_cube.shape
        if verbose:
            print(f"Background FITS: {bg_cube.shape}")

        # Extract galactic latitude from FITS header if not provided
        if galactic_b_deg is None:
            galactic_b_deg = _compute_galactic_b(bg_hdr, center_yx=center_pixel)
            if verbose:
                print(f"  Galactic b = {galactic_b_deg:.2f} deg (from FITS header)")

        # Extract velocity axis from FITS header
        crval3 = bg_hdr.get('CRVAL3', 0.0)
        cdelt3 = bg_hdr.get('CDELT3', -200.0)
        crpix3 = bg_hdr.get('CRPIX3', 1)
        velo_kms = (crval3 + (np.arange(n_v) - (crpix3 - 1)) * cdelt3) / 1000.0

        # Pixel scale
        cdelt1 = abs(bg_hdr.get('CDELT1', Config.default_cdelt_deg))
        cdelt2 = abs(bg_hdr.get('CDELT2', Config.default_cdelt_deg))
        if distance_pc is None:
            distance_pc = Config.distance_pc
        pix_scale_rad = np.sqrt(cdelt1 * cdelt2) * np.pi / 180.0
        pixel_scale_pc = pix_scale_rad * distance_pc

        # Cloud center
        if center_pixel is not None:
            yc, xc = center_pixel
        else:
            yc, xc = ny // 2, nx // 2

        # Auto R_out from angular extent if not specified
        if R_out_pc is None:
            angular_radius_arcmin = max(nx, ny) / 2.0 * np.sqrt(cdelt1 * cdelt2) * 60.0
            R_out_pc = angular_radius_arcmin / 60.0 * np.pi / 180.0 * distance_pc
            if verbose:
                print(f"  Auto R_out = {R_out_pc:.3f} pc from cube angular size")

        out_hdr_template = bg_hdr

    else:
        # --- Constant background mode ---
        if verbose:
            print("Mode: constant background (no FITS input)")

        if galactic_b_deg is None:
            raise ValueError(
                "Constant background mode requires galactic_b_deg (degrees). "
                "No FITS coordinates available to auto-detect.")

        if distance_pc is None:
            from .config import Config
            distance_pc = Config.distance_pc
        if R_out_pc is None:
            R_out_pc = Config.R_out_pc

        vlsr_val = vlsr_kms if vlsr_kms is not None else 0.0
        v_min = vlsr_val - (Config.v_max_kms - Config.v_min_kms) / 2.0
        v_max = vlsr_val + (Config.v_max_kms - Config.v_min_kms) / 2.0
        n_v = Config.n_v_channels
        velo_kms = np.linspace(v_min, v_max, n_v)

        pixel_scale_pc = R_out_pc * 2.0 / 21.0
        npix = int(2 * R_out_pc / pixel_scale_pc) + 1
        npix = max(npix, 21) | 1
        ny, nx = npix, npix
        yc, xc = ny // 2, nx // 2

        T_bg = 40.0
        bg_cube = np.full((n_v, ny, nx), T_bg, dtype=np.float32)
        out_hdr_template = None
        if verbose:
            print(f"  Grid: {n_v} x {ny} x {nx}, T_bg={T_bg} K")

    # --- Velocity range for Config ---
    v_min_kms = float(velo_kms.min())
    v_max_kms = float(velo_kms.max())

    if vlsr_kms is None:
        vlsr_kms = 0.0

    cfg = Config(
        n_shells=n_shells,
        R_out_pc=R_out_pc,
        vlsr_kms=vlsr_kms,
        v_min_kms=v_min_kms,
        v_max_kms=v_max_kms,
        n_v_channels=n_v,
        distance_pc=distance_pc,
    )
    cfg.n_jobs = n_jobs

    if multipliers is None and abundance is None:
        multipliers = np.array([max(0.1, 1.0 - 0.1 * i) for i in range(n_shells - 1)])

    params = {
        'rho0': rho0, 'r0': r0, 'alpha': alpha,
        'T0': T0, 'T1': T1, 'rT': rT,
        'f_ff': f_ff, 'turb_kms': turb_kms, 'v_offset': v_offset,
        'v_rot_kms': v_rot_kms, 'rot_pa_deg': rot_pa_deg,
    }

    if abundance is not None:
        params['f_HI'] = np.asarray(abundance, dtype=float)
    else:
        params['peak_shell'] = peak_shell
        params['f_HI_peak'] = f_HI_peak
        params['multipliers'] = multipliers

    if verbose:
        print(f"Model: rho0={rho0}, r0={r0}, alpha={alpha}, T0={T0}, T1={T1}")
        print(f"       f_ff={f_ff}, turb={turb_kms} km/s, v_rot={v_rot_kms} km/s")
        print(f"       R_out={R_out_pc:.3f} pc, vlsr={vlsr_kms} km/s")
        print(f"       center=({yc}, {xc}), pix_scale={pixel_scale_pc:.4f} pc")
        print(f"Forward-modeling {ny}x{nx} pixels ...")

    out_cube = build_synthetic_hinsa(
        cfg, params, bg_cube, (yc, xc), pixel_scale_pc,
        galactic_b_deg=galactic_b_deg, n_jobs=n_jobs
    )

    # --- Compute optical depth range at center pixel ---
    n_HI, T_spin, sigma_v, v_infall = compute_radial_profiles(cfg, params)
    layers_center = los_path_lengths(0.0, cfg.r_outer, cfg.n_shells)
    if layers_center:
        tau_center = [compute_layer_tau0(cfg, n_HI[k], T_spin[k], dl, sigma_v[k])
                      for k, dl, z in layers_center]
        tau_total_center = sum(tau_center)
        tau_max_layer = max(tau_center)
    else:
        tau_total_center = 0.0
        tau_max_layer = 0.0

    # Also compute at 1-pixel offset for context
    b1 = pixel_scale_pc
    layers_1px = los_path_lengths(b1, cfg.r_outer, cfg.n_shells)
    if layers_1px:
        tau_1px = [compute_layer_tau0(cfg, n_HI[k], T_spin[k], dl, sigma_v[k])
                   for k, dl, z in layers_1px]
        tau_total_1px = sum(tau_1px)
    else:
        tau_total_1px = 0.0

    if verbose:
        print(f"Optical depth (center):  total={tau_total_center:.3f}, max_layer={tau_max_layer:.3f}")
        print(f"Optical depth (1px off): total={tau_total_1px:.3f}")
        if tau_max_layer > 3.0:
            print(f"  WARNING: max layer tau > 3 — inverse RT (second_derivative mode) will be unreliable")
        elif tau_total_center > 5.0:
            print(f"  WARNING: total center tau > 5 — inverse RT may be noisy at line center")

    # --- Spatial + velocity smoothing ---
    if spatial_res_pc is not None or vel_res_kms is not None:
        from scipy.ndimage import gaussian_filter
        sigma_v = 0.0
        sigma_xy = 0.0
        if vel_res_kms is not None:
            dv = abs(velo_kms[1] - velo_kms[0]) if len(velo_kms) > 1 else 1.0
            sigma_v = (vel_res_kms / dv) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        if spatial_res_pc is not None:
            sigma_xy = (spatial_res_pc / pixel_scale_pc) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        out_cube = gaussian_filter(out_cube, sigma=(sigma_v, sigma_xy, sigma_xy))
        if verbose:
            parts = []
            if spatial_res_pc is not None:
                parts.append(f"spatial={spatial_res_pc:.3f} pc ({sigma_xy:.1f} pix)")
            if vel_res_kms is not None:
                parts.append(f"velocity={vel_res_kms:.3f} km/s ({sigma_v:.1f} ch)")
            print(f"Smoothed: {', '.join(parts)}")

    if verbose:
        print(f"Done. Cube range: [{out_cube.min():.2f}, {out_cube.max():.2f}] K")
        bg_spec = bg_cube[:, yc, xc]
        absorption = bg_spec - out_cube[:, yc, xc]
        peak_idx = np.argmax(absorption)
        print(f"Center: peak_abs={absorption.max():.2f} K at v={velo_kms[peak_idx]:.2f} km/s")

    # --- Build FITS header ---
    out_hdr = pyfits.Header()
    out_hdr['SIMPLE'] = True
    out_hdr['BITPIX'] = -32
    out_hdr['NAXIS'] = 3
    out_hdr['NAXIS1'] = nx
    out_hdr['NAXIS2'] = ny
    out_hdr['NAXIS3'] = n_v

    if out_hdr_template is not None:
        out_hdr['CTYPE1'] = out_hdr_template.get('CTYPE1', 'RA--CAR')
        out_hdr['CTYPE2'] = out_hdr_template.get('CTYPE2', 'DEC-CAR')
        out_hdr['CRPIX1'] = out_hdr_template.get('CRPIX1', 1)
        out_hdr['CRPIX2'] = out_hdr_template.get('CRPIX2', 1)
        out_hdr['CRVAL1'] = out_hdr_template.get('CRVAL1', 0.0)
        out_hdr['CRVAL2'] = out_hdr_template.get('CRVAL2', 0.0)
        out_hdr['CDELT1'] = out_hdr_template.get('CDELT1', -0.025)
        out_hdr['CDELT2'] = out_hdr_template.get('CDELT2', 0.025)
    else:
        out_hdr['CTYPE1'] = 'RA--CAR'
        out_hdr['CTYPE2'] = 'DEC-CAR'
        out_hdr['CRPIX1'] = 1
        out_hdr['CRPIX2'] = 1
        out_hdr['CRVAL1'] = 0.0
        out_hdr['CRVAL2'] = 0.0
        out_hdr['CDELT1'] = -pixel_scale_pc / distance_pc * 180.0 / np.pi
        out_hdr['CDELT2'] = pixel_scale_pc / distance_pc * 180.0 / np.pi

    out_hdr['EPOCH'] = 2000.0
    out_hdr['CTYPE3'] = 'VELO-LSR'
    out_hdr['CRPIX3'] = 1
    out_hdr['CRVAL3'] = float(velo_kms[0]) * 1000.0
    out_hdr['CDELT3'] = float(velo_kms[1] - velo_kms[0]) * 1000.0 if n_v > 1 else 0.0

    out_hdr['MOD_Rhoc'] = (rho0, '[cm^-3] central H nucleon density')
    out_hdr['MOD_R0'] = (r0, '[pc] Plummer core radius')
    out_hdr['MOD_ALPH'] = (alpha, 'Plummer power-law index')
    out_hdr['MOD_T0'] = (T0, '[K] central spin temperature')
    out_hdr['MOD_T1'] = (T1, '[K] ambient spin temperature')
    out_hdr['MOD_RT'] = (rT, '[pc] temperature transition radius')
    out_hdr['MOD_ROUT'] = (R_out_pc, '[pc] cloud outer radius')
    out_hdr['MOD_FFF'] = (f_ff, 'free-fall velocity fraction')
    out_hdr['MOD_TURB'] = (turb_kms, '[km/s] turbulent velocity')
    out_hdr['MOD_VOFS'] = (v_offset, '[km/s] velocity offset from Vlsr')
    out_hdr['MOD_VLSR'] = (vlsr_kms, '[km/s] cloud systemic velocity')
    out_hdr['MOD_VROT'] = (v_rot_kms, '[km/s] rotation velocity at R_out')
    out_hdr['MOD_RPA'] = (rot_pa_deg, '[deg] rotation axis position angle')
    out_hdr['MOD_DIST'] = (distance_pc, '[pc] cloud distance')
    if abundance is not None:
        for i, val in enumerate(abundance):
            out_hdr[f'MOD_FHI{i+1}'] = (float(val), f'f_HI shell {i+1}')

    pyfits.writeto(output_path, out_cube.astype(np.float32), out_hdr, overwrite=True)
    if verbose:
        print(f"Saved: {output_path}")

    # --- Save metadata JSON alongside FITS ---
    import json
    meta_path = os.path.splitext(output_path)[0] + '.json'
    meta = {
        'output_fits': os.path.basename(output_path),
        'input_background': os.path.basename(background) if background else None,
        'center_pixel': list(center_pixel) if center_pixel is not None else [yc, xc],
        'pixel_scale_pc': float(pixel_scale_pc),
        'config': {
            'n_shells': cfg.n_shells,
            'R_out_pc': cfg.R_out_pc,
            'vlsr_kms': cfg.vlsr_kms,
            'v_min_kms': cfg.v_min_kms,
            'v_max_kms': cfg.v_max_kms,
            'n_v_channels': cfg.n_v_channels,
            'distance_pc': cfg.distance_pc,
            'beam_fwhm_arcmin': cfg.beam_fwhm_arcmin,
        },
        'params': {k: v.tolist() if hasattr(v, 'tolist') else v for k, v in params.items()},
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    if verbose:
        print(f"Metadata: {meta_path}")

    # --- Save diagnostic PNG ---
    png_path = os.path.splitext(output_path)[0] + '.png'
    _save_diagnostic_png(
        png_path, cfg, params, out_cube, velo_kms, bg_cube,
        center_yx=(yc, xc), v_offset=v_offset, abundance=abundance,
    )
    if verbose:
        print(f"Diagnostic: {png_path}")

    return {
        'output_path': output_path,
        'out_cube': out_cube,
        'velo_kms': velo_kms,
        'cfg': cfg,
        'params': params,
        'hdr': out_hdr,
        'tau_total_center': tau_total_center,
        'tau_max_layer': tau_max_layer,
        'tau_total_1px': tau_total_1px,
    }


def _save_diagnostic_png(png_path, cfg, params, out_cube, velo_kms,
                         bg_cube, center_yx, v_offset, abundance=None):
    """Save a 6-panel diagnostic figure for generate_sim_hinsa output."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .profiles import density_plummer, temperature_plummer, abundance_profile_111n

    yc, xc = center_yx
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    r_mid = cfg.r_mid  # pc

    # 1. Density profile (log-log)
    n_H = density_plummer(r_mid, params['rho0'], params['r0'], params['alpha'])
    axes[0, 0].loglog(r_mid, n_H, 'o-', color='C0')
    axes[0, 0].set_xlabel('r (pc)')
    axes[0, 0].set_ylabel('n_H (cm$^{-3}$)')
    axes[0, 0].set_title('Density')

    # 2. Temperature profile (linear-linear)
    T = temperature_plummer(r_mid, params['T0'], params['T1'], params['rT'])
    axes[0, 1].plot(r_mid, T, 'o-', color='C1')
    axes[0, 1].set_xlabel('r (pc)')
    axes[0, 1].set_ylabel('T_spin (K)')
    axes[0, 1].set_title('Temperature')

    # 3. Abundance profile
    if abundance is not None:
        f_HI = np.asarray(abundance, dtype=float)
    else:
        f_HI = abundance_profile_111n(cfg.n_shells, params['peak_shell'],
                                       params['multipliers'],
                                       f_HI_peak=params.get('f_HI_peak', 1.0))
    axes[0, 2].plot(r_mid, f_HI, 'o-', color='C2')
    axes[0, 2].set_xlabel('r (pc)')
    axes[0, 2].set_ylabel('f_HI')
    axes[0, 2].set_title('HI Abundance')

    # 4. Moment 0 map (Cloud+BG, integrate within ±1 km/s of vlsr+v_offset)
    v_center = cfg.vlsr_kms + v_offset
    dv = np.abs(velo_kms - v_center)
    mask = dv <= 1.0
    if np.any(mask):
        moment0 = np.sum(out_cube[mask, :, :], axis=0) * np.abs(velo_kms[1] - velo_kms[0])
    else:
        moment0 = np.zeros(out_cube.shape[1:])
    im = axes[1, 0].imshow(moment0, origin='lower', cmap='viridis')
    axes[1, 0].plot(xc, yc, 'w+', ms=10, mew=1.5)
    axes[1, 0].set_title(f'Moment 0 ({v_center:.1f}±1 km/s)')
    fig.colorbar(im, ax=axes[1, 0], label='K km/s')

    # 5. Absorption map (peak absorption)
    absorption = np.max(bg_cube - out_cube, axis=0)
    axes[1, 1].imshow(absorption, origin='lower', cmap='Reds')
    axes[1, 1].plot(xc, yc, 'b+', ms=10, mew=1.5)
    axes[1, 1].set_title('Peak Absorption')

    # 6. Spectrum at center pixel
    spec_obs = out_cube[:, yc, xc].astype(float)
    spec_bg = bg_cube[:, yc, xc].astype(float)
    axes[1, 2].plot(velo_kms, spec_bg, 'gray', alpha=0.5, label='Obs (no cloud)')
    axes[1, 2].plot(velo_kms, spec_obs, 'k', label='Cloud model')
    axes[1, 2].axvline(v_center, color='red', ls='--', alpha=0.5, label=f'Vlsr+voff={v_center:.1f}')
    axes[1, 2].set_xlabel('v (km/s)')
    axes[1, 2].set_ylabel('T_B (K)')
    axes[1, 2].set_title(f'Spectrum ({yc},{xc})')
    axes[1, 2].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
