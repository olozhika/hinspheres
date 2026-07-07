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
    radiative_transfer_pixel, los_velocity
)


def build_synthetic_hinsa(cfg, params, T_HI_true_map=None, n_jobs=1):
    """Forward-model a 2D HINSA intensity map from spherical cloud parameters.

    Parameters
    ----------
    cfg : Config
    params : dict with keys:
        rho0, r0, alpha      — density (may be pre-fitted from Planck)
        T0, T1, rT           — temperature profile
        peak_shell            — HI abundance peak (1-indexed)
        multipliers           — 1D array len = n_shells-1
        f_ff                  — free-fall fraction
        turb_kms              — turbulence (km/s)
    T_HI_true_map : 2D array or None
        Background HI map (K) at each spatial pixel.
        If None, a constant is used.
    n_jobs : int

    Returns
    -------
    hinsa_map : 2D array
        Synthetic HINSA absorption intensity (positive = absorption).
    cube : 3D array (optionally, if requested)
        Full velocity-resolved data cube.
    """
    ns = cfg.n_shells

    # --- 1. Radial profiles ---
    n_HI, T_spin, sigma_v, v_infall_kms = compute_radial_profiles(cfg, params)

    # --- 2. Velocity grid ---
    v_grid = np.linspace(cfg.v_min_kms, cfg.v_max_kms, cfg.n_v_channels)

    # --- 3. Spatial grid ---
    npix = int(2 * cfg.R_out_pc / cfg.pc_per_pix) + 1
    npix = max(npix, 3) | 1  # odd
    x = np.linspace(-cfg.R_out_pc, cfg.R_out_pc, npix)
    X, Y = np.meshgrid(x, x)
    B = np.sqrt(X**2 + Y**2)

    # --- 4. Background/foreground HI ---
    # If user provides T_HI_true_map, interpolate onto model grid (per-pixel)
    if T_HI_true_map is not None and np.any(np.isfinite(T_HI_true_map)):
        from scipy.ndimage import zoom as ndzoom
        zoom_y = T_HI_true_map.shape[0] / npix
        zoom_x = T_HI_true_map.shape[1] / npix
        T_bg_2d = ndzoom(T_HI_true_map, (zoom_y, zoom_x), order=1)
        # If zoom produced wrong shape, use nearest
        if T_bg_2d.shape != (npix, npix):
            T_bg_2d = ndzoom(T_HI_true_map, (npix / T_HI_true_map.shape[0],
                                               npix / T_HI_true_map.shape[1]),
                             order=0)
    else:
        T_bg_2d = np.full((npix, npix), 30.0)  # default Galactic HI brightness
    tau_bg = 0.1     # optically thin approximation
    tau_fg = 0.02    # foreground (typically small)

    v_offset = params.get('v_offset', 0.0)
    v_rot_kms = params.get('v_rot_kms', 0.0)
    rot_pa_deg = params.get('rot_pa_deg', 0.0)

    # --- 5. Ray-trace per pixel ---
    def _compute_pixel(ip, jp):
        b = B[ip, jp]
        dx = X[ip, jp]
        if b > cfg.r_outer[-1]:
            return 0.0

        layers = los_path_lengths(b, cfg.r_outer, ns)
        if not layers:
            return 0.0

        T_bg_px = T_bg_2d[ip, jp]
        n_layer = len(layers)
        tau0_arr = np.zeros(n_layer)
        v_center_arr = np.zeros(n_layer)
        sigma_arr = np.zeros(n_layer)
        T_s_arr = np.zeros(n_layer)

        for il, (k, dl, z) in enumerate(layers):
            tau0_arr[il] = compute_layer_tau0(n_HI[k], T_spin[k], dl,
                                               sigma_v[k], cfg.pc_cm)
            v_los = los_velocity(b, z, cfg.r_mid, v_infall_kms,
                                 v_rot_kms=v_rot_kms, rot_pa_deg=rot_pa_deg,
                                 dx=dx)
            v_center_arr[il] = v_los + cfg.vlsr_kms + v_offset
            sigma_arr[il] = sigma_v[k]
            T_s_arr[il] = T_spin[k]

        T_B = radiative_transfer_pixel(
            v_grid, n_layer, tau0_arr, v_center_arr,
            sigma_arr, T_s_arr, T_bg_px, T_bg_px, tau_bg, tau_fg
        )

        # HINSA absorption = unabsorbed HI - absorbed spectrum
        hinsa_spec = T_bg_px - T_B
        return np.nanmax(hinsa_spec)  # peak absorption

    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_pixel)(ip, jp)
        for ip in range(npix) for jp in range(npix)
    )

    hinsa_map = np.array(results).reshape(npix, npix)
    hinsa_map = np.maximum(hinsa_map, 0)
    return hinsa_map


def compute_enclosed_mass(n_H, cfg):
    """Enclosed mass profile (M_sun) at each shell's outer boundary.

    Returns 1D array of length n_shells, where M_enc[k] = mass inside r_outer[k].
    """
    M_enc = np.zeros(cfg.n_shells)
    mass_cum_g = 0.0
    for k in range(cfg.n_shells):
        n_k = n_H[k]
        vol = cfg.shell_vol[k]
        mass_cum_g += n_k * cfg.m_H2 * (vol * cfg.pc_cm**3)
        M_enc[k] = mass_cum_g / (2.0 * 1.989e33)  # M_sun
    return M_enc


def compute_radial_profiles(cfg, params):
    """Compute shell-by-shell physical profiles from parameters.

    Returns (n_HI, T_spin, sigma_v, v_infall_kms, M_enc) all as 1-D arrays.
    """
    n_H = density_plummer(cfg.r_mid, params['rho0'], params['r0'], params['alpha'])
    T = temperature_plummer(cfg.r_mid, params['T0'], params['T1'], params['rT'])
    if 'f_HI' in params:
        f_HI = np.asarray(params['f_HI'], dtype=float)
    else:
        f_HI = abundance_profile_111n(cfg.n_shells, params['peak_shell'], params['multipliers'])
    n_HI = n_H * f_HI
    T_spin = np.maximum(T, cfg.T_cmb)

    M_enc = compute_enclosed_mass(n_H, cfg)
    v_infall_kms = infall_velocity(
        cfg.r_mid * cfg.pc_cm, cfg.r_outer * cfg.pc_cm,
        params['f_ff'], M_enc, cfg.G, cfg.m_H2 / 2.0
    ) / 1e5

    sigma_thermal = np.sqrt(cfg.k_B * T / (cfg.m_H * cfg.mu)) / 1e5
    sigma_v = np.sqrt(sigma_thermal**2 + params['turb_kms']**2)

    return n_HI, T_spin, sigma_v, v_infall_kms


def synthetic_spectrum_at_pixel(cfg, params, b_pc, v_grid_kms, T_bg):
    """Compute the full HI spectrum at a single impact parameter.

    Parameters
    ----------
    cfg : Config
    params : dict
    b_pc : float — impact parameter (pc)
    v_grid_kms : 1-D array — velocity channels (km/s)
    T_bg : float or 1-D array — background HI brightness temperature (K).
        If 1D array, must have same length as v_grid_kms.

    Returns
    -------
    T_B : 1-D array — brightness temperature at each velocity channel
    """
    n_HI, T_spin, sigma_v, v_infall_kms = compute_radial_profiles(cfg, params)
    ns = cfg.n_shells
    v_offset = params.get('v_offset', 0.0)

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
        tau0_arr[il] = compute_layer_tau0(n_HI[k], T_spin[k], dl,
                                           sigma_v[k], cfg.pc_cm)
        v_los = los_velocity(b_pc, z, cfg.r_mid, v_infall_kms)
        v_center_arr[il] = v_los + cfg.vlsr_kms + v_offset  # observed frame
        sigma_arr[il] = sigma_v[k]
        T_s_arr[il] = T_spin[k]

    tau_bg = 0.1
    tau_fg = 0.02
    T_fg_scalar = np.mean(T_bg) if np.ndim(T_bg) > 0 else T_bg
    T_B = radiative_transfer_pixel(
        v_grid_kms, n_layer, tau0_arr, v_center_arr,
        sigma_arr, T_s_arr, T_bg, T_fg_scalar, tau_bg, tau_fg
    )
    return T_B


def residual_map(obs_map, model_map):
    """Chi-squared residual between observed and modeled HINSA."""
    # Resize model to match observed if shapes differ
    if model_map.shape != obs_map.shape:
        from scipy.ndimage import zoom
        zoom_factors = (obs_map.shape[0] / model_map.shape[0],
                        obs_map.shape[1] / model_map.shape[1])
        model_map = zoom(model_map, zoom_factors, order=1)
    mask = (obs_map > 0) & np.isfinite(obs_map) & (model_map > 0)
    if not np.any(mask):
        return 1e10
    diff = obs_map[mask] - model_map[mask]
    return np.nansum(diff**2 / np.maximum(obs_map[mask], 0.01))


def forward_model_cube(cfg, params, bg_cube, velo_kms, center_yx,
                       pixel_scale_pc, vlsr_kms=None, R_out_pc=None):
    """Forward-model a synthetic HI cube with HINSA absorption.

    Takes a background HI brightness-temperature cube, places a spherical
    cloud model in front of it, computes ray-tracing through the cloud at
    every spatial pixel, and returns a cube of the same shape containing
    the resulting spectrum (background modified by cloud absorption/emission).

    Parameters
    ----------
    cfg : Config
        Must have n_shells, R_out_pc, etc. Set vlsr_kms via cfg or vlsr_kms arg.
    params : dict
        Cloud model parameters (rho0, r0, alpha, T0, T1, rT, peak_shell,
        multipliers, f_ff, turb_kms, v_offset, ...).
    bg_cube : ndarray, shape (n_v, ny, nx)
        Background HI brightness-temperature cube (K). The velocity axis
        must correspond to velo_kms.
    velo_kms : 1D array, shape (n_v,)
        Velocity axis of bg_cube in km/s (observed frame, e.g. LSR).
    center_yx : tuple (yc, xc)
        Pixel position of the cloud centre in the bg_cube spatial grid.
    pixel_scale_pc : float
        Spatial pixel scale in pc/pixel.
    vlsr_kms : float or None
        Cloud systemic velocity (km/s).  If None, taken from cfg.vlsr_kms.
    R_out_pc : float or None
        Override cloud outer radius (pc). If None, uses cfg.R_out_pc.
        Useful when CO-observed radius differs from HI extent.

    Returns
    -------
    out_cube : ndarray, shape (n_v, ny, nx)
        Synthetic cube: background spectrum modified by cloud RT at each pixel.
    """
    if vlsr_kms is None:
        vlsr_kms = cfg.vlsr_kms
    # Temporarily override R_out_pc and rebuild shell radii if requested
    orig_R_out = cfg.R_out_pc
    if R_out_pc is not None and R_out_pc != cfg.R_out_pc:
        cfg.R_out_pc = R_out_pc
        cfg._build_shell_radii()
    # Temporarily set vlsr on cfg for los_velocity / v_center computation
    orig_vlsr = cfg.vlsr_kms
    cfg.vlsr_kms = vlsr_kms

    n_v, ny, nx = bg_cube.shape
    yc, xc = center_yx

    # --- 1. Radial profiles (same for all pixels) ---
    n_HI, T_spin, sigma_v, v_infall_kms = compute_radial_profiles(cfg, params)
    v_offset = params.get('v_offset', 0.0)
    v_grid = np.asarray(velo_kms, dtype=float)

    # Pre-compute shell geometry
    r_outer = cfg.r_outer
    r_mid = cfg.r_mid
    ns = cfg.n_shells
    pc_cm = cfg.pc_cm

    # --- 2. Build impact-parameter and offset maps ---
    yy, xx = np.mgrid[0:ny, 0:nx]
    dx_map = (xx - xc).astype(float) * pixel_scale_pc  # pc, x-offset from center
    b_map = np.sqrt(dx_map**2 + ((yy - yc).astype(float) * pixel_scale_pc)**2)

    # Rotation parameters
    v_rot_kms = params.get('v_rot_kms', 0.0)
    rot_pa_deg = params.get('rot_pa_deg', 0.0)

    # --- 3. Ray-trace per pixel ---
    out_cube = np.empty_like(bg_cube, dtype=np.float32)

    def _compute_pixel(j, i):
        b = b_map[j, i]
        dx = dx_map[j, i]
        T_bg_spec = bg_cube[:, j, i].astype(float)

        if b >= r_outer[-1]:
            return T_bg_spec

        layers = los_path_lengths(b, r_outer, ns)
        if not layers:
            return T_bg_spec

        n_layer = len(layers)
        tau0_arr = np.zeros(n_layer)
        v_center_arr = np.zeros(n_layer)
        sigma_arr = np.zeros(n_layer)
        T_s_arr = np.zeros(n_layer)

        for il, (k, dl, z) in enumerate(layers):
            tau0_arr[il] = compute_layer_tau0(n_HI[k], T_spin[k], dl,
                                               sigma_v[k], pc_cm)
            v_los = los_velocity(b, z, r_mid, v_infall_kms,
                                 v_rot_kms=v_rot_kms, rot_pa_deg=rot_pa_deg,
                                 dx=dx)
            v_center_arr[il] = v_los + vlsr_kms + v_offset
            sigma_arr[il] = sigma_v[k]
            T_s_arr[il] = T_spin[k]

        tau_bg = 0.1
        tau_fg = 0.02
        T_B = radiative_transfer_pixel(
            v_grid, n_layer, tau0_arr, v_center_arr,
            sigma_arr, T_s_arr, T_bg_spec, np.mean(T_bg_spec), tau_bg, tau_fg
        )
        return T_B.astype(np.float32)

    from joblib import Parallel, delayed
    n_jobs = getattr(cfg, 'n_jobs', 1)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_pixel)(j, i)
        for j in range(ny) for i in range(nx)
    )

    for idx, j in enumerate(range(ny)):
        for i in range(nx):
            flat_idx = idx * nx + i
            out_cube[:, j, i] = results[flat_idx]

    # Restore original cfg values
    cfg.vlsr_kms = orig_vlsr
    cfg.R_out_pc = orig_R_out
    cfg._build_shell_radii()
    return out_cube


def generate_sim_hinsa(output_path, background=None, center_pixel=None,
                       vlsr_kms=None, R_out_pc=None, distance_pc=None,
                       rho0=500, r0=0.05, alpha=1.0,
                       T0=10.0, T1=40.0, rT=0.15,
                       peak_shell=1, multipliers=None, abundance=None,
                       f_ff=0.1, turb_kms=0.15, v_offset=0.0,
                       v_rot_kms=0.0, rot_pa_deg=0.0,
                       spatial_res_pc=None, vel_res_kms=None,
                       n_shells=9, n_jobs=1, verbose=True):
    """Generate a synthetic HINSA datacube with a spherical cloud model.

    Supports two modes:
    1. FITS background: place cloud in front of a real HI brightness-T cube.
    2. Constant background: generate a uniform-T background cube automatically.

    Parameters
    ----------
    output_path : str
        Path for the output FITS file.
    background : str or None
        Path to a background HI brightness-temperature FITS cube (n_v, ny, nx).
        If None, a constant-background cube is generated automatically.
    center_pixel : tuple (yc, xc) or None
        Cloud center pixel in the background cube. If None, uses cube center.
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

        # Extract velocity axis from FITS header
        crval3 = bg_hdr.get('CRVAL3', 0.0)
        cdelt3 = bg_hdr.get('CDELT3', -200.0)
        crpix3 = bg_hdr.get('CRPIX3', 1)
        velo_kms = (crval3 + (np.arange(n_v) - (crpix3 - 1)) * cdelt3) / 1000.0

        # Pixel scale
        cdelt1 = abs(bg_hdr.get('CDELT1', 0.025))
        cdelt2 = abs(bg_hdr.get('CDELT2', 0.025))
        if distance_pc is None:
            distance_pc = 140.0
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

        if distance_pc is None:
            distance_pc = 140.0
        if R_out_pc is None:
            R_out_pc = 0.91

        vlsr_val = vlsr_kms if vlsr_kms is not None else 0.0
        v_min = vlsr_val - 15.0
        v_max = vlsr_val + 15.0
        n_v = 301
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
        params['multipliers'] = multipliers

    if verbose:
        print(f"Model: rho0={rho0}, r0={r0}, alpha={alpha}, T0={T0}, T1={T1}")
        print(f"       f_ff={f_ff}, turb={turb_kms} km/s, v_rot={v_rot_kms} km/s")
        print(f"       R_out={R_out_pc:.3f} pc, vlsr={vlsr_kms} km/s")
        print(f"       center=({yc}, {xc}), pix_scale={pixel_scale_pc:.4f} pc")
        print(f"Forward-modeling {ny}x{nx} pixels ...")

    out_cube = forward_model_cube(
        cfg, params, bg_cube, velo_kms, (yc, xc), pixel_scale_pc
    )

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
        f_HI = abundance_profile_111n(cfg.n_shells, params['peak_shell'], params['multipliers'])
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
    axes[1, 2].plot(velo_kms, spec_bg, 'gray', alpha=0.5, label='Background')
    axes[1, 2].plot(velo_kms, spec_obs, 'k', label='Cloud+BG')
    axes[1, 2].axvline(v_center, color='red', ls='--', alpha=0.5, label=f'Vlsr+voff={v_center:.1f}')
    axes[1, 2].set_xlabel('v (km/s)')
    axes[1, 2].set_ylabel('T_B (K)')
    axes[1, 2].set_title(f'Spectrum ({yc},{xc})')
    axes[1, 2].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
