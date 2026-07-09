"""
I/O, Planck loading, beam convolution, and diagnostic plotting.
"""

import numpy as np


def load_planck_av(fits_path):
    """Load Planck Av map from FITS and return (map, wcs)."""
    from astropy.io import fits
    hdul = fits.open(fits_path)
    return hdul[0].data, hdul[0].header


def load_hinsa_map(fits_path):
    """Load user-provided HINSA intensity FITS."""
    from astropy.io import fits
    hdul = fits.open(fits_path)
    return hdul[0].data, hdul[0].header


def convolve_beam(image, beam_sigma_pix):
    """Gaussian convolution matching telescope beam."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(image, beam_sigma_pix, mode='constant', cval=0.0)


def radial_profile_2d(image, center=None, bin_width=1.0):
    """Azimuthally averaged radial profile of a 2D image."""
    ny, nx = image.shape
    if center is None:
        cy, cx = ny // 2, nx // 2
    else:
        cy, cx = center
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    r_flat = r.ravel()
    vals = image.ravel()
    r_max = int(r_flat.max()) + 1
    profile = np.zeros(r_max)
    count = np.zeros(r_max)
    for ri, val in zip(r_flat, vals):
        idx = int(ri / bin_width)
        if idx < r_max:
            profile[idx] += np.nan_to_num(val, 0.0)
            count[idx] += 1.0 - float(np.isnan(val))
    count = np.maximum(count, 1)
    return np.arange(len(profile)) * bin_width, profile / count


def diagnostic_plot(cfg, params, obs_map, model_map, save_path=None):
    """Side-by-side diagnostic plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Observed HINSA map
    im0 = axes[0, 0].imshow(obs_map, origin='lower', cmap='RdYlBu_r')
    axes[0, 0].set_title('Observed HINSA')
    plt.colorbar(im0, ax=axes[0, 0])

    # Model HINSA map
    im1 = axes[0, 1].imshow(model_map, origin='lower', cmap='RdYlBu_r')
    axes[0, 1].set_title('Model HINSA')
    plt.colorbar(im1, ax=axes[0, 1])

    # Residual
    residual = obs_map - model_map
    im2 = axes[0, 2].imshow(residual, origin='lower', cmap='RdBu')
    axes[0, 2].set_title('Residual')
    plt.colorbar(im2, ax=axes[0, 2])

    # Radial profiles
    r_obs, p_obs = radial_profile_2d(obs_map)
    r_mod, p_mod = radial_profile_2d(model_map)
    axes[1, 0].plot(r_obs, p_obs, 'o-', ms=3, label='Observed')
    axes[1, 0].plot(r_mod, p_mod, 's-', ms=3, label='Model')
    axes[1, 0].set_xlabel('Radius (pix)')
    axes[1, 0].set_ylabel('HINSA (K)')
    axes[1, 0].legend()

    # Density and temperature profiles
    from .profiles import density_plummer, temperature_plummer
    r = np.linspace(0, cfg.R_out_pc, 100)
    n_H = density_plummer(r, params['rho0'], params['r0'], params['alpha'])
    T = temperature_plummer(r, params['T0'], params['T1'], params['rT'])
    ax_t = axes[1, 1].twinx() if True else axes[1, 1]
    axes[1, 1].plot(r, n_H, 'b-', label='n_H')
    axes[1, 1].set_ylabel('n_H (cm^-3)', color='b')
    ax_t.plot(r, T, 'r-', label='T')
    ax_t.set_ylabel('T (K)', color='r')
    axes[1, 1].set_xlabel('r (pc)')
    axes[1, 1].set_title('Radial Profiles')

    # HI abundance profile
    from .profiles import abundance_profile_111n
    f_HI = abundance_profile_111n(cfg.n_shells, params['peak_shell'],
                                   params['multipliers'],
                                   f_HI_peak=params.get('f_HI_peak', 1.0))
    r_mid = cfg.r_mid
    axes[1, 2].plot(r_mid, f_HI, 'o-', ms=5)
    axes[1, 2].axvline(r_mid[params['peak_shell'] - 1], ls='--', color='gray', alpha=0.5)
    axes[1, 2].set_xlabel('r (pc)')
    axes[1, 2].set_ylabel('f_HI')
    axes[1, 2].set_title('HI Abundance')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"  Saved: {save_path}")
    plt.close()
