"""
Data preparation for HINSA spherical RT fitting.

Extracts a sub-cube from an HI FITS datacube, fits a polynomial baseline
to separate the background from HINSA absorption, and saves the products
needed by the fitting pipeline.

A generalized version of hinsapack.prepare_hinspheres_input that does NOT
depend on any hinsapack modules.  All sub-cube extraction, baseline fitting,
interactive mask selection, and FITS I/O are implemented from scratch using
only astropy, numpy, scipy, joblib, and matplotlib.

Usage
-----
    from astropy.coordinates import SkyCoord
    from hinspheres.prepare import prepare_hinspheres_input

    result = prepare_hinspheres_input(
        target_id='B227',
        datacube_path='./ot1_hi_destripe.fits',
        center_coord=SkyCoord(ra=83.5, dec=-5.0, unit='deg'),
        vlsr_kms=10.5,
    )
"""

import os
import sys
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord

# ---------------------------------------------------------------------------
# Sub-cube extraction (replaces hinsapack.io_cube)
# ---------------------------------------------------------------------------

def _make_wcs2d(header):
    """Strip the 3rd (spectral) axis from a FITS header to get a 2D spatial WCS."""
    wcs_header = fits.Header(header)
    wcs_header['NAXIS'] = 2
    for key in ['NAXIS3', 'CTYPE3', 'CRVAL3', 'CDELT3', 'CRPIX3', 'CROTA3', 'CUNIT3']:
        if key in wcs_header:
            del wcs_header[key]
    if 'CROTA1' not in wcs_header and 'CROTA2' not in wcs_header:
        wcs_header['CUNIT1'] = 'deg'
        wcs_header['CUNIT2'] = 'deg'
    return wcs_header


def _make_wcscut(lim, wcs2d):
    """Shift CRPIX after cropping a spatial sub-region."""
    wcs2d.wcs.crpix[0] = wcs2d.wcs.crpix[0] - round(lim[0][0])
    wcs2d.wcs.crpix[1] = wcs2d.wcs.crpix[1] - round(lim[1][0])
    return wcs2d


def _get_using_data(hdu, ra_deg, dec_deg, vlsr_ms,
                    x_radius_arcmin, y_radius_arcmin, z_radius_ms, all_z=0):
    """
    Slice a 3D rectangular sub-cube from an FITS HDU around (ra, dec, vlsr).

    Returns (lim, data) where lim = [x_lim, y_lim, z_lim] in pixel coords.
    """
    wcs = WCS(hdu.header)
    center = wcs.wcs_world2pix([[ra_deg, dec_deg, vlsr_ms]], 0)[0]

    cdelt1 = hdu.header.get('CDELT1', -1.0)
    cdelt2 = hdu.header.get('CDELT2', 1.0)
    cdelt3 = hdu.header.get('CDELT3', 1.0)

    x_lim = [
        center[0] + (x_radius_arcmin / 60.0) / cdelt1,
        center[0] - (x_radius_arcmin / 60.0) / cdelt1,
    ]
    y_lim = [
        center[1] - (y_radius_arcmin / 60.0) / cdelt2,
        center[1] + (y_radius_arcmin / 60.0) / cdelt2,
    ]
    z_lim = [
        center[2] - z_radius_ms / cdelt3,
        center[2] + z_radius_ms / cdelt3,
    ]

    z_start, z_end = round(min(z_lim)), round(max(z_lim)) + 1
    y_start, y_end = round(min(y_lim)), round(max(y_lim)) + 1
    x_start, x_end = round(min(x_lim)), round(max(x_lim)) + 1

    z_start, z_end = max(0, z_start), min(hdu.data.shape[0], z_end)
    y_start, y_end = max(0, y_start), min(hdu.data.shape[1], y_end)
    x_start, x_end = max(0, x_start), min(hdu.data.shape[2], x_end)

    if all_z == 1:
        using_data = hdu.data[:, y_start:y_end, x_start:x_end]
    else:
        using_data = hdu.data[z_start:z_end, y_start:y_end, x_start:x_end]

    return [x_lim, y_lim, z_lim], using_data


def _extract_hi_subcube(datacube_fits, ra_deg, dec_deg, vlsr_ms,
                        spatial_radius_arcmin, velo_radius_ms):
    """
    Open a FITS datacube and extract a sub-cube around (ra, dec, vlsr).

    Returns (data_3d, wcs_2d, velo_axis_ms) where
      data_3d   : ndarray (nv, ny, nx)
      wcs_2d    : astropy.wcs.WCS (2D spatial)
      velo_axis_ms : 1D velocity array in m/s
    """
    with fits.open(datacube_fits) as hdul:
        hdu = hdul[0]
        hdr = hdu.header

        velocity_axis = (np.arange(0.0, hdr['NAXIS3']) * hdr.get('CDELT3', 1.0)
                         + hdr.get('CRVAL3', 0.0))

        lim, using_data = _get_using_data(
            hdu, ra_deg, dec_deg, vlsr_ms,
            spatial_radius_arcmin, spatial_radius_arcmin, velo_radius_ms,
        )

        wcs2d = _make_wcscut(lim, WCS(_make_wcs2d(hdr)))

        z_start = max(0, round(min(lim[2])))
        z_end = min(len(velocity_axis), round(max(lim[2])) + 1)
        velo_range = velocity_axis[z_start:z_end]

        return using_data, wcs2d, velo_range


# ---------------------------------------------------------------------------
# Baseline fitting (replaces hinsapack.fitters)
# ---------------------------------------------------------------------------

def _get_pure_hinsa_by_3polyfit(spec, velo_list, velo_range, poly_order=5,
                                 peak_unmask_radius=1, extra_mask_ranges=None,
                                 exclude_channels=None, max_unmask_iters=5):
    """
    Fit a polynomial background and subtract it to isolate the HINSA absorption dip.

    Parameters
    ----------
    spec : 1D array — observed spectrum
    velo_list : 1D array — velocity axis (m/s)
    velo_range : tuple(float, float) — mask range (m/s) around absorption
    poly_order : int — polynomial order
    peak_unmask_radius : int — initial pixels around peak to unmask
    extra_mask_ranges : list[tuple(float,float)] or None
    exclude_channels : 1D bool array or None
        Per-channel exclusion (True = exclude from fit AND residual).  Used for
        spatially interpolated masks (complex mode) where different pixels need
        different excluded velocity channels.
    max_unmask_iters : int
    """
    def _fit_with_mask(mask_channels):
        n_pts = mask_channels.sum()
        if n_pts <= poly_order:
            return np.full_like(spec, np.nanmax(spec))
        coefficients = np.polyfit(velo_list[mask_channels], spec[mask_channels], poly_order)
        poly_func = np.poly1d(coefficients)
        return poly_func(velo_list)

    mask = ~((velo_list > velo_range[0]) & (velo_list < velo_range[1]))

    if extra_mask_ranges is not None:
        for (lo, hi) in extra_mask_ranges:
            mask &= ~((velo_list > lo) & (velo_list < hi))
    if exclude_channels is not None:
        mask &= ~np.asarray(exclude_channels, dtype=bool)

    poly_vals = _fit_with_mask(mask)

    hinsa_mask = (velo_list > velo_range[0]) & (velo_list < velo_range[1])
    current_radius = peak_unmask_radius
    for _ in range(max_unmask_iters):
        if np.max(poly_vals) >= np.max(spec):
            break
        hinsa_spec = spec.copy()
        hinsa_spec[~hinsa_mask] = -np.inf
        peak_idx = np.argmax(hinsa_spec)
        unmask_lo = max(0, peak_idx - current_radius)
        unmask_hi = min(len(spec), peak_idx + current_radius + 1)
        mask[unmask_lo:unmask_hi] = True
        poly_vals = _fit_with_mask(mask)
        current_radius *= 2

    residual = spec - poly_vals
    # Extra masked regions are also excluded from the residual
    # (e.g. other HINSA structures we do not care about).
    if extra_mask_ranges is not None:
        for (lo, hi) in extra_mask_ranges:
            residual[(velo_list > lo) & (velo_list < hi)] = 0.0
    if exclude_channels is not None:
        residual[np.asarray(exclude_channels, dtype=bool)] = 0.0
    return residual


def _get_pure_hinsa_by_ngauss(spec, velo_list, velo_range, n_components=2,
                                peak_unmask_radius=1, extra_mask_ranges=None,
                                max_unmask_iters=5):
    """
    Fit a Gaussian-mixture background and subtract it to isolate HINSA.

    Falls back to polynomial fitting if curve_fit fails or insufficient data.
    """
    from scipy.optimize import curve_fit

    mask = ~((velo_list > velo_range[0]) & (velo_list < velo_range[1]))
    if extra_mask_ranges is not None:
        for (lo, hi) in extra_mask_ranges:
            mask &= ~((velo_list > lo) & (velo_list < hi))

    velo_fit = velo_list[mask]
    spec_fit = spec[mask]

    if len(velo_fit) < 3 * n_components:
        return _get_pure_hinsa_by_3polyfit(spec, velo_list, velo_range,
                                            poly_order=5,
                                            peak_unmask_radius=peak_unmask_radius,
                                            extra_mask_ranges=extra_mask_ranges)

    def _ngauss(v, *params):
        result = np.zeros_like(v)
        for i in range(n_components):
            A = params[3 * i]
            mu = params[3 * i + 1]
            sigma = params[3 * i + 2]
            result += A * np.exp(-(v - mu) ** 2 / (2 * sigma ** 2))
        return result

    def _fit_with_mask(m):
        v_m = velo_list[m]
        s_m = spec[m]
        if len(v_m) < 3 * n_components:
            return None
        v_lo, v_hi = velo_list.min(), velo_list.max()
        p0 = []
        for i in range(n_components):
            mu_i = v_lo + (v_hi - v_lo) * (i + 1) / (n_components + 1)
            A_i = np.max(s_m) * 0.8
            sigma_i = (v_hi - v_lo) / (4 * n_components)
            p0.extend([A_i, mu_i, sigma_i])
        lo_bounds = []
        hi_bounds = []
        for i in range(n_components):
            lo_bounds.extend([0, v_lo, 1e-3])
            hi_bounds.extend([np.inf, v_hi, v_hi - v_lo])
        try:
            popt, _ = curve_fit(_ngauss, v_m, s_m, p0=p0,
                                bounds=(lo_bounds, hi_bounds), maxfev=5000)
            return _ngauss(velo_list, *popt)
        except (RuntimeError, ValueError):
            return None

    gauss_vals = _fit_with_mask(mask)
    if gauss_vals is None:
        return _get_pure_hinsa_by_3polyfit(spec, velo_list, velo_range,
                                            poly_order=5,
                                            peak_unmask_radius=peak_unmask_radius)

    hinsa_mask = (velo_list > velo_range[0]) & (velo_list < velo_range[1])
    current_radius = peak_unmask_radius
    for _ in range(max_unmask_iters):
        if np.max(gauss_vals) >= np.max(spec):
            break
        hinsa_spec = spec.copy()
        hinsa_spec[~hinsa_mask] = -np.inf
        peak_idx = np.argmax(hinsa_spec)
        unmask_lo = max(0, peak_idx - current_radius)
        unmask_hi = min(len(spec), peak_idx + current_radius + 1)
        mask[unmask_lo:unmask_hi] = True
        gauss_vals_new = _fit_with_mask(mask)
        if gauss_vals_new is not None:
            gauss_vals = gauss_vals_new
        current_radius *= 2

    return spec - gauss_vals


def _get_pure_hinsa_cube(using_data, velo_new, vlsr, ignore_radius_ms=3000.0,
                           poly_order=5, peak_unmask_radius=1, n_jobs=1,
                           extra_mask_ranges=None):
    """
    Pixel-by-pixel baseline fitting across a 3D cube to produce a HINSA
    absorption cube (positive = absorption).

    Parameters
    ----------
    poly_order : int
        Positive → polynomial of that order; negative → |poly_order| Gaussians.
    """
    from joblib import Parallel, delayed

    ny, nx = using_data.shape[1], using_data.shape[2]
    velo_range = [vlsr - ignore_radius_ms, vlsr + ignore_radius_ms]

    use_gauss = poly_order < 0
    n_gauss = abs(poly_order) if use_gauss else 0

    def _fit_pixel(j, i):
        spec_ij = using_data[:, j, i]
        if np.all(~np.isfinite(spec_ij)) or np.nanmax(np.abs(spec_ij)) < 1e-10:
            return np.zeros(len(spec_ij))
        try:
            if use_gauss:
                return _get_pure_hinsa_by_ngauss(
                    spec_ij, velo_new, velo_range,
                    n_components=n_gauss,
                    peak_unmask_radius=peak_unmask_radius,
                    extra_mask_ranges=extra_mask_ranges,
                )
            else:
                return _get_pure_hinsa_by_3polyfit(
                    spec_ij, velo_new, velo_range,
                    poly_order=poly_order,
                    peak_unmask_radius=peak_unmask_radius,
                    extra_mask_ranges=extra_mask_ranges,
                )
        except Exception:
            return np.zeros(len(spec_ij))

    results = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_fit_pixel)(j, i) for j in range(ny) for i in range(nx))

    pure_hinsa = np.zeros(using_data.shape)
    for idx, (j, i) in enumerate([(j, i) for j in range(ny) for i in range(nx)]):
        pure_hinsa[:, j, i] = results[idx]

    pure_hinsa = -pure_hinsa  # convert dip (negative) → absorption (positive)
    return pure_hinsa


# ---------------------------------------------------------------------------
# Complex interactive mask selection (grid of 3x3-binned spectra)
# ---------------------------------------------------------------------------

def _bin_3x3_spec(using_data, cell3=3):
    """Return (cell_specs, cell_pos) for a grid of 3x3-binned spectra.

    cell_specs : list of 1D arrays — binned spectrum per cell (ascending v).
    cell_pos   : list of (x0, y0, cx, cy) — cell pixel bounds + center.
    """
    n_v, ny, nx = using_data.shape
    xg = np.arange(0, nx, cell3)
    yg = np.arange(0, ny, cell3)
    cell_specs, cell_pos = [], []
    for x0 in xg:
        for y0 in yg:
            x1 = min(x0 + cell3, nx)
            y1 = min(y0 + cell3, ny)
            sp = np.nanmean(using_data[:, y0:y1, x0:x1], axis=(1, 2))
            cell_specs.append(sp)
            cell_pos.append((x0, y0, x0 + (x1 - x0) / 2.0, y0 + (y1 - y0) / 2.0))
    return cell_specs, cell_pos


def _interactive_grid_mask_selection(using_data, velo_kms, vlsr_kms, target_id):
    """Interactive per-cell mask selection on a grid of 3x3-binned spectra.

    Stage 1 (HINSA main mask): click a cell -> its spectrum is loaded into the
    editor panel; click TWO velocity points to set the main HINSA mask for that
    cell, then auto-return to cell picking.  Click another cell to set another.
    Re-click an already-set cell to redefine it.  Right-click / Esc clears that
    cell.  Press E to advance to Stage 2.

    Stage 2 (extra excluded regions): click a cell -> click velocity point pairs
    (each pair = one extra excluded region).  Press Enter to exit that cell and
    return to picking; click another cell to continue.  Press E to finish.

    Returns a list of anchors:
      [{'cy': cy, 'cx': cx, 'main': (lo, hi) or None, 'extras': [(lo,hi), ...]}, ...]
    where (cy, cx) is the cell centre in pixel coords and velocities are km/s.
    """
    import matplotlib
    _orig_backend = matplotlib.get_backend().lower()
    _in_jupyter = ('zmq' in _orig_backend or 'ipympl' in _orig_backend
                   or 'nbagg' in _orig_backend or 'inline' in _orig_backend
                   or 'ipykernel' in sys.modules)
    _has_display = os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')
    if _in_jupyter:
        matplotlib.use('nbagg')
    elif _has_display:
        try:
            matplotlib.use('TkAgg')
        except Exception:
            try:
                matplotlib.use('Qt5Agg')
            except Exception:
                raise RuntimeError('No interactive backend available for complex masking')
    else:
        raise RuntimeError('No display detected; complex masking requires a GUI. '
                           'Use polyfit_mask_kms=-1 (simple) or >0 (fixed).')

    from matplotlib import pyplot as plt
    from matplotlib.gridspec import GridSpec

    n_v, ny, nx = using_data.shape
    velo_kms = np.asarray(velo_kms, dtype=float)
    if velo_kms[-1] < velo_kms[0]:
        velo_kms = velo_kms[::-1]
        using_data = using_data[::-1]

    cell_specs, cell_pos = _bin_3x3_spec(using_data)
    bg_map = np.nanmean(using_data, axis=0)

    # anchor state keyed by (iy, ix) cell grid index
    anchors = {}          # (iy,ix) -> {'main': (lo,hi)|None, 'extras': [(lo,hi),...]}
    stage = 1
    mode = 'pick'         # 'pick' | 'edit'
    active = None         # (iy, ix) currently being edited
    pending = []          # velocity clicks in edit mode
    finished = False

    fig = plt.figure(figsize=(13, 7))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.15, 1.0], wspace=0.15)
    ax_map = fig.add_subplot(gs[0])
    ax_edit = fig.add_subplot(gs[1])

    vbg_lo, vbg_hi = np.nanpercentile(bg_map, [2, 98])
    ax_map.imshow(bg_map, origin='lower', cmap='cividis',
                  vmin=vbg_lo, vmax=vbg_hi, aspect='equal', alpha=0.4)
    # grid lines every 3 px
    for gx in range(0, nx, 3):
        ax_map.axvline(gx - 0.5, color='0.7', lw=0.3)
    for gy in range(0, ny, 3):
        ax_map.axhline(gy - 0.5, color='0.7', lw=0.3)

    cell_artist = {}      # (iy,ix) -> (patch, spec_lines)
    cell_spec_by_key = {}  # (iy,ix) -> binned spectrum
    for k, ((x0, y0, cx, cy), sp) in enumerate(zip(cell_pos, cell_specs)):
        iy, ix = y0 // 3, x0 // 3
        cell_spec_by_key[(iy, ix)] = sp
        # mini spectrum normalized into cell box
        sp_min, sp_max = np.nanmin(sp), np.nanmax(sp)
        spn = (sp - sp_min) / max(sp_max - sp_min, 1e-12)
        vn = (velo_kms - velo_kms.min()) / max(velo_kms.max() - velo_kms.min(), 1e-12)
        lx = x0 + 0.4 + vn * (min(x0 + 3, nx) - x0 - 0.8)
        ly = y0 + 0.4 + spn * (min(y0 + 3, ny) - y0 - 0.8)
        line, = ax_map.plot(lx, ly, color='steelblue', lw=0.6, alpha=0.7)
        cell_artist[(iy, ix)] = ([], [line])
    ax_map.set_xlim(-0.5, nx - 0.5)
    ax_map.set_ylim(-0.5, ny - 0.5)
    ax_map.set_xlabel('Pixel X'); ax_map.set_ylabel('Pixel Y')

    ax_edit.set_xlabel('V$_{\\rm LSR}$ (km s$^{-1}$)')
    ax_edit.set_ylabel('T$_\\mathrm{B}$ (K)')
    ax_edit.axvline(vlsr_kms, color='red', ls='--', alpha=0.5)
    ax_edit.grid(True, alpha=0.3)
    editor_spec = None
    editor_main_line = None
    editor_extra_lines = []

    def _cell_at(px, py):
        ix = int(np.floor(px / 3.0))
        iy = int(np.floor(py / 3.0))
        if 0 <= ix < (nx + 2) // 3 and 0 <= iy < (ny + 2) // 3:
            return (iy, ix)
        return None

    def _load_editor(spec, cell=None, clear=True):
        nonlocal editor_spec, editor_main_line, editor_extra_lines
        editor_spec = spec
        ax_edit.cla()
        ax_edit.plot(velo_kms, spec, '-', color='navy', lw=1.5)
        ax_edit.axvline(vlsr_kms, color='red', ls='--', alpha=0.5,
                        label=f'V_LSR={vlsr_kms:.1f}')
        editor_main_line = None
        editor_extra_lines = []
        if cell is not None and cell in anchors:
            a = anchors[cell]
            if a['main'] is not None:
                lo, hi = a['main']
                editor_main_line = ax_edit.axvspan(lo, hi, color='orange', alpha=0.15)
            for (elo, ehi) in a['extras']:
                editor_extra_lines.append(
                    ax_edit.axvspan(elo, ehi, color='magenta', alpha=0.15))
        ax_edit.set_title(f'Cell {cell} spectrum' if cell is not None
                          else 'Central average')
        ax_edit.grid(True, alpha=0.3)
        ax_edit.legend(fontsize=8, loc='upper right')
        fig.canvas.draw_idle()

    def _redraw_map():
        for (iy, ix), (patches, lines) in cell_artist.items():
            for p in patches:
                p.remove()
            patches.clear()
            a = anchors.get((iy, ix))
            col = None
            if a is not None:
                if a['main'] is not None:
                    col = 'orange'
                if a['extras']:
                    col = 'magenta' if col is None else 'crimson'
            if col is not None:
                p = ax_map.add_patch(plt.Rectangle(
                    (ix * 3 - 0.5, iy * 3 - 0.5), 3, 3, fill=False,
                    edgecolor=col, lw=2.0))
                patches.append(p)
            if (iy, ix) == active:
                p = ax_map.add_patch(plt.Rectangle(
                    (ix * 3 - 0.5, iy * 3 - 0.5), 3, 3, fill=False,
                    edgecolor='lime', lw=2.5))
                patches.append(p)
        fig.canvas.draw_idle()

    def _on_click(event):
        nonlocal stage, mode, active, pending
        if finished or event.xdata is None:
            return
        # Right-click / button 3 clears active cell (current stage)
        if event.button == 3 and event.inaxes == ax_map:
            cell = _cell_at(event.xdata, event.ydata)
            if cell is not None:
                a = anchors.setdefault(cell, {'main': None, 'extras': []})
                if stage == 1:
                    a['main'] = None
                else:
                    a['extras'] = []
                if active == cell:
                    active = None
                    mode = 'pick'
                    pending = []
                _redraw_map()
                print(f'  Cleared cell {cell} (stage {stage})')
            return
        if event.inaxes == ax_map and mode == 'pick':
            cell = _cell_at(event.xdata, event.ydata)
            if cell is None:
                return
            active = cell
            mode = 'edit'
            pending = []
            sp = cell_spec_by_key.get(cell)
            if sp is None:
                sp = np.nanmean(using_data, axis=(1, 2))
            _load_editor(sp, cell=cell)
            _redraw_map()
            print(f'  [Stage {stage}] Editing cell {cell}. '
                  f'Click {"two points for main mask" if stage == 1 else "point pairs for extras"} '
                  f'on the right panel.')
        elif event.inaxes == ax_edit and mode == 'edit' and active is not None:
            v = event.xdata
            pending.append(v)
            ax_edit.axvline(v, color='orange' if stage == 1 else 'magenta',
                            ls='-', alpha=0.8, lw=1.3)
            fig.canvas.draw_idle()
            if stage == 1:
                if len(pending) == 2:
                    lo, hi = sorted(pending)
                    anchors.setdefault(active, {'main': None, 'extras': []})['main'] = (lo, hi)
                    print(f'  [Stage 1] Cell {active} main mask: [{lo:.2f}, {hi:.2f}] km/s')
                    pending = []
                    mode = 'pick'
                    active = None
                    _load_editor(np.nanmean(using_data, axis=(1, 2)), cell=None)
                    _redraw_map()
            else:
                if len(pending) == 2:
                    lo, hi = sorted(pending)
                    anchors.setdefault(active, {'main': None, 'extras': []})['extras'].append((lo, hi))
                    print(f'  [Stage 2] Cell {active} extra: [{lo:.2f}, {hi:.2f}] km/s '
                          f'({len(anchors[active]["extras"])} total)')
                    pending = []
                    _load_editor(editor_spec, cell=active)
                    _redraw_map()

    def _on_key(event):
        nonlocal stage, mode, active, pending, finished
        if finished:
            return
        if event.key in ('escape',):
            if active is not None:
                a = anchors.get(active)
                if a is not None:
                    if stage == 1:
                        a['main'] = None
                    else:
                        a['extras'] = []
                pending = []
                active = None
                mode = 'pick'
                _load_editor(np.nanmean(using_data, axis=(1, 2)), cell=None)
                _redraw_map()
                print(f'  Cleared cell (stage {stage})')
            return
        if event.key in ('e', 'E'):
            if mode == 'edit':
                return
            if stage == 1:
                stage = 2
                print('  [Stage 2] Click cells to add extra excluded regions; '
                      'Enter exits a cell, E finishes.')
            else:
                finished = True
                fig.canvas.mpl_disconnect(cid)
                fig.canvas.mpl_disconnect(cid_key)
                plt.close(fig)
            return
        if event.key in ('enter', 'return'):
            if stage == 2 and mode == 'edit' and active is not None:
                pending = []
                active = None
                mode = 'pick'
                _load_editor(np.nanmean(using_data, axis=(1, 2)), cell=None)
                _redraw_map()
                print('  Exited cell. Click another cell for extras, or press E to finish.')

    print('\n  [Complex mask] Map shows 3x3-binned spectra (spatial layout).')
    print('  Stage 1: click a cell, then click TWO velocity points on the right '
          'panel to set its HINSA main mask.')
    print('  Re-click a set cell to redefine; right-click/Esc clears it. Press E '
          'for Stage 2.')
    print('  Stage 2: click a cell, click point pairs for extra excluded regions; '
          'Enter exits the cell; E finishes.\n')

    # default editor: central average
    tmp = using_data.copy()
    tmp[tmp == 0] = np.nan
    _load_editor(np.nanmean(tmp, axis=(1, 2)), cell=None)

    cid = fig.canvas.mpl_connect('button_press_event', _on_click)
    cid_key = fig.canvas.mpl_connect('key_press_event', _on_key)
    plt.show()
    while not finished:
        plt.pause(0.1)
    matplotlib.use(_orig_backend)

    result = []
    for (iy, ix), a in anchors.items():
        if a['main'] is None and not a['extras']:
            continue
        cx = ix * 3 + 1.5
        cy = iy * 3 + 1.5
        result.append({'cy': cy, 'cx': cx, 'main': a['main'], 'extras': list(a['extras'])})
    print(f'\n  Complex mask: {len(result)} anchors set.')
    return result


def _interpolate_pixel_masks(anchors, ny, nx, velo_kms, vlsr_kms, power=2.0):
    """Interpolate per-anchor masks to a per-pixel mask grid (IDW).

    Parameters
    ----------
    anchors : list of {'cy', 'cx', 'main': (lo,hi)|None, 'extras': [(lo,hi),...]}
    ny, nx : int — output grid size
    velo_kms : 1D array — ascending velocity axis (km/s)
    vlsr_kms : float — cloud systemic velocity (km/s)
    power : float — IDW power (default 2)

    Returns
    -------
    mask_lo_ms : (ny, nx) float — main HINSA lower bound (m/s)
    mask_hi_ms : (ny, nx) float — main HINSA upper bound (m/s)
    extra_mask_3d : (n_v, ny, nx) uint8 — 0 = excluded, 1 = kept
    has_hinsa : bool — center-pixel interpolated mask crosses V_LSR
    """
    n_v = len(velo_kms)
    Y, X = np.mgrid[0:ny, 0:nx].astype(float)
    n_anch = len(anchors)
    if n_anch == 0:
        raise ValueError('No anchors to interpolate')

    # IDW weights
    w = np.zeros((ny, nx, n_anch))
    for k, a in enumerate(anchors):
        d2 = (Y - a['cy'])**2 + (X - a['cx'])**2
        w[..., k] = 1.0 / np.maximum(d2, 1e-12) ** (power / 2.0)
    wsum = w.sum(axis=-1)
    wsum[wsum == 0] = 1.0
    wn = w / wsum[..., None]

    # main mask lo/hi (km/s) via IDW over anchors that have main
    main_w = np.zeros((ny, nx))
    lo_acc = np.zeros((ny, nx))
    hi_acc = np.zeros((ny, nx))
    for k, a in enumerate(anchors):
        if a['main'] is None:
            continue
        lo, hi = a['main']
        main_w += wn[..., k]
        lo_acc += wn[..., k] * lo
        hi_acc += wn[..., k] * hi
    have_main = main_w > 0
    mask_lo = np.where(have_main, lo_acc / np.maximum(main_w, 1e-12), 0.0)
    mask_hi = np.where(have_main, hi_acc / np.maximum(main_w, 1e-12), 0.0)
    # enforce lo < hi per pixel
    lo_p = np.minimum(mask_lo, mask_hi)
    hi_p = np.maximum(mask_lo, mask_hi)

    # extra mask per velocity channel: weighted fraction of anchors that
    # exclude this channel; exclude where fraction >= 0.5
    frac = np.zeros((n_v, ny, nx))
    for k, a in enumerate(anchors):
        if not a['extras']:
            continue
        excl = np.zeros(n_v, dtype=bool)
        for (lo, hi) in a['extras']:
            excl |= (velo_kms > lo) & (velo_kms < hi)
        frac += wn[None, ..., k] * excl[:, None, None]
    extra_mask_3d = (frac < 0.5).astype(np.uint8)   # 1 = keep, 0 = excluded

    cy, cx = ny // 2, nx // 2
    has_hinsa = have_main[cy, cx] and (lo_p[cy, cx] < vlsr_kms < hi_p[cy, cx])
    return (lo_p * 1000.0, hi_p * 1000.0, extra_mask_3d, bool(has_hinsa))


# ---------------------------------------------------------------------------
# Interactive mask selection (replaces hinsapack.pipeline._interactive_mask_selection)
# ---------------------------------------------------------------------------

def _interactive_mask_selection(using_data, velo_kms, vlsr_kms, target_id):
    """
    Interactive GUI for selecting the velocity mask range(s).

    Two stages:
      Stage 1: click TWO points -> primary HINSA mask (the one of interest).
      Stage 2: click point pairs elsewhere; each pair masks one ADDITIONAL
               velocity region (e.g. other HINSA structures we do not care
               about).  Multiple regions allowed.  Press Enter to finish.
               These extra regions are excluded from background fitting and
               residual computation.

    Returns (mask_lo_kms, mask_hi_kms, extra_ranges_kms, has_hinsa) where
      extra_ranges_kms : list[(lo, hi)] of additional excluded ranges (km/s).
    """
    import matplotlib
    _orig_backend = matplotlib.get_backend().lower()

    _in_jupyter = ('zmq' in _orig_backend or 'ipympl' in _orig_backend
                    or 'nbagg' in _orig_backend or 'inline' in _orig_backend
                    or 'ipykernel' in sys.modules)
    _has_display = os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')

    if _in_jupyter:
        matplotlib.use('nbagg')
    elif _has_display:
        try:
            matplotlib.use('TkAgg')
        except Exception:
            try:
                matplotlib.use('Qt5Agg')
            except Exception:
                print("[Warning] No interactive backend available. Falling back to default mask.")
                mid = len(velo_kms) // 2
                half = max(1, mid // 3)
                return velo_kms[mid - half], velo_kms[mid + half], [], True
    else:
        print("[Warning] No display detected (headless). Falling back to default ±3 km/s mask.")
        mid = len(velo_kms) // 2
        half = max(1, mid // 3)
        return velo_kms[mid - half], velo_kms[mid + half], [], True

    from matplotlib import pyplot as plt
    from matplotlib.gridspec import GridSpec

    n_v, ny, nx = using_data.shape

    if ny >= 5 and nx >= 5:
        central_crop = using_data[:, ny // 2 - 2: ny // 2 + 5, nx // 2 - 2: nx // 2 + 5].copy()
        central_crop[central_crop == 0] = np.nan
        avg_spec = np.nanmean(central_crop, axis=(1, 2))
    else:
        tmp = using_data.copy()
        tmp[tmp == 0] = np.nan
        avg_spec = np.nanmean(tmp, axis=(1, 2))

    dv = np.nanmean(np.diff(velo_kms)) if len(velo_kms) > 1 else 1.0
    d2_spec = np.full_like(avg_spec, np.nan)
    if np.sum(np.isfinite(avg_spec)) >= 3:
        d2_spec[1:-1] = (avg_spec[2:] + avg_spec[:-2] - 2 * avg_spec[1:-1]) / (dv ** 2)

    fig = plt.figure(figsize=(10, 5.8))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[3, 1.2], hspace=0.08)

    ax = fig.add_subplot(gs[0])
    for j in range(ny):
        for i in range(nx):
            spec = using_data[:, j, i]
            if np.any(np.isfinite(spec)) and np.nanmax(spec) > 0.1:
                ax.plot(velo_kms, spec, '-', color='steelblue', alpha=0.05, lw=0.5)

    ax.plot(velo_kms, avg_spec, '-', color='navy', alpha=0.9, lw=1.5, label='Central avg')
    ax.axvline(vlsr_kms, color='red', ls='--', alpha=0.5, label=f'V_LSR={vlsr_kms:.1f} km/s')
    ax.set_ylabel('T$_\\mathrm{B}$ (K)')
    ax.set_title(f'{target_id} — Stage 1: click TWO points to define the HINSA mask')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelbottom=False)

    ax2 = fig.add_subplot(gs[1], sharex=ax)
    ax2.plot(velo_kms, d2_spec, '-', color='crimson', lw=1.2)
    ax2.axhline(0, color='gray', ls='-', lw=0.5, alpha=0.6)
    ax2.axvline(vlsr_kms, color='red', ls='--', alpha=0.5)
    ax2.set_xlabel('V$_{\\rm LSR}$ (km s$^{-1}$)')
    ax2.set_ylabel("d$^2$T/dv$^2$", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='y', labelsize=8)
    ax2.text(0.02, 0.90, '(HINSA -> W-shaped dip)', transform=ax2.transAxes,
             fontsize=8, va='top', color='crimson', alpha=0.7)

    main_clicked = []
    extra_ranges = []
    pending_extra = []
    finished = False

    def _draw_vline(x, color):
        ax.axvline(x, color=color, ls='-', alpha=0.8, lw=1.5)
        ax2.axvline(x, color=color, ls='-', alpha=0.5, lw=1.0)

    def _on_click(event):
        nonlocal pending_extra, finished
        if event.xdata is None or event.inaxes != ax or finished:
            return
        x = event.xdata
        if len(main_clicked) < 2:
            main_clicked.append(x)
            _draw_vline(x, 'orange')
            fig.canvas.draw_idle()
            if len(main_clicked) == 2:
                lo, hi = sorted(main_clicked)
                print(f"  [Stage 1] Primary HINSA mask: [{lo:.2f}, {hi:.2f}] km/s")
                ax.axvspan(lo, hi, color='orange', alpha=0.12)
                ax2.axvspan(lo, hi, color='orange', alpha=0.12)
                ax.set_title(f'{target_id} — Stage 2: click pairs to add excluded regions (Enter to finish)')
                print(f"  [Stage 2] Now click pairs of points to mask additional velocity regions.")
                print(f"  Each pair = one excluded region (shaded). Press Enter when done.")
                fig.canvas.draw_idle()
        else:
            pending_extra.append(x)
            _draw_vline(x, 'magenta')
            fig.canvas.draw_idle()
            if len(pending_extra) == 2:
                lo, hi = sorted(pending_extra)
                extra_ranges.append((lo, hi))
                ax.axvspan(lo, hi, color='magenta', alpha=0.15)
                ax2.axvspan(lo, hi, color='magenta', alpha=0.15)
                print(f"  [Stage 2] Extra excluded region: [{lo:.2f}, {hi:.2f}] km/s "
                      f"({len(extra_ranges)} total)")
                pending_extra = []
                fig.canvas.draw_idle()

    def _on_key(event):
        nonlocal finished
        if event.key in ('enter', 'return'):
            finished = True
            fig.canvas.mpl_disconnect(cid)
            fig.canvas.mpl_disconnect(cid_key)
            plt.close(fig)

    if _in_jupyter:
        print(f"\n  [Interactive mask] Stage 1: click two velocity points for the HINSA region.")
        print(f"  Stage 2: click pairs of points to add excluded regions (each pair = one region).")
        print(f"  Press Enter to finish.  Lower panel: 2nd derivative (HINSA -> 'W' shape).\n")
    else:
        print(f"\n  [Interactive mask] Stage 1: click two velocity points for the HINSA region.")
        print(f"  Stage 2: click pairs of points to add excluded regions (each pair = one region).")
        print(f"  Press Enter to finish.  Lower panel: 2nd derivative (HINSA -> 'W' shape).\n")

    cid = fig.canvas.mpl_connect('button_press_event', _on_click)
    cid_key = fig.canvas.mpl_connect('key_press_event', _on_key)
    plt.show()

    matplotlib.use(_orig_backend)

    if len(main_clicked) != 2:
        print("  [Warning] Did not receive 2 clicks. Using default ±3 km/s around V_LSR.")
        return vlsr_kms - 3.0, vlsr_kms + 3.0, [], True

    mask_lo = min(main_clicked[0], main_clicked[1])
    mask_hi = max(main_clicked[0], main_clicked[1])

    has_hinsa = (mask_lo < vlsr_kms < mask_hi)
    if not has_hinsa:
        print(f"  Mask [{mask_lo:.2f}, {mask_hi:.2f}] does NOT cross V_LSR={vlsr_kms:.2f}")
        print(f"  -> No HINSA detected, skipping {target_id}")
    else:
        print(f"  Mask range set: [{mask_lo:.2f}, {mask_hi:.2f}] km/s (crosses V_LSR)")
    if extra_ranges:
        print(f"  Extra excluded regions: {[(round(a,2), round(b,2)) for (a,b) in extra_ranges]}")

    return mask_lo, mask_hi, extra_ranges, has_hinsa


# ---------------------------------------------------------------------------
# Grid spectra plotting (replaces hinsapack.pipeline._plot_grid_spectra_polyfit)
# ---------------------------------------------------------------------------

def _plot_grid_spectra_polyfit(png_path, using_data, velo_kms, vlsr_kms,
                                pure_hinsa, target_id):
    """
    Generate a grid-spectra overlay plot.  Each grid cell shows:
      - blue: observed T_B spectrum
      - orange: recovered T_HI (= observed + HINSA absorption)
    """
    from matplotlib import pyplot as plt

    n_v, ny, nx = using_data.shape

    velo_kms = np.asarray(velo_kms, dtype=float)
    if velo_kms[-1] < velo_kms[0]:
        velo_kms = velo_kms[::-1]
        using_data = using_data[::-1]
        pure_hinsa = pure_hinsa[::-1]

    bg_map = np.nanmean(using_data, axis=0)
    vbg_lo, vbg_hi = np.nanpercentile(bg_map, [2, 98])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(bg_map, origin='lower', cmap='cividis',
              vmin=vbg_lo, vmax=vbg_hi, aspect='equal', alpha=0.3)

    cell3 = 3
    xg = np.arange(0, nx, cell3)
    yg = np.arange(0, ny, cell3)

    cell_hinsa_list = []
    cell_pos_list = []
    cell_spec_list = []
    for x0 in xg:
        for y0 in yg:
            x1 = min(x0 + cell3, nx)
            y1 = min(y0 + cell3, ny)
            sp = np.nanmean(using_data[:, y0:y1, x0:x1], axis=(1, 2))
            ha = np.nanmean(pure_hinsa[:, y0:y1, x0:x1], axis=(1, 2))
            cell_hinsa_list.append(ha)
            cell_pos_list.append((x0, y0, x0 + cell3 / 2, y0 + cell3 / 2))
            cell_spec_list.append(sp)

    if cell_spec_list:
        all_lo = min(np.nanmin(s) for s in cell_spec_list)
        all_hi = max(np.nanmax(s) for s in cell_spec_list)
        margin = max((all_hi - all_lo) * 0.15, 1.0)
        global_ylo = all_lo - margin
        global_yhi = all_hi + margin
    else:
        global_ylo, global_yhi = 0, 10

    for ha, sp, (x0, y0, ccx, ccy) in zip(cell_hinsa_list, cell_spec_list, cell_pos_list):
        THI = sp + ha
        ins = ax.inset_axes(
            [x0 - 0.5, y0 - 0.5, cell3, cell3],
            transform=ax.transData)
        ins.plot(velo_kms, sp, '-', color='#3366cc', linewidth=0.5)
        ins.plot(velo_kms, THI, '-', color='orange', linewidth=0.5)
        ins.axvline(vlsr_kms, ls='--', color='0.5', alpha=0.3, lw=0.3)
        ins.set_ylim(global_ylo, global_yhi)
        ins.set_xlim(velo_kms[0], velo_kms[-1])
        ins.tick_params(axis='both', which='both', length=0,
                        labelleft=False, labelbottom=False)
        ins.set_facecolor('none')
        for spn in ins.spines.values():
            spn.set_visible(False)
        ax.plot(ccx, ccy, 'wo', ms=2, alpha=0.5)

    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    ax.set_xlabel('Pixel X')
    ax.set_ylabel('Pixel Y')
    ax.set_title(f'{target_id} - Grid: Observed (blue) vs Recovered T_HI (orange)')

    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Planck Av dust extinction map (standalone, no hinsapack dependency)
# ---------------------------------------------------------------------------

def _retrieve_planck_av(ra_deg, dec_deg, radius_arcmin, output_dir, fname_prefix):
    """
    Retrieve a Planck A_V dust extinction map over a square region.

    Returns a 2D numpy array of A_V values, or None on failure.
    Prompts the user to install ``dustmaps`` or download Planck data if needed.
    """
    # --- Check / install dustmaps package ---
    try:
        from dustmaps.config import config as dm_config
    except ImportError:
        print("\n  [Planck Av] The 'dustmaps' package is required but not installed.")
        resp = input("  Install it now? (y/n) [y]: ").strip().lower()
        if resp in ('', 'y', 'yes'):
            import subprocess, sys
            print("  Installing dustmaps ...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'dustmaps'])
            from dustmaps.config import config as dm_config
        else:
            print("  Skipping Planck Av (dustmaps not installed).")
            return None

    # --- Set data directory ---
    dm_config['data_dir'] = os.path.expanduser('~/dustmaps_data')

    # --- Check / download Planck data ---
    try:
        from dustmaps.planck import PlanckQuery
        planck = PlanckQuery()
    except Exception as e:
        print(f"\n  [Planck Av] Failed to initialise PlanckQuery: {e}")
        resp = input("  Download Planck dust data now? (y/n) [y]: ").strip().lower()
        if resp in ('', 'y', 'yes'):
            from dustmaps.planck import fetch as planck_fetch
            print("  Downloading Planck data (this may take a while) ...")
            try:
                planck_fetch()
                from dustmaps.planck import PlanckQuery
                planck = PlanckQuery()
            except Exception as e2:
                print(f"  Planck data download failed: {e2}")
                print("  Skipping Planck Av.")
                return None
        else:
            print("  Skipping Planck Av.")
            return None

    # --- Query the map ---
    from astropy import units as u
    from astropy.coordinates import SkyCoord as _SkyCoord

    radius_deg = radius_arcmin / 60.0
    cdelt_deg = 0.025  # ~2.5 arcmin pixel scale

    raa = np.arange(ra_deg - radius_deg, ra_deg + radius_deg, cdelt_deg)
    decc = np.arange(dec_deg - radius_deg, dec_deg + radius_deg, cdelt_deg)
    ra_grid, dec_grid = np.meshgrid(raa, decc)
    coords = _SkyCoord(ra=ra_grid * u.deg, dec=dec_grid * u.deg, frame='icrs')

    Av_planck = 3.1 * planck(coords)
    # Flip horizontally for RA-origin convention (RA increases leftward)
    Av_planck = Av_planck[:, ::-1]

    print(f"  Planck Av loaded: shape={Av_planck.shape}")
    return Av_planck


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def prepare_hinspheres_input(target_id, datacube_path,
                              output_dir='./HIprepared/',
                              center_coord=None,
                              spatial_radius_arcmin=None,
                              velo_radius_kms=None,
                              polyfit_mask_kms=-1,
                              poly_order=5,
                              peak_unmask_radius=1,
                              extra_mask_ranges=None,
                              vlsr_override=None,
                              file_prefix=None,
                              distance_override=None,
                              fetch_planck_av=False):
    """
    Prepare all data needed by the hinspheres spherical RT fitting package.

    This is a standalone version that does NOT depend on hinsapack.  Instead of
    looking up coordinates/velocities from catalogs and auto-finding the HI cube,
    the caller provides everything directly.

    Parameters
    ----------
    target_id : str
        Source name used for labelling (e.g. 'B227').  Does NOT drive any
        catalog lookup — it is purely cosmetic.
    datacube_path : str
        Full path to the HI FITS datacube (e.g. './ot1_hi_destripe.fits').
    output_dir : str
        Where to save the prepared FITS files.
    center_coord : astropy.coordinates.SkyCoord
        Sky position of the source center.
    spatial_radius_arcmin : float, optional
        Spatial extraction radius in arcmin.  Default is 15.0.
    velo_radius_kms : float, optional
        Velocity extraction half-width in km/s around V_LSR.  Default is 6.0.
    polyfit_mask_kms : float, optional
        Velocity radius in km/s around V_LSR to mask during polyfit baseline
        fitting.  Set to -1 for interactive mode: click two points for the
        primary HINSA region, then click pairs of points to add excluded
        regions (multiple allowed, Enter to finish).  The primary region
        crosses V_LSR and defines the HINSA; the extra regions (other HINSA
        structures) are excluded from background fit and residual
        computation.  Set to a positive value (e.g. 3.0) to skip the
        interactive GUI and use a fixed ±polyfit_mask_kms range.
        Default is -1 (interactive).
    polyfit_mask_kms : float, optional (complex mode)
        Set to -2 to enable the complex per-cell mask mode: a map of
        3x3-binned spectra is shown; each cell's main HINSA mask and extra
        excluded regions are set individually and interpolated to per-pixel
        masks (IDW).  Extra exclusions are stored as an embedded 0/1 cube
        (EXMASK extension) inside the background FITS, consumed by
        fit_hinsa_model's residual computation.
    poly_order : int, optional
        Fitting method for HI background baseline:
        - Positive (e.g. 3, 5): polynomial order.  Default is 5.
        - Negative (e.g. -2, -3): use |poly_order| Gaussian components.
    peak_unmask_radius : int, optional
        If fitted background peak < observed peak, unmask the local peak
        channel +/- this many pixels and re-fit.  Default is 1.
    extra_mask_ranges : list[tuple(float,float)] or None, optional
        Additional velocity ranges (km/s) to mask during fitting.
    vlsr_override : float or None, optional
        V_LSR in km/s.  Required (no catalog fallback).
    file_prefix : str or None, optional
        Prefix for output FITS filenames.  Defaults to target_id.
    distance_override : float or None, optional
        Distance in kpc.  If None, defaults to 1.0 kpc.
    fetch_planck_av : bool, optional
        If True, retrieve and save a Planck A_V dust extinction map over the
        same spatial region.  Requires the ``dustmaps`` package
        (``pip install dustmaps``).  If the package or the Planck data files
        are missing, the user will be prompted to install/download them.
        Default is False.

    Returns
    -------
    data : dict
        Keys: 'hinsa_map', 'T_HI_true', 'HI_cube', 'wcs', 'velo_kms',
              'center_ra', 'center_dec', 'distance_pc', 'R_out_pc',
              'vlsr_kms', 'R_arcmin', plus file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    fname_prefix = file_prefix if file_prefix is not None else target_id

    # ---- Resolve coordinates ----
    if not isinstance(center_coord, SkyCoord):
        raise TypeError("center_coord must be an astropy.coordinates.SkyCoord")
    icrs = center_coord.icrs
    ra = icrs.ra.deg
    dec = icrs.dec.deg

    # ---- Resolve V_LSR ----
    if vlsr_override is None:
        raise ValueError("vlsr_override is required (no catalog fallback).")
    vlsr_kms_val = float(vlsr_override)

    # ---- Resolve distance ----
    D_kpc = float(distance_override) if distance_override is not None else 1.0
    if not np.isfinite(D_kpc) or D_kpc <= 0:
        D_kpc = 1.0
    D_pc = D_kpc * 1000.0

    # ---- Angular radius ----
    R_arcmin = spatial_radius_arcmin if spatial_radius_arcmin is not None else 15.0
    R_out_pc = R_arcmin * np.pi / (180.0 * 60.0) * D_pc

    print(f"[prepare_hinspheres_input] {target_id}")
    print(f"  RA={ra:.4f}  Dec={dec:.4f}  Vlsr={vlsr_kms_val:.1f} km/s")
    print(f"  D={D_pc:.0f} pc  R={R_arcmin:.1f} arcmin -> {R_out_pc:.3f} pc")

    # ---- Extract HI sub-cube ----
    extraction_radius = spatial_radius_arcmin if spatial_radius_arcmin is not None else 15.0
    velo_radius_ms = velo_radius_kms * 1000.0 if velo_radius_kms is not None else 6000.0
    vlsr_ms = vlsr_kms_val * 1000.0

    using_data, wcs, velo_range = _extract_hi_subcube(
        datacube_path, ra, dec, vlsr_ms, extraction_radius, velo_radius_ms)

    velo_kms = np.asarray(velo_range / 1000.0, dtype=float)

    print(f"  Cube shape: {using_data.shape}  (extracted {extraction_radius:.0f} arcmin)")

    if using_data.ndim == 3 and (using_data.shape[1] == 0 or using_data.shape[2] == 0):
        raise ValueError(
            f"Extracted HI cube is empty ({using_data.shape}). "
            f"The center coordinates (RA={ra:.4f}, Dec={dec:.4f}) may not overlap "
            f"with the data cube. Check that RA/Dec are in ICRS (not Galactic)."
        )

    # ---- Polyfit baseline -> HI background + HINSA ----
    if polyfit_mask_kms == -2:
        # Complex mode: per-cell masks on a grid of 3x3-binned spectra,
        # interpolated to per-pixel masks.  Extra exclusions are stored as an
        # embedded 0/1 cube in the background FITS (EXMASK extension).
        anchors = _interactive_grid_mask_selection(
            using_data, velo_kms, vlsr_kms_val, target_id)
        ny, nx = using_data.shape[1], using_data.shape[2]
        if not anchors:
            raise SystemExit('No cells selected in complex mask; aborting.')
        mask_lo_ms, mask_hi_ms, extra_mask_3d, has_hinsa = _interpolate_pixel_masks(
            anchors, ny, nx, velo_kms, vlsr_kms_val, power=2.0)
        if not has_hinsa:
            print(f"  Skipping {target_id} (no HINSA: centre interpolated mask "
                  f"does not cross V_LSR={vlsr_kms_val:.2f})")
            return None
        extra_ranges_kms = []   # per-pixel extras live in EXMASK cube, not header
        print(f"  Complex mask: {len(anchors)} anchors, "
              f"per-pixel main mask + EXMASK cube")

        from joblib import Parallel, delayed

        def _fit_pixel_mask(j, i):
            return _get_pure_hinsa_by_3polyfit(
                using_data[:, j, i], velo_range,
                [mask_lo_ms[j, i], mask_hi_ms[j, i]],
                poly_order=poly_order,
                exclude_channels=(extra_mask_3d[:, j, i] == 0))

        results = Parallel(n_jobs=4, prefer='threads')(
            delayed(_fit_pixel_mask)(j, i) for j in range(ny) for i in range(nx))
        pure_hinsa = np.zeros(using_data.shape)
        for idx, (j, i) in enumerate([(j, i) for j in range(ny) for i in range(nx)]):
            pure_hinsa[:, j, i] = results[idx]
        pure_hinsa = -pure_hinsa
    elif polyfit_mask_kms == -1:
        mask_lo_kms, mask_hi_kms, extra_ranges_kms, has_hinsa = _interactive_mask_selection(
            using_data, velo_kms, vlsr_kms_val, target_id)
        if not has_hinsa:
            print(f"  Skipping {target_id} (no HINSA)")
            return None
        mask_lo_ms = mask_lo_kms * 1000.0
        mask_hi_ms = mask_hi_kms * 1000.0
        extra_ranges_ms = [(lo * 1000.0, hi * 1000.0) for (lo, hi) in extra_ranges_kms]

        from joblib import Parallel, delayed

        def _fit_pixel_mask(j, i):
            return _get_pure_hinsa_by_3polyfit(
                using_data[:, j, i], velo_range,
                [mask_lo_ms, mask_hi_ms],
                poly_order=poly_order,
                extra_mask_ranges=extra_ranges_ms)

        ny, nx = using_data.shape[1], using_data.shape[2]
        results = Parallel(n_jobs=4, prefer='threads')(
            delayed(_fit_pixel_mask)(j, i) for j in range(ny) for i in range(nx))
        pure_hinsa = np.zeros(using_data.shape)
        for idx, (j, i) in enumerate([(j, i) for j in range(ny) for i in range(nx)]):
            pure_hinsa[:, j, i] = results[idx]
        pure_hinsa = -pure_hinsa
    else:
        ignore_radius_ms = polyfit_mask_kms * 1000.0
        pure_hinsa = _get_pure_hinsa_cube(
            using_data, velo_range, vlsr_ms,
            ignore_radius_ms=ignore_radius_ms,
            poly_order=poly_order,
            peak_unmask_radius=peak_unmask_radius,
            n_jobs=4,
            extra_mask_ranges=extra_mask_ranges)
        extra_ranges_kms = list(extra_mask_ranges) if extra_mask_ranges else []

    T_HI_true = using_data + pure_hinsa

    hinsa_map = np.nanmax(pure_hinsa, axis=0)
    hinsa_map = np.maximum(hinsa_map, 0)

    # ---- Save FITS files ----
    def _save_fits(data_3d_or_2d, name, add_velocity=False, extra_hdu=None):
        path = os.path.join(output_dir, f'{fname_prefix}_{name}.fits')
        arr = np.asarray(data_3d_or_2d, dtype=np.float32)
        if add_velocity and arr.ndim == 3 and velo_kms is not None:
            n_v, n_y, n_x = arr.shape
            world0 = wcs.pixel_to_world_values(0, 0)
            hdr = fits.Header()
            hdr['SIMPLE'] = True
            hdr['BITPIX'] = -32
            hdr['NAXIS'] = 3
            hdr['NAXIS1'] = n_x
            hdr['NAXIS2'] = n_y
            hdr['NAXIS3'] = n_v
            hdr['EXTEND'] = True
            hdr['CTYPE1'] = 'RA---CAR'
            hdr['CTYPE2'] = 'DEC--CAR'
            hdr['CRPIX1'] = 1
            hdr['CRPIX2'] = 1
            hdr['CRVAL1'] = float(world0[0])
            hdr['CRVAL2'] = float(world0[1])
            hdr['CDELT1'] = float(wcs.wcs.cdelt[0])
            hdr['CDELT2'] = float(wcs.wcs.cdelt[1])
            hdr['EPOCH'] = 2000.0
            hdr['CTYPE3'] = 'VELO-LSR'
            hdr['CRPIX3'] = 1
            hdr['CRVAL3'] = float(velo_kms[0]) * 1000.0
            hdr['CDELT3'] = float(velo_kms[1] - velo_kms[0]) * 1000.0 if n_v > 1 else 0.0
            # Extra velocity ranges excluded from background fit AND residual
            hdr['XRMASKCN'] = len(extra_ranges_kms)
            for _k, (_lo, _hi) in enumerate(extra_ranges_kms):
                hdr[f'XRM{_k+1}LO'] = float(_lo) * 1000.0  # m/s
                hdr[f'XRM{_k+1}HI'] = float(_hi) * 1000.0  # m/s
            if extra_hdu is not None:
                # Embedded per-pixel exclusion cube: 0 = excluded, 1 = kept.
                hdr['EXMASK'] = True
                primary = fits.PrimaryHDU(arr, header=hdr)
                ex_hdu = fits.ImageHDU(np.asarray(extra_hdu, dtype=np.uint8),
                                       name='EXMASK')
                fits.HDUList([primary, ex_hdu]).writeto(path, overwrite=True)
            else:
                hdu = fits.PrimaryHDU(arr, header=hdr)
                hdu.writeto(path, overwrite=True)
        else:
            hdr = wcs.to_header() if wcs is not None else fits.Header()
            fits.writeto(path, arr, hdr, overwrite=True)
        return path

    hinsa_path = _save_fits(hinsa_map, 'hinsa')
    _exmask_hdu = extra_mask_3d if polyfit_mask_kms == -2 else None
    THI_path = _save_fits(T_HI_true, 'HI_background',
                          add_velocity=(T_HI_true.ndim == 3), extra_hdu=_exmask_hdu)
    cube_path = _save_fits(using_data, 'HI_cube', add_velocity=True)
    velo_path = os.path.join(output_dir, f'{fname_prefix}_velo_kms.npy')
    np.save(velo_path, velo_kms)

    # Complex mode: persist anchors + interpolated main mask maps
    if polyfit_mask_kms == -2:
        import json as _json
        anchors_path = os.path.join(output_dir, f'{fname_prefix}_mask_anchors.json')
        with open(anchors_path, 'w') as f:
            _json.dump(anchors, f, indent=2)
        mask_lo_path = _save_fits(mask_lo_ms, 'hinsa_masklo')
        mask_hi_path = _save_fits(mask_hi_ms, 'hinsa_maskhi')
        print(f"  Saved anchors:        {anchors_path}")
        print(f"  Saved mask lo/hi:     {mask_lo_path} / {mask_hi_path}")

    print(f"  Saved HINSA map:     {hinsa_path}")
    print(f"  Saved HI background: {THI_path}")
    print(f"  Saved HI cube:       {cube_path}")
    print(f"  Saved velocity axis: {velo_path}")

    # ---- Grid spectra overlay plot ----
    grid_png_path = os.path.join(output_dir, f'{fname_prefix}_grid_spectra_polyfit.png')
    _plot_grid_spectra_polyfit(
        grid_png_path, using_data, velo_kms, vlsr_kms_val,
        pure_hinsa, target_id)
    print(f"  Saved grid spectra:  {grid_png_path}")

    # ---- Planck Av (optional) ----
    planck_av = None
    planck_path = None
    if fetch_planck_av:
        planck_av = _retrieve_planck_av(ra, dec, extraction_radius, output_dir, fname_prefix)
        if planck_av is not None:
            planck_path = os.path.join(output_dir, f'{fname_prefix}_Planck_Av.fits')
            fits.writeto(planck_path, np.asarray(planck_av, dtype=np.float32),
                         overwrite=True)
            print(f"  Saved Planck Av:     {planck_path}")

    return {
        'target_id': target_id,
        'hinsa_map': hinsa_map,
        'T_HI_true': np.nanmean(T_HI_true, axis=0) if T_HI_true.ndim == 3 else T_HI_true,
        'T_HI_true_cube': T_HI_true if T_HI_true.ndim == 3 else None,
        'HI_cube': using_data,
        'wcs': wcs,
        'velo_kms': velo_kms,
        'pixel_scale_pc': float(np.mean([s.value for s in wcs.proj_plane_pixel_scales()])) * np.pi / 180.0 * D_pc,
        'center_ra': float(ra),
        'center_dec': float(dec),
        'distance_pc': D_pc,
        'R_out_pc': R_out_pc,
        'R_arcmin': R_arcmin,
        'vlsr_kms': vlsr_kms_val,
        'hinsa_map_path': hinsa_path,
        'T_HI_true_path': THI_path,
        'HI_cube_path': cube_path,
        'velo_path': velo_path,
        'planck_av': planck_av,
        'planck_av_path': planck_path,
        'output_dir': output_dir,
    }
