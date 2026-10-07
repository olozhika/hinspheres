# hinspheres

Forward radiative-transfer modeling and fitting of HINSA (H I Narrow Self-Absorption) in cold cloud cores.

`hinspheres` builds a spherically symmetric, multi-shell 21-cm absorption model of a cloud core (Plummer density and spin-temperature profiles, shell-wise HI abundance, infall and rotation kinematics), synthesizes absorption cubes against any background HI survey, and recovers the physical parameters of observed cubes with CMA-ES optimization, optionally refined with emcee MCMC. Data preparation from raw FITS cubes is built in. The package runs standalone or as the fitting backend of the `hinsapack` pipeline through `prepare_hinspheres_input`.

## Installation

```bash
pip install numpy scipy astropy matplotlib joblib cma
# optional, for MCMC refinement
pip install emcee corner
```

## Quick Start

### 1. Prepare data

`prepare_hinspheres_input` is the first pipeline step: it extracts a sub-cube around the target, fits a polynomial baseline, builds the velocity masks, and writes all files needed for fitting.

```python
from astropy.coordinates import SkyCoord
from hinspheres import prepare_hinspheres_input

result = prepare_hinspheres_input(
    target_id='G206',
    datacube_path='./ot1_hi_destripe.fits',
    output_dir='HIfig/hinspheres_input/',
    center_coord=SkyCoord(ra=206.1, dec=-15.77, unit='deg'),
    vlsr_kms=9.3,
    spatial_radius_arcmin=12.0,   # sub-cube spatial radius (arcmin)
    velo_radius_kms=15.0,         # sub-cube velocity radius (km/s)
    poly_order=5,                 # baseline polynomial order
    n_jobs=4,
)
```

Returns a dict with everything the fitting step needs:

```python
result['hinsa_map']       # 3D HINSA absorption cube (n_v, ny, nx)
result['T_HI_true']       # 3D background HI brightness cube (n_v, ny, nx)
result['R_out_pc']        # core radius (pc)
result['vlsr_kms']        # systemic velocity (km/s)
result['pixel_scale_pc']  # pixel scale (pc/pixel)
result['output_dir']      # output directory
```

Key optional arguments:

| Argument | Default | Meaning |
|---|---|---|
| `polyfit_mask_kms` | `-1` | Masking mode: `-1` simple interactive, `-2` per-cell grid masking, `>0` fixed ±window (km/s), `0`/`None` fixed ±3 |
| `extra_mask_ranges` | `None` | Extra velocity ranges to exclude `[(v1, v2), ...]` |
| `vlsr_override` | `None` | Override Vlsr (km/s) instead of fitting the line |
| `distance_override` | `None` | Override distance (pc) |
| `fetch_planck_av` | `True` | Fetch Planck column density |

Output files: `hinsa_map.fits`, `T_HI_true.fits`, `baseline.fits`, `mask.fits`, `metadata.json`, `diagnostic.png`.

### 2. Fit

```python
from hinspheres import Config, fit_hinspheres

cfg = Config(
    n_shells=9,
    R_out_pc=result['R_out_pc'],
    vlsr_kms=result['vlsr_kms'],
    v_min_kms=-10.0, v_max_kms=25.0,
    n_v_channels=301,
)

best_params, history, param_stds = fit_hinspheres(
    cfg=cfg,
    obs_hinsa_map=result['hinsa_map'],       # 3D observed cube
    T_HI_true_map=result['T_HI_true'],       # 3D background cube
    max_gen=200,
    popsize=24,
    n_jobs=4,
)
```

File-based entry point with automatic output files:

```python
from hinspheres import fit_hinsa_model

result = fit_hinsa_model(
    obs_hinsa_fits='G206_HI_cube.fits',
    obs_background_fits='G206_HI_background.fits',
    mode='forward',
)
print(result['best_params'], result['residual'], result['param_stds'])
```

The integer parameter `peak_shell` is scanned on a grid (each value triggers an independent CMA-ES run); the remaining continuous parameters are optimized by CMA-ES. Pass `method='cma+emcee'` or `'mcmc'` to refine the posterior with emcee (corner and chain plots are saved automatically).

### 3. Forward modeling (synthetic data)

```python
from hinspheres import generate_sim_hinsa

result = generate_sim_hinsa(
    output_path='output.fits',
    background='G206_HI_background.fits',   # None = uniform 40 K background
    vlsr_kms=9.3,
    R_out_pc=0.91,
    distance_pc=1290.0,
    rho0=500, r0=0.05, alpha=1.0,
    T0=10.0, T1=40.0, rT=0.15,
    f_ff=0.15, turb_kms=0.15,
    v_rot_kms=0.3, rot_pa_deg=45.0,
    n_shells=9, n_jobs=4,
    spatial_res_arcmin=4.0,   # optional beam convolution
    vel_res_kms=0.3,          # optional velocity smoothing
)
cube = result['out_cube']     # (n_v, ny, nx)
velo = result['velo_kms']     # velocity axis (km/s)
```

HI abundance can also be given shell by shell with `abundance=[0.02, 0.03, ...]` (takes precedence over the peak-shell form).

## Fitting Modes

| Mode | Needs background | Objective |
|---|---|---|
| `mode='forward'` (default, recommended) | yes (`obs_background_fits`) | Weighted residual between observed and modeled cube |
| `mode='second_derivative'` | no | R-value of the second derivative of the background reconstructed by inverse RT (Liu et al. 2021) |

Known limitation: `second_derivative` has a tau -> 0 degeneracy (the optimizer can drive the optical depth to zero and collapse the R-value), so use `mode='forward'` for real fits unless you have a specific reason not to.

## Objective

The fitter minimizes the radially weighted mean of squared residuals in K^2:

- weight `w = 1/r` (r in pc, center pixel set to one pixel scale), which compensates for the growing ring area with radius,
- restricted to `r <= R_out`,
- restricted to the velocity window `|v - v_LSR| <= 3 km/s`; additional channels are excluded through the `XRM*` header keywords written by `prepare_hinspheres_input`, `extra_mask_ranges_kms=[(-8.0, -4.0), ...]`, or the per-pixel `EXMASK` extension,
- normalized by the total weight, so a perfect fit gives 0 and the value is independent of map size.

## Physical Parameters

| Parameter | Typical range | Meaning |
|---|---|---|
| `rho0` | 100-10000 cm^-3 | central H density |
| `r0` | 0.01-0.15 pc | Plummer core radius |
| `alpha` | 1.0-3.0 | density power-law index (smaller = flatter) |
| `T0` | 3-15 K | central spin temperature (cold core) |
| `T1` | 10-60 K | envelope spin temperature (warm) |
| `rT` | 0.01-0.15 pc | temperature transition radius |
| `f_ff` | 0-0.5 | fraction of free-fall infall speed |
| `turb_kms` | 0.05-0.5 km/s | micro-turbulent speed |
| `v_rot_kms` | 0-5 km/s | rotation speed at `R_out` |
| `rot_pa_deg` | 0-180 deg | rotation axis position angle (N to E) |
| `v_offset` | -3 to +3 km/s | velocity offset from Vlsr |
| `peak_shell` | 4..n_shells | shell of peak HI abundance |
| `f_HI_peak` | 0.001-1.0 | peak HI abundance |

Fit bounds live in `Config.bounds_pc`; physical parameter values are in `Config`.

## Outputs

`generate_sim_hinsa` writes:

| File | Content |
|---|---|
| `output.fits` | synthetic cube |
| `output.json` | configuration and inputs |
| `output.png` | 6-panel diagnostic (profiles, moment 0, absorption, spectra) |

`fit_hinsa_model` writes:

| File | Content |
|---|---|
| `{name}_bestfit.fits` | best-fit model cube |
| `{name}_bestfit.png` | 6-panel diagnostic |
| `{name}_grid_spectra.png` | observed vs modeled spectra on a spatial grid |
| `{name}_fit_result.npz` | params, uncertainties, cubes, residual |
| `{name}_corner.png`, `{name}_chains.png` | MCMC plots (if MCMC ran) |

## Full Workflow

```python
from astropy.coordinates import SkyCoord
from hinspheres import Config, fit_hinspheres, prepare_hinspheres_input

# 1. data preparation
result = prepare_hinspheres_input(
    target_id='G206',
    datacube_path='./ot1_hi_destripe.fits',
    output_dir='HIfig/hinspheres_input/',
    center_coord=SkyCoord(ra=206.1, dec=-15.77, unit='deg'),
    vlsr_kms=9.3,
    spatial_radius_arcmin=12.0,
    velo_radius_kms=15.0,
)

# 2. fit
cfg = Config(n_shells=9, R_out_pc=result['R_out_pc'],
             vlsr_kms=result['vlsr_kms'])
best_params, history, param_stds = fit_hinspheres(
    cfg=cfg,
    obs_hinsa_map=result['hinsa_map'],
    T_HI_true_map=result['T_HI_true'],
    max_gen=200,
    popsize=24,
    n_jobs=4,
)
```

## Package Structure

```
hinspheres/
├── __init__.py      # public API
├── config.py        # physical constants and configuration (Config, bounds_pc)
├── profiles.py      # Plummer density/temperature/abundance/infall profiles
├── rt.py            # ray tracing and radiative transfer (incl. inverse RT)
├── models.py        # forward modeling (build_synthetic_hinsa, generate_sim_hinsa)
├── fitters.py       # CMA-ES fitter (fit_hinsa_model, fit_hinspheres, reload_fit_result)
├── mcmc.py          # emcee posterior sampling
├── prepare.py       # data preparation (prepare_hinspheres_input)
└── utils.py         # diagnostic plots
```

## References

- Goldsmith, P. F. 2007, ApJ, 668, 1043
- Li, D. & Goldsmith, P. F. 2003, ApJ, 585, 823
- Liu, T. et al. 2021, ApJS, 252, 5
- Zuo, P. et al. 2018, ApJ, 858, 89
