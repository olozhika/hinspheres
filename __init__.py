"""
hinspheres - Spherically symmetric, multi-layer HINSA radiative transfer modeling.
Forward-fits radial profiles (density, temperature, HI abundance, infall velocity)
to observed HINSA absorption maps.

Reference models:
  - Li & Goldsmith (2003)  : three-component HI 21cm RT
  - Goldsmith, Li & Krco    (2007) : HI->H2 time evolution in slab clouds
  - Zuo+2018                : onion-like HI shells in dark clouds
  - Plummer density profile : n(r) = n0 / (1 + (r/r0)^alpha)
"""

from .config import Config
from .profiles import density_plummer, temperature_plummer, \
    abundance_profile_111n, infall_velocity
from .rt import make_shells, los_path_lengths, radiative_transfer_pixel
from .models import build_synthetic_hinsa, forward_model_cube, generate_sim_hinsa
from .fitters import fit_hinspheres

__all__ = [
    'Config',
    'density_plummer', 'temperature_plummer',
    'abundance_profile_111n', 'infall_velocity',
    'make_shells', 'los_path_lengths', 'radiative_transfer_pixel',
    'build_synthetic_hinsa', 'forward_model_cube', 'generate_sim_hinsa',
    'fit_hinspheres',
]
