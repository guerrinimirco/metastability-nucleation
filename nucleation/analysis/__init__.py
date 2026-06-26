"""Analysis utilities built on the nucleation + eos packages.

``sigma_crit``: critical-surface-tension parameter scans (CFL filters +
single-point nucleation + grid scan + plots) for 2-family quark nucleation.
"""
from .sigma_crit import (
    FilterConfig, NucConfig, StarMatch,
    build_PH_of_muB, make_star_match,
    cfl_eos_at_params, zero_crossing, passes_cfl_filters, replay_cfl, ud_eps_per_nB,
    crossover_radius, central_state, tau_pt, sigma_target_pt,
    scan_cfl_filters, compute_sigma_crit, run_sigma_crit_scan,
    plot_sigma_crit_grid, REASON_CODE,
)
# Re-export the private joblib-capability flag (the notebook reads nuc_an._HAVE_JOBLIB);
# the `as` form marks it a deliberate re-export so linters don't flag it unused.
from .sigma_crit import _HAVE_JOBLIB as _HAVE_JOBLIB

__all__ = [
    "FilterConfig", "NucConfig", "StarMatch",
    "build_PH_of_muB", "make_star_match",
    "cfl_eos_at_params", "zero_crossing", "passes_cfl_filters", "replay_cfl",
    "ud_eps_per_nB",
    "crossover_radius", "central_state", "tau_pt", "sigma_target_pt",
    "scan_cfl_filters", "compute_sigma_crit", "run_sigma_crit_scan",
    "plot_sigma_crit_grid", "REASON_CODE",
]
