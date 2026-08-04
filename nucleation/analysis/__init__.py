"""Analysis layer: this paper's specific choices, built on nucleation + eos.

Where the core package (`nucleation.barrier` ... `nucleation.conditions`) knows
nothing about stars, quark-parameter grids or matplotlib, everything here does.
This is where the 2-family quark-nucleation study lives.

Layout (bottom to top):

    config      -- FilterConfig / NucConfig / StarMatch: the setup objects that
                   replaced the notebook's loose globals
    filters     -- which (alpha_s, B^1/4, Delta_0) cells are admissible at all
                   (Witten bound, no re-hadronization, TOV M_max window)
    sigma_crit  -- the single-point core: the surface tension at which the
                   nucleation time hits its target, at a PNS centre or shell
    scan        -- sweep filters + sigma_crit over the parameter plane (Part II
                   data production), with caching and .npz output
    replay      -- re-solve the accepted cells into M-R / P(mu_B) curve bundles
    stellar     -- TOV sequence bookkeeping
    droplet     -- droplet observables and parameter-plane maps
    outcomes    -- the PNS-evolution NS / QS / BH decision engine
    stability   -- absolute-stability boundaries at P=0
    figure      -- publication style (from eos) + this paper's drawing primitives

Everything is re-exported here, so `import nucleation.analysis as nuc_an`
followed by `nuc_an.<name>` works regardless of which module a name lives in.
"""
from .config import (
    FilterConfig, NucConfig, StarMatch,
    build_PH_of_muB, make_star_match, REASON_CODE,
)
from .filters import (
    cfl_eos_at_params, unpaired_eos_at_params, zero_crossing, ud_eps_per_nB,
    passes_cfl_filters, passes_unpaired_filters,
    rehad_pressure_profile, rehad_flags,
)
from .replay import replay_cfl, replay_accepted
from .sigma_crit import (
    crossover_radius, hadronic_point, central_state, star_shell_states,
    critical_droplet_pt, tau_pt, sigma_target_pt,
)
from .scan import (
    scan_cfl_filters, scan_unpaired_filters, compute_sigma_crit,
    run_sigma_crit_scan, plot_sigma_crit_grid,
)
from .stellar import (
    TOV_COL, stable_branch, max_masses, branch_interp, snapshot_key,
    load_tov_trapped, nearest_trapped_sequence, cold_quark_star_branch,
)
from .stability import (
    E_PER_BARYON_STABLE, energy_per_baryon_at_P0, two_flavour_energy_at_P0,
    B4_at_energy, stability_curve, resample_curve,
)
from .droplet import (
    REGIME, droplet_regime_grid, barrier_ratio_map, sigma_crit_along_sequence,
)
from .outcomes import (
    M_SUN_C2_ERG, Snapshot, EvolutionTrack, nucleates, sigma_crit_at_snapshot,
    outcome_row, outcomes_table, to_latex_tabular, pick_cell_by_Mmax,
)
# Re-export the private joblib-capability flag (the notebook reads
# nuc_an._HAVE_JOBLIB); the `as` form marks it a deliberate re-export so
# linters do not flag it unused.
from .scan import _HAVE_JOBLIB as _HAVE_JOBLIB

__all__ = [
    # config
    "FilterConfig", "NucConfig", "StarMatch",
    "build_PH_of_muB", "make_star_match", "REASON_CODE",
    # filters
    "cfl_eos_at_params", "unpaired_eos_at_params", "zero_crossing",
    "ud_eps_per_nB", "passes_cfl_filters", "passes_unpaired_filters",
    "rehad_pressure_profile", "rehad_flags",
    # replay
    "replay_cfl", "replay_accepted",
    # sigma_crit core
    "crossover_radius", "hadronic_point", "central_state", "star_shell_states",
    "critical_droplet_pt", "tau_pt", "sigma_target_pt",
    # scan
    "scan_cfl_filters", "scan_unpaired_filters", "compute_sigma_crit",
    "run_sigma_crit_scan", "plot_sigma_crit_grid",
    # stellar
    "TOV_COL", "stable_branch", "max_masses", "branch_interp", "snapshot_key",
    "load_tov_trapped", "nearest_trapped_sequence", "cold_quark_star_branch",
    # stability
    "E_PER_BARYON_STABLE", "energy_per_baryon_at_P0", "two_flavour_energy_at_P0",
    "B4_at_energy", "stability_curve", "resample_curve",
    # droplet
    "REGIME", "droplet_regime_grid", "barrier_ratio_map",
    "sigma_crit_along_sequence",
    # outcomes
    "M_SUN_C2_ERG", "Snapshot", "EvolutionTrack", "nucleates",
    "sigma_crit_at_snapshot", "outcome_row", "outcomes_table",
    "to_latex_tabular", "pick_cell_by_Mmax",
]
