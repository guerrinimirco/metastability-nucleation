# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Coexistence of strange quark stars and neutron stars
# ## Metastability and quark-droplet nucleation in proto-neutron stars
#
# Companion notebook to the paper. It computes the thermal nucleation of a
# deconfined quark droplet inside the hot, dense hadronic matter of a
# proto-neutron star (PNS), in the **two-family scenario**: neutron stars and
# strange quark stars coexist as distinct families, because strange quark matter
# is absolutely stable but ordinary nuclei are protected by a nucleation barrier.
#
# ### The physical question
#
# If strange quark matter is the true ground state, why is any neutron star still
# a neutron star? Because converting one requires nucleating a quark droplet, and
# a droplet costs surface energy before it repays bulk energy. The barrier
# $W_*$ that results is crossed at a rate $\propto e^{-W_*/T}$, so the answer
# depends on a quantity nobody has measured: the quark–hadron **surface tension**
# $\sigma$.
#
# Rather than assume a value for $\sigma$, we invert the problem and report the
# threshold $\sigma_{\rm crit}$ at which nucleation happens within the PNS
# lifetime. A star converts if $\sigma < \sigma_{\rm crit}$. The reader can then
# place any preferred $\sigma$ against our map.
#
# ### Structure
#
# | Part | What it does | Needs |
# |------|--------------|-------|
# | **I**   | Imports, parameters, grids | — |
# | **II**  | Generate every table and scan; write to `output/` | I |
# | **III** | Load what Part II produced | II (once, ever) |
# | **IV**  | The paper figures + the outcomes table | III |
# | **V**   | Supplementary material, same style | III |
#
# Part II is the only expensive part. Once it has run, Parts III–V replay from
# disk in seconds, so day-to-day work on the figures never re-runs the physics.
#
# ### Reproducing
#
# Set `REDUCED_GRID = True` (Part I) for a coarse end-to-end run in ~10 minutes,
# which writes to `output/smoke/` and validates that every cell works. Set it to
# `False` for the production grids behind the published figures, which write to
# `output/paper/` and take hours. See `docs/reproducing.md`.

# %% [markdown]
# # Part I — Setup & parameters
#
# Everything tunable lives in this Part. Later Parts read these names and should
# not introduce new constants of their own.

# %% [markdown]
# ## I.1 — Imports
#
# All imports are here, not scattered through the notebook: a cell that imports
# what it needs looks self-contained but silently depends on having been run.

# %%
# --- standard library ---------------------------------------------------------
import os
import glob
from pathlib import Path
from dataclasses import replace as dc_replace

# --- third party --------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import interp1d
from scipy.optimize import brentq

# --- eos: equation of state, TOV structure, publication figure style -----------
from eos.sfho.parameters import create_custom_parametrization
from eos.sfho.compute_tables import (
    TableSettings, compute_table,
    load_eos_table as load_eos_table_sfho,
    build_interpolators as build_interpolators_sfho,
)
from eos.alphabag.parameters import get_alphabag_custom
from eos.alphabag.compute_tables import AlphaBagTableSettings, compute_alphabag_table
from eos.alphabag.thermodynamics_quarks import T_critical
from eos.tov.solver import (EOSTable_for_TOV, generate_ec_logspace,
                            compute_tov_sequence, truncate_to_stable_branch)

# --- nucleation: the core engine ----------------------------------------------
from nucleation import (
    compute_Qstar_table, build_Qstar_interpolators, load_Qstar_table,
    export_table, QstarTableData,
    compute_thermal_nucleation_observables, load_thermal_nucleation_table,
    build_thermal_nucleation_interpolators,
    compute_energy_barrier,
    NucleationCondition, nucleation_point, hadronic_point, crossover_radius,
    compute_nucleation_density, nucleation_curve,
)

# --- nucleation.analysis: this paper's specific choices ------------------------
import nucleation.analysis as nuc_an
from nucleation.analysis import (
    FilterConfig, NucConfig, make_star_match, build_PH_of_muB,
    central_state, star_shell_states, critical_droplet_pt, sigma_target_pt,
    passes_cfl_filters, replay_cfl, replay_accepted,
    cfl_eos_at_params, unpaired_eos_at_params, zero_crossing, ud_eps_per_nB,
    scan_unpaired_filters, compute_sigma_crit, run_sigma_crit_scan,
    REASON_CODE,
    TOV_COL, stable_branch, branch_interp, snapshot_key, load_tov_trapped,
    nearest_trapped_sequence, cold_quark_star_branch,
    energy_per_baryon_at_P0, two_flavour_energy_at_P0, B4_at_energy,
    stability_curve, resample_curve,
    droplet_regime_grid, barrier_ratio_map, sigma_crit_along_sequence,
    EvolutionTrack, outcome_row, to_latex_tabular, pick_cell_by_Mmax,
    M_SUN_C2_ERG,
)
from nucleation.analysis.figure import (
    set_paper_style, paper_grid, panel_label,
    STANDARD_COLORS, OKAB, OKAB_CAT, PHASE_LW, PHASE_ALPHA,
    add_observational_constraints,
    mass_marks, label_with_arrows, isentrope_mass_markers,
    sigma_map, diverging_map, symmetric_vlim, iso_lines,
    reject_outlines, regime_outlines,
    ISO_SIGMA_COLOUR, ISO_MASS_COLOUR,
    resample_profiles, band, quantile_bins, summary_points,
)

print(f"joblib parallelism available: {nuc_an._HAVE_JOBLIB}")

# %% [markdown]
# ## I.2 — Run mode: smoke test or production
#
# One switch controls every grid in the notebook. The **physics is identical** in
# both modes — only the sampling changes — and the two write to *different*
# directories, so a quick validation run can never overwrite or be mistaken for
# paper data.

# %%
# =============================================================================
#  THE run switch.
#    True  -> coarse grids; Part II finishes in ~10 min and Parts III-V run
#             end-to-end. Figures are jagged and the numbers are NOT paper
#             values. Writes to output/smoke/.
#    False -> the production grids behind the published figures. Part II takes
#             hours. Writes to output/paper/.
# =============================================================================
REDUCED_GRID = True

OUT = Path('../output') / ('smoke' if REDUCED_GRID else 'paper')
DIRS = {
    'H_eos':     OUT / 'tables' / 'hadronic_eos',
    'H_tov':     OUT / 'tables' / 'hadronic_tov',
    'Q_eos':     OUT / 'tables' / 'quark_eos',
    'Q_tov':     OUT / 'tables' / 'quark_tov',
    'Qstar':     OUT / 'tables' / 'qstar',
    'nucleation': OUT / 'tables' / 'nucleation',
    'sigma_crit': OUT / 'tables' / 'sigma_crit',
    'fig_paper': OUT / 'figures' / 'paper',
    'fig_supp':  OUT / 'figures' / 'supplementary',
    'fig_data':  OUT / 'figure_data',
}
for _d in DIRS.values():
    _d.mkdir(parents=True, exist_ok=True)

if REDUCED_GRID:
    print("!" * 78)
    print("!  REDUCED_GRID = True -- SMOKE TEST MODE")
    print("!  Coarse sampling: figures are jagged, numbers are NOT paper values.")
    print(f"!  Writing to {OUT}")
    print("!" * 78)
else:
    print(f"Production mode. Writing to {OUT}")

# %% [markdown]
# ## I.3 — Hadronic phase: parametrization and grids
#
# The hadronic side is SFHo extended with hyperons and $\Delta$ resonances
# (`2fam_phi`). The $\sigma$–$\Delta$ coupling $x_{\sigma\Delta}$ is the most
# uncertain of the couplings and is the one we tag runs by.
#
# Four independent variables appear, and which ones are active depends on the
# equilibrium being imposed:
#
# * $n_B$ — baryon density, always;
# * $T$ — temperature, for the isothermal tables;
# * $Y_L$ — lepton fraction, when neutrinos are **trapped** ($\mu_\nu\neq0$), in
#   which case $Y_L$ is conserved and replaces $\beta$-equilibrium;
# * $S = s/n_B$ — entropy per baryon, for the **isentropic** profiles that
#   describe a real PNS snapshot.

# %%
# =============================================================================
#  Hadronic parametrization
# =============================================================================
n_sat = 0.1583                # nuclear saturation density [fm^-3]

parametrization  = '2fam_phi'                    # SFHo + hyperons + Deltas
particle_content = 'nucleons_hyperons_deltas'

# Subdominant contributions to the EoS. All three matter at PNS temperatures.
include_photons             = True   # blackbody photons
include_pseudoscalar_mesons = True   # thermal pi, K, eta
include_thermal_neutrinos   = True   # thermal nu/nubar (only where mu_nu = 0)

# --- couplings -----------------------------------------------------------------
# x_yD = g_yD / g_yN: Delta-meson couplings relative to the nucleon. The sigma-Delta
# ratio is the uncertain one; the literature range is ~1.0-1.25 and it controls
# how early Deltas appear, hence how soft the EoS gets.
# U_HN = depth of the hyperon-nucleon potential at n_sat in symmetric matter,
# which pins each hyperon family's scalar coupling.
x_sigma_delta = 1.15          # TUNABLE: the headline coupling; tags the run
x_omega_delta = 1.0           # SU(6) default
x_rho_delta   = 1.0           # SU(6) default
U_Lambda_N = -28.0            # MeV
U_Sigma_N  = +30.0            # MeV, positive = strongly repulsive
U_Xi_N     = -18.0            # MeV

xsd_tag = f"xsd{int(round(x_sigma_delta * 100))}"      # 1.15 -> 'xsd115'

params_H = create_custom_parametrization(
    U_Lambda_N=U_Lambda_N, U_Sigma_N=U_Sigma_N, U_Xi_N=U_Xi_N,
    x_sigma_delta=x_sigma_delta, x_omega_delta=x_omega_delta,
    x_rho_delta=x_rho_delta, name=f"2fam_phi_{xsd_tag}")

# =============================================================================
#  Grids. Every entry is TUNABLE; the reduced values only change sampling.
# =============================================================================
GRIDS = dict(
    # n_B: 0.1 -> 12 n_sat, from the crust edge to well above any stellar centre.
    n_B=np.linspace(0.1, 12, 60 if REDUCED_GRID else 300) * n_sat,

    # T: two near-zero anchors (the cold limit) + a uniform ladder to 100 MeV,
    # which brackets the PNS / merger-remnant regime.
    T=np.concatenate([[0.01, 0.1],
                      np.arange(2, 101., 8 if REDUCED_GRID else 2)]),

    # Y_L: NEVER reduced. 0.25 and 0.35 must remain grid nodes because they are
    # the two PNS snapshots (I.6); dropping them breaks Figs 2/3/4 and the
    # outcomes table without any error being raised.
    Y_L=np.arange(0.1, 0.401, 0.05),

    # S: entropy per baryon for the isentropic tables.
    S=np.arange(0.5, 4.01, 0.5),

    # Quark EoS: denser in n_B than the hadronic table (quark matter only matters
    # at high density and its EoS is smooth, so this stays cheap).
    n_B_quark=np.linspace(0.5, 15, 80 if REDUCED_GRID else 400) * n_sat,
)

# TOV: entropies for which trapped sequences are built. Both 1.5 and 2.0 are
# REQUIRED -- they are the two PNS snapshots.
S_TOV = [1.5, 2.0] if REDUCED_GRID else [1.0, 1.5, 2.0, 2.5, 3.0]
Y_L_TOV = [0.25, 0.35] if REDUCED_GRID else list(np.arange(0.10, 0.401, 0.05))

# Central energy densities for the TOV sequences [MeV/fm^3].
E_C_VEC = generate_ec_logspace(e_min=150, e_max=2500,
                               n_points=25 if REDUCED_GRID else 100)

print(f"n_B: {len(GRIDS['n_B'])} pts, {GRIDS['n_B'].min():.3f}-{GRIDS['n_B'].max():.3f} fm^-3"
      f"  ({GRIDS['n_B'].min()/n_sat:.1f}-{GRIDS['n_B'].max()/n_sat:.1f} n_sat)")
print(f"T  : {len(GRIDS['T'])} pts, {GRIDS['T'].min()}-{GRIDS['T'].max()} MeV")
print(f"Y_L: {len(GRIDS['Y_L'])} pts | S: {len(GRIDS['S'])} pts")

# %% [markdown]
# ## I.4 — Quark phase: parameter sets
#
# The quark side is the $\alpha$-Bag model: a bag constant $B^{1/4}$, a
# perturbative-QCD correction $\alpha_s$, the strange-quark mass $m_s$, and the
# CFL pairing gap $\Delta_0$.
#
# Two representative sets are carried through the figures. They are chosen to
# bracket the viable region found by the Part-II scan, not tuned by hand.

# %%
# =============================================================================
#  Quark parameter sets carried through the figures.
#  TUNABLE: add a dict here and every per-set figure picks it up.
# =============================================================================
quark_param_sets = [
    dict(alpha=0.1 * np.pi / 2, B4=145.0, Delta0=80.0,  m_s=100.0),
    dict(alpha=0.08 * np.pi / 2, B4=158.0, Delta0=157.0, m_s=100.0),
]


def q_tag(p):
    """Filename / dict tag uniquely identifying a quark parameter set."""
    return (f"B4{p['B4']:.0f}_D{p['Delta0']:.0f}"
            f"_a{p['alpha']:.2f}_ms{p['m_s']:.0f}")


for _p in quark_param_sets:
    print(f"  {q_tag(_p)}:  alpha_s={_p['alpha']:.3f}  B^1/4={_p['B4']:.0f} MeV  "
          f"Delta_0={_p['Delta0']:.0f} MeV  m_s={_p['m_s']:.0f} MeV")

# %% [markdown]
# ## I.5 — Nucleation setup
#
# **Surface tension** $\sigma$ is the free parameter of the whole problem, so
# tables are built at several values and $\sigma_{\rm crit}$ is solved for.
#
# **Nucleation volume** $V$: the region over which the central conditions are
# roughly uniform, taken as a sphere of radius 100 m. It only rescales $\tau$
# (never the barrier), and $\tau$ spans tens of decades, so the result is
# insensitive to it.
#
# **Target time** $\tau_{\rm target}$: the timescale a droplet must appear within
# to matter. We use 1 ms — short compared with the deleptonization time, so a
# star that has not converted by then survives the snapshot.
#
# **Droplet phase** — three options, and the paper uses all three:
# `unpaired` (no pairing), `cfl` (colour-flavour locked throughout), and
# `unpCFL`, the physical composite where the droplet is unpaired inside the
# pairing coherence radius $R_\Delta = \hbar c/\Delta(T)$ and CFL outside it.

# %%
# =============================================================================
#  Nucleation parameters
# =============================================================================
V_NUC = 4.18879e51            # fm^3; sphere of R = 100 m
TAU_TARGET = 1e-3             # s; must nucleate within ~1 ms to matter

# Surface tensions for which Q* / nucleation tables are built [MeV/fm^2].
SIGMA_LIST = ([50., 100., 150.] if REDUCED_GRID
              else [50., 80., 100., 150., 200., 250.])

# The (flavour, charge, phase) combinations tabulated in Part II.
#   flavour: 'saddlepoint' (composition minimises the barrier) or 'frozen'
#   charge : 'lcn' local neutrality | 'gcn' global | 'coulomb_minimize'
#            self-consistent Coulomb (the physical one) | 'screening' Debye
CHARGE_MODES = ['lcn', 'gcn', 'coulomb_minimize']
QUARK_PHASES = ['unpaired', 'cfl']       # unpCFL is composed from these two

# The (flavour, charge, phase) used for every headline number in the paper.
MAIN_FLAVOR, MAIN_CHARGE, MAIN_PHASE = 'saddlepoint', 'coulomb_minimize', 'unpCFL'

# %% [markdown]
# ## I.6 — The two PNS snapshots
#
# A proto-neutron star is not one object but a sequence of them. We follow two
# moments, because the nucleation answer differs between them:
#
# * $t_0$ — **just after bounce**: lepton-rich ($Y_L=0.35$) and comparatively
#   cold ($S=1.5$). Neutrino trapping keeps the EoS stiff, so $M_{\max}$ is high.
# * $t_{T_{\max}}$ — **peak temperature**, a few seconds later: deleptonized to
#   $Y_L=0.25$ and hotter ($S=2.0$). The EoS has softened, so $M_{\max}$ has
#   *fallen* — a star that was stable at $t_0$ may not be now.
#
# Baryon number is conserved between them; gravitational mass is not. That
# asymmetry is what makes the outcome (NS / QS / BH) non-trivial.

# %%
PNS_T0   = dict(YLH=0.35, S=1.5, label=r'$t_0$')
PNS_TMAX = dict(YLH=0.25, S=2.0, label=r'$t_{T_{\rm max}}$')

# The snapshot the sigma_crit maps are evaluated at (the hotter, more permissive
# one -- if it does not nucleate there, it does not nucleate).
SCAN_SNAPSHOT = PNS_TMAX

# Gravitational masses [M_sun] of the cold, catalogued star whose baryon number
# labels a PNS track. TUNABLE: the outcomes table sweeps these.
MT0_GRID = [1.4] if REDUCED_GRID else [1.17, 1.4, 1.6]

# %% [markdown]
# ## I.7 — Parameter-plane scan setup
#
# The central result is a map of $\sigma_{\rm crit}$ over the quark parameter
# plane $(B^{1/4}, \Delta_0)$ at fixed $\alpha_s$, with the regions excluded by
# each acceptance filter outlined on top.
#
# **Star-wide vs centre-only $\sigma_{\rm crit}$.** Nucleation is not always
# fastest at the centre: near the strange-matter-stability corner the driving
# force peaks around $2\,n_{\rm sat}$ and *weakens* toward the centre, so an
# off-centre shell nucleates first. $\sigma_{\rm crit}$ is therefore the maximum
# over `N_SHELLS` shells, not the central value — the centre-only number
# underestimates it by up to a factor ~2 there.

# %%
# =============================================================================
#  Acceptance filters (which quark parameters describe a viable strange star)
#
#  These are the TUNABLE knobs. The FilterConfig itself is built in Part II,
#  because it also needs the cold hadronic P_H(mu_B) comparator, which does not
#  exist until the hadronic tables have been computed.
# =============================================================================
FILTER_KW = dict(
    m_s=100.0,
    n_B_grid=np.linspace(0.05, 2.5, 250),   # fm^-3, for the cold quark EoS solve
    e_c_vec_tov=E_C_VEC,
    M_max_window=(2.0, np.inf),   # must support the heaviest observed pulsar
    e_over_nB_max=930.0,          # Witten: 3-flavour SQM must be BOUND
    e_over_nB_2flavor=930.0,      # ... and 2-flavour matter must NOT be
    check_2flavor=True,
    tov_backend='fast',           # numba solver; 'scipy' is the trusted reference
)

nuc_cfg = NucConfig(
    sig_lo=0.001, sig_hi=300.0,     # sigma bracket for the sigma_crit root-find
    n_sigma_scan=40,                # NOT safely reducible: this coarse scan picks
                                    # WHICH bracket brentq then refines, so
                                    # shrinking it changes the answer, not just
                                    # its precision.
    tau_target=TAU_TARGET, V=V_NUC,
    include_photons=include_photons, include_gluons=True,
    include_thermal_neutrinos=include_thermal_neutrinos)

# Shells sampled from nB_min to the centre when computing star-wide sigma_crit.
# 6 is converged to <0.5% against 12; 2 (smoke mode) is NOT converged.
N_SHELLS = 2 if REDUCED_GRID else 6
NB_SHELL_MIN = 0.25              # fm^-3, the innermost shell's lower bound

# =============================================================================
#  The (alpha_s, B^1/4, Delta_0) scan grid
# =============================================================================
SCAN = dict(
    alpha=([0.1 * np.pi / 2] if REDUCED_GRID
           else [0.0, 0.1 * np.pi / 2, 0.2 * np.pi / 2, 0.3 * np.pi / 2]),
    B4=np.linspace(130., 180., 11 if REDUCED_GRID else 51),
    Delta0=np.linspace(0., 200., 11 if REDUCED_GRID else 51),
    m_s=100.0,
    n_jobs=-1,
)

# %% [markdown]
# ## I.8 — Figure style and consistency checks
#
# The publication style is applied **once**, here. Every figure then goes through
# `paper_grid(**PAPER_STYLE)`; no cell writes `plt.rcParams` directly. (Doing so
# is what previously made the observational contours render at the wrong size in
# one panel but not another.)
#
# The assertion below is the one that matters: it fails *now*, in Part I, if a
# figure asks for a surface tension that Part II will never tabulate. Without it
# the figure silently drops a curve.

# %%
# TUNABLE: pt sizes for every paper figure. fontsize=title/fallback,
# labelsize=axis names + tick numbers, legendsize=legend, aspect=panel W/H.
PAPER_STYLE = dict(fontsize=11, labelsize=11, legendsize=9, aspect=1.1)
set_paper_style(**{k: v for k, v in PAPER_STYLE.items() if k != 'aspect'})

# --- which sigma values the figures will ask for -------------------------------
FIG_SIGMAS = {
    'fig1_barrier':   [SIGMA_LIST[0]],
    'fig1b_sweep':    list(SIGMA_LIST),
    'fig3_centre':    [SIGMA_LIST[0], SIGMA_LIST[-1]],
    'fig4_Tnuc':      [SIGMA_LIST[0], SIGMA_LIST[len(SIGMA_LIST) // 2]],
    'appA_charge':    [SIGMA_LIST[0], SIGMA_LIST[-1]],
    'appB_flavour':   [SIGMA_LIST[0]],
    'outcomes_table': [SIGMA_LIST[0], SIGMA_LIST[len(SIGMA_LIST) // 2]],
}
_wanted = sorted(set().union(*FIG_SIGMAS.values()))
_missing = sorted(set(_wanted) - set(SIGMA_LIST))
assert not _missing, (
    f"figures request sigma {_missing} but SIGMA_LIST is {SIGMA_LIST} -- "
    f"Part II will never tabulate those, and the figures would silently drop "
    f"the corresponding curves. Add them to SIGMA_LIST or fix FIG_SIGMAS.")

# --- the PNS snapshots must be grid nodes --------------------------------------
for _snap in (PNS_T0, PNS_TMAX):
    assert np.isclose(GRIDS['Y_L'], _snap['YLH']).any(), (
        f"Y_L={_snap['YLH']} is not a node of GRIDS['Y_L']; the snapshot would "
        f"be interpolated rather than computed.")
    assert _snap['S'] in S_TOV, (
        f"S={_snap['S']} is not in S_TOV; no trapped TOV sequence would exist.")

print(f"sigma values needed by figures: {_wanted}")
print(f"sigma values tabulated        : {SIGMA_LIST}")
print("Part I consistency checks passed.")

# %% [markdown]
# # Part II — Table generation
#
# **This is the only expensive Part.** It writes everything Parts III–V read, so
# it is run once per configuration and then left alone. Nothing here plots.
#
# | Section | Produces | Into |
# |---------|----------|------|
# | II.1 | Hadronic EoS tables (4 equilibria) | `tables/hadronic_eos/` |
# | II.2 | Hadronic TOV sequences (cold + trapped) | `tables/hadronic_tov/` |
# | II.3 | Quark EoS tables (unpaired + CFL) | `tables/quark_eos/` |
# | II.4 | Cold quark-star TOV sequences | `tables/quark_tov/` |
# | II.5 | $Q^*$ critical-droplet tables | `tables/qstar/` |
# | II.6 | Thermal nucleation observables | `tables/nucleation/` |
# | II.7 | $\sigma_{\rm crit}$ parameter scans | `tables/sigma_crit/` |
#
# Every section skips work whose output file already exists, so an interrupted
# run resumes rather than restarting. Delete a file to force its recomputation.

# %% [markdown]
# ## II.1 — Hadronic EoS tables
#
# Four tables, one per equilibrium condition. The **trapped** one carries the
# physics: at PNS densities neutrinos are trapped, so lepton number $Y_L$ is
# conserved and replaces $\beta$-equilibrium. The **isentropic** variants give
# $T(n_B)$ along a constant-$S$ profile, which is what an actual PNS follows.

# %%
_H_JOBS = [
    # (key, equilibrium, extra axes, filename stem)
    ('betaeq',      'beta_eq',             dict(T_values=GRIDS['T']),
     f'eos_hadronic_betaeq_sfho_2famphi_{xsd_tag}'),
    ('iso_betaeq',  'isentropic_beta_eq',  dict(S_values=GRIDS['S']),
     f'eos_hadronic_betaeq_sfho_2famphi_{xsd_tag}_isentropic'),
    ('trapped',     'trapped_neutrinos',   dict(Y_L_values=GRIDS['Y_L'],
                                                T_values=GRIDS['T']),
     f'eos_hadronic_trapped_sfho_2famphi_{xsd_tag}'),
    ('iso_trapped', 'isentropic_trapped',  dict(Y_L_values=GRIDS['Y_L'],
                                                S_values=GRIDS['S']),
     f'eos_hadronic_trapped_sfho_2famphi_{xsd_tag}_isentropic'),
]

_common_H = dict(
    parametrization=parametrization,
    custom_params=params_H,            # the actual physics; the name above is a label
    particle_content=particle_content,
    n_B_values=GRIDS['n_B'],
    include_photons=include_photons,
    include_pseudoscalar_mesons=include_pseudoscalar_mesons,
    include_thermal_neutrinos=include_thermal_neutrinos,
    print_results=False, print_errors=True, print_timing=True,
    save_to_file=True,
)

for _key, _eq, _extra, _stem in _H_JOBS:
    _path = DIRS['H_eos'] / f'{_stem}.dat'
    if _path.exists():
        print(f"  [skip] {_stem} already exists")
        continue
    print(f"\n-- hadronic EoS: {_key} ({_eq}) --")
    compute_table(TableSettings(**_common_H, equilibrium=_eq, **_extra,
                                output_filename=str(_path)))

# %% [markdown]
# ## II.2 — Hadronic TOV sequences
#
# Stellar structure for the hadronic EoS: one **cold** sequence (the catalogued
# neutron star a PNS ends up as) and one **trapped** sequence per $(Y_L, S)$
# snapshot.
#
# The baryonic mass is computed because a PNS evolves at fixed **baryon number**,
# not fixed gravitational mass — that is what lets Part IV follow one star across
# snapshots.

# %%
# The hadronic EoS table is spliced onto a COMPOSE SFHo crust below
# n_transition, with a tanh blend of width delta_n. Without a crust the low-mass
# end of the sequence (and the radius of every star) is wrong.
# The hot/trapped case splices LOWER, because a hot crust extends to higher
# density and the trapped COMPOSE tables reach further up.
N_TRANSITION_COLD = 0.5 * n_sat
N_TRANSITION_HOT = 0.3 * n_sat
DELTA_N_BLEND = 0.05 * n_sat


def _hadronic_tov(itp, eq_type, out_path, *, Y_L=None, S=None, T=None, label=""):
    """Solve one hadronic TOV sequence (with crust) and cache it to `out_path`."""
    if out_path.exists():
        return np.loadtxt(out_path)

    n_B = GRIDS['n_B']
    if eq_type == 'isentropic_trapped':
        P, e = itp['P'](n_B, Y_L, S), itp['eps'](n_B, Y_L, S)
        crust_kw = dict(add_crust_table='compose_sfho_nYLS_trap',
                        crust_YL=Y_L, crust_S=S,
                        n_transition=N_TRANSITION_HOT)
    else:
        P, e = itp['P'](n_B, T), itp['eps'](n_B, T)
        crust_kw = dict(add_crust_table='compose_sfho_nT0_beta',
                        n_transition=N_TRANSITION_COLD)

    tov = compute_tov_sequence(
        EOSTable_for_TOV(P=P, epsilon=e, nB=n_B), e_c_vec=E_C_VEC,
        add_crust_mode='interpolate', delta_n=DELTA_N_BLEND, **crust_kw,
        compute_baryonic_mass=True, compute_tidal=False,
        backend=FILTER_KW['tov_backend'], verbose=False)

    # Drop non-finite masses before truncating: central states past the table's
    # validity return NaN M, which would crash the M_max spline inside
    # truncate_to_stable_branch.
    finite = np.isfinite(tov[:, TOV_COL['M']])
    if finite.sum() < 5:
        print(f"    !! {label}: only {int(finite.sum())} finite-M points -- skipped")
        return None
    if not finite.all():
        print(f"    ({label}: dropped {int((~finite).sum())} non-finite-M points)")

    seq, M_max, _ = truncate_to_stable_branch(
        tov[finite], output_file=str(out_path),
        header_info=f"Hadronic {eq_type}, {params_H.name} {label}", verbose=False)
    print(f"    -> {out_path.name}: M_max = {M_max:.3f} M_sun")
    return seq


# --- cold beta-equilibrium sequence (the final, catalogued star) ---------------
print("-- hadronic TOV: cold beta-eq --")
_tov_cold_path = DIRS['H_tov'] / f'tov_hadronic_betaeq_2famphi_{xsd_tag}_T0.dat'
_H_betaeq_itp = build_interpolators_sfho(load_eos_table_sfho(
    str(DIRS['H_eos'] / f'eos_hadronic_betaeq_sfho_2famphi_{xsd_tag}.dat'), 'beta_eq'))
_ = _hadronic_tov(_H_betaeq_itp, 'beta_eq', _tov_cold_path,
                  T=float(GRIDS['T'][0]), label='T=0')

# --- trapped, isentropic sequences: one per (Y_L, S) snapshot -----------------
print("-- hadronic TOV: trapped isentropic --")
_H_iso_itp = build_interpolators_sfho(load_eos_table_sfho(
    str(DIRS['H_eos'] / f'eos_hadronic_trapped_sfho_2famphi_{xsd_tag}_isentropic.dat'),
    'isentropic_trapped'))
for _YL in Y_L_TOV:
    for _S in S_TOV:
        _ = _hadronic_tov(
            _H_iso_itp, 'isentropic_trapped',
            DIRS['H_tov'] / f'tov_hadronic_trapped_2famphi_{xsd_tag}_YL{_YL:.2f}_S{_S:.1f}.dat',
            Y_L=_YL, S=_S, label=f'YL={_YL:.2f} S={_S:.1f}')

# %% [markdown]
# ## II.3 — Quark EoS tables
#
# The $\alpha$-Bag equation of state for each parameter set, in both the
# `unpaired` and `cfl` phases. Needed for the bulk no-rehadronization filter and
# for the $P(\mu_B)$ panel of Fig. 5a.

# %%
for _p in quark_param_sets:
    _stag = q_tag(_p)
    for _label, _extra in (('unpaired', dict(phase='unpaired', equilibrium='beta_eq')),
                           ('cfl',      dict(phase='cfl', Delta0_values=[_p['Delta0']]))):
        _path = DIRS['Q_eos'] / f'eos_quark_{_label}_{_stag}.dat'
        if _path.exists():
            print(f"  [skip] {_path.name}")
            continue
        print(f"\n-- quark EoS [{_stag}] {_label} --")
        compute_alphabag_table(AlphaBagTableSettings(
            alpha=_p['alpha'], B4=_p['B4'], m_s=_p['m_s'],
            n_B_values=GRIDS['n_B_quark'], T_values=GRIDS['T'],
            include_photons=True, include_gluons=True,
            include_thermal_neutrinos=True,
            print_results=False, print_errors=True, print_timing=True,
            save_to_file=True, output_filename=str(_path), **_extra))

# %% [markdown]
# ## II.4 — Hadronic comparator and the filter config
#
# The acceptance filters compare the quark pressure against the **cold hadronic**
# one at equal $\mu_B$. That comparator is built once here and carried in
# `filt_cfg`, so no filter re-solves the hadronic EoS.

# %%
H_table = {}
H = {}
for _key, _eq, _, _stem in _H_JOBS:
    _tab = load_eos_table_sfho(str(DIRS['H_eos'] / f'{_stem}.dat'), _eq)
    H_table[_key] = _tab
    H[_key] = build_interpolators_sfho(_tab)

_PH_fn, _mu_sorted, _P_sorted = build_PH_of_muB(
    H['betaeq'], FILTER_KW['n_B_grid'], T=float(GRIDS['T'][0]))

filt_cfg = FilterConfig(P_H_of_muB=_PH_fn, mu_B_H_sorted=_mu_sorted,
                        P_H_sorted=_P_sorted, **FILTER_KW)
print(f"hadronic comparator built over mu_B = "
      f"{_mu_sorted.min():.0f}-{_mu_sorted.max():.0f} MeV")

# %% [markdown]
# ## II.5 — Cold quark-star TOV sequences
#
# A strange quark star is **self-bound**: it needs no crust and the sequence
# starts at $P=0$ with finite density. These give the remnant mass in Part IV,
# read off at fixed baryon number.

# %%
for _p in quark_param_sets:
    _path = DIRS['Q_tov'] / f'tov_quark_cfl_{q_tag(_p)}.dat'
    _seq, _Mmax = cold_quark_star_branch(
        _p, filt_cfg, e_c_vec=E_C_VEC,
        backend=FILTER_KW['tov_backend'], cache_path=str(_path))
    print(f"  {q_tag(_p)}: M_max^QS = {_Mmax:.3f} M_sun  ({_path.name})")

# %% [markdown]
# ## II.6 — $Q^*$ critical-droplet tables
#
# For every hadronic grid point, the composition of the quark droplet that would
# form there. This is the expensive intermediate that both the thermal and the
# quantum nucleation drivers consume.
#
# **A shortcut worth stating**: for the `lcn` / `gcn` charge modes the droplet
# composition does not depend on $\sigma$, and $R_c \propto \sigma$ exactly. So
# those are solved once and rescaled. For `coulomb_minimize`, $\sigma$ enters the
# self-consistent solve itself and each $\sigma$ must be solved separately.

# %%
def qstar_stem(stag, charge, phase, sigma):
    """Filename stem / dict key for one Q* table."""
    return f"Htrapped_saddlepoint_{charge}_{phase}_{stag}_s{int(sigma)}"


for _p in quark_param_sets:
    _stag = q_tag(_p)
    _pars = get_alphabag_custom(alpha=_p['alpha'], B4=_p['B4'], m_s=_p['m_s'])
    for _charge in CHARGE_MODES:
        _cn = 'local' if _charge == 'lcn' else 'global'
        _sigma_in_solve = (_charge == 'coulomb_minimize')
        for _phase, _D0 in (('unpaired', None), ('cfl', _p['Delta0'])):
            _stems = [qstar_stem(_stag, _charge, _phase, s) for s in SIGMA_LIST]
            if all((DIRS['Qstar'] / f'Qstar_{s}.dat').exists() for s in _stems):
                print(f"  [skip] {_stag} {_charge} {_phase}")
                continue
            if _sigma_in_solve:
                for _sg in SIGMA_LIST:
                    _t = compute_Qstar_table(
                        H_table['trapped'], flavor_mode='saddlepoint',
                        electric_charge_mode=_charge, params=_pars, sigma=_sg,
                        quark_phase=_phase, Delta0=_D0, verbose=False,
                        save_table=True,
                        output_file=str(DIRS['Qstar'] /
                                        f'Qstar_{qstar_stem(_stag, _charge, _phase, _sg)}.dat'))
                _n_ok = int(_t.data['converged'].sum())
                _n_tot = _t.data['converged'].size
            else:
                _base = compute_Qstar_table(
                    H_table['trapped'], flavor_mode='saddlepoint',
                    electric_charge_mode=_charge, params=_pars,
                    sigma=SIGMA_LIST[0], quark_phase=_phase, Delta0=_D0,
                    verbose=False, save_table=False)
                _n_ok = int(_base.data['converged'].sum())
                _n_tot = _base.data['converged'].size
                for _sg in SIGMA_LIST:
                    # R_c scales exactly linearly with sigma at fixed composition.
                    _data = {**_base.data,
                             'R_c': _base.data['R_c'] * (_sg / SIGMA_LIST[0])}
                    export_table(
                        QstarTableData(eq_type=_base.eq_type,
                                       hadronic_grids=_base.hadronic_grids,
                                       data=_data),
                        _pars,
                        output_file=str(DIRS['Qstar'] /
                                        f'Qstar_{qstar_stem(_stag, _charge, _phase, _sg)}.dat'),
                        charge_neutrality=_cn, sigma=_sg)
            print(f"  [{_stag}] {_charge:17s} {_phase:8s}: {_n_ok}/{_n_tot} converged")

# %% [markdown]
# ## II.7 — Thermal nucleation observables
#
# $(R_*, W_*, \Gamma, \tau)$ over the hadronic grid, for every method and
# surface tension. The `unpCFL` phase is built here too: its barrier is the
# kinked composite of the unpaired and CFL ones, and locating that composite's
# global peak is exactly what the driver does.

# %%
def nuc_stem(stag, charge, phase, sigma):
    return f"Htrapped_saddlepoint_{charge}_{phase}_{stag}_s{int(sigma)}"


# (charge, phase) combinations tabulated. unpCFL is only meaningful with the
# self-consistent Coulomb treatment, which is the paper's main method.
_NUC_MODES = ([(c, p) for c in CHARGE_MODES for p in QUARK_PHASES]
              + [('coulomb_minimize', 'unpCFL')])

for _p in quark_param_sets:
    _stag = q_tag(_p)
    _pars = get_alphabag_custom(alpha=_p['alpha'], B4=_p['B4'], m_s=_p['m_s'])
    for _charge, _phase in _NUC_MODES:
        for _sg in SIGMA_LIST:
            _path = DIRS['nucleation'] / f'nucleation_{nuc_stem(_stag, _charge, _phase, _sg)}.dat'
            if _path.exists():
                continue
            _kw = dict(sigma=_sg, params=_pars, V=V_NUC,
                       flavor_mode='saddlepoint', electric_charge_mode=_charge,
                       include_photons=include_photons, include_gluons=True,
                       include_thermal_neutrinos=include_thermal_neutrinos,
                       verbose=False, save_table=True, output_file=str(_path))
            if _phase == 'unpCFL':
                # Rx = the pairing coherence radius at the reference temperature.
                # Inside it the droplet is unpaired, outside it CFL.
                _Rx = float(crossover_radius(SCAN_SNAPSHOT['S'] * 10.0, _p['Delta0']))
                compute_thermal_nucleation_observables(
                    H_table['trapped'], quark_phase='unpCFL',
                    Delta0=_p['Delta0'], Rx=_Rx, switching_mode='step', **_kw)
            else:
                compute_thermal_nucleation_observables(
                    H_table['trapped'], quark_phase=_phase,
                    Delta0=(_p['Delta0'] if _phase == 'cfl' else None), **_kw)
    print(f"  [{_stag}] nucleation tables done "
          f"({len(_NUC_MODES)} modes x {len(SIGMA_LIST)} sigma)")

# %% [markdown]
# ## II.8 — $\sigma_{\rm crit}$ parameter-plane scans
#
# The central result. For every $(\alpha_s, B^{1/4}, \Delta_0)$ cell:
#
# 1. apply the acceptance filters — is this a viable strange star at all?
# 2. if so, find the surface tension at which the PNS nucleates within
#    $\tau_{\rm target}$.
#
# **Star-wide, not central.** $\sigma_{\rm crit}$ is the maximum over `N_SHELLS`
# shells from `NB_SHELL_MIN` to the centre, because the fastest-nucleating shell
# is not always at $r=0$. Every downstream number must use the same definition —
# mixing star-wide maps with centre-only table entries compares two different
# quantities.
#
# Results are cached as `.npz`; delete one to force a rescan.

# %%
# The star whose central conditions the scan is evaluated at.
star_scan = make_star_match(
    H, SCAN_SNAPSHOT['YLH'], SCAN_SNAPSHOT['S'],
    tov_betaeq_path=str(_tov_cold_path),
    tov_trapped_path=str(DIRS['H_tov'] /
                         f"tov_hadronic_trapped_2famphi_{xsd_tag}"
                         f"_YL{SCAN_SNAPSHOT['YLH']:.2f}_S{SCAN_SNAPSHOT['S']:.1f}.dat"))

SCAN_PATH_FMT = str(DIRS['sigma_crit'] /
                    'sigma_crit_grid_{xsd}_MT0{MT0:.2f}_{flavor}-{charge}-{phase}.npz')

# Which (flavour, charge, phase) planes to scan. The first is the paper's main
# method; the others support the method-comparison figures.
SCAN_CASES = [(MAIN_FLAVOR, MAIN_CHARGE, MAIN_PHASE)]
if not REDUCED_GRID:
    SCAN_CASES.append((MAIN_FLAVOR, MAIN_CHARGE, 'cfl'))

for _flav, _chg, _ph in SCAN_CASES:
    _todo = [m for m in MT0_GRID
             if not Path(SCAN_PATH_FMT.format(xsd=xsd_tag, MT0=m, flavor=_flav,
                                              charge=_chg, phase=_ph)).exists()]
    if not _todo:
        print(f"  [skip] {_flav}-{_chg}-{_ph}: all MT0 already scanned")
        continue
    print(f"\n-- sigma_crit scan: {_flav} / {_chg} / {_ph}  MT0={_todo} --")
    print(f"   grid: {len(SCAN['alpha'])} alpha x {len(SCAN['B4'])} B4 "
          f"x {len(SCAN['Delta0'])} Delta0, {N_SHELLS} shells")
    run_sigma_crit_scan(
        _todo, _flav, _chg, _ph,
        SCAN['alpha'], SCAN['B4'], SCAN['Delta0'],
        filt_cfg, nuc_cfg, star_scan,
        n_jobs=SCAN['n_jobs'], save_path_fmt=SCAN_PATH_FMT, xsd_tag=xsd_tag,
        n_shells=N_SHELLS, nB_shell_min=NB_SHELL_MIN)

print("\nPart II complete.")

# %% [markdown]
# # Part III — Load
#
# Rebuild every object Parts IV and V need, purely from what Part II wrote.
#
# **Run Parts I + III to work on figures.** Part II can be skipped entirely once
# its output exists, which is the point of the split: a figure tweak should cost
# seconds, not hours.

# %% [markdown]
# ## III.1 — Hadronic interpolators and TOV sequences

# %%
# H / H_table were built in II.4; rebuild here so Part III stands alone.
H_table = {}
H = {}
for _key, _eq, _, _stem in _H_JOBS:
    _tab = load_eos_table_sfho(str(DIRS['H_eos'] / f'{_stem}.dat'), _eq)
    H_table[_key] = _tab
    H[_key] = build_interpolators_sfho(_tab)

tov_cold = np.loadtxt(_tov_cold_path)
tov_trapped = load_tov_trapped(str(DIRS['H_tov']), xsd_tag)

print(f"hadronic tables : {list(H)}")
print(f"cold TOV        : M_max = {tov_cold[:, TOV_COL['M']].max():.3f} M_sun")
print(f"trapped TOV     : {len(tov_trapped)} (Y_L, S) sequences")
for _k in sorted(tov_trapped):
    print(f"   Y_L={_k[0]:.2f} S={_k[1]:.1f}  "
          f"M_max = {tov_trapped[_k][:, TOV_COL['M']].max():.3f}")

# %% [markdown]
# ## III.2 — Cold quark-star sequences
#
# One per quark parameter set: the remnant a converted star becomes.

# %%
qs_tov = {}
for _p in quark_param_sets:
    _path = DIRS['Q_tov'] / f'tov_quark_cfl_{q_tag(_p)}.dat'
    qs_tov[q_tag(_p)] = np.loadtxt(_path)
    print(f"  {q_tag(_p)}: M_max^QS = "
          f"{qs_tov[q_tag(_p)][:, TOV_COL['M']].max():.3f} M_sun")

# %% [markdown]
# ## III.3 — Nucleation observable tables
#
# Keyed by `(quark set, charge mode, phase, sigma)` so a figure asks for exactly
# the method it wants and fails loudly if Part II never produced it.

# %%
nuc_sets = {}
for _f in sorted(DIRS['nucleation'].glob('nucleation_*.dat')):
    nuc_sets[_f.stem.replace('nucleation_', '')] = load_thermal_nucleation_table(str(_f))


def nuc_table(pset, charge, phase, sigma):
    """The thermal observables for one method, or a clear error if absent."""
    stem = nuc_stem(q_tag(pset), charge, phase, sigma)
    if stem not in nuc_sets:
        raise KeyError(
            f"no nucleation table for {stem}.\n"
            f"Part II tabulates charge={CHARGE_MODES}, "
            f"phase={QUARK_PHASES + ['unpCFL']}, sigma={SIGMA_LIST}.")
    return nuc_sets[stem]


print(f"loaded {len(nuc_sets)} nucleation tables")
_n_conv = {k: int(v.converged.sum()) for k, v in nuc_sets.items()}
print(f"  converged points: min {min(_n_conv.values())}, max {max(_n_conv.values())} "
      f"of {next(iter(nuc_sets.values())).converged.size}")

# %% [markdown]
# ## III.4 — $Q^*$ tables and their interpolators
#
# Only loaded on demand: there are many, and most figures need one or two.

# %%
_qstar_cache = {}


def qstar(pset, charge, phase, sigma, interp=False):
    """Load (and cache) one Q* table; `interp=True` returns its interpolators."""
    key = (q_tag(pset), charge, phase, sigma)
    if key not in _qstar_cache:
        path = DIRS['Qstar'] / f'Qstar_{qstar_stem(*key)}.dat'
        if not path.exists():
            raise KeyError(f"no Q* table at {path}")
        tab = load_Qstar_table(str(path))
        _qstar_cache[key] = (tab, build_Qstar_interpolators(tab))
    return _qstar_cache[key][1 if interp else 0]


print(f"Q* tables available: {len(list(DIRS['Qstar'].glob('Qstar_*.dat')))}")

# %% [markdown]
# ## III.5 — $\sigma_{\rm crit}$ scans
#
# The `.npz` grids from II.8, plus the star match and evolution track the
# outcomes table needs.

# %%
def load_scan(MT0, flavor=MAIN_FLAVOR, charge=MAIN_CHARGE, phase=MAIN_PHASE):
    """One saved sigma_crit grid as a dict, or None if it was never scanned."""
    path = Path(SCAN_PATH_FMT.format(xsd=xsd_tag, MT0=MT0, flavor=flavor,
                                     charge=charge, phase=phase))
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        # .item() on the 0-d entries: ONE idiom, so a scalar never comes back as
        # a 0-d array that then formats as 'array(1.4)' in a title.
        return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}


SCAN_MT0 = MT0_GRID[0]
scan_main = load_scan(SCAN_MT0)
if scan_main is None:
    raise RuntimeError(
        f"no sigma_crit scan for MT0={SCAN_MT0}. Run Part II.8 first.")

SIG = scan_main['sig_crit']          # (alpha, Delta0, B4)
MMAX = scan_main['M_max']
REASON = scan_main['reason']
AL, B4G, D0G = scan_main['alpha_slices'], scan_main['B4_grid'], scan_main['Delta0_grid']
SIG_VMIN = float(np.nanmin(SIG))
SIG_VMAX = float(np.nanmax(SIG[np.isfinite(SIG)]))

print(f"sigma_crit scan MT0={SCAN_MT0}: {SIG.shape} "
      f"(alpha x Delta_0 x B^1/4)")
print(f"  viable cells   : {int(np.isfinite(SIG).sum())} / {SIG.size}")
print(f"  sigma_crit span: {SIG_VMIN:.1f} - {SIG_VMAX:.1f} MeV/fm^2")

# The star match + evolution track used by Part IV's outcomes table.
star_t0 = make_star_match(
    H, PNS_T0['YLH'], PNS_T0['S'], str(_tov_cold_path),
    str(DIRS['H_tov'] / f"tov_hadronic_trapped_2famphi_{xsd_tag}"
                        f"_YL{PNS_T0['YLH']:.2f}_S{PNS_T0['S']:.1f}.dat"))
star_tmax = make_star_match(
    H, PNS_TMAX['YLH'], PNS_TMAX['S'], str(_tov_cold_path),
    str(DIRS['H_tov'] / f"tov_hadronic_trapped_2famphi_{xsd_tag}"
                        f"_YL{PNS_TMAX['YLH']:.2f}_S{PNS_TMAX['S']:.1f}.dat"))

track = EvolutionTrack.build(
    H_trapped=H['trapped'], H_iso_trapped=H['iso_trapped'], tov_cold=tov_cold,
    tov_t0=nearest_trapped_sequence(tov_trapped, PNS_T0['YLH'], PNS_T0['S']),
    tov_tmax=nearest_trapped_sequence(tov_trapped, PNS_TMAX['YLH'], PNS_TMAX['S']),
    pns_t0=PNS_T0, pns_tmax=PNS_TMAX)

print(f"\nPNS evolution track:")
print(f"  t_0      Y_L={track.t0.Y_L:.2f} S={track.t0.S:.1f}  "
      f"M_max={track.t0.M_max:.3f}  M_B,max={track.t0.M_B_max:.3f}")
print(f"  t_Tmax   Y_L={track.t_max.Y_L:.2f} S={track.t_max.S:.1f}  "
      f"M_max={track.t_max.M_max:.3f}  M_B,max={track.t_max.M_B_max:.3f}")
print(f"  -> M_max falls by {track.t0.M_max - track.t_max.M_max:+.3f} M_sun "
      f"during deleptonization")
print("\nPart III complete: ready for figures.")
