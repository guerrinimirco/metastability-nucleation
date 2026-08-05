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
REDUCED_GRID = False

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
_dM = track.t0.M_max - track.t_max.M_max
print(f"  -> deleptonization changes M_max by {-_dM:+.3f} M_sun "
      f"({'falls' if _dM > 0 else 'rises'}): a star with M_B between the two "
      f"M_B,max values survives t_0 but collapses by t_Tmax")
print("\nPart III complete: ready for figures.")

# %% [markdown]
# # Part IV — Paper figures
#
# Everything here reads Part III and draws. No physics is computed except the
# outcomes table (IV.7), which is the paper's new result and cheap enough to
# evaluate inline.
#
# The publication style was set once in Part I; every figure goes through
# `paper_grid(**PAPER_STYLE)` and no cell touches `plt.rcParams`.

# %%
# --- shared drawing conventions ------------------------------------------------
# Phase line style: unpCFL is the physical composite (thick, opaque, on top);
# CFL and unpaired are its two limiting cases (thinner, equal weight).
PHASE_LABEL = {'unpaired': 'unpaired', 'cfl': 'CFL', 'unpCFL': 'unpCFL'}
PHASE_COLOUR = {'unpaired': OKAB['orange'], 'cfl': OKAB['sky'],
                'unpCFL': OKAB['blue']}
PHASE_ORDER = ['unpaired', 'cfl', 'unpCFL']

SET_A, SET_B = quark_param_sets[0], quark_param_sets[1]


def _save(fig, name, paper=True, data=None):
    """Save a figure (and optionally the arrays behind it) with one convention."""
    d = DIRS['fig_paper'] if paper else DIRS['fig_supp']
    path = d / f'{name}_{xsd_tag}.pdf'
    fig.savefig(path)
    print(f"  saved {path.relative_to(OUT)}")
    if data is not None:
        csv = DIRS['fig_data'] / f'{name}_{xsd_tag}.csv'
        data.to_csv(csv, index=False)
        print(f"  saved {csv.relative_to(OUT)}")
    return path


def _T_index(nuc_obs, T):
    """Index of the grid temperature nearest T."""
    return int(np.argmin(np.abs(nuc_obs.hadronic_grids['T'] - T)))


def _YL_index(nuc_obs, Y_L):
    return int(np.argmin(np.abs(nuc_obs.hadronic_grids['Y_L_H'] - Y_L)))

# %% [markdown]
# ## IV.1 — Figure 1: the nucleation barrier
#
# **(a)** The work of formation $W(R)$ at three densities, for each droplet
# phase. The peak is the barrier: its height sets the rate, its position the
# size of the critical fluctuation.
#
# **(b–d)** The three quantities that follow from that peak, against density and
# coloured by temperature: the critical radius $R_*$, the barrier in units of the
# temperature $W_*/T$ (the exponent that actually decides the rate), and the
# resulting nucleation time.
#
# The message: $W_*/T$ falls steeply with density, so nucleation is controlled by
# a narrow window near the stellar centre.

# %%
# TUNABLE
F1_SET = SET_A                      # which quark parameter set
F1_SIGMA = FIG_SIGMAS['fig1_barrier'][0]
F1_T = 30.0                         # MeV, for panel (a)
F1_DENS = [1.0, 4.0, 8.0]           # n_B / n_sat shown in panel (a)
F1_TEMPS = [10.0, 30.0, 50.0]       # MeV, the coloured curves in (b)-(d)
F1_YL = PNS_TMAX['YLH']
F1_R = np.linspace(0.05, 12.0, 300)  # fm, the R grid for panel (a)

fig, ((axA, axB), (axC, axD)) = paper_grid('2x2', mode='double',
                                           placeholder=False, **PAPER_STYLE)

# --- (a) W(R) --------------------------------------------------------------
_pars_f1 = get_alphabag_custom(alpha=F1_SET['alpha'], B4=F1_SET['B4'],
                               m_s=F1_SET['m_s'])
_dens_col = plt.get_cmap('viridis')(np.linspace(0.15, 0.85, len(F1_DENS)))
for _i, _nd in enumerate(F1_DENS):
    _nB = _nd * n_sat
    for _ph in ['unpaired', 'cfl']:
        _eb = compute_energy_barrier(
            H['trapped'], _nB, F1_T, F1_SIGMA, params=_pars_f1,
            Y_L_H=F1_YL, electric_charge_mode=MAIN_CHARGE, quark_phase=_ph,
            Delta0=(F1_SET['Delta0'] if _ph == 'cfl' else None),
            R_values=F1_R)
        axA.plot(_eb.R, _eb.W, color=_dens_col[_i],
                 lw=PHASE_LW[_ph], alpha=PHASE_ALPHA[_ph],
                 ls='-' if _ph == 'unpaired' else '--',
                 label=(f'${_nd:g}\\,n_0$' if _ph == 'unpaired' else None))
axA.axhline(0, color='0.7', lw=0.6, zorder=0)
axA.set_xlabel(r'$R$ [fm]')
axA.set_ylabel(r'$W(R)$ [MeV]')
axA.set_xlim(0, F1_R.max())
axA.legend(loc='upper right', title=f'$T={F1_T:g}$ MeV')
panel_label(axA, '(a)')

# --- (b)-(d) R*, W*/T, log10 tau vs density, coloured by T -----------------
_T_col = plt.get_cmap('plasma')(np.linspace(0.1, 0.75, len(F1_TEMPS)))
_f1_rows = []
for _ax, _key, _ylab, _logy in (
        (axB, 'R_c',  r'$R_*$ [fm]', False),
        (axC, 'W_T',  r'$W_*/T$', True),
        (axD, 'tau',  r'$\log_{10}(\tau\,/\,{\rm s})$', False)):
    for _ph in PHASE_ORDER:
        _obs = nuc_table(F1_SET, MAIN_CHARGE, _ph, F1_SIGMA)
        _iYL = _YL_index(_obs, F1_YL)
        _nB = _obs.hadronic_grids['n_B_H']
        for _k, _T in enumerate(F1_TEMPS):
            _iT = _T_index(_obs, _T)
            if _key == 'R_c':
                _y = _obs.R_c[:, _iYL, _iT]
            elif _key == 'W_T':
                _y = _obs.W_c[:, _iYL, _iT] / _obs.hadronic_grids['T'][_iT]
            else:
                with np.errstate(divide='ignore', invalid='ignore'):
                    _y = np.log10(_obs.tau[:, _iYL, _iT])
            _ok = _obs.converged[:, _iYL, _iT] & np.isfinite(_y)
            _ax.plot(_nB[_ok] / n_sat, _y[_ok], color=_T_col[_k],
                     lw=PHASE_LW[_ph], alpha=PHASE_ALPHA[_ph],
                     ls={'unpaired': ':', 'cfl': '--', 'unpCFL': '-'}[_ph],
                     label=(f'${_T:g}$ MeV' if _ph == 'unpCFL' else None))
            if _key == 'R_c' and _ph == 'unpCFL':
                _f1_rows.append(pd.DataFrame({
                    'n_B_over_n0': _nB[_ok] / n_sat, 'T_MeV': _T,
                    'R_star_fm': _y[_ok]}))
    _ax.set_xlabel(r'$n_B^H / n_0$')
    _ax.set_ylabel(_ylab)
    if _logy:
        _ax.set_yscale('log')
axD.axhline(np.log10(TAU_TARGET), color='0.4', lw=0.8, ls=':', zorder=1)
axB.legend(loc='upper right', title=r'$T$', ncol=1)
for _ax, _lab in ((axB, '(b)'), (axC, '(c)'), (axD, '(d)')):
    panel_label(_ax, _lab)

fig.suptitle(rf"$\sigma = {F1_SIGMA:g}$ MeV/fm$^2$,  $Y_L = {F1_YL:g}$,  "
             rf"{q_tag(F1_SET)}", fontsize=PAPER_STYLE['fontsize'] - 1)
_save(fig, 'paper_fig1_barrier',
      data=pd.concat(_f1_rows, ignore_index=True) if _f1_rows else None)
plt.show()

# %% [markdown]
# ## IV.2 — Figure 2: stellar sequences
#
# The two families side by side. **(a)** mass–radius against the observational
# constraints; **(b)** gravitational vs baryonic mass, which is the plane the
# conversion actually happens in, because baryon number — not mass — is
# conserved; **(c, d)** central temperature and density along each PNS sequence.
#
# The quark-star branch is systematically more compact. Where it lies *below* the
# neutron-star branch in (b), a converted star is lighter than the neutron star
# of the same baryon number, and that mass difference is the energy released.

# %%
# TUNABLE
F2_QS_SETS = quark_param_sets
F2_SEQS = [(PNS_T0, OKAB['vermillion']), (PNS_TMAX, OKAB['purple'])]
F2_QS_COLOURS = [OKAB['green'], OKAB['sky']]

fig, ((axA, axB), (axC, axD)) = paper_grid('2x2', mode='double',
                                           placeholder=False, **PAPER_STYLE)

# --- (a) mass-radius, over the observational constraints -------------------
add_observational_constraints(axA, show_mass_bands=True, inline_labels=False)
_cold_st = stable_branch(tov_cold)
axA.plot(_cold_st[:, TOV_COL['R']], _cold_st[:, TOV_COL['M']],
         color='k', lw=1.8, label='cold NS')
for _snap, _col in F2_SEQS:
    _seq = stable_branch(nearest_trapped_sequence(tov_trapped, _snap['YLH'], _snap['S']))
    axA.plot(_seq[:, TOV_COL['R']], _seq[:, TOV_COL['M']], color=_col, lw=1.5,
             ls='--', label=f"PNS {_snap['label']}")
for _k, _p in enumerate(F2_QS_SETS):
    _qs = stable_branch(qs_tov[q_tag(_p)])
    axA.plot(_qs[:, TOV_COL['R']], _qs[:, TOV_COL['M']],
             color=F2_QS_COLOURS[_k % len(F2_QS_COLOURS)], lw=1.8,
             label=rf"QS $B^{{1/4}}={_p['B4']:.0f}$")
axA.set_xlabel(r'$R$ [km]')
axA.set_ylabel(r'$M$ [$M_\odot$]')
axA.set_xlim(8, 16)
axA.set_ylim(0, 2.8)
axA.legend(loc='lower left', fontsize=PAPER_STYLE['legendsize'] - 1)
panel_label(axA, '(a)')

# --- (b) M vs M_B: conversion conserves M_B, not M ------------------------
axB.plot(_cold_st[:, TOV_COL['M_B']], _cold_st[:, TOV_COL['M']],
         color='k', lw=1.8, label='cold NS')
for _snap, _col in F2_SEQS:
    _seq = stable_branch(nearest_trapped_sequence(tov_trapped, _snap['YLH'], _snap['S']))
    axB.plot(_seq[:, TOV_COL['M_B']], _seq[:, TOV_COL['M']], color=_col,
             lw=1.5, ls='--')
    mass_marks(axB, _seq, _seq[:, TOV_COL['M']], _col, xcol='M_B', label=False)
for _k, _p in enumerate(F2_QS_SETS):
    _qs = stable_branch(qs_tov[q_tag(_p)])
    axB.plot(_qs[:, TOV_COL['M_B']], _qs[:, TOV_COL['M']],
             color=F2_QS_COLOURS[_k % len(F2_QS_COLOURS)], lw=1.8)
axB.set_xlabel(r'$M_B$ [$M_\odot$]')
axB.set_ylabel(r'$M$ [$M_\odot$]')
panel_label(axB, '(b)')

# --- (c, d) central conditions along each PNS sequence --------------------
for _snap, _col in F2_SEQS:
    _seq = stable_branch(nearest_trapped_sequence(tov_trapped, _snap['YLH'], _snap['S']))
    _nBc, _M = _seq[:, TOV_COL['n_Bc']], _seq[:, TOV_COL['M']]
    _Tc = np.array([float(H['iso_trapped']['T'](n, _snap['YLH'], _snap['S']))
                    for n in _nBc])
    axC.plot(_M, _Tc, color=_col, lw=1.6, label=_snap['label'])
    axD.plot(_M, _nBc / n_sat, color=_col, lw=1.6)
axC.set_xlabel(r'$M$ [$M_\odot$]'); axC.set_ylabel(r'$T_c$ [MeV]')
axC.legend(loc='upper left')
axD.set_xlabel(r'$M$ [$M_\odot$]'); axD.set_ylabel(r'$n_{B,c} / n_0$')
panel_label(axC, '(c)')
panel_label(axD, '(d)')

_save(fig, 'paper_fig2_stellar_sequences')
plt.show()

# %% [markdown]
# ## IV.3 — Figure 3: nucleation at the PNS centre
#
# The same quantities as Fig. 1, but now *along a stellar sequence*: every point
# is the centre of a PNS of that mass. This is what connects the microphysics to
# an observable — a heavier star is denser and hotter at its centre, so it
# nucleates more easily.
#
# Panel **(d)** is the headline: $\sigma_{\rm crit}(M_{\rm PNS})$. A star whose
# $\sigma$ lies below the curve converts.
#
# Note this panel uses the **centre-only** $\sigma_{\rm crit}$, because it is a
# statement about the centre. The parameter-plane maps (Fig. 5) use the
# **star-wide** definition. The two differ, and are labelled accordingly.

# %%
# TUNABLE
F3_SET = SET_A
F3_SIGMAS = FIG_SIGMAS['fig3_centre']
F3_SNAP = PNS_TMAX
F3_NM = 10 if REDUCED_GRID else 30       # points along the sequence

_f3_seq = stable_branch(
    nearest_trapped_sequence(tov_trapped, F3_SNAP['YLH'], F3_SNAP['S']))
_f3_M = np.linspace(_f3_seq[:, TOV_COL['M']].min() + 0.05,
                    _f3_seq[:, TOV_COL['M']].max() - 0.01, F3_NM)
_f3_nBc = branch_interp(_f3_seq, 'M', 'n_Bc')(_f3_M)
_f3_Tc = np.array([float(H['iso_trapped']['T'](n, F3_SNAP['YLH'], F3_SNAP['S']))
                   for n in _f3_nBc])
_pars_f3 = get_alphabag_custom(alpha=F3_SET['alpha'], B4=F3_SET['B4'],
                               m_s=F3_SET['m_s'])

fig, ((axA, axB), (axC, axD)) = paper_grid('2x2', mode='double',
                                           placeholder=False, **PAPER_STYLE)

_sig_col = plt.get_cmap('viridis')(np.linspace(0.15, 0.8, len(F3_SIGMAS)))
_f3_rows = []
for _s, _sg in enumerate(F3_SIGMAS):
    for _ph in PHASE_ORDER:
        _R = np.full(F3_NM, np.nan)
        _WT = np.full(F3_NM, np.nan)
        _tau = np.full(F3_NM, np.nan)
        _cache = {}
        for _i, (_nB, _T) in enumerate(zip(_f3_nBc, _f3_Tc)):
            if not (np.isfinite(_nB) and np.isfinite(_T)):
                continue
            _pt = nucleation_point(
                H['trapped'], _nB, _T, _sg, params=_pars_f3,
                Y_L_H=F3_SNAP['YLH'], quark_phase=_ph,
                Delta0=(F3_SET['Delta0'] if _ph != 'unpaired' else None),
                electric_charge_mode=MAIN_CHARGE, V=V_NUC, cache=_cache)
            _R[_i], _WT[_i], _tau[_i] = _pt.R_star, _pt.W_over_T, _pt.tau
        _kw = dict(color=_sig_col[_s], lw=PHASE_LW[_ph], alpha=PHASE_ALPHA[_ph],
                   ls={'unpaired': ':', 'cfl': '--', 'unpCFL': '-'}[_ph])
        axA.plot(_f3_M, _R, **_kw,
                 label=(rf'$\sigma={_sg:g}$' if _ph == 'unpCFL' else None))
        axB.plot(_f3_M, _WT, **_kw)
        with np.errstate(divide='ignore', invalid='ignore'):
            axC.plot(_f3_M, np.log10(_tau), **_kw)
        if _ph == 'unpCFL':
            with np.errstate(divide='ignore', invalid='ignore'):
                _f3_rows.append(pd.DataFrame({
                    'M_PNS': _f3_M, 'sigma': _sg, 'R_star_fm': _R,
                    'W_over_T': _WT, 'log10_tau_s': np.log10(_tau)}))

# --- (d) sigma_crit along the sequence (CENTRE-ONLY definition) ------------
_f3_sc = sigma_crit_along_sequence(
    _f3_nBc, _f3_Tc, Y_L_H=F3_SNAP['YLH'], H_trapped=H['trapped'],
    params=_pars_f3, Delta0=F3_SET['Delta0'], nuc=nuc_cfg, shells=None,
    flavor=MAIN_FLAVOR, charge=MAIN_CHARGE, phase=MAIN_PHASE)
axD.plot(_f3_M, _f3_sc, color=OKAB['blue'], lw=2.0)
for _s, _sg in enumerate(F3_SIGMAS):
    axD.axhline(_sg, color=_sig_col[_s], lw=0.9, ls=':')

axA.set_xlabel(r'$M_{\rm PNS}$ [$M_\odot$]'); axA.set_ylabel(r'$R_*$ [fm]')
axB.set_xlabel(r'$M_{\rm PNS}$ [$M_\odot$]'); axB.set_ylabel(r'$W_*/T$')
axB.set_yscale('log')
axC.set_xlabel(r'$M_{\rm PNS}$ [$M_\odot$]')
axC.set_ylabel(r'$\log_{10}(\tau\,/\,{\rm s})$')
axC.axhline(np.log10(TAU_TARGET), color='0.4', lw=0.8, ls=':')
axD.set_xlabel(r'$M_{\rm PNS}$ [$M_\odot$]')
axD.set_ylabel(r'$\sigma_{\rm crit}^{\rm centre}$ [MeV/fm$^2$]')
axA.legend(loc='best', fontsize=PAPER_STYLE['legendsize'] - 1)
for _ax, _lab in ((axA, '(a)'), (axB, '(b)'), (axC, '(c)'), (axD, '(d)')):
    panel_label(_ax, _lab)
fig.suptitle(rf"PNS at {F3_SNAP['label']} ($Y_L={F3_SNAP['YLH']:g}$, "
             rf"$S={F3_SNAP['S']:g}$),  {q_tag(F3_SET)}",
             fontsize=PAPER_STYLE['fontsize'] - 1)
_save(fig, 'paper_fig3_centre_vs_Mpns',
      data=pd.concat(_f3_rows, ignore_index=True) if _f3_rows else None)
plt.show()

# %% [markdown]
# ## IV.4 — Figure 4: nucleation conditions $T_{\rm nuc}(n_B^H)$
#
# The **nucleation condition** itself: the locus in the $(n_B, T)$ plane along
# which $\tau = \tau_{\rm target}$. Above it a droplet forms in time; below it
# the hadronic star survives.
#
# Overlaid is the PNS isentrope — the path an actual star's centre traverses. The
# figure is read by asking where the isentrope crosses the locus: that crossing
# is the stellar mass at which conversion becomes possible.
#
# Rows are quark parameter sets, columns are the two PNS snapshots. This uses the
# `NucleationCondition` API, so the same object serves the curve and the markers.

# %%
# TUNABLE
F4_SETS = quark_param_sets
F4_SNAPS = [PNS_T0, PNS_TMAX]
F4_SIGMAS = FIG_SIGMAS['fig4_Tnuc']

fig, axes = paper_grid('2x2', mode='double', placeholder=False, **PAPER_STYLE)
_f4_rows = []
for _r, _set in enumerate(F4_SETS):
    for _c, _snap in enumerate(F4_SNAPS):
        _ax = axes[_r, _c]
        for _s, _sg in enumerate(F4_SIGMAS):
            for _ph in PHASE_ORDER:
                _obs = nuc_table(_set, MAIN_CHARGE, _ph, _sg)
                # scan='n_B' (root-find along T at each density) is REQUIRED for
                # unpCFL: tau is non-monotonic in T there, because the CFL gap
                # melts as T rises, and scanning the other way picks the wrong
                # branch without complaining.
                _cond = NucleationCondition.from_table(
                    _obs, tau_target=TAU_TARGET, scan='n_B')
                _nB, _T = _cond.curve(Y_L=_snap['YLH'])
                if _nB.size == 0:
                    continue
                _ax.plot(_nB / n_sat, _T,
                         color=_sig_col[_s] if _s < len(_sig_col) else 'k',
                         lw=PHASE_LW[_ph], alpha=PHASE_ALPHA[_ph],
                         ls={'unpaired': ':', 'cfl': '--', 'unpCFL': '-'}[_ph],
                         label=(rf'$\sigma={_sg:g}$'
                                if (_ph == 'unpCFL' and _r == 0 and _c == 0)
                                else None))
                if _ph == MAIN_PHASE:
                    _f4_rows.append(pd.DataFrame({
                        'set': q_tag(_set), 'snapshot': _snap['label'],
                        'sigma': _sg, 'n_B_over_n0': _nB / n_sat, 'T_nuc_MeV': _T}))

        # --- the PNS isentrope: the path the star's centre actually takes ---
        _seq = stable_branch(
            nearest_trapped_sequence(tov_trapped, _snap['YLH'], _snap['S']))
        _nBc = _seq[:, TOV_COL['n_Bc']]
        _Tiso = np.array([float(H['iso_trapped']['T'](n, _snap['YLH'], _snap['S']))
                          for n in _nBc])
        _ax.plot(_nBc / n_sat, _Tiso, color='k', lw=1.4, ls='-.',
                 label='PNS isentrope' if (_r == 0 and _c == 0) else None)
        isentrope_mass_markers(
            _ax, _seq,
            lambda n, _s=_snap: float(H['iso_trapped']['T'](n, _s['YLH'], _s['S'])),
            'k', n_sat=n_sat)

        _ax.set_xlabel(r'$n_B^H / n_0$')
        _ax.set_ylabel(r'$T$ [MeV]')
        _ax.set_title(rf"{q_tag(_set)}  |  {_snap['label']}",
                      fontsize=PAPER_STYLE['legendsize'])
        panel_label(_ax, f"({'abcd'[_r * 2 + _c]})")
axes[0, 0].legend(loc='upper right', fontsize=PAPER_STYLE['legendsize'] - 1)
_save(fig, 'paper_fig4_Tnuc',
      data=pd.concat(_f4_rows, ignore_index=True) if _f4_rows else None)
plt.show()

# %% [markdown]
# ## IV.5 — Figure 5: $\sigma_{\rm crit}$ over the quark parameter plane
#
# **The central result.** For every $(B^{1/4}, \Delta_0)$ at fixed $\alpha_s$:
# the surface tension at which a PNS nucleates within $\tau_{\rm target}$.
#
# Layers, in order:
#
# * **heatmap** — $\sigma_{\rm crit}$ (star-wide, max over shells);
# * **white contours** — iso-$\sigma_{\rm crit}$, to read values off;
# * **grey dashed** — iso-$M_{\max}$ of the resulting quark star;
# * **coloured outlines** — the regions each acceptance filter excludes, so the
#   reader can see *why* a parameter choice is unavailable, not just that it is.
#
# Grey cells are not viable at all. A star converts if its $\sigma$ lies below the
# local value.

# %%
# TUNABLE
F5_ALPHA_SHOW = list(range(len(AL)))          # which alpha_s slices to draw
F5_ISO_SIGMA = [50, 100, 150, 200, 250]       # white contour levels
F5_ISO_MMAX = [2.0, 2.2, 2.4, 2.6]            # grey dashed contour levels
F5_SHOW_REJECT = True

_n_panels = min(len(F5_ALPHA_SHOW), 2)
fig, axes = paper_grid('1x2' if _n_panels == 2 else '1x2', mode='double',
                       placeholder=False, aspect=1.0,
                       **{k: v for k, v in PAPER_STYLE.items() if k != 'aspect'})
_axl = np.atleast_1d(axes).ravel()

for _k, _ia in enumerate(F5_ALPHA_SHOW[:len(_axl)]):
    _ax = _axl[_k]
    _pcm = sigma_map(_ax, B4G, D0G, SIG[_ia], vmin=SIG_VMIN, vmax=SIG_VMAX)
    iso_lines(_ax, B4G, D0G, SIG[_ia], F5_ISO_SIGMA, ISO_SIGMA_COLOUR,
              fmt='%.0f', linewidths=0.9)
    iso_lines(_ax, B4G, D0G, MMAX[_ia], F5_ISO_MMAX, ISO_MASS_COLOUR,
              fmt='%.1f', linewidths=0.9, linestyles='--')
    if F5_SHOW_REJECT:
        reject_outlines(_ax, B4G, D0G, REASON[_ia], panel_index=_k)
    _ax.set_xlabel(r'$B^{1/4}$ [MeV]')
    _ax.set_ylabel(r'$\Delta_0$ [MeV]')
    _ax.set_title(rf'$\alpha_s = {AL[_ia]:.2f}$',
                  fontsize=PAPER_STYLE['legendsize'])
    panel_label(_ax, f"({'ab'[_k]})")

# One colorbar for the whole figure: sigma_crit means the same thing in
# every panel, so a per-panel bar would invite the wrong comparison.
_cb = fig.colorbar(_pcm, ax=list(_axl), fraction=0.046, pad=0.02)
_cb.set_label(r'$\sigma_{\rm crit}^{\rm star}$ [MeV/fm$^2$]')

# Flat CSV of the grid, so the map can be re-read without the .npz.
_ii, _jj = np.meshgrid(np.arange(len(D0G)), np.arange(len(B4G)), indexing='ij')
_f5 = pd.DataFrame({
    'alpha_s': np.repeat(AL[F5_ALPHA_SHOW[0]], _ii.size),
    'B4_MeV': B4G[_jj].ravel(), 'Delta0_MeV': D0G[_ii].ravel(),
    'sigma_crit_star': SIG[F5_ALPHA_SHOW[0]].ravel(),
    'M_max_Msun': MMAX[F5_ALPHA_SHOW[0]].ravel(),
    'reject_reason_code': REASON[F5_ALPHA_SHOW[0]].ravel()})
_save(fig, 'paper_fig5_sigmacrit_map', data=_f5)
plt.show()

# %% [markdown]
# ## IV.6 — Appendices: method dependence
#
# **A — electric-charge prescription.** How the droplet's charge is treated
# changes the barrier. `lcn` forces local neutrality (no Coulomb cost, but an
# unphysically constrained composition); `gcn` allows global neutrality;
# `coulomb_minimize` solves the droplet radius and composition self-consistently
# *with* the Coulomb energy, and is what the paper uses.
#
# **B — flavour treatment.** `frozen` keeps the hadronic flavour composition in
# the droplet (weak interactions too slow to act during the fluctuation);
# `saddlepoint` lets the composition relax to minimise the barrier. The truth is
# between them, but the saddle point is the conservative choice: it gives the
# *lowest* barrier, hence the most optimistic nucleation.

# %%
fig, ((axA, axB),) = paper_grid('1x2', mode='double', placeholder=False,
                                **PAPER_STYLE)
_appA_set = SET_A
_appA_sigma = FIG_SIGMAS['appA_charge'][0]
_appA_charges = [('lcn', OKAB['orange']), ('gcn', OKAB['green']),
                 ('coulomb_minimize', OKAB['blue'])]

# --- (a) T_nuc under each charge prescription -----------------------------
for _chg, _col in _appA_charges:
    _obs = nuc_table(_appA_set, _chg, 'unpaired', _appA_sigma)
    _cond = NucleationCondition.from_table(_obs, tau_target=TAU_TARGET, scan='n_B')
    _nB, _T = _cond.curve(Y_L=PNS_TMAX['YLH'])
    if _nB.size:
        axA.plot(_nB / n_sat, _T, color=_col, lw=1.7, label=_chg)
axA.set_xlabel(r'$n_B^H / n_0$')
axA.set_ylabel(r'$T_{\rm nuc}$ [MeV]')
axA.legend(loc='best', title='charge mode',
           fontsize=PAPER_STYLE['legendsize'] - 1)
panel_label(axA, '(a)')

# --- (b) frozen vs saddle-point flavour -----------------------------------
_appB_sigma = FIG_SIGMAS['appB_flavour'][0]
_pars_app = get_alphabag_custom(alpha=_appA_set['alpha'], B4=_appA_set['B4'],
                                m_s=_appA_set['m_s'])
_nB_app = np.linspace(0.3, 1.2, 25 if REDUCED_GRID else 60)
_T_app = 30.0
for _flav, _col in (('frozen', OKAB['vermillion']),
                    ('saddlepoint', OKAB['blue'])):
    _WT = np.full(_nB_app.size, np.nan)
    _cache = {}
    for _i, _n in enumerate(_nB_app):
        _pt = nucleation_point(
            H['trapped'], _n, _T_app, _appB_sigma, params=_pars_app,
            Y_L_H=PNS_TMAX['YLH'], flavor_mode=_flav,
            electric_charge_mode='gcn',   # frozen supports lcn/gcn only
            quark_phase='unpaired', V=V_NUC, cache=_cache)
        _WT[_i] = _pt.W_over_T
    axB.plot(_nB_app / n_sat, _WT, color=_col, lw=1.7, label=_flav)
axB.set_yscale('log')
axB.set_xlabel(r'$n_B^H / n_0$')
axB.set_ylabel(r'$W_*/T$')
axB.legend(loc='best', title=f'flavour, $T={_T_app:g}$ MeV',
           fontsize=PAPER_STYLE['legendsize'] - 1)
panel_label(axB, '(b)')

fig.suptitle(rf"Method dependence  ({q_tag(_appA_set)})",
             fontsize=PAPER_STYLE['fontsize'] - 1)
_save(fig, 'paper_appendix_methods')
plt.show()

# %% [markdown]
# ## IV.7 — Outcomes table: neutron star, quark star, or black hole
#
# **The paper's new result.** Everything above asks *whether a droplet forms*.
# This asks what is left at the end.
#
# One PNS of fixed baryon number is followed from $t_0$ to $t_{T_{\max}}$ and
# asked the nucleation question at each snapshot. Three ways out:
#
# * **QS** — it nucleates. The remnant mass follows from **baryon** conservation,
#   and the gravitational-mass deficit $(M_{\rm PNS}-M_{\rm QS})c^2$ is released.
# * **BH** — either born above the $t_0$ maximum mass, or deleptonization drops
#   $M_{\max}$ below its baryon mass before it can convert.
# * **NS** — it survives both, and cools into an ordinary neutron star.
#
# The $\sigma_{\rm crit}$ at each snapshot is reported alongside the verdict,
# because it *is* the decision: the outcome column alone hides how close the call
# was.

# %%
# TUNABLE
TABLE_SETS = quark_param_sets
TABLE_SIGMAS = FIG_SIGMAS['outcomes_table']
TABLE_MPNS = ([1.3, 1.5] if REDUCED_GRID else [1.2, 1.4, 1.6, 1.8])
# shells=None -> the CENTRE-ONLY sigma_crit, stated explicitly rather than
# defaulted, because it is a different quantity from the star-wide map in Fig 5.
TABLE_SHELLS = None

_rows = []
for _p in TABLE_SETS:
    _pars = get_alphabag_custom(alpha=_p['alpha'], B4=_p['B4'], m_s=_p['m_s'])
    # M_QS(M_B): conversion conserves baryon number, so the remnant is read off
    # the cold quark-star sequence at fixed M_B.
    _M_QS = branch_interp(qs_tov[q_tag(_p)], 'M_B', 'M')
    for _sg in TABLE_SIGMAS:
        for _M0 in TABLE_MPNS:
            _row, _sc0, _scT = outcome_row(
                track, _p, _pars, _p['Delta0'], _sg, _M0,
                nuc=nuc_cfg, shells=TABLE_SHELLS, M_QS_of_Mb=_M_QS,
                n_sat=n_sat, flavor=MAIN_FLAVOR, charge=MAIN_CHARGE,
                phase=MAIN_PHASE, label=q_tag(_p))
            _row['sigma_crit_centre_t0'] = _sc0
            _row['sigma_crit_centre_tT'] = _scT
            _rows.append(_row)

outcomes = pd.DataFrame(_rows)
_show = ['params', 'sigma', 'M_t0', 'Mb_t0', 'nBc_t0', 'YSc_t0',
         'sigma_crit_centre_t0', 'M_tT', 'nBc_tT', 'sigma_crit_centre_tT',
         'type', 'M_rem', 'E_conv_e53']
print(outcomes[_show].to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
print(f"\noutcome counts: {outcomes['type'].value_counts().to_dict()}")

outcomes.to_csv(DIRS['fig_data'] / f'table_outcomes_{xsd_tag}.csv', index=False)
(DIRS['fig_data'] / f'table_outcomes_{xsd_tag}.tex').write_text(
    to_latex_tabular(outcomes[_show]))
print(f"  saved table_outcomes_{xsd_tag}.csv / .tex")

# %% [markdown]
# # Part V — Supplementary material
#
# Same style and same data as Part IV; these support the paper's claims without
# being headline figures.
#
# | Section | Question it answers |
# |---------|---------------------|
# | V.1 | What do the *viable* parameter sets look like as stars? |
# | V.2 | How sensitive is $\sigma_{\rm crit}$ to the choices we made? |
# | V.3 | Is $W_*/T$ an invariant across the viable region? |
# | V.4 | What happens without CFL pairing? |

# %% [markdown]
# ## V.1 — The viable region as stars: M–R and $P(\mu_B)$ coloured by $\sigma_{\rm crit}$
#
# Fig. 5 shows $\sigma_{\rm crit}$ over an abstract parameter plane. This shows
# the same information as *observable* stars: every accepted cell is replayed
# into a mass–radius curve and an equation of state, then binned by
# $\sigma_{\rm crit}$ and drawn as a median with a p16–p84 envelope.
#
# Reading it: if the bands separate, then $\sigma_{\rm crit}$ is correlated with
# something measurable and a radius measurement constrains the surface tension.
# If they overlap, it is not.

# %%
# TUNABLE
V1_NBINS = 3
V1_MAX_CURVES = 60 if REDUCED_GRID else 400

_curves = replay_accepted(SIG, AL, B4G, D0G, filt_cfg,
                          max_curves=V1_MAX_CURVES,
                          n_jobs=SCAN['n_jobs'], verbose=True)
print(f"replayed {len(_curves)} accepted cells")

if len(_curves) >= 4:
    fig, ((axA, axB), (axC, axD)) = paper_grid(
        '2x2', mode='double', placeholder=False, **PAPER_STYLE)
    _norm = Normalize(SIG_VMIN, SIG_VMAX)
    _cmap = plt.get_cmap('viridis')
    _bins = quantile_bins(_curves, V1_NBINS, _cmap, _norm)

    # (a) M-R: interpolate R at fixed M, but plot R horizontally -> swap=True
    _Mg = np.linspace(0.6, 2.6, 60)
    for _lo, _hi, _members, _col in _bins:
        _prof = resample_profiles(_members, 'M', 'R', _Mg, stable=True)
        band(axA, _prof, _Mg, _col, label=rf'{_lo:.0f}-{_hi:.0f}', swap=True)
    add_observational_constraints(axA, show_mass_bands=False, inline_labels=False)
    axA.set_xlabel(r'$R$ [km]'); axA.set_ylabel(r'$M$ [$M_\odot$]')
    axA.set_xlim(8, 14); axA.set_ylim(0.5, 2.8)
    axA.legend(loc='lower left', title=r'$\sigma_{\rm crit}$',
               fontsize=PAPER_STYLE['legendsize'] - 1)
    panel_label(axA, '(a)')

    # (b) the equation of state behind those curves
    _mug = np.linspace(900, 1600, 80)
    for _lo, _hi, _members, _col in _bins:
        _prof = resample_profiles(_members, 'mu', 'P', _mug)
        band(axB, _prof, _mug, _col)
    axB.set_xlabel(r'$\mu_B$ [MeV]')
    axB.set_ylabel(r'$P$ [MeV fm$^{-3}$]')
    panel_label(axB, '(b)')

    # (c) each cell as one point: (M_max, R_1.4), coloured by sigma_crit
    _pts = summary_points(_curves, 'R_1.4')
    _sc = np.array([s for _, s in _curves])
    _sm = axC.scatter(_pts[:, 1], _pts[:, 0], c=_sc, cmap=_cmap, norm=_norm,
                      s=14, lw=0.3, edgecolor='white')
    axC.set_xlabel(r'$R_{1.4}$ [km]')
    axC.set_ylabel(r'$M_{\max}$ [$M_\odot$]')
    panel_label(axC, '(c)')

    # (d) the distribution of sigma_crit over the accepted cells
    axD.hist(_sc[np.isfinite(_sc)], bins=20, color=OKAB['blue'], alpha=0.8)
    axD.set_xlabel(r'$\sigma_{\rm crit}^{\rm star}$ [MeV/fm$^2$]')
    axD.set_ylabel('accepted cells')
    panel_label(axD, '(d)')

    fig.colorbar(_sm, ax=[axC], fraction=0.046, pad=0.02,
                 label=r'$\sigma_{\rm crit}$ [MeV/fm$^2$]')
    _save(fig, 'supp_viable_region_stars', paper=False,
          data=pd.DataFrame({'M_max': _pts[:, 0], 'R_1.4': _pts[:, 1],
                             'sigma_crit_star': _sc}))
    plt.show()
else:
    print("too few accepted cells to draw the bundle "
          "(expected on the smoke grid); rerun with REDUCED_GRID=False")

# %% [markdown]
# ## V.2 — Sensitivity of $\sigma_{\rm crit}$
#
# $\sigma_{\rm crit}$ is a threshold, so it is only useful if it is robust
# against the choices behind it. The two that matter most:
#
# * **stellar mass** $M_{T_0}$ — a heavier star is denser and hotter, so it
#   nucleates more easily and tolerates a *larger* $\sigma$;
# * **droplet phase** — a CFL droplet has a different composition and surface
#   than the physical unpaired-core composite.
#
# A difference map is drawn on a diverging scale symmetric about zero, so white
# means "no change" everywhere; grey means a cell viable in one scan but not the
# other.

# %%
_avail = {m: load_scan(m) for m in MT0_GRID}
_avail = {m: g for m, g in _avail.items() if g is not None}

if len(_avail) >= 2:
    _ms = sorted(_avail)
    _a, _b = _avail[_ms[0]], _avail[_ms[-1]]
    _D = _b['sig_crit'] - _a['sig_crit']
    _vlim = symmetric_vlim([_D])
    fig, ((axA, axB),) = paper_grid('1x2', mode='double', placeholder=False,
                                    aspect=1.0,
                                    **{k: v for k, v in PAPER_STYLE.items()
                                       if k != 'aspect'})
    _pcm = diverging_map(axA, B4G, D0G, _D[0], _vlim)
    axA.set_xlabel(r'$B^{1/4}$ [MeV]'); axA.set_ylabel(r'$\Delta_0$ [MeV]')
    axA.set_title(rf'$\Delta\sigma_{{\rm crit}}$: '
                  rf'$M_{{T_0}}={_ms[-1]:.2f}$ vs ${_ms[0]:.2f}$',
                  fontsize=PAPER_STYLE['legendsize'])
    fig.colorbar(_pcm, ax=[axA], fraction=0.046, pad=0.02)
    _fin = _D[np.isfinite(_D)]
    axB.hist(_fin, bins=25, color=OKAB['purple'], alpha=0.85)
    axB.set_xlabel(r'$\Delta\sigma_{\rm crit}$ [MeV/fm$^2$]')
    axB.set_ylabel('cells')
    panel_label(axA, '(a)'); panel_label(axB, '(b)')
    print(f"Delta sigma_crit over {_fin.size} cells: "
          f"median {np.median(_fin):+.2f}, "
          f"p16-p84 [{np.percentile(_fin, 16):+.2f}, {np.percentile(_fin, 84):+.2f}], "
          f"max |Delta| {np.abs(_fin).max():.2f} MeV/fm^2")
    _save(fig, 'supp_dsigma_MT0', paper=False)
    plt.show()
else:
    print(f"sensitivity map needs >=2 scanned M_T0 values; MT0_GRID={MT0_GRID} "
          f"gave {len(_avail)}. Set MT0_GRID = [1.17, 1.4, 1.6] and re-run "
          f"Part II.8 (this is why the map is skipped, not a silent no-op).")

# %% [markdown]
# ## V.3 — Is $W_*/T$ an invariant?
#
# If the barrier-to-temperature ratio at threshold were the same everywhere in
# the viable region, then $\sigma_{\rm crit}$ would be capturing essentially all
# of the parameter dependence — which is what justifies quoting it as a single
# number per parameter set.
#
# $W_*/T$ is read at the **deciding shell**: the one that nucleates fastest, not
# the centre, since that is the shell the star-wide threshold is set by.

# %%
# TUNABLE -- expensive: several droplet solves per cell.
V3_SLICES = [0]

_shells_v3 = star_shell_states(SCAN_MT0, star_scan, N_SHELLS,
                               nB_min=NB_SHELL_MIN)
_WT_path = DIRS['sigma_crit'] / f'WoverT_{xsd_tag}_MT0{SCAN_MT0:.2f}.npz'
if _WT_path.exists():
    with np.load(_WT_path) as _z:
        _WT, _Tdec = _z['W_over_T'], _z['T_dec']
    print(f"loaded cached {_WT_path.name}")
else:
    _WT, _Tdec, _kdec = barrier_ratio_map(
        SIG, AL, B4G, D0G, shells=_shells_v3, nuc=nuc_cfg, m_s=SCAN['m_s'],
        flavor=MAIN_FLAVOR, charge=MAIN_CHARGE, phase=MAIN_PHASE,
        slices=V3_SLICES, n_jobs=SCAN['n_jobs'])
    np.savez(_WT_path, W_over_T=_WT, T_dec=_Tdec, shell_index=_kdec)
    print(f"  saved {_WT_path.name}")

fig, ((axA, axB),) = paper_grid('1x2', mode='double', placeholder=False,
                                aspect=1.0,
                                **{k: v for k, v in PAPER_STYLE.items()
                                   if k != 'aspect'})
_ia = V3_SLICES[0]
_pcm = sigma_map(axA, B4G, D0G, _WT[_ia], cmap='magma')
axA.set_xlabel(r'$B^{1/4}$ [MeV]'); axA.set_ylabel(r'$\Delta_0$ [MeV]')
axA.set_title(rf'$W_*/T$ at threshold, $\alpha_s={AL[_ia]:.2f}$',
              fontsize=PAPER_STYLE['legendsize'])
fig.colorbar(_pcm, ax=[axA], fraction=0.046, pad=0.02)
_wt_fin = _WT[np.isfinite(_WT)]
if _wt_fin.size:
    axB.hist(_wt_fin, bins=25, color=OKAB['vermillion'], alpha=0.85)
    axB.axvline(np.median(_wt_fin), color='k', lw=1.2, ls='--')
    print(f"W*/T over {_wt_fin.size} cells: median {np.median(_wt_fin):.1f}, "
          f"p16-p84 [{np.percentile(_wt_fin, 16):.1f}, "
          f"{np.percentile(_wt_fin, 84):.1f}]  "
          f"-> spread/median = "
          f"{(np.percentile(_wt_fin, 84) - np.percentile(_wt_fin, 16)) / np.median(_wt_fin):.2f}")
axB.set_xlabel(r'$W_*/T$'); axB.set_ylabel('cells')
panel_label(axA, '(a)'); panel_label(axB, '(b)')
_save(fig, 'supp_WoverT_map', paper=False,
      data=pd.DataFrame({'W_over_T': _wt_fin}))
plt.show()

# %% [markdown]
# ## V.4 — Unpaired matter: $\sigma_{\rm crit}$ over $(B^{1/4}, \alpha_s)$
#
# The control case. With no CFL pairing there is no $\Delta_0$ axis, so the
# natural plane is $(B^{1/4}, \alpha_s)$. Comparing this against Fig. 5 isolates
# how much of the answer is due to pairing rather than to the bulk equation of
# state.

# %%
# TUNABLE
V4_ALPHA = np.linspace(0.0, 0.4 * np.pi / 2, 6 if REDUCED_GRID else 21)
V4_B4 = np.linspace(130., 180., 11 if REDUCED_GRID else 41)

_unp_path = DIRS['sigma_crit'] / f'sigma_crit_unpaired_{xsd_tag}_MT0{SCAN_MT0:.2f}.npz'
if _unp_path.exists():
    with np.load(_unp_path) as _z:
        _u_sig, _u_reason, _u_al, _u_b4 = (_z['sig_crit'], _z['reason'],
                                           _z['alpha_slices'], _z['B4_grid'])
    print(f"loaded cached {_unp_path.name}")
else:
    _u_ok, _u_mm, _u_reason = scan_unpaired_filters(
        V4_ALPHA, V4_B4, filt_cfg, n_jobs=SCAN['n_jobs'])
    _u_sig = compute_sigma_crit(
        _u_ok, SCAN_MT0, MAIN_FLAVOR, MAIN_CHARGE, 'unpaired',
        V4_ALPHA, V4_B4, np.array([0.0]), star_scan, nuc_cfg,
        m_s=SCAN['m_s'], n_jobs=SCAN['n_jobs'],
        n_shells=N_SHELLS, nB_shell_min=NB_SHELL_MIN)
    _u_al, _u_b4 = V4_ALPHA, V4_B4
    np.savez(_unp_path, sig_crit=_u_sig, reason=_u_reason, M_max=_u_mm,
             alpha_slices=_u_al, B4_grid=_u_b4, MT0=SCAN_MT0)
    print(f"  saved {_unp_path.name}")

# The Delta_0 axis is a singleton for unpaired matter -> squeeze it out and put
# alpha_s on the y-axis instead.
_u_field = _u_sig[:, 0, :].T            # (B4, alpha)
_u_reason2 = _u_reason[:, 0, :].T

fig, ((axA, axB),) = paper_grid('1x2', mode='double', placeholder=False,
                                aspect=1.0,
                                **{k: v for k, v in PAPER_STYLE.items()
                                   if k != 'aspect'})
_pcm = sigma_map(axA, _u_al, _u_b4, _u_field)
reject_outlines(axA, _u_al, _u_b4, _u_reason2, labels=False)
axA.set_xlabel(r'$\alpha_s$'); axA.set_ylabel(r'$B^{1/4}$ [MeV]')
axA.set_title(rf'unpaired, $m_s={SCAN["m_s"]:.0f}$ MeV',
              fontsize=PAPER_STYLE['legendsize'])
fig.colorbar(_pcm, ax=[axA], fraction=0.046, pad=0.02,
             label=r'$\sigma_{\rm crit}$ [MeV/fm$^2$]')

# The absolute-stability window: 3-flavour matter bound, 2-flavour not.
_stab3 = stability_curve(
    lambda a, b: energy_per_baryon_at_P0(a, b, filt_cfg),
    _u_al, _u_b4.min(), _u_b4.max())
_stab2 = stability_curve(
    lambda a, b: two_flavour_energy_at_P0(a, b, filt_cfg),
    _u_al, _u_b4.min(), _u_b4.max())
axB.plot(_u_al, resample_curve(_stab3, _u_al, _u_al, np.nan),
         color=OKAB['green'], lw=1.8, label='3-flavour bound')
axB.plot(_u_al, resample_curve(_stab2, _u_al, _u_al, np.nan),
         color=OKAB['grey'], lw=1.8, ls='--', label='2-flavour bound')
axB.set_xlabel(r'$\alpha_s$'); axB.set_ylabel(r'$B^{1/4}$ [MeV]')
axB.set_ylim(_u_b4.min(), _u_b4.max())
axB.legend(loc='best', fontsize=PAPER_STYLE['legendsize'] - 1)
panel_label(axA, '(a)'); panel_label(axB, '(b)')
_save(fig, 'supp_unpaired_sigmacrit', paper=False)
plt.show()

print("\n" + "=" * 78)
print("Notebook complete.")
print(f"  figures     -> {DIRS['fig_paper'].relative_to(OUT)} / "
      f"{DIRS['fig_supp'].relative_to(OUT)}")
print(f"  figure data -> {DIRS['fig_data'].relative_to(OUT)}")
if REDUCED_GRID:
    print("  NOTE: smoke-test output. Set REDUCED_GRID=False for paper values.")
print("=" * 78)
