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
from pathlib import Path
from dataclasses import replace as dc_replace

# --- third party --------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator, FuncFormatter
from scipy.interpolate import interp1d
from scipy.stats import spearmanr

# --- eos: equation of state, TOV structure, publication figure style -----------
# Both model packages name their grid driver TableSettings/compute_table, so the
# alphaBag pair is aliased -- two settings dataclasses cannot share one name, and
# the call sites should say which phase they build. EOSTable_for_TOV comes from
# eos.general, not from astro: it is the contract surface both layers may import.
from eos.sfho.nmp import create_custom_parametrization
from eos.sfho.table import (
    TableSettings, compute_table,
    load_eos_table as load_eos_table_sfho,
    build_interpolators as build_interpolators_sfho,
)
from eos.alphabag.table import (TableSettings as AlphaBagTableSettings,
                                compute_table as compute_alphabag_table)
from eos.alphabag.thermodynamics import T_critical
from eos.general.state import EOSTable_for_TOV
from eos.astro.tov.solver import (generate_ec_logspace,
                                  compute_tov_sequence, truncate_to_stable_branch)

# --- nucleation: the core engine ----------------------------------------------
# Aliased: `custom_params` is also the name of an sfho TableSettings field
# used a few cells below, and one notebook namespace cannot carry both.
from nucleation.quark import custom_params as quark_params
from nucleation import (
    work_of_formation, effective_inertia, quantum_nucleation_time,
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
    # configuration + the star match the sigma_crit engine is bound to
    FilterConfig, NucConfig, make_star_match, build_PH_of_muB,
    central_state, star_shell_states, critical_droplet_pt, sigma_target_pt,
    # acceptance filters and the parameter-plane scans
    replay_cfl, replay_accepted, scan_unpaired_filters, compute_sigma_crit,
    run_sigma_crit_scan, REASON_CODE,
    # TOV sequence bookkeeping
    TOV_COL, stable_branch, branch_interp, load_tov_trapped,
    nearest_trapped_sequence, cold_quark_star_branch,
    # absolute-stability boundaries at P = 0
    energy_per_baryon_at_P0, two_flavour_energy_at_P0, B4_at_energy,
    stability_curve, resample_curve,
    # droplet observables over the plane / along a stellar sequence
    barrier_ratio_map, sigma_crit_along_sequence,
    # the PNS-evolution outcome engine
    EvolutionTrack, outcome_row, to_latex_tabular,
)
from nucleation.analysis.figure import (
    set_paper_style, paper_grid, panel_label,
    STANDARD_COLORS, OKAB, OKAB_CAT, PHASE_LW, PHASE_ALPHA,
    add_observational_constraints,
    mass_marks, label_with_arrows,
    diverging_map, symmetric_vlim, reject_outlines,
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

# Central energy densities for the HADRONIC TOV sequences [MeV/fm^3].
E_C_VEC = generate_ec_logspace(e_min=150, e_max=2500,
                               n_points=25 if REDUCED_GRID else 100)

# Central energy densities for the QUARK-star TOV sequences [MeV/fm^3]. A quark
# star is self-bound and much more compact, so its sequence lives at lower e_c
# and does not need the hadronic grid's reach. This is the grid the published
# M_max filter and every quark M-R curve were computed on -- changing it moves
# M_max, hence which cells the sigma_crit scan accepts, hence Fig. 5.
E_C_VEC_QUARK = generate_ec_logspace(e_min=100, e_max=2000,
                                     n_points=25 if REDUCED_GRID else 80)

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
# All six are needed: Fig. 4 draws {50, 100, 150} for Set A and {150, 200, 250}
# for Set B, Fig. 3 uses {80, 100, 150}, Appendix A {100, 150}. Dropping one does
# not raise -- the curve simply vanishes from the figure -- which is why I.8
# cross-checks this list against what the figures ask for.
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
# 'color' is part of the figure contract, not decoration: orange = t_0 and red =
# t_Tmax in EVERY panel that shows both (Figs 2 and 4), so a reader learns the
# pairing once. One source of truth -- change here, not in a figure cell.
PNS_T0   = dict(YLH=0.35, S=1.5, color='#fd8d3c', label=r'$t_0$')
PNS_TMAX = dict(YLH=0.25, S=2.0, color='#e31a1c',
                label=r'$t_{T_\mathrm{max}}$')

# The snapshot the sigma_crit maps are evaluated at (the hotter, more permissive
# one -- if it does not nucleate there, it does not nucleate).
SCAN_SNAPSHOT = PNS_TMAX

# Gravitational masses [M_sun] of the cold, catalogued star whose baryon number
# labels a PNS track. TUNABLE: the outcomes table sweeps these.
#   MT0_REF is the baseline: Fig. 5, the W*/T map and the outcomes table are all
#   evaluated at it, and the Delta-sigma_crit maps (V.2) difference the others
#   against it. It MUST be in MT0_GRID or nothing downstream has a grid to read.
MT0_GRID = [1.4] if REDUCED_GRID else [1.17, 1.4, 1.6]
MT0_REF = 1.4
assert MT0_REF in MT0_GRID

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
    e_c_vec_tov=E_C_VEC_QUARK,    # quark-star grid: see I.3
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

# Where a scanned grid lives, and how to read one back. Defined HERE, in Part I,
# because both II.8 (which skips a scan whose file already exists) and III.5
# (which loads them for the figures) need the same convention -- and a scan that
# writes one path while the loader reads another silently rescans for hours.
SCAN_PATH_FMT = str(DIRS['sigma_crit'] /
                    'sigma_crit_grid_{xsd}_MT0{MT0:.2f}_{flavor}-{charge}-{phase}.npz')


def scan_path(MT0, flavor=MAIN_FLAVOR, charge=MAIN_CHARGE, phase=MAIN_PHASE):
    """Path of one saved sigma_crit grid (it need not exist yet)."""
    return Path(SCAN_PATH_FMT.format(xsd=xsd_tag, MT0=MT0, flavor=flavor,
                                     charge=charge, phase=phase))


def load_scan(MT0, flavor=MAIN_FLAVOR, charge=MAIN_CHARGE, phase=MAIN_PHASE):
    """One saved sigma_crit grid as a dict, or None if it was never scanned."""
    path = scan_path(MT0, flavor, charge, phase)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        # .item() on the 0-d entries: ONE idiom, so a scalar never comes back as
        # a 0-d array that then formats as 'array(1.4)' in a figure title.
        return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}

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

# Figures 1-3 of the manuscript were produced BEFORE PAPER_STYLE was changed
# from 10 pt / aspect 1.2 to the values above, so reproducing those pages needs
# the style they were actually drawn with -- there is no single style that
# reproduces the whole published set. Set this to PAPER_STYLE to put the entire
# figure set on one modern style instead; every figure is then self-consistent
# and Figs. 1-3 change proportions relative to the current PDFs.
PAPER_STYLE_FIG123 = dict(fontsize=10, labelsize=10, legendsize=9, aspect=1.2)

set_paper_style(**{k: v for k, v in PAPER_STYLE.items() if k != 'aspect'})

# --- which sigma values the figures will ask for -------------------------------
# The published choices, per figure. Every entry must be in SIGMA_LIST or the
# curve is silently missing from the figure -- which is exactly what the
# assertion below catches, in Part I, before hours of Part II.
FIG_SIGMAS = {
    'fig1_barrier':   [100.],            # panel (a) W(R) and (b-d) vs n_B
    'fig3_centre':    [80., 100., 150.],  # colour = sigma along the PNS sequence
    'fig4_setA':      [50., 100., 150.],  # Fig 4 rows: one sigma list per set
    'fig4_setB':      [150., 200., 250.],  # Set B nucleates at higher sigma
    'appA_charge':    [100., 150.],       # charge prescriptions, two sigmas
    'appB_flavour':   [100.],             # frozen vs saddlepoint
    'outcomes_table': [50., 100.],
}
_wanted = sorted(set().union(*FIG_SIGMAS.values()))
_missing = sorted(set(_wanted) - set(SIGMA_LIST))
if _missing and not REDUCED_GRID:
    raise AssertionError(
        f"figures request sigma {_missing} but SIGMA_LIST is {SIGMA_LIST} -- "
        f"Part II will never tabulate those and the figures would silently drop "
        f"the corresponding curves. Add them to SIGMA_LIST or fix FIG_SIGMAS.")
if _missing:
    print(f"NOTE (smoke mode): figures ask for sigma {_missing}, which the "
          f"reduced SIGMA_LIST does not tabulate -- those curves are absent. "
          f"Expected; the production list carries all of them.")

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
        _p, filt_cfg, e_c_vec=E_C_VEC_QUARK,
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
    _pars = quark_params(alpha=_p['alpha'], B4=_p['B4'], m_s=_p['m_s'])
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
def nuc_stem(stag, flavor, charge, phase, sigma):
    """Filename stem / dict key for one nucleation table."""
    return f"Htrapped_{flavor}_{charge}_{phase}_{stag}_s{int(sigma)}"


# (flavour, charge, [phases]) tabulated. `frozen` freezes the droplet at the
# hadronic composition, which only has a meaning under local neutrality, so it
# is tabulated for LCN / unpaired alone -- that is the pair Appendix B compares.
_NUC_MODES = [
    ('saddlepoint', 'lcn',              ['unpaired', 'cfl', 'unpCFL']),
    ('saddlepoint', 'gcn',              ['unpaired', 'cfl', 'unpCFL']),
    ('saddlepoint', 'coulomb_minimize', ['unpaired', 'cfl', 'unpCFL']),
    ('frozen',      'lcn',              ['unpaired']),
]

_T_grid_H = H_table['trapped'].grids['T']

for _p in quark_param_sets:
    _stag = q_tag(_p)
    _pars = quark_params(alpha=_p['alpha'], B4=_p['B4'], m_s=_p['m_s'])
    # R_x(T) = hbar c / Delta(T), the pairing coherence radius, evaluated on the
    # WHOLE hadronic T grid. It must be an array: the gap melts as T rises, so a
    # single scalar R_x would put the unpaired->CFL switch at the same radius in
    # a 2 MeV droplet as in a 90 MeV one.
    _Rx = crossover_radius(_T_grid_H, _p['Delta0'])
    for _flavor, _charge, _phases in _NUC_MODES:
        for _phase in _phases:
            for _sg in SIGMA_LIST:
                _stem = nuc_stem(_stag, _flavor, _charge, _phase, _sg)
                _path = DIRS['nucleation'] / f'nucleation_{_stem}.dat'
                if _path.exists():
                    continue
                _kw = dict(sigma=_sg, V=V_NUC, flavor_mode=_flavor,
                           electric_charge_mode=_charge,
                           include_photons=include_photons, include_gluons=True,
                           include_thermal_neutrinos=include_thermal_neutrinos,
                           verbose=False, save_table=True,
                           output_file=str(_path))

                def _Qs(phase):
                    """The saved Q* table for this phase, or None to recompute.
                    Reusing it is what keeps this section from re-solving the
                    droplet composition it already solved in II.6; `frozen` has
                    no saved table (they are all saddlepoint) so it recomputes."""
                    q = DIRS['Qstar'] / f'Qstar_{qstar_stem(_stag, _charge, phase, _sg)}.dat'
                    return (load_Qstar_table(str(q))
                            if (_flavor == 'saddlepoint' and q.exists()) else None)

                if _phase == 'unpCFL':
                    # The composite barrier is assembled from BOTH pure phases,
                    # so both Q* tables go in and the kink lands at R_x(T).
                    compute_thermal_nucleation_observables(
                        H_table['trapped'], quark_phase='unpCFL',
                        Delta0=_p['Delta0'], Rx=_Rx, switching_mode='step',
                        Qstar_table_unp=_Qs('unpaired'),
                        Qstar_table_cfl=_Qs('cfl'),
                        params_unp=_pars, params_cfl=_pars, **_kw)
                else:
                    compute_thermal_nucleation_observables(
                        H_table['trapped'], quark_phase=_phase,
                        Delta0=(_p['Delta0'] if _phase == 'cfl' else None),
                        Qstar_table=_Qs(_phase), params=_pars, **_kw)
    _n_modes = sum(len(ph) for _, _, ph in _NUC_MODES)
    print(f"  [{_stag}] nucleation tables done "
          f"({_n_modes} modes x {len(SIGMA_LIST)} sigma)")

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

# Which (flavour, charge, phase) planes to scan. The first is the paper's main
# method; the others support the method-comparison figures.
SCAN_CASES = [(MAIN_FLAVOR, MAIN_CHARGE, MAIN_PHASE)]
if not REDUCED_GRID:
    SCAN_CASES.append((MAIN_FLAVOR, MAIN_CHARGE, 'cfl'))

# Scan only what is missing, then LOAD every grid -- computed or pre-existing --
# into one dict. Skipping without loading is the trap this avoids: the cell
# looks like it succeeded, `scan_grids` is empty, and the failure only shows up
# several cells later as a KeyError with no hint that a scan was skipped.
scan_grids = {}          # (flavor, charge, phase, MT0) -> grid dict
for _flav, _chg, _ph in SCAN_CASES:
    _todo = [m for m in MT0_GRID if not scan_path(m, _flav, _chg, _ph).exists()]
    if _todo:
        print(f"\n-- sigma_crit scan: {_flav} / {_chg} / {_ph}  MT0={_todo} --")
        print(f"   grid: {len(SCAN['alpha'])} alpha x {len(SCAN['B4'])} B4 "
              f"x {len(SCAN['Delta0'])} Delta0, {N_SHELLS} shells")
        run_sigma_crit_scan(
            _todo, _flav, _chg, _ph,
            SCAN['alpha'], SCAN['B4'], SCAN['Delta0'],
            filt_cfg, nuc_cfg, star_scan,
            n_jobs=SCAN['n_jobs'], save_path_fmt=SCAN_PATH_FMT, xsd_tag=xsd_tag,
            n_shells=N_SHELLS, nB_shell_min=NB_SHELL_MIN)
    for _m in MT0_GRID:
        _g = load_scan(_m, _flav, _chg, _ph)
        if _g is None:
            print(f"  [MISSING] {_flav}-{_chg}-{_ph} MT0={_m:.2f} -- the scan "
                  f"produced no file; downstream cells will skip it")
            continue
        scan_grids[(_flav, _chg, _ph, _m)] = _g
        _n_ok = int(np.isfinite(_g['sig_crit']).sum())
        print(f"  [{'computed' if _m in _todo else 'loaded  '}] "
              f"{_flav}-{_chg}-{_ph} MT0={_m:.2f}: {_n_ok}/{_g['sig_crit'].size} "
              f"cells with a finite sigma_crit")

print(f"\nPart II complete. {len(scan_grids)} sigma_crit grid(s) in memory.")

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


def nuc_table(pset, charge, phase, sigma, flavor=MAIN_FLAVOR):
    """The thermal observables for one method, or a clear error if absent."""
    stem = nuc_stem(q_tag(pset), flavor, charge, phase, sigma)
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
# load_scan / scan_path come from Part I, so Part III stands alone without
# re-running Part II and both parts agree on where a grid lives.
SCAN_MT0 = MT0_REF          # the baseline grid Fig. 5 and the W*/T map draw
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

print("\nPNS evolution track:")
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
# ## III.6 — Using the package on one point, without a table
#
# The figures below all read the tables Part II produced. If you want to ask the
# same questions about a parameter set that was never tabulated, the package
# answers them directly — this cell is the worked example, and nothing later
# depends on it.
#
# * `nucleation_point(...)` — everything about one $(n_B, Y_L, T, \sigma)$ point:
#   $R_*$, $W_*$, $W_*/T$, the baryon number of the critical droplet, $\tau$.
# * `NucleationCondition` — the $\tau=\tau_{\rm target}$ locus. Build it
#   `from_table` (what Fig. 4 does), or `from_point_solver`, which needs no table
#   at all and is the one for exploring parameter space.
# * `sigma_target_pt(...)` — the inverse question: which $\sigma$ puts this point
#   exactly at $\tau_{\rm target}$. That is $\sigma_{\rm crit}$.

# %%
_demo_p = quark_param_sets[0]
_demo_pars = quark_params(alpha=_demo_p['alpha'], B4=_demo_p['B4'],
                          m_s=_demo_p['m_s'])
_demo_nB, _demo_T, _demo_sig = 3.5 * n_sat, 30.0, 100.0

# (1) one point, everything about it
_pt = nucleation_point(
    H['trapped'], _demo_nB, _demo_T, _demo_sig, params=_demo_pars,
    Y_L_H=PNS_TMAX['YLH'], quark_phase=MAIN_PHASE, Delta0=_demo_p['Delta0'],
    electric_charge_mode=MAIN_CHARGE, V=V_NUC)
print(f"one point  n_B={_demo_nB/n_sat:.2f} n_sat, T={_demo_T:g} MeV, "
      f"sigma={_demo_sig:g} MeV/fm^2:")
print(f"   R_* = {_pt.R_star:.2f} fm   W_* = {_pt.W_star:.1f} MeV   "
      f"W_*/T = {_pt.W_over_T:.1f}")
print(f"   N_B inside the critical droplet = {_pt.N_B_star:.1f}")
print(f"   tau = {_pt.tau:.3e} s  -> nucleates within {TAU_TARGET:g} s? "
      f"{_pt.nucleates(TAU_TARGET)}")

# (2) the nucleation condition, from a table that already exists
_cond = NucleationCondition.from_table(
    nuc_table(_demo_p, MAIN_CHARGE, MAIN_PHASE, _demo_sig),
    tau_target=TAU_TARGET, scan='n_B')
print(f"\nT_nuc({_demo_nB/n_sat:.1f} n_sat) = "
      f"{_cond.T_of_nB(_demo_nB, Y_L=PNS_TMAX['YLH']):.2f} MeV")

# (3) the inverse: the sigma at which THIS point sits exactly at tau_target.
#     shells=None -> the centre-only definition (see I.7).
_sig_c = sigma_target_pt(
    hadronic_point(H['trapped'], _demo_nB, PNS_TMAX['YLH'], _demo_T),
    _demo_T, MAIN_FLAVOR, MAIN_CHARGE, MAIN_PHASE, _demo_pars,
    _demo_p['Delta0'], nuc_cfg, shells=None)
print(f"sigma_crit (centre-only) at that point = {_sig_c:.1f} MeV/fm^2")


# %% [markdown]
# # Part IV — Paper figures
#
# Everything here reads Part III and draws. The only physics computed inline is
# what a saved table cannot hold: the $W(R)$ profiles of Figs. 1 and A, the
# $\sigma_{\rm crit}$ root-finds of Fig. 3(d), and the outcomes table.
#
# **These cells reproduce the published figures.** Colours, sizes, limits,
# legends and panel anchors are the ones the paper carries; the knob blocks at
# the top of each cell are what to edit, not the drawing code below them.
#
# The style is applied once in Part I and every figure goes through
# `paper_grid(**PAPER_STYLE)`; no cell writes `plt.rcParams` directly. Doing so
# is what once made the observational contours render at the wrong size in one
# panel but not another.
#
# | Figure | File |
# |---|---|
# | 1 — barrier and critical quantities | `paper_fig1_barrier_{tag}_{set}.pdf` |
# | 2 — stellar sequences | `paper_fig2_stellar_sequences_{tag}.pdf` |
# | 3 — nucleation at the PNS centre | `paper_fig3_Rstar_Wc_tau_sigmacrit_{tag}_{set}.pdf` |
# | 4 — nucleation conditions | `paper_fig4_Tnuc_{tag}.pdf` |
# | 5 — $\sigma_{\rm crit}$ parameter plane | `paper_fig5_sigcrit_map_isolines_{tag}.pdf` |
# | App. A — charge prescriptions | `paper_appA_charge_prescriptions_{tag}.pdf` |
# | App. B — frozen vs saddle point | `paper_appB_frozen_vs_saddlepoint_{tag}.pdf` |

# %%
# Shared drawing conventions. ONE source of truth: the reader learns
# "solid = unpCFL, dashed = CFL, dotted = unpaired" once and it holds in every
# panel of every figure. Thickness/opacity come from eos.general.figure_style
# so a change there rescales the whole paper.
PHASE_LS = {'unpaired': ':', 'cfl': '--', 'unpCFL': '-'}
PHASE_LBL = {'unpaired': 'unpaired', 'cfl': 'CFL', 'unpCFL': 'unpCFL'}

FIG_P = DIRS['fig_paper']       # the seven figures the paper includes
FIG_S = DIRS['fig_supp']        # supporting material (Part V)
FIG_D = DIRS['fig_data']        # the arrays behind them, as CSV


def save_paper_figure(fig, name, supp=False):
    """Write a figure with the paper's file convention.

    bbox_inches='tight' is NOT cosmetic here: paper_grid fixes the figure size
    and lets the panels centre inside it, so without the tight crop every
    figure carries a margin of slack that LaTeX then scales down.
    """
    path = (FIG_S if supp else FIG_P) / f'{name}.pdf'
    fig.savefig(path, bbox_inches='tight')
    print(f"  saved {path.relative_to(OUT)}")
    return path


# %% [markdown]
# ## IV.1 — Figure 1: the nucleation barrier and what follows from it
#
# **(a)** The work of formation $W(R)$ at $T=30$ MeV for three densities
# (colour) and all three droplet phases (line style). The peak is the barrier:
# its height sets the rate, its position the size of the critical fluctuation.
# Filled dots mark the unpCFL critical point, open dots the CFL one; the shaded
# strip is $R<R_\Delta$, where an unpCFL droplet is still unpaired.
#
# **(b–d)** The three quantities that follow from that peak, against density and
# coloured by temperature: $R_*$, the barrier in units of the temperature
# $W_*/T$ (the exponent that actually decides the rate), and the resulting
# nucleation time $\tau$. The dotted line in (d) is $\tau_{\rm target}$.
#
# The message: $W_*/T$ falls steeply with density, so nucleation is controlled
# by a narrow window near the stellar centre.

# %%
# =============================================================================
#  Figure 1.  Knobs -> which set / method / fixed values.
# =============================================================================
for F1_SET in [quark_param_sets[0]]:
    F1_FLAVOR = 'saddlepoint'                        # 'frozen' | 'saddlepoint'
    F1_CHARGE = 'coulomb_minimize'                   # 'lcn' | 'gcn' | 'coulomb_minimize'
    F1_YLH    = 0.25
    F1_SIGMA  = FIG_SIGMAS['fig1_barrier'][0]        # MeV/fm^2
    F1_TW     = 30.0                                 # panel (a) temperature [MeV]
    F1_DENS   = [1.0, 4, 8]                          # panel (a) n_B^H / n_sat
    F1_TEMPS  = [20.0, 30.0, 40.0, 50.0, 60.0]       # panels (b,c,d) T [MeV]

    _PHASE_FILL = {'unpCFL': True, 'cfl': False, 'unpaired': False}   # dot fill

    _stag   = q_tag(F1_SET)
    _params = quark_params(alpha=F1_SET['alpha'], B4=F1_SET['B4'],
                           m_s=F1_SET['m_s'])

    def _f1_get(phase, sg=F1_SIGMA):
        """The nucleation table for this method/phase/set/sigma (or None)."""
        return nuc_sets.get(nuc_stem(_stag, F1_FLAVOR, F1_CHARGE, phase, sg))

    def _f1_Rx(T):
        """unpCFL crossover radius R_x(T) = hbar c / Delta(T) -- the SAME CFL gap
        as the EoS uses (T_c = T_critical(Delta_0)); delegated to the engine so
        the figure and the tables can never disagree about where the kink is."""
        return float(crossover_radius(T, F1_SET['Delta0']))

    _ref = _f1_get('unpCFL')
    _nBg = _ref.hadronic_grids['n_B_H']
    _Tg  = _ref.hadronic_grids['T']
    _iYL = int(np.argmin(np.abs(_ref.hadronic_grids['Y_L_H'] - F1_YLH)))

    def _iT(T):
        return int(np.argmin(np.abs(_Tg - T)))

    fig, ((axA, axB), (axC, axD)) = paper_grid('2x2', mode='double',
                                               placeholder=False,
                                               **PAPER_STYLE_FIG123)

    # ---- (a) W(R): density = colour, phase = line style ----
    _cA = plt.cm.viridis(np.linspace(0.12, 0.85, len(F1_DENS)))
    _Rg = np.linspace(0.01, 14.0, 400)
    for _ci, _x in enumerate(F1_DENS):
        for _ph, _ls in PHASE_LS.items():
            _eb = compute_energy_barrier(
                H['trapped'], _x * n_sat, F1_TW, F1_SIGMA,
                electric_charge_mode=F1_CHARGE, params=_params,
                flavor_mode=F1_FLAVOR, quark_phase=_ph,
                Delta0=F1_SET['Delta0'], Y_L_H=F1_YLH, R_values=_Rg,
                switching_mode='step',
                Rx=(_f1_Rx(F1_TW) if _ph == 'unpCFL' else None))
            axA.plot(_Rg, _eb.W, color=_cA[_ci], ls=_ls,
                     lw=PHASE_LW[_ph], alpha=PHASE_ALPHA[_ph])
            if np.isfinite(_eb.W).any() and _ph != 'unpaired':
                # Critical point = the barrier peak. No marker on `unpaired`:
                # its peak sits off the frame at these densities and the dot
                # would land on the axis edge.
                _k = int(np.nanargmax(_eb.W))
                axA.plot(_Rg[_k], _eb.W[_k], 'o', ms=6, color=_cA[_ci],
                         mec=_cA[_ci],
                         mfc=(_cA[_ci] if _PHASE_FILL[_ph] else 'white'),
                         zorder=(6 if _PHASE_FILL[_ph] else 5))
    # shade R <= R_Delta: inside it an unpCFL droplet is still unpaired
    _Rdelta = _f1_Rx(F1_TW)
    if np.isfinite(_Rdelta):
        axA.axvspan(0, _Rdelta, color='tab:blue', alpha=0.07, zorder=0)
    axA.axhline(0, color='0.6', lw=0.7, zorder=0)
    axA.set_xlim(0, 7); axA.set_ylim(0, 10000)
    axA.set_xlabel(r'$R$ [fm]'); axA.set_ylabel(r'$W$ [MeV]')
    axA.set_title(rf'$Y_L^H={F1_YLH}$, $T={F1_TW:.0f}$ MeV, '
                  rf'$\sigma={F1_SIGMA:.0f}$ MeV/fm$^2$')
    panel_label(axA, '(a)', corner='lower')
    # TWO legends on one axes: matplotlib keeps only the last, so the first is
    # re-added as an artist. Opposite corners, both empty at these limits.
    _lgA = axA.legend([Line2D([], [], color=_cA[i], lw=2)
                       for i in range(len(F1_DENS))],
                      [rf'$n_B^H/n_\mathrm{{sat}}={int(x)}$' for x in F1_DENS],
                      loc='upper right', labelspacing=0.25)
    axA.add_artist(_lgA)
    axA.legend([Line2D([], [], color='0.3', ls=PHASE_LS[p]) for p in PHASE_LS],
               [PHASE_LBL[p] for p in PHASE_LS], loc='upper left')

    # ---- (b,c,d) vs n_B^H: temperature = colour, phase = line style ----
    _cT = plt.cm.plasma(np.linspace(0.05, 0.85, len(F1_TEMPS)))

    def _f1_vs_nBH(ax, getter, ylabel):
        for _ti, _T in enumerate(F1_TEMPS):
            for _ph, _ls in PHASE_LS.items():
                _o = _f1_get(_ph)
                if _o is None:
                    continue
                with np.errstate(divide='ignore', invalid='ignore'):
                    ax.plot(_nBg / n_sat, getter(_o, _iT(_T)), color=_cT[_ti],
                            ls=_ls, lw=PHASE_LW[_ph], alpha=PHASE_ALPHA[_ph])
        ax.set_xlabel(r'$n_B^H/n_\mathrm{sat}$'); ax.set_ylabel(ylabel)
        ax.set_xlim(0.5, 10)

    _title_bcd = rf'$Y_L^H={F1_YLH}$, $\sigma={F1_SIGMA:.0f}$ MeV/fm$^2$'

    _f1_vs_nBH(axB, lambda o, it: o.R_c[:, _iYL, it], r'$R_*$ [fm]')
    axB.set_ylim(1, 7); axB.set_title(_title_bcd)
    panel_label(axB, '(b)', corner='lower')
    axB.legend([Line2D([], [], color=_cT[i], lw=2) for i in range(len(F1_TEMPS))],
               [rf'$T={int(T)}\,\,\rm MeV$' for T in F1_TEMPS], loc='upper right')

    _f1_vs_nBH(axC, lambda o, it: o.W_c[:, _iYL, it] / _Tg[it], r'$W_*/T$')
    axC.set_ylim(0, 500); axC.set_title(_title_bcd)
    panel_label(axC, '(c)', corner='lower')

    _f1_vs_nBH(axD, lambda o, it: np.log10(o.tau[:, _iYL, it]),
               r'$\log_{10}\,\tau$ [s]')
    axD.axhline(np.log10(TAU_TARGET), color='k', ls=(0, (1, 1)), lw=0.9)
    axD.set_ylim(-60, 60); axD.set_title(_title_bcd)
    panel_label(axD, '(d)', corner='lower')

    # CSV of panels (b,c,d): every curve that is drawn, one row per point.
    _d1 = []
    for _T in F1_TEMPS:
        _it = _iT(_T)
        for _ph in PHASE_LS:
            _o = _f1_get(_ph)
            if _o is None:
                continue
            with np.errstate(divide='ignore', invalid='ignore'):
                _R = _o.R_c[:, _iYL, _it]
                _WT = _o.W_c[:, _iYL, _it] / _Tg[_it]
                _LT = np.log10(_o.tau[:, _iYL, _it])
            for _nb, _r, _w, _lt in zip(_nBg / n_sat, _R, _WT, _LT):
                _d1.append((_T, _ph, float(_nb), float(_r), float(_w), float(_lt)))
    pd.DataFrame(_d1, columns=['T_MeV', 'phase', 'nBH_over_n0', 'R_star_fm',
                               'W_over_T', 'log10_tau_s']).to_csv(
        FIG_D / f'fig1_bcd_{xsd_tag}_{_stag}.csv', index=False)
    save_paper_figure(fig, f'paper_fig1_barrier_{xsd_tag}_{_stag}')
    plt.show()

# %% [markdown]
# ## IV.2 — Figure 2: stellar sequences
#
# The two families side by side, in the four planes the argument needs.
#
# **(a)** mass–radius against the NICER/HESS/GW170817 regions and the J0740 /
# J0952 mass bands. **(b)** $M$ against **baryonic** mass — the plane the
# conversion actually happens in, because baryon number, not mass, is conserved.
# **(c, d)** central temperature and central density along each sequence, again
# against $M_B$, with dots at $M=1,1.2,1.4,1.6\,M_\odot$ and a star at
# $M_{\max}$.
#
# NS is the cold hadronic star, QS the cold CFL quark star, and the two PNS
# curves are the snapshots of I.6 (orange $t_0$, red $t_{T_{\max}}$). The quark
# branch lies *below* the neutron-star branch in (b): a converted star is lighter
# at the same baryon number, and that difference is the energy released.

# %%
# =============================================================================
#  Figure 2.  Knobs.
# =============================================================================
F2_SET = quark_param_sets[0]              # quark parametrization (reference)
F2_QS_SETS = [quark_param_sets[0]]        # QS curves drawn; add sets to overlay
_QS_GREENS = ['#31a354', '#006d2c']       # one green per QS set (light -> dark)

# Fonts come from set_paper_style() via paper_grid, the same as every other
# paper figure -- do NOT override them here or Fig 2 desyncs from the rest
# (this is what once made the PSR/HESS labels render tiny).

# Per-panel inline-label anchors: (frac_along_branch, dx, dy, ha, va).
# Mass-marker label offsets in (c)/(d): cdot/ddot for the M-dot value labels,
# cstar/dstar for the M_max star, each (dx_pt, dy_pt, ha, va) in POINTS so the
# text clears the curves regardless of the axis limits. Tune against the render.
_F2SEQ = [
    dict(arr=tov_cold, c='#000000', lbl='NS',
         cdot=(-9, 26, 'center', 'bottom'), cstar=(-9, 34, 'center', 'bottom'),
         ddot=(-8, 2, 'right', 'bottom'),   dstar=(-6, -2, 'right', 'top')),
]
for _i, _p in enumerate(F2_QS_SETS):
    _F2SEQ.append(dict(
        arr=qs_tov[q_tag(_p)], c=_QS_GREENS[_i % len(_QS_GREENS)],
        lbl=(f'QS ({q_tag(_p)})' if len(F2_QS_SETS) > 1 else 'QS'),
        cdot=(-13, 13, 'center', 'bottom'), cstar=(-13, 13, 'center', 'bottom'),
        ddot=(0, -9, 'center', 'top'),      dstar=(-7, 0, 'right', 'center')))
_F2SEQ += [
    dict(arr=nearest_trapped_sequence(tov_trapped, PNS_T0['YLH'], PNS_T0['S']),
         c=PNS_T0['color'], lbl=r'PNS ($t_0$)',
         yls=(PNS_T0['YLH'], PNS_T0['S']),
         cdot=(0, -9, 'center', 'top'),   cstar=(8, -3, 'left', 'top'),
         ddot=(8, -1, 'left', 'top'),     dstar=(7, -2, 'left', 'top')),
    dict(arr=nearest_trapped_sequence(tov_trapped, PNS_TMAX['YLH'], PNS_TMAX['S']),
         c=PNS_TMAX['color'], lbl=r'PNS ($t_{T_\mathrm{max}}$)',
         yls=(PNS_TMAX['YLH'], PNS_TMAX['S']),
         cdot=(0, 8, 'center', 'bottom'), cstar=(8, -4, 'left', 'top'),
         ddot=(-8, 0, 'right', 'center'), dstar=(-4, 7, 'right', 'bottom')),
]

fig, ((axA, axB), (axC, axD)) = paper_grid('2x2', mode='double',
                                           placeholder=False,
                                           **PAPER_STYLE_FIG123)

# (a) M-R. Fix the view FIRST so the constraint fills (some reach R ~ 21 km)
#     cannot auto-expand the axis and fling the inline labels into the margin.
axA.set_xlim(8, 16); axA.set_ylim(0.3, 2.8)
for _s in _F2SEQ:
    _a = stable_branch(_s['arr'])
    axA.plot(_a[:, TOV_COL['R']], _a[:, TOV_COL['M']], color=_s['c'], lw=2.4,
             zorder=4)                      # curve labels live in (b)'s legend
add_observational_constraints(axA, show_mass_bands=True, inline_labels=True)
axA.set_xlim(8, 16); axA.set_ylim(0.3, 2.8)              # re-assert after fills
axA.set_xlabel(r'$R$ [km]'); axA.set_ylabel(r'$M$ [$M_\odot$]')
panel_label(axA, '(a)', corner='lower right')

# (b) M vs M_baryonic. Carries the shared curve legend for all four panels.
for _s in _F2SEQ:
    _a = stable_branch(_s['arr'])
    axB.plot(_a[:, TOV_COL['M_B']], _a[:, TOV_COL['M']], color=_s['c'], lw=2.4,
             label=_s['lbl'])
axB.set_xlabel(r'$M_B$ [$M_\odot$]'); axB.set_ylabel(r'$M$ [$M_\odot$]')
axB.set_xlim(0.6, 3.4); axB.set_ylim(0.6, 2.6)
axB.legend(loc='upper left', fontsize=10, frameon=False)
panel_label(axB, '(b)', corner='lower right')

# (c) central temperature vs M_B. NS and QS are cold, so they are drawn as a
#     coloured T_c = 0 line over their own M_B span; the PNS sequences carry
#     T_c(n_B^c) from their isentrope. Every curve gets dots + an M_max star.
for _s in _F2SEQ:
    _a = stable_branch(_s['arr'])
    if _s.get('yls') is None:
        _mb = _a[:, TOV_COL['M_B']]
        axC.plot([_mb.min(), _mb.max()], [0.0, 0.0], color=_s['c'], lw=2.4)
        # The cold markers crowd the T_c = 0 line, so their labels get leader
        # arrows to lift them clear of it.
        mass_marks(axC, _a, np.zeros(len(_a)), _s['c'],
                   dot_off=_s['cdot'], star_off=_s['cstar'], dot_arrow=True)
        continue
    _yl, _S = _s['yls']
    _Tc = np.asarray(H['iso_trapped']['T'](_a[:, TOV_COL['n_Bc']], _yl, _S))
    axC.plot(_a[:, TOV_COL['M_B']], _Tc, color=_s['c'], lw=2.4)
    mass_marks(axC, _a, _Tc, _s['c'], dot_off=_s['cdot'], star_off=_s['cstar'])
axC.set_xlabel(r'$M_B$ [$M_\odot$]'); axC.set_ylabel(r'$T_c$ [MeV]')
axC.set_xlim(0.6, 3.4); axC.set_ylim(bottom=-1)
panel_label(axC, '(c)', corner='upper right')

# (d) central density vs M_B, same marker set.
for _s in _F2SEQ:
    _a = stable_branch(_s['arr'])
    _y = _a[:, TOV_COL['n_Bc']] / n_sat
    axD.plot(_a[:, TOV_COL['M_B']], _y, color=_s['c'], lw=2.4)
    mass_marks(axD, _a, _y, _s['c'], dot_off=_s['ddot'], star_off=_s['dstar'])
axD.set_xlabel(r'$M_B$ [$M_\odot$]')
axD.set_ylabel(r'$n_B^c/n_\mathrm{sat}$')
axD.set_xlim(0.6, 3.4)
panel_label(axD, '(d)', corner='lower right')

_d2 = []
for _s in _F2SEQ:
    _a = stable_branch(_s['arr'])
    _yls = _s.get('yls')
    _Tc = (np.zeros(len(_a)) if _yls is None
           else np.asarray(H['iso_trapped']['T'](_a[:, TOV_COL['n_Bc']],
                                                 _yls[0], _yls[1])))
    for _row, _tc in zip(_a, _Tc):
        _d2.append((_s['lbl'], float(_row[TOV_COL['M_B']]),
                    float(_row[TOV_COL['M']]), float(_row[TOV_COL['R']]),
                    float(_row[TOV_COL['n_Bc']]) / n_sat, float(_tc)))
pd.DataFrame(_d2, columns=['sequence', 'M_B_Msun', 'M_Msun', 'R_km',
                           'nBc_over_n0', 'Tc_MeV']).to_csv(
    FIG_D / f'fig2_sequences_{xsd_tag}.csv', index=False)
save_paper_figure(fig, f'paper_fig2_stellar_sequences_{xsd_tag}')
plt.show()

# %% [markdown]
# ## IV.3 — Figure 3: nucleation at the PNS centre vs $M_{\rm PNS}$
#
# The same quantities as Fig. 1, but now *along a stellar sequence*: every point
# on the x-axis is the centre of a proto-neutron star of that mass, at the
# $t_{T_{\max}}$ snapshot. This is what connects the microphysics to an
# observable — a heavier star is denser and hotter at its centre, so it
# nucleates more easily.
#
# **(a,b,c)** $R_*$, $W_*/T$ and $\log_{10}\tau$, colour = $\sigma$, line style =
# phase. The faint horizontal guides in (c) are the three $\tau$ targets, in the
# same colours panel (d) uses.
#
# **(d)** the headline: $\sigma_{\rm crit}(M_{\rm PNS})$, the surface tension at
# which the *central* $\tau$ equals each target. A star whose $\sigma$ lies below
# the curve converts. This is the **centre-only** definition — Fig. 5 uses the
# star-wide one, and they are different quantities.

# %%
# =============================================================================
#  Figure 3.  R*, W*/T, log10(tau) and sigma_crit at the PNS centre vs M_PNS.
#  colour = sigma (a-c) or tau target (d);  line style = droplet phase.
#  The (M_PNS -> n_Bc -> T_c) map is hadronic-only: built ONCE, reused.
# =============================================================================
set_paper_style()

# ---- knobs ----
FN2_YL, FN2_S = PNS_TMAX['YLH'], PNS_TMAX['S']   # the snapshot: t_Tmax
FN2_FLAVOR    = 'saddlepoint'
FN2_CHARGE    = 'coulomb_minimize'
FN2_MVERT     = []                       # masses for vertical guides (none)
FN2_SIGMAS    = FIG_SIGMAS['fig3_centre']  # sigma set drawn in (a-c) [MeV/fm^2]
FN2_VGUIDE_COL = STANDARD_COLORS['Red']
FN2_D_MODE    = 'tau'                    # (d) colour: 'tau' | 'quarks'
FN2_D_TAUS    = [1e-3, 1.0, 100.0]       # s -- tau targets when mode == 'tau'
FN2_D_NM      = 100                      # M_PNS samples in (d); each is a root-find

_sig_col = {sg: mpl.cm.viridis(t) for sg, t in
            zip(FN2_SIGMAS, np.linspace(0.15, 0.85, len(FN2_SIGMAS)))}
_phase_ls = list(PHASE_LS.items())

# ---- the PNS sequence (quark-independent): M_PNS -> (n_Bc, T_c) ----
_tov = nearest_trapped_sequence(tov_trapped, FN2_YL, FN2_S)
_tov_st = stable_branch(_tov)
_msk = _tov_st[:, TOV_COL['M']] >= 0.6              # show from 0.6 M_sun up
M_seq   = _tov_st[_msk, TOV_COL['M']]
nBc_seq = _tov_st[_msk, TOV_COL['n_Bc']]
T_seq   = np.asarray(H['iso_trapped']['T'](nBc_seq, FN2_YL, FN2_S))
M_max   = float(_tov_st[:, TOV_COL['M']].max())

# Vertical guides: dashed at fixed M_PNS, dotted at the M_PNS whose BARYONIC
# mass matches a cold star of the same gravitational mass (the same star, cooled).
_cold_M_to_Mb = branch_interp(tov_cold, 'M', 'M_B')
_trap_Mb_to_M = branch_interp(_tov, 'M_B', 'M')
M_dot = [float(_trap_Mb_to_M(_cold_M_to_Mb(m0))) for m0 in FN2_MVERT]

# panels (a,b,c): (ylabel, getter(interpolators), yscale, ylim)
_abc = [
    (r'$R_*$ [fm]',
     lambda itp: np.array([itp['R_c'](nb, _YL, T)
                           for nb, T in zip(nBc_seq, T_seq)]),
     'linear', (1, 7)),
    (r'$W_*/T$',
     lambda itp: np.array([itp['W_c'](nb, _YL, T)
                           for nb, T in zip(nBc_seq, T_seq)]) / T_seq,
     'linear', (0, 500)),
    (r'$\log_{10}\,\tau$ [s]',
     lambda itp: np.array([itp['log10_tau'](nb, _YL, T)
                           for nb, T in zip(nBc_seq, T_seq)]),
     'linear', (-40, 80)),
]


def _tau_lbl(t):
    return f'{t*1e3:g} ms' if t < 1 else f'{t:g} s'


for FN2_SET in quark_param_sets:
    _tag    = q_tag(FN2_SET)
    _params = quark_params(alpha=FN2_SET['alpha'], B4=FN2_SET['B4'],
                           m_s=FN2_SET['m_s'])
    _D0     = FN2_SET['Delta0']
    _T_CFL  = float(T_critical(_D0))          # CFL is undefined above this

    fig, ((axA, axB), (axC, axD)) = paper_grid('2x2', mode='double',
                                               placeholder=False,
                                               **PAPER_STYLE_FIG123)
    _abc_axes = [axA, axB, axC]

    # The Y_L node the tables actually carry (the getters read there).
    _a0 = next((nuc_sets[k] for k in nuc_sets
                if k.endswith(f"_{_tag}_s{int(FN2_SIGMAS[0])}")), None)
    _YL = (float(_a0.hadronic_grids['Y_L_H'][
               np.argmin(np.abs(_a0.hadronic_grids['Y_L_H'] - FN2_YL))])
           if _a0 is not None else FN2_YL)

    # (a,b,c) from the stored tables: colour = sigma, line style = phase
    for sg in FN2_SIGMAS:
        for ph, ls in _phase_ls:
            stem = nuc_stem(_tag, FN2_FLAVOR, FN2_CHARGE, ph, sg)
            if stem not in nuc_sets:
                continue
            itp = build_thermal_nucleation_interpolators(nuc_sets[stem])
            for ax, (_, getter, _sc, _yl) in zip(_abc_axes, _abc):
                ax.plot(M_seq, getter(itp), color=_sig_col[sg], ls=ls,
                        lw=PHASE_LW[ph], alpha=PHASE_ALPHA[ph])
    for ax, (ylab, _, sc, ylim) in zip(_abc_axes, _abc):
        ax.set_ylabel(ylab); ax.set_yscale(sc)
        if ylim is not None:
            ax.set_ylim(*ylim)

    # ---- (d) sigma_crit(tau = target) vs M_PNS ----
    # Each point is a root-find, so it runs on a coarser M grid sampled from the
    # SAME central (n_B, T) sequence as (a-c).
    _nBc_of_M = interp1d(M_seq, nBc_seq, kind='cubic', bounds_error=False,
                         fill_value=np.nan)
    _T_of_M   = interp1d(M_seq, T_seq, kind='cubic', bounds_error=False,
                         fill_value=np.nan)
    _Md = np.linspace(M_seq.min(), M_seq.max(), FN2_D_NM)
    _nBd, _Td = _nBc_of_M(_Md), _T_of_M(_Md)

    def _sigma_crit_vs_M(params, D0, phase, nuc):
        """sigma_crit(M_PNS): the sigma at which the CENTRAL tau equals
        nuc.tau_target. NaN where no crossing exists, and where the CFL gap has
        already melted (T > T_c) so a CFL droplet is not defined at all."""
        out = sigma_crit_along_sequence(
            _nBd, _Td, Y_L_H=FN2_YL, H_trapped=H['trapped'], params=params,
            Delta0=D0, nuc=nuc, shells=None,          # shells=None -> centre-only
            flavor=FN2_FLAVOR, charge=FN2_CHARGE, phase=phase)
        if phase == 'cfl':
            out = np.where(_Td > float(T_critical(D0)), np.nan, out)
        return np.where(np.isfinite(out), out, np.nan)   # +/-inf -> gap

    if FN2_D_MODE == 'quarks':          # colour = quark set, one tau
        _dcol = {q_tag(p): mpl.cm.plasma(t) for p, t in
                 zip(quark_param_sets,
                     np.linspace(0.15, 0.8, max(len(quark_param_sets), 2)))}
        for p in quark_param_sets:
            _pp = quark_params(alpha=p['alpha'], B4=p['B4'], m_s=p['m_s'])
            for ph, ls in _phase_ls:
                axD.plot(_Md, _sigma_crit_vs_M(_pp, p['Delta0'], ph, nuc_cfg),
                         color=_dcol[q_tag(p)], ls=ls, lw=PHASE_LW[ph],
                         alpha=PHASE_ALPHA[ph])
        _d_handles = [Line2D([], [], color=_dcol[q_tag(p)], label=q_tag(p))
                      for p in quark_param_sets]
        _d_title = 'quark set'
    else:                               # colour = tau target, one quark set
        _dcol = {t: mpl.cm.plasma(u) for t, u in
                 zip(FN2_D_TAUS, np.linspace(0.15, 0.8, max(len(FN2_D_TAUS), 2)))}
        for tau in FN2_D_TAUS:
            for ph, ls in _phase_ls:
                axD.plot(_Md, _sigma_crit_vs_M(_params, _D0, ph,
                                               dc_replace(nuc_cfg, tau_target=tau)),
                         color=_dcol[tau], ls=ls, lw=PHASE_LW[ph],
                         alpha=PHASE_ALPHA[ph])
        _d_handles = [Line2D([], [], color=_dcol[t], label=_tau_lbl(t))
                      for t in FN2_D_TAUS]
        _d_title = r'$\tau$'
    axD.set_ylabel(r'$\sigma_{\rm crit}$ [MeV/fm$^2$]')
    axD.set_ylim(50, 200)
    axD.legend(handles=_d_handles, loc='best', title=_d_title, fontsize=8)

    # tau reference levels on (c), coloured to MATCH (d) -- the SAME plasma
    # recipe over FN2_D_TAUS -- so the two panels read together. Faded and thin
    # so they register as guides, not data.
    _tau_ref_col = {t: mpl.cm.plasma(u) for t, u in
                    zip(FN2_D_TAUS,
                        np.linspace(0.15, 0.8, max(len(FN2_D_TAUS), 2)))}
    for _t in FN2_D_TAUS:
        axC.axhline(np.log10(_t), color=_tau_ref_col[_t], ls='-', lw=1.2,
                    alpha=0.4, zorder=1)

    for _lab, ax in zip(('(a)', '(b)', '(c)', '(d)'), (axA, axB, axC, axD)):
        ax.set_xlim(0.7, M_max)
        ax.set_xlabel(r'$M_{\rm PNS}\ [M_\odot]$')       # no sharex: label all
        for mv in FN2_MVERT:
            ax.axvline(mv, color=FN2_VGUIDE_COL, ls='-', lw=1.0, zorder=1)
        for md in M_dot:
            if np.isfinite(md):
                ax.axvline(md, color=FN2_VGUIDE_COL, ls=':', lw=1.0, zorder=1)
        panel_label(ax, _lab, corner='upper left')

    # Legends: sigma colour on (a), phase line style on (b). They were once two
    # legend() calls on the same axes, and matplotlib keeps only the last -- so
    # the phase key silently vanished. One legend per axes.
    axA.legend(handles=[Line2D([], [], color=_sig_col[sg], ls='-',
                               label=rf'$\sigma={int(sg)}$') for sg in FN2_SIGMAS],
               loc='upper right', title=r'$\sigma$ [MeV/fm$^2$]')
    axB.legend(handles=[Line2D([], [], color='k', ls=PHASE_LS[p],
                               lw=PHASE_LW[p], label=PHASE_LBL[p])
                        for p in PHASE_LS], loc='best')

    _d3 = []
    for sg in FN2_SIGMAS:
        for ph, ls in _phase_ls:
            stem = nuc_stem(_tag, FN2_FLAVOR, FN2_CHARGE, ph, sg)
            if stem not in nuc_sets:
                continue
            itp = build_thermal_nucleation_interpolators(nuc_sets[stem])
            _c3 = [getter(itp) for _, getter, _, _ in _abc]
            for _m, _r, _w, _lt in zip(M_seq, *_c3):
                _d3.append((sg, ph, float(_m), float(_r), float(_w), float(_lt)))
    pd.DataFrame(_d3, columns=['sigma', 'phase', 'M_PNS_Msun', 'R_star_fm',
                               'W_over_T', 'log10_tau_s']).to_csv(
        FIG_D / f'fig3abc_vs_MPNS_{xsd_tag}_{_tag}.csv', index=False)
    save_paper_figure(
        fig, f'paper_fig3_Rstar_Wc_tau_sigmacrit_{xsd_tag}_{_tag}')
    plt.show()

# %% [markdown]
# ## IV.4 — Figure 4: the nucleation condition $T_{\rm nuc}(n_B^H)$
#
# The **nucleation condition** itself: the locus in the $(n_B^H, T)$ plane along
# which $\tau=\tau_{\rm target}$. Above it a droplet forms in time; below it the
# hadronic star survives.
#
# Rows are quark parameter sets (A top, B bottom), columns the two PNS snapshots
# ($t_0$ left, $t_{T_{\max}}$ right). Colour is $\sigma$ — chosen **per set**,
# because the two sets nucleate in different $\sigma$ ranges — and line style is
# the droplet phase.
#
# Drawn over it in the snapshot colour is the stellar isentrope, the path an
# actual star's centre traverses. Markers on it are stars: a filled dot at fixed
# gravitational mass, an open dot at the density where the PNS has the *baryonic*
# mass of a cold star of that mass, and stars at $M_{\max}$. The figure is read by
# asking where the isentrope crosses a $T_{\rm nuc}$ curve.

# %%
# =============================================================================
#  Figure 4.  T_nuc(n_B^H) -- one 2x2, two quark sets x two snapshots.
# =============================================================================
set_paper_style()

# ---- knobs ----
FN_FLAVOR = 'saddlepoint'
FN_CHARGE = 'coulomb_minimize'
FN_TAU    = TAU_TARGET
# sigma drawn, SELECTED PER quark set (colours are viridis over each set's own
# list, so the two rows do NOT share a colour scale). Each value must be in
# SIGMA_LIST or its curve is simply absent -- see the note in I.5 about 250.
F6_SIGMAS_BY_SET = {q_tag(quark_param_sets[0]): FIG_SIGMAS['fig4_setA'],
                    q_tag(quark_param_sets[1]): FIG_SIGMAS['fig4_setB']}
FN_CUT_CFL_ABOVE_TC = True     # mask CFL where T_nuc > T_c (the gap has melted)

_MDOTS_F6 = (1.0, 1.2, 1.4, 1.6)     # gravitational masses marked [M_sun]


def _iso_markers(ax, tov_tr, T_iso, color):
    """PNS central-density markers on the isentrope, Fig-2 (c/d) styling.

    For each gravitational mass in _MDOTS_F6: a SOLID dot at the fixed-M PNS
    density, and a WHITE-filled dot at the density where the PNS baryonic mass
    equals M_B of the COLD star of the same M. One value label arrows to both,
    because they are the same star before and after cooling. Stars mark M_max:
    solid = the trapped branch tip, white = the cold M_max, baryon-matched.
    """
    nBc_M     = branch_interp(tov_tr, 'M', 'n_Bc')       # trapped: M    -> n_Bc
    Mb_of_M   = branch_interp(tov_cold, 'M', 'M_B')      # cold:    M    -> M_B
    nBc_of_Mb = branch_interp(tov_tr, 'M_B', 'n_Bc')     # trapped: M_B  -> n_Bc

    def _pt(nb):                                   # density -> (n_B/n_sat, T)
        return (nb / n_sat, float(T_iso(nb))) if np.isfinite(nb) else (np.nan,
                                                                       np.nan)

    for mt in _MDOTS_F6:
        pf = _pt(float(nBc_M(mt)))                       # solid: fixed M
        pw = _pt(float(nBc_of_Mb(Mb_of_M(mt))))          # white: M_B-matched
        if np.isfinite(pf[0]):
            ax.plot(*pf, 'o', ms=5.5, color=color, mec='white', mew=0.7,
                    zorder=7)
        if np.isfinite(pw[0]):
            ax.plot(*pw, 'o', ms=5.5, color='white', mec=color, mew=1.3,
                    zorder=7)
        label_with_arrows(ax, [pf, pw], f"{mt:g}", color)

    sf = _pt(tov_tr[int(np.argmax(tov_tr[:, TOV_COL['M']])), TOV_COL['n_Bc']])
    sw = _pt(float(nBc_of_Mb(
        tov_cold[int(np.argmax(tov_cold[:, TOV_COL['M']])), TOV_COL['M_B']])))
    if np.isfinite(sf[0]):
        ax.plot(*sf, '*', ms=13, color=color, mec='white', mew=0.6, zorder=7)
    if np.isfinite(sw[0]):
        ax.plot(*sw, '*', ms=13, color='white', mec=color, mew=1.1, zorder=7)
    label_with_arrows(ax, [sf, sw], r"$M_{\max}$", color)


def _draw_f4_panel(ax, FN_SET, pan, panel_lab, set_label):
    """One T_nuc(n_B^H) panel for (quark set, snapshot).

    Returns (sigma -> colour, sigma list) so the caller can build that ROW's
    legend: the sigma palette differs between the two sets."""
    _tag = q_tag(FN_SET)
    F6_SIGMAS = F6_SIGMAS_BY_SET.get(_tag, FIG_SIGMAS['fig4_setA'])
    _sc = {sg: mpl.cm.viridis(t) for sg, t in
           zip(F6_SIGMAS, np.linspace(0.15, 0.85, len(F6_SIGMAS)))}
    _T_CFL = float(T_critical(FN_SET['Delta0']))
    YL, S, col = pan['YLH'], pan['S'], pan['color']
    YL_used = None
    for sg in F6_SIGMAS:
        for ph, ls in PHASE_LS.items():           # unpCFL last -> drawn on top
            stem = nuc_stem(_tag, FN_FLAVOR, FN_CHARGE, ph, sg)
            if stem not in nuc_sets:
                continue
            grids = nuc_sets[stem].hadronic_grids
            iYL = int(np.argmin(np.abs(grids['Y_L_H'] - YL)))
            YL_used = grids['Y_L_H'][iYL]
            # scan='n_B' (root-find along T at each density) is REQUIRED for
            # unpCFL: tau is non-monotonic in T there, because the gap melts as
            # T rises, and scanning the other way picks the wrong branch
            # without complaining.
            res = compute_nucleation_density(nuc_sets[stem], tau_target=FN_TAU,
                                             scan='n_B')
            nB, T = nucleation_curve(res, iYL)
            m = np.isfinite(nB) & np.isfinite(T)
            if FN_CUT_CFL_ABOVE_TC and ph == 'cfl':
                m &= (T <= _T_CFL)
            if m.any():
                ax.plot(nB[m] / n_sat, T[m], color=_sc[sg], ls=ls,
                        lw=PHASE_LW[ph], alpha=PHASE_ALPHA[ph])
    ax.set_xlim(0.5, 12); ax.set_ylim(1, 80)

    # the isentrope T(n_B^H) at (Y_L, S), in the snapshot colour, + PNS markers
    YL_iso = YL if YL_used is None else float(YL_used)
    T_iso = lambda nBc: H['iso_trapped']['T'](nBc, YL_iso, S)
    tov_tr = nearest_trapped_sequence(tov_trapped, YL_iso, S)
    nBc_mx = tov_tr[int(np.argmax(tov_tr[:, TOV_COL['M']])), TOV_COL['n_Bc']]
    nB_iso = np.linspace(0.5 * n_sat, nBc_mx, 200)
    ax.plot(nB_iso / n_sat, T_iso(nB_iso), color=col, ls='-', lw=1.8, zorder=3)
    _iso_markers(ax, tov_tr, T_iso, col)

    ax.set_xlabel(r'$n_B^H / n_{\rm sat}$')
    ax.set_ylabel(r'$T_{\rm nuc}$ [MeV]')
    ax.set_title(rf"{pan['label']},  $Y_L^H={YL_iso:.2f}$,  {set_label}")
    # No set_box_aspect here on purpose: panel shape is paper_grid's job, and
    # pinning it fights PAPER_STYLE's aspect -- the width budgeted for the wider
    # shape then turns into a gutter between the columns.
    panel_label(ax, panel_lab)
    return _sc, F6_SIGMAS


_F4_SETS = [('Set A', quark_param_sets[0]), ('Set B', quark_param_sets[1])]

# square=False: let each panel fill its slot so leftover width becomes panel
# instead of gutter -- the ~0.6" between the columns is the right column's
# y-label and tick numbers, which is ink, not waste.
fig, ((axA, axB), (axC, axD)) = paper_grid('2x2', mode='double',
                                           placeholder=False, square=False,
                                           **PAPER_STYLE)
(_nameA, _setA), (_nameB, _setB) = _F4_SETS
_scA, _sgsA = _draw_f4_panel(axA, _setA, PNS_T0,   '(a)', _nameA)
_draw_f4_panel(axB, _setA, PNS_TMAX, '(b)', _nameA)
_scB, _sgsB = _draw_f4_panel(axC, _setB, PNS_T0,   '(c)', _nameB)
_draw_f4_panel(axD, _setB, PNS_TMAX, '(d)', _nameB)

# Legends: phase (line style) on (a); the per-row sigma legend on the RIGHT
# panel of each row, because the sigma palette differs between rows.
axA.legend(handles=[Line2D([], [], color='k', ls=PHASE_LS[p], lw=PHASE_LW[p],
                           alpha=PHASE_ALPHA[p], label=PHASE_LBL[p])
                    for p in PHASE_LS], loc='upper right')
for _ax, _sc, _sgs in ((axB, _scA, _sgsA), (axD, _scB, _sgsB)):
    _ax.legend(handles=[Line2D([], [], color=_sc[sg], ls='-', label=f'{int(sg)}')
                        for sg in _sgs],
               loc='upper right', title=r'$\sigma\;[\mathrm{MeV\,fm^{-2}}]$')

_d4 = []
for _setname, _pset in _F4_SETS:
    _tag = q_tag(_pset)
    for _pan in (PNS_T0, PNS_TMAX):
        for sg in F6_SIGMAS_BY_SET.get(_tag, FIG_SIGMAS['fig4_setA']):
            for ph in PHASE_LS:
                stem = nuc_stem(_tag, FN_FLAVOR, FN_CHARGE, ph, sg)
                if stem not in nuc_sets:
                    continue
                _grids = nuc_sets[stem].hadronic_grids
                _iYLp = int(np.argmin(np.abs(_grids['Y_L_H'] - _pan['YLH'])))
                _res = compute_nucleation_density(nuc_sets[stem],
                                                  tau_target=FN_TAU, scan='n_B')
                _nB, _Tn = nucleation_curve(_res, _iYLp)
                _mk = np.isfinite(_nB) & np.isfinite(_Tn)
                for _n, _t in zip(_nB[_mk] / n_sat, _Tn[_mk]):
                    _d4.append((_setname, _tag, _pan['label'], sg, ph,
                                float(_n), float(_t)))
pd.DataFrame(_d4, columns=['set', 'quark_tag', 'snapshot', 'sigma', 'phase',
                           'nBH_over_n0', 'T_nuc_MeV']).to_csv(
    FIG_D / f'fig4_Tnuc_{xsd_tag}.csv', index=False)
save_paper_figure(fig, f'paper_fig4_Tnuc_{xsd_tag}')
plt.show()

# %% [markdown]
# ## IV.5 — Figure 5: $\sigma_{\rm crit}$ over the quark parameter plane
#
# **The central result.** For every $(B^{1/4},\Delta_0)$ at fixed $\alpha_s$, the
# surface tension at which a proto-neutron star nucleates within
# $\tau_{\rm target}$ — star-wide, i.e. the maximum over `N_SHELLS` shells, not
# the central value.
#
# Layers, each with its own `F8_*` toggle at the top of the cell:
#
# * **heatmap** — $\sigma_{\rm crit}$; cells excluded by a filter are left white;
# * **white contours** — iso-$\sigma_{\rm crit}$, so a value can be read off;
# * **light-grey dashed** — iso-$M_{\max}$ of the resulting quark star;
# * **coloured outlines + labels** — *why* each excluded region is excluded:
#   vermillion $M_{\max}<2M_\odot$, green unbound 3-flavour matter (Witten), grey
#   2-flavour bound, blue re-hadronizing, purple quasi-re-hadronizing.
#
# A star converts if its $\sigma$ lies below the local value.

# %%
# =============================================================================
#  Figure 5.  Excluded regions are drawn as coloured BOUNDARY outlines (not
#  fills, which would hide the map underneath); the viable region carries the
#  sigma_crit heatmap with iso-sigma_crit and iso-M_max contours.
# =============================================================================
# ---- layer toggles: set any False to drop that layer -----------------------
# Which alpha_s slices to draw. The paper shows two, [1, 3] = pi/2 x 0.1 and
# pi/2 x 0.3: they bracket the perturbative correction, and the intermediate
# slices interpolate between them without adding anything a reader can use.
# Set [0, 1, 2, 3] for all four -- the layout switches to 2x2 on its own.
F8_SHOW          = [1, 3]
F8_HEATMAP       = True           # sigma_crit colour fill + colorbar
F8_ISO_SIGMA     = True           # iso-sigma_crit contour lines (white)
F8_ISO_MMAX      = True           # iso-M_max contour lines (light grey dashed)
F8_REJECT        = True           # excluded-region boundary outlines
F8_REJECT_LABELS = True           # labels on those regions
F8_REGIME        = False          # R* droplet-regime zones (costly; off)
F8_UNP_STABLE    = False          # vertical line: unpaired SQM abs. stability
# ----------------------------------------------------------------------------
set_paper_style()
_F8 = scan_main                              # the MT0_REF grid loaded in III.5
_al8, _B48, _D08 = _F8['alpha_slices'], _F8['B4_grid'], _F8['Delta0_grid']
_SIG8, _RS8, _MM8, _OK8 = (_F8['sig_crit'], _F8['reason'], _F8['M_max'],
                           _F8['cfl_ok'])
_vmin8 = float(np.nanmin(_SIG8))
_vmax8 = float(np.nanmax(_SIG8[np.isfinite(_SIG8)]))
# The smoke grid scans a single alpha_s, so a hard-coded slice list would index
# off the end. Clip rather than assert: fewer panels is still a valid figure.
F8_SHOW = [i for i in F8_SHOW if i < len(_al8)]

# Contour colours over viridis: only white / very light hues stay legible
# (blue, green and teal blend into the map; orange clashes with the vermillion
# M<2 edge). iso-sigma = white, iso-M_max = light grey.
_ISO_SIG_COL = ISO_SIGMA_COLOUR
_ISO_M_COL   = ISO_MASS_COLOUR

# The grid SHAPE follows len(F8_SHOW) and panels are taken with axes.flat, so
# adding or removing an alpha_s slice needs no edit here: 1-2 slices -> one row,
# 3-4 -> two rows. (Hard-coding '2x2' while indexing axes[0, c] is what broke on
# F8_SHOW = [0,1,2,3]: a 2x2 has only two columns, so c = 2 ran off the row.)
fig, axes = paper_grid('1x2' if len(F8_SHOW) <= 2 else '2x2', mode='double',
                       placeholder=False, fontsize=11, labelsize=11,
                       legendsize=9, aspect=1)         # square: it is a plane
for _ax in axes.flat[len(F8_SHOW):]:
    _ax.set_visible(False)
pcm = None
for _c, _ia in enumerate(F8_SHOW):
    ax = axes.flat[_c]
    _sa, _ra, _ma, _ok = _SIG8[_ia], _RS8[_ia], _MM8[_ia], _OK8[_ia]
    # (1) sigma_crit heatmap. Excluded cells are NaN and left WHITE here -- the
    #     coloured outlines say why they are excluded, so grey would only add
    #     ink. (The diverging maps in V.2 grey them instead, because there white
    #     is a data value.)
    if F8_HEATMAP:
        pcm = ax.pcolormesh(_B48, _D08, np.ma.masked_invalid(_sa),
                            cmap='viridis', vmin=_vmin8, vmax=_vmax8,
                            shading='nearest', zorder=2)
    # (2) iso-M_max, over the mass-relevant cells only (viable + the M<2 band).
    #     Levels start at 2.2: the 2 M_sun edge itself is the vermillion
    #     rejection boundary and drawing both would double the line.
    if F8_ISO_MMAX:
        _mm = np.where(_ok | (_ra == REASON_CODE['mmax']), _ma, np.nan)
        _lv = [round(x, 1) for x in np.arange(2.2, 4.001, 0.2)
               if np.isfinite(_mm).any()
               and np.nanmin(_mm) <= round(x, 1) <= np.nanmax(_mm)]
        if _lv:
            _cm = ax.contour(_B48, _D08, _mm, levels=_lv, colors=[_ISO_M_COL],
                             linewidths=0.9, linestyles='--', alpha=1.0,
                             zorder=2.5)               # below the iso-sigma set
            ax.clabel(_cm, fmt='%.1f', fontsize=7.5, inline=True)
    # (3) iso-sigma_crit
    if F8_ISO_SIGMA:
        _cs = ax.contour(_B48, _D08, _sa, levels=[50, 100, 150, 200, 250],
                         colors=[_ISO_SIG_COL], linewidths=0.8, alpha=0.9,
                         zorder=3)
        ax.clabel(_cs, fmt='%.0f', fontsize=9, inline=True)
    # (4) excluded-region outlines + labels
    if F8_REJECT:
        reject_outlines(ax, _B48, _D08, _ra, labels=F8_REJECT_LABELS,
                        panel_index=_c)
    # (5) unpaired-SQM absolute-stability threshold: left of it unpaired matter
    #     is absolutely stable, which would make the CFL story moot there.
    if F8_UNP_STABLE:
        _bstab = B4_at_energy(
            lambda a, b: energy_per_baryon_at_P0(a, b, filt_cfg),
            float(_al8[_ia]), float(_B48.min()), float(_B48.max()))
        if np.isfinite(_bstab):
            ax.axvline(_bstab, color='k', lw=1.6, ls=(0, (5, 2)), zorder=8)
            ax.text(_bstab, _D08.max(), r'unp. SQM abs. stable ', color='k',
                    fontsize=8.5, ha='right', va='top', rotation=90, zorder=8)
    ax.set_title(rf"$\alpha_s=\pi/2\times{_al8[_ia]/(np.pi/2):.1f}$")
    ax.set_xlabel(r'$B^{1/4}$ [MeV]')
    ax.set_ylabel(r'$\Delta_0$ [MeV]')          # every panel: sharey is off
    ax.set_xlim(_B48.min(), _B48.max()); ax.set_ylim(0.1, _D08.max())
    panel_label(ax, f"({chr(97 + _c)})")
if F8_HEATMAP and pcm is not None:
    # Colorbar exactly as tall as the (square) panels: an inset on the last
    # panel at full axes height. Attaching it to the figure instead would
    # stretch it over the whole figure height and desync from the panels.
    _cax = axes[0, -1].inset_axes([1.05, 0.0, 0.05, 1.0])
    fig.colorbar(pcm, cax=_cax, label=r'$\sigma_{\rm crit}$ [MeV/fm$^2$]')

# Flat CSV of the plane, so the map can be re-read without the .npz.
_ii, _jj = np.meshgrid(np.arange(len(_D08)), np.arange(len(_B48)), indexing='ij')
pd.concat([pd.DataFrame({
    'alpha_s': _al8[_ia], 'B4_MeV': _B48[_jj].ravel(),
    'Delta0_MeV': _D08[_ii].ravel(),
    'sigma_crit_star': _SIG8[_ia].ravel(), 'M_max_Msun': _MM8[_ia].ravel(),
    'reject_reason_code': _RS8[_ia].ravel()}) for _ia in F8_SHOW],
    ignore_index=True).to_csv(
        FIG_D / f'fig5_sigmacrit_map_{xsd_tag}.csv', index=False)
save_paper_figure(fig, f'paper_fig5_sigcrit_map_isolines_{xsd_tag}')
plt.show()

# %% [markdown]
# ## IV.6 — Appendix A: the electric-charge prescription
#
# How the droplet's net charge is treated changes the barrier, and this appendix
# is about how much.
#
# **(a)** $W(R)$ at the reference PNS centre under five prescriptions. **LCN**
# forces local neutrality (no Coulomb cost, but an unphysically constrained
# composition) and **GCN** allows global neutrality with the cost ignored: those
# two bracket the truth. **GCN + Coulomb** adds the *unscreened* $\propto R^5$
# term, which overshoots and destroys the barrier peak entirely. **Minimization**
# (`coulomb_minimize`, the paper's scheme) and **Debye screening** sit inside the
# bracket and on top of each other — that agreement is the point.
#
# **(b)** the same prescriptions through the observable that matters, the
# nucleation curve $T_{\rm nuc}(n_B^H)$, at two surface tensions. The spread is
# far smaller than in (a), because $W_*\sim\sigma^{2\text{–}3}$ compresses the
# barrier uncertainty — and it stays small at both $\sigma$.

# %%
# =============================================================================
#  Appendix A.  Electric-charge prescription: barrier vs observable.
# =============================================================================
APP_SET = quark_param_sets[0]
APP_TAG = q_tag(APP_SET)
APP_PAR = quark_params(alpha=APP_SET['alpha'], B4=APP_SET['B4'],
                       m_s=APP_SET['m_s'])
APP_SIG = FIG_SIGMAS['appA_charge'][0]           # sigma of both panels
APP_MT0 = MT0_REF                                # reference PNS
APP_YL  = PNS_TMAX['YLH']                        # the t_Tmax snapshot
APP_PH  = 'unpaired'                             # droplet phase of panel (b)

# The reference PNS centre, taken from the star match rather than hardcoded, so
# it can never drift from the star Figs. 3 and 5 use.
APP_NB, APP_T, APP_HPT = central_state(APP_MT0, star_scan)
print(f"reference PNS centre: n_B = {APP_NB/n_sat:.2f} n_sat, T = {APP_T:.2f} MeV")

# Where panel (a) is evaluated. (None, None) falls back to the centre above; a
# round (n_B, T) near it reads better in a caption and none of the appendix's
# conclusions depend on the exact point.
APPA_NB, APPA_T = 3.5 * n_sat, 30.0
if APPA_NB is None or APPA_T is None:
    APPA_NB, APPA_T = APP_NB, APP_T
print(f"panel (a) evaluated at: n_B = {APPA_NB/n_sat:.2f} n_sat, "
      f"T = {APPA_T:.2f} MeV")

# (charge mode, label, colour, linestyle). Only lcn/gcn/coulomb_minimize have
# saved nucleation tables, so only those appear in panel (b). NOTE: screening
# draws on top of and hides minimization -- they agree to <1%, which is the
# result; give screening a dashed style here if you want to see both.
_APP_MODES = [
    ('lcn',              'LCN',          OKAB_CAT[0], '-'),
    ('gcn',              'GCN',          OKAB_CAT[1], '-'),
    ('gcn_coulomb',      'GCN + Coul.',  OKAB_CAT[3], '-'),
    ('coulomb_minimize', 'minimization', OKAB_CAT[2], '-'),
    ('screening',        'screening',    OKAB_CAT[4], '-'),
]
_APP_TABLE_MODES = ['lcn', 'gcn', 'coulomb_minimize']

fig, ((axA, axB),) = paper_grid('1x2', mode='double', placeholder=False,
                                square=False,
                                **(PAPER_STYLE | {'aspect': 1.15}))

# ---- (a) W(R) ---------------------------------------------------------------
APP_Rg = np.linspace(0.02, 12.0, 500)            # droplet radius grid [fm]
_dA, _lamD = [], np.nan
for _ch, _lbl, _col, _ls in _APP_MODES:
    _eb = compute_energy_barrier(
        H['trapped'], APPA_NB, APPA_T, APP_SIG, electric_charge_mode=_ch,
        params=APP_PAR, flavor_mode='saddlepoint', quark_phase='unpaired',
        Y_L_H=APP_YL, R_values=APP_Rg)
    axA.plot(APP_Rg, _eb.W, color=_col, ls=_ls, lw=1.7, label=_lbl)
    if _ch == 'screening':
        _lamD = float(_eb.lambda_D)
    if np.isfinite(_eb.W).any():
        # The critical droplet is the FIRST local maximum of W(R), not the
        # global one: for GCN+Coulomb the global max sits at the right edge of
        # the R grid (the R^5 runaway), which is not a critical point at all.
        _dW = np.diff(_eb.W)
        _k = int(np.argmax(_dW < 0)) if (_dW < 0).any() else int(np.nanargmax(_eb.W))
        axA.plot(APP_Rg[_k], _eb.W[_k], 'o', ms=5, color=_col, mfc='white',
                 zorder=5)
        print(f"  {_lbl:14s} R_* = {APP_Rg[_k]:.2f} fm, "
              f"W_*/T = {_eb.W[_k]/APPA_T:.1f}")
    _dA += [(_lbl, float(r), float(w)) for r, w in zip(APP_Rg, _eb.W)]
axA.axhline(0, color='0.6', lw=0.7, zorder=0)
# Fixed frame showing the barrier region only. The GCN+Coulomb curve dips below
# zero past its peak and then runs off the top as R^5 (~ +1e5 MeV by R = 12 fm);
# both excursions are outside the frame on purpose.
axA.set_xlim(0, 8.5); axA.set_ylim(0, 10000)
axA.set_xlabel(r'$R$ [fm]'); axA.set_ylabel(r'$W$ [MeV]')
axA.legend(loc='upper left')
panel_label(axA, '(a)', 'upper right')

# ---- (b) T_nuc(n_B^H) from the saved tables --------------------------------
# TWO surface tensions, so the reader sees the prescription spread is small NOT
# just at one sigma: colour = prescription, line style = sigma.
APPA_SIGMAS = FIG_SIGMAS['appA_charge']
_SIG_LS = {sg: ls for sg, ls in zip(APPA_SIGMAS, ((0, (5, 2)), '-'))}

_dB, _sig_anchor = [], {}
for _sg in APPA_SIGMAS:
    for _ch, _lbl, _col, _ls in _APP_MODES:
        if _ch not in _APP_TABLE_MODES:
            continue
        stem = nuc_stem(APP_TAG, 'saddlepoint', _ch, APP_PH, _sg)
        if stem not in nuc_sets:
            print(f"  [skip] missing table {stem}")
            continue
        _grids = nuc_sets[stem].hadronic_grids
        _iYL = int(np.argmin(np.abs(_grids['Y_L_H'] - APP_YL)))
        _res = compute_nucleation_density(nuc_sets[stem],
                                          tau_target=TAU_TARGET, scan='n_B')
        _nB, _Tn = nucleation_curve(_res, _iYL)
        _m = np.isfinite(_nB) & np.isfinite(_Tn)
        axB.plot(_nB[_m] / n_sat, _Tn[_m], color=_col, ls=_SIG_LS[_sg], lw=1.7)
        _sig_anchor.setdefault(_sg, []).append((_nB[_m] / n_sat, _Tn[_m]))
        _dB += [(_lbl, _sg, float(n), float(t))
                for n, t in zip(_nB[_m] / n_sat, _Tn[_m])]

# sigma is written NEXT TO its group of curves rather than in the legend: the
# three prescriptions at one sigma sit close together, so one label does for all.
_xa = 3.0                                     # n_B/n_sat where the labels sit
for _sg, _curves in _sig_anchor.items():
    _ys = [float(np.interp(_xa, _x, _y)) for _x, _y in _curves]
    _above = _sg > min(_sig_anchor)           # lowest sigma labelled BELOW
    axB.text(_xa, (max(_ys) + 2.5) if _above else (min(_ys) - 3.0),
             rf'$\sigma={int(_sg)}$ MeV fm$^{{-2}}$', fontsize=9,
             va='bottom' if _above else 'top', rotation=-10,
             rotation_mode='anchor')

# The stellar track stops at the central density of the HEAVIEST PNS on the
# sequence -- beyond that no star exists. Where the track lies ABOVE a T_nuc
# curve, that star nucleates.
_tov_tr = nearest_trapped_sequence(tov_trapped, APP_YL, PNS_TMAX['S'])
_nBc_max = float(_tov_tr[int(np.argmax(_tov_tr[:, TOV_COL['M']])),
                         TOV_COL['n_Bc']])
print(f"isentrope drawn to n_B(M_PNS^max) = {_nBc_max/n_sat:.2f} n_sat "
      f"(M_PNS^max = {_tov_tr[:, TOV_COL['M']].max():.2f} Msun)")

# The x-axis stops where the stellar track does; the top is 95 MeV so the
# sigma = 150 curves are not cropped at low density.
axB.set_xlim(1, np.ceil(_nBc_max / n_sat)); axB.set_ylim(1, 95)
axB.set_xlabel(r'$n_B^H / n_{\rm sat}$'); axB.set_ylabel(r'$T_{\rm nuc}$ [MeV]')
axB.legend(handles=[Line2D([], [], color=_c, label=_l)
                    for _ch, _l, _c, _ in _APP_MODES
                    if _ch in _APP_TABLE_MODES], loc='upper right')
panel_label(axB, '(b)')

pd.DataFrame(_dA, columns=['prescription', 'R_fm', 'W_MeV']).to_csv(
    FIG_D / f'appA_WR_charge_{xsd_tag}.csv', index=False)
pd.DataFrame(_dB, columns=['prescription', 'sigma', 'nBH_over_n0',
                           'T_nuc_MeV']).to_csv(
    FIG_D / f'appA_Tnuc_charge_{xsd_tag}.csv', index=False)
save_paper_figure(fig, f'paper_appA_charge_prescriptions_{xsd_tag}')
plt.show()
print(f"lambda_D = {_lamD:.1f} fm at the reference centre")

# %% [markdown]
# ## IV.7 — Appendix B: frozen flavour vs saddle-point composition
#
# **(a)** $T_{\rm nuc}(n_B^H)$ for **frozen flavour** — the droplet inherits the
# hadronic composition, $Y_C^{Q}=Y_C^{H}$ and $Y_S^{Q}=Y_S^{H}$, because the weak
# interaction is too slow to act during the fluctuation — against **saddle
# point**, where the composition relaxes to minimise the barrier. The frozen
# curve sits far above the others: it needs $T\sim70$ MeV where the saddle-point
# ones need $\sim35$.
#
# **(b)** why. The composition of the critical droplet itself at $T=30$ MeV. The
# saddle-point droplet goes straight to $Y_S\simeq0.9$ — nearly one strange quark
# per baryon, which is what makes it bulk-favoured — while the frozen droplet is
# pinned to the small hadronic $Y_S^H$. A deconfined droplet is bulk-favoured
# only if it is strange enough, and in the frozen limit it can only inherit the
# strangeness the bulk hadronic phase already has, so it cannot nucleate until
# hyperons are abundant.
#
# Both prescriptions are compared under LCN, because frozen flavour only has a
# meaning under local neutrality.

# %%
# =============================================================================
#  Appendix B.  Frozen-flavour vs saddle-point composition (LCN).
# =============================================================================
APPB_SIG = FIG_SIGMAS['appB_flavour'][0]   # sigma of panel (a) [MeV/fm^2]
APPB_SET = quark_param_sets[0]
APPB_CH  = 'lcn'                           # frozen exists for LCN only
APPB_T   = 30.0                            # temperature of panel (b) [MeV]
APPB_NB2 = np.linspace(1.0, 10.0, 70) * n_sat        # density grid of panel (b)

# (flavour, phase, label, colour, linestyle): colour = composition prescription,
# line style = droplet phase, the same convention as Figs. 1 and 4.
APPB_CURVES = [
    ('saddlepoint', 'unpaired', 'saddle point, unpaired', OKAB_CAT[0], ':'),
    ('saddlepoint', 'cfl',      'saddle point, CFL',      OKAB_CAT[0], '--'),
    ('saddlepoint', 'unpCFL',   'saddle point, unpCFL',   OKAB_CAT[0], '-'),
    ('frozen',      'unpaired', 'frozen, unpaired',       OKAB_CAT[3], ':'),
]
# Species in panel (b). Y_e and Y_nu are not drawn: under LCN Y_e = Y_C
# identically, and Y_nu = 0 because no neutrinos are trapped inside the droplet.
APPB_SPECIES = [('Y_C', r'$Y_C$', OKAB_CAT[3]),
                ('Y_S', r'$Y_S$', OKAB_CAT[0])]

fig, ((axA, axB),) = paper_grid('1x2', mode='double', placeholder=False,
                                square=False,
                                **(PAPER_STYLE | {'aspect': 1.15}))

# ---- (a) T_nuc(n_B^H) -------------------------------------------------------
_tag = q_tag(APPB_SET)
_T_CFL = float(T_critical(APPB_SET['Delta0']))
_dB2 = []
for _fl, _ph, _lbl, _col, _ls in APPB_CURVES:
    stem = nuc_stem(_tag, _fl, APPB_CH, _ph, APPB_SIG)
    if stem not in nuc_sets:
        print(f"  [skip] missing table {stem}")
        continue
    _grids = nuc_sets[stem].hadronic_grids
    _iYL = int(np.argmin(np.abs(_grids['Y_L_H'] - APP_YL)))
    _res = compute_nucleation_density(nuc_sets[stem], tau_target=TAU_TARGET,
                                      scan='n_B')
    _nB, _Tn = nucleation_curve(_res, _iYL)
    _m = np.isfinite(_nB) & np.isfinite(_Tn)
    if _ph == 'cfl':
        _m &= (_Tn <= _T_CFL)                       # the gap vanishes above Tc
    # PHASE_ALPHA fades the non-emphasised phases WITHIN the saddle-point
    # family; the frozen curve is the comparison this panel is about, so it
    # stays fully opaque.
    axA.plot(_nB[_m] / n_sat, _Tn[_m], color=_col, ls=_ls, lw=PHASE_LW[_ph],
             alpha=(1.0 if _fl == 'frozen' else PHASE_ALPHA[_ph]), label=_lbl)
    print(f"  {_lbl:24s}: {int(_m.sum())} points"
          + (f", n_B >= {_nB[_m].min()/n_sat:.2f} n_sat" if _m.any()
             else " (no nucleation)"))
    _dB2 += [(_fl, _ph, float(n), float(t))
             for n, t in zip(_nB[_m] / n_sat, _Tn[_m])]
axA.set_xlim(1, 10); axA.set_ylim(5, 80)
axA.set_xlabel(r'$n_B^H / n_{\rm sat}$'); axA.set_ylabel(r'$T_{\rm nuc}$ [MeV]')
# The legend lives in the band between the frozen curve (top of the panel) and
# the saddle-point ones (middle); anchored in axes coords so it stays there.
axA.legend(loc='upper left', bbox_to_anchor=(0.02, 0.90), fontsize=8)
panel_label(axA, '(a)')

# ---- (b) critical-droplet composition --------------------------------------
# The composition is sigma-independent under LCN, so the sigma passed here only
# fixes R_c (unused): these Y_i enter W(R) at every sigma. Q* is returned even
# where no critical droplet exists (Delta_f >= 0), which is the frozen case over
# most of this range -- and showing that is the point of the panel.
_comp = {}
for _fl in ('saddlepoint', 'frozen'):
    _rows = []
    for _nb in APPB_NB2:
        _hp = hadronic_point(H['trapped'], _nb, APP_YL, APPB_T)
        _Qs = critical_droplet_pt(APPB_SIG, _hp, APPB_T, _fl, APPB_CH,
                                  'unpaired', {}, APP_PAR,
                                  APPB_SET['Delta0'], nuc_cfg)[2]
        _rows.append([np.nan if _Qs is None else float(getattr(_Qs, _y))
                      for _y, *_ in APPB_SPECIES])
    _comp[_fl] = np.array(_rows)                    # (n_density, n_species)

for _j, (_y, _ylbl, _col) in enumerate(APPB_SPECIES):
    axB.plot(APPB_NB2 / n_sat, _comp['saddlepoint'][:, _j], color=_col, ls='-',
             lw=1.8)
    axB.plot(APPB_NB2 / n_sat, _comp['frozen'][:, _j], color=_col, ls='--',
             lw=1.4)
axB.set_xlim(1, 10); axB.set_ylim(-0.03, 1.0)
axB.set_xlabel(r'$n_B^H / n_{\rm sat}$'); axB.set_ylabel(r'$Y_i^{Q*}$')
axB.legend(handles=[Line2D([], [], color=_c, ls='-', lw=1.8, label=_lb)
                    for _y, _lb, _c in APPB_SPECIES]
                   + [Line2D([], [], color='k', ls='-',  lw=1.8,
                             label='saddle point'),
                      Line2D([], [], color='k', ls='--', lw=1.4,
                             label='frozen')],
           loc='center left', ncol=2)
panel_label(axB, '(b)')

pd.DataFrame(_dB2, columns=['flavor_mode', 'phase', 'nBH_over_n0',
                            'T_nuc_MeV']).to_csv(
    FIG_D / f'appB_Tnuc_{xsd_tag}.csv', index=False)
pd.DataFrame({'nBH_over_n0': APPB_NB2 / n_sat,
              **{f'{_y}_{_fl}': _comp[_fl][:, _j]
                 for _fl in _comp for _j, (_y, *_r) in enumerate(APPB_SPECIES)}}
             ).to_csv(FIG_D / f'appB_composition_{xsd_tag}.csv', index=False)
save_paper_figure(fig, f'paper_appB_frozen_vs_saddlepoint_{xsd_tag}')
plt.show()

# %% [markdown]
# ## IV.8 — Outcomes table: neutron star, quark star, or black hole
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
# * **NS** — it survives both and cools into an ordinary neutron star.
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
    _pars = quark_params(alpha=_p['alpha'], B4=_p['B4'], m_s=_p['m_s'])
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

outcomes.to_csv(FIG_D / f'table_outcomes_{xsd_tag}.csv', index=False)
(FIG_D / f'table_outcomes_{xsd_tag}.tex').write_text(
    to_latex_tabular(outcomes[_show]))
print(f"  saved table_outcomes_{xsd_tag}.csv / .tex")

# %% [markdown]
# # Part V — Supplementary material
#
# Same style, same data, same machinery as Part IV. These support the paper's
# claims without being headline figures, and they go to
# `figures/supplementary/`.
#
# | Section | Question it answers |
# |---------|---------------------|
# | V.1 | What do the *viable* parameter sets look like as actual stars? |
# | V.2 | How much does $\sigma_{\rm crit}$ move when an input changes? |
# | V.3 | Is $W_*/T$ an invariant across the viable region? |
# | V.4 | What happens with no CFL pairing at all? |
# | V.5 | Does the star-wide vs centre-only definition matter? |
# | V.6 | Which term of $W(R)$ sets the critical radius? |
# | V.7 | Can a droplet tunnel instead of being thermally activated? |

# %% [markdown]
# ## V.1 — The viable region as stars: does $\sigma_{\rm crit}$ show up in an observable?
#
# Fig. 5 shows $\sigma_{\rm crit}$ over an abstract parameter plane. The question
# a reader will ask next is whether that plane connects to anything measurable —
# and this figure is built to answer it, not just to look at the bundle.
#
# **(a, b)** every accepted cell replayed into a mass–radius curve and its
# equation of state, binned by $\sigma_{\rm crit}$ into equal-**count** groups and
# drawn as a median with a p16–p84 envelope. Equal-count rather than equal-width
# because $\sigma_{\rm crit}$ is far from uniform over the plane, and a band over
# three cells next to one over three hundred invites the wrong comparison. Each
# band takes its colour at its median $\sigma_{\rm crit}$ on the **Fig. 5 scale**,
# so a colour means the same surface tension in both figures. The two parameter
# sets carried through the paper are drawn on top in their Fig. 2 colours.
#
# **(c, d)** the point of the figure: $\sigma_{\rm crit}$ against a stellar
# observable, one point per parameter set, with a running median and a Spearman
# rank correlation. Colour is no longer carrying the quantity — it is on an axis,
# where a correlation can actually be read. If $\rho$ is large, then measuring
# that observable **bounds the quark–hadron surface tension**; if it is near
# zero, the two are independent and no radius measurement will help.
#
# Rank correlation rather than Pearson: the relation need not be linear, and
# $\rho$ only asks whether it is monotone.

# %%
# =============================================================================
#  V.1 knobs.
# =============================================================================
V1_NBINS      = 5                 # equal-count sigma_crit bins drawn as bands
V1_MAX_CURVES = 60 if REDUCED_GRID else 400   # even subsample of accepted cells
V1_RDEF       = 'R_1.4'           # radius in (c): 'R_1.4' | 'R_Mmax' | 'R_max'
                                  #   R_1.4 is what NICER-type measurements
                                  #   pin; R_max runs to ~22 km in the
                                  #   low-B4 corner and stretches the panel.
V1_MED_BINS   = 8                 # bins behind the running median in (c), (d)
# Keep only cells whose sigma_crit and M_max fall in these windows. Both cuts are
# applied BEFORE the replay -- replay_accepted treats a non-finite sigma_crit as
# "not accepted" -- so a narrow window is proportionally CHEAPER, not just sparser.
V1_SIG_RANGE  = (-np.inf, np.inf)  # sigma_crit window [MeV/fm^2]
V1_MMAX_RANGE = (-np.inf, np.inf)  # M_max window [M_sun]

_v1_keep = (np.isfinite(SIG) &
            (SIG >= V1_SIG_RANGE[0]) & (SIG <= V1_SIG_RANGE[1]) &
            (MMAX >= V1_MMAX_RANGE[0]) & (MMAX <= V1_MMAX_RANGE[1]))
print(f"V.1: {int(_v1_keep.sum())}/{int(np.isfinite(SIG).sum())} viable cells "
      f"kept (sigma_crit in {V1_SIG_RANGE}, M_max in {V1_MMAX_RANGE})")

# with_params: (c) and (d) colour each point by its Delta_0, which is what
# turns "scatter" into "structure" -- see the panel comment below.
_curves3 = replay_accepted(np.where(_v1_keep, SIG, np.nan), AL, B4G, D0G,
                           filt_cfg, max_curves=V1_MAX_CURVES,
                           n_jobs=SCAN['n_jobs'], verbose=True,
                           with_params=True)
# Draw low sigma_crit first so high sigma_crit ends up on top; without a sort
# that ordering is whatever the grid cells happened to come back in.
_curves3 = sorted(_curves3, key=lambda t: t[1])
_curves = [(c, sc) for c, sc, _ in _curves3]     # the band helpers take pairs
_D0_of_curve = np.array([p[2] for _, _, p in _curves3])


def _running_median(x, y, n_bins):
    """Median of y in equal-count bins of x, plus the p16-p84 spread.

    A scatter of a few hundred points shows whether a trend EXISTS; the running
    median shows its shape. Equal-count bins keep every point of the curve
    backed by the same number of cells, so a sparse tail cannot masquerade as
    structure. Bins holding fewer than 3 points are dropped for that reason.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2 * n_bins:
        return np.array([]), np.array([]), np.array([]), np.array([])
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    xc, med, lo, hi = [], [], [], []
    for k in range(n_bins):
        m = (x >= edges[k]) & (x < edges[k + 1])
        if m.sum() < 3:
            continue
        xc.append(np.median(x[m]))
        med.append(np.median(y[m]))
        lo.append(np.percentile(y[m], 16))
        hi.append(np.percentile(y[m], 84))
    return (np.array(xc), np.array(med), np.array(lo), np.array(hi))


def _corr_panel(ax, x, y, xlabel, cvals):
    """sigma_crit against one observable: points, running median, Spearman rho.

    Points are coloured by the pairing gap, because the vertical spread at fixed
    R or M_max is not noise -- it is Delta_0, and colouring by it turns a cloud
    into two questions answered at once: does the observable track sigma_crit,
    and what moves a star off that track.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    sca = ax.scatter(x[ok], y[ok], c=cvals[ok], cmap='plasma', s=11, lw=0,
                     alpha=0.75, zorder=2)
    _xc, _md, _lo, _hi = _running_median(x[ok], y[ok], V1_MED_BINS)
    if _xc.size:
        ax.fill_between(_xc, _lo, _hi, color='0.35', alpha=0.16, lw=0, zorder=3)
        # White casing under the median so it stays legible over the points.
        ax.plot(_xc, _md, color='white', lw=3.4, zorder=4, solid_capstyle='round')
        ax.plot(_xc, _md, color='k', lw=1.8, zorder=5, solid_capstyle='round')
    if ok.sum() > 3:
        _rho, _p = spearmanr(x[ok], y[ok])
        # p is quoted as an upper bound below 1e-3: the exact value of a tiny
        # p-value over a few hundred grid cells is not meaningful, the sign and
        # size of rho are.
        _ptxt = 'p < 0.001' if _p < 1e-3 else f'p = {_p:.3f}'
        ax.text(0.97, 0.05, rf'$\rho = {_rho:+.2f}$' + f'\n{_ptxt}',
                transform=ax.transAxes, va='bottom', ha='right', fontsize=8.5,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.75',
                          alpha=0.92), zorder=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'$\sigma_{\rm crit}^{\rm star}$ [MeV/fm$^2$]')
    ax.margins(x=0.03)
    return sca


if len(_curves) >= 4:
    fig, ((axA, axB), (axC, axD)) = paper_grid(
        '2x2', mode='double', placeholder=False, **PAPER_STYLE)
    _norm = Normalize(SIG_VMIN, SIG_VMAX)      # == the Fig. 5 heatmap scale
    _cmap = plt.get_cmap('viridis')
    _bins = quantile_bins(_curves, V1_NBINS, _cmap, _norm)

    # --- (a) M-R -------------------------------------------------------------
    # R is interpolated at fixed M -- the single-valued direction on the stable
    # branch -- but plotted horizontally, hence swap=True.
    _Mg = np.linspace(0.6, 2.6, 60)
    for _lo, _hi, _members, _col in _bins:
        band(axA, resample_profiles(_members, 'M', 'R', _Mg, stable=True), _Mg,
             _col, swap=True)
    # The constraint blobs are context, not the subject: they go on WITHOUT
    # labels of either kind (inline names collided with the bands; legend
    # entries put five pulsars in a box titled "sigma_crit"). Fig. 2(a) is where
    # they are identified -- this panel only needs to show where the viable
    # stars fall relative to them.
    add_observational_constraints(axA, show_mass_bands=False,
                                  inline_labels=False)
    # The two sets the paper follows, so the reader can locate them in the bundle.
    for _k, _p in enumerate(quark_param_sets):
        _rc = replay_cfl(_p['alpha'], _p['B4'], _p['Delta0'], filt_cfg)
        if _rc is None:
            continue
        axA.plot(_rc['R'], _rc['M'], color=_QS_GREENS[_k % len(_QS_GREENS)],
                 lw=1.6, ls='--', zorder=6)
        axB.plot(_rc['mu'], _rc['P'],
                 color=_QS_GREENS[_k % len(_QS_GREENS)], lw=1.6, ls='--',
                 zorder=6)
    axA.set_xlabel(r'$R$ [km]'); axA.set_ylabel(r'$M$ [$M_\odot$]')
    axA.set_xlim(8, 14); axA.set_ylim(0.5, 2.8)
    # Explicit handles, so nothing the constraint overlay registers can leak
    # into a legend that is about sigma_crit.
    axA.legend(handles=([Line2D([], [], color=_c, lw=1.7,
                                label=rf'{_lo:.0f}-{_hi:.0f}')
                         for _lo, _hi, _, _c in _bins]
                        + [Line2D([], [], color=_QS_GREENS[_k % len(_QS_GREENS)],
                                  lw=1.6, ls='--',
                                  label=rf"$B^{{1/4}}\!=\!{_p['B4']:.0f}$, "
                                        rf"$\Delta_0\!=\!{_p['Delta0']:.0f}$")
                           for _k, _p in enumerate(quark_param_sets)]),
               loc='upper left', bbox_to_anchor=(0.0, 0.86), ncol=2,
               title=r'$\sigma_{\rm crit}$ [MeV/fm$^2$]',
               fontsize=PAPER_STYLE['legendsize'] - 2.5,
               title_fontsize=PAPER_STYLE['legendsize'] - 1.5, framealpha=0.92,
               handlelength=1.4, columnspacing=1.0, labelspacing=0.3,
               borderpad=0.35)
    panel_label(axA, '(a)')

    # --- (b) the equation of state behind those curves -----------------------
    # Same plane the no-rehadronization filter works in: a viable set must stay
    # above the hadronic comparator past their crossing.
    _mug = np.linspace(900, 1600, 80)
    for _lo, _hi, _members, _col in _bins:
        band(axB, resample_profiles(_members, 'mu', 'P', _mug), _mug, _col)
    axB.plot(filt_cfg.mu_B_H_sorted, filt_cfg.P_H_sorted, 'k-', lw=1.6,
             zorder=5, label=r'$P_H\ (T\approx0)$')
    axB.set_xlabel(r'$\mu_B$ [MeV]')
    axB.set_ylabel(r'$P$ [MeV fm$^{-3}$]')
    axB.set_xlim(_mug.min(), _mug.max())
    axB.legend(loc='lower right', fontsize=PAPER_STYLE['legendsize'] - 1)
    panel_label(axB, '(b)')

    # --- (c, d) sigma_crit against an observable -----------------------------
    _pts = summary_points(_curves, V1_RDEF)          # (M_max, R) per cell
    _sc = np.array([s for _, s in _curves])
    _RLAB = {'R_Mmax': r'$R(M_{\max})$', 'R_1.4': r'$R_{1.4}$',
             'R_max': r'$R_{\max}$'}
    _corr_panel(axC, _pts[:, 1], _sc, _RLAB[V1_RDEF] + ' [km]', _D0_of_curve)
    _sca = _corr_panel(axD, _pts[:, 0], _sc, r'$M_{\max}$ [$M_\odot$]',
                       _D0_of_curve)
    axD.axvline(2.0, color='0.5', ls=':', lw=1.0, zorder=1)   # the M_max filter
    # One colorbar for both correlation panels: Delta_0 means the same thing in
    # each, and a per-panel bar would invite reading them on different scales.
    fig.colorbar(_sca, ax=[axC, axD], fraction=0.046, pad=0.02,
                 label=r'$\Delta_0$ [MeV]')
    panel_label(axC, '(c)'); panel_label(axD, '(d)')

    pd.DataFrame({'M_max_Msun': _pts[:, 0], f'{V1_RDEF}_km': _pts[:, 1],
                  'sigma_crit_star': _sc}).to_csv(
        FIG_D / f'supp_viable_region_stars_{xsd_tag}.csv', index=False)
    save_paper_figure(fig, f'supp_viable_region_stars_{xsd_tag}', supp=True)
    plt.show()

    # The numbers the panels are claiming, printed so they can be quoted.
    for _lbl, _x in ((_RLAB[V1_RDEF], _pts[:, 1]), ('M_max', _pts[:, 0])):
        _ok = np.isfinite(_x) & np.isfinite(_sc)
        if _ok.sum() > 3:
            _r, _pv = spearmanr(_x[_ok], _sc[_ok])
            print(f"  sigma_crit vs {_lbl:12s}: Spearman rho = {_r:+.3f} "
                  f"(p = {_pv:.2e}, n = {int(_ok.sum())})")
else:
    print("too few accepted cells to draw the bundle "
          "(expected on the smoke grid); rerun with REDUCED_GRID=False")

# %% [markdown]
# ## V.2 — $\Delta\sigma_{\rm crit}$: sensitivity to $M_{T0}$ and to the phase
#
# How much does $\sigma_{\rm crit}(B^{1/4},\Delta_0)$ actually move when an
# *input* changes rather than the quark parameters? Two comparisons, the same
# machinery and the same plane as Fig. 5, showing a **signed difference**:
#
# * **(A)** $\sigma_{\rm crit}(M_{T0})-\sigma_{\rm crit}(1.4)$ — the threshold
#   mass sets which star must not yet have nucleated, so a heavier $M_{T0}$
#   probes a denser centre and generally needs a *larger* $\sigma$ to stay quiet.
# * **(B)** $\sigma_{\rm crit}({\rm unpCFL})-\sigma_{\rm crit}({\rm CFL})$ at
#   $M_{T0}=1.4$ — the droplet-phase choice. These coincide *exactly* wherever
#   the critical droplet is larger than the pairing coherence radius, so they can
#   differ only in the small-droplet corner.
#
# Colour is **diverging with a neutral midpoint and symmetric limits**, so white
# means "no change" and the hue is the sign — deliberately *not* Fig. 5's
# sequential viridis, which encodes magnitude. Grey means the cell is viable in
# one scan but not the other, so no difference exists.
#
# The statistics are printed twice, over all compared cells and over the cells
# that actually move: for (B) the two phases are bit-identical over most of the
# plane, so an all-cells median is 0 and says nothing about the size of the
# effect where there IS one.

# %%
# =============================================================================
#  Delta-sigma_crit maps + summary statistics. Everything is read from the
#  saved II.8 grids -- nothing is re-scanned here.
# =============================================================================
set_paper_style()
# ---- knobs -----------------------------------------------------------------
DSIG_REF_MT0 = MT0_REF                  # the baseline everything differences against
DSIG_MT0S    = [m for m in MT0_GRID if m != MT0_REF]
DSIG_PHASE   = MAIN_PHASE               # droplet phase used in comparison (A)
DSIG_PHASES  = ('unpCFL', 'cfl')        # the pair compared in (B)
DSIG_ALPHA   = 0                        # alpha_s slice index mapped in (A)
DSIG_VLIM    = None                     # colour limit; None = robust from data
DSIG_PCT     = 99                       # percentile of |Delta| setting it
DSIG_TOL     = 1e-3                     # |Delta| above this "actually differs"
# ----------------------------------------------------------------------------
_DSIG_AXES = ('alpha_slices', 'B4_grid', 'Delta0_grid')


def _dsig_diff(gA, gB):
    """Signed sigma_crit difference gA - gB on cells finite in BOTH, else NaN.

    Refuses to difference grids built on different axes -- the arrays would
    still subtract element-wise and hand back a silently meaningless map. The
    metadata check is deliberately wider than the axes: differencing an
    M_T0=1.17 grid against a 1.4 one is the POINT of panel (A), but doing it
    across a different m_s or tau would not be.
    """
    for k in _DSIG_AXES:
        if not np.array_equal(gA[k], gB[k]):
            raise ValueError(f"grids differ in '{k}' -- cannot be differenced "
                             f"cell-by-cell (re-scan both on the same grid)")
    for k in ('m_s', 'tau'):
        if not np.isclose(gA[k], gB[k]):
            raise ValueError(f"grids differ in '{k}' ({gA[k]} vs {gB[k]}) -- "
                             f"the difference would not be a sensitivity")
    A, B = gA['sig_crit'], gB['sig_crit']
    both = np.isfinite(A) & np.isfinite(B)
    return np.where(both, A - B, np.nan), B    # (difference, reference for %)


def _dsig_stats(D, ref, tol=DSIG_TOL):
    """Summary of a Delta-sigma_crit field, over all compared cells AND over the
    subset that actually moves (|Delta| > tol). Relative values are Delta/ref."""
    m = np.isfinite(D)
    d, r = D[m], ref[m]
    if d.size == 0:
        return None
    mv = np.abs(d) > tol
    s = dict(n=d.size, n_move=int(mv.sum()), frac=100.0 * mv.mean(),
             med=np.median(d), mean=d.mean(),
             p16=np.percentile(d, 16), p84=np.percentile(d, 84),
             lo=d.min(), hi=d.max())
    if mv.any():
        dm, rm = d[mv], r[mv]
        ok = np.isfinite(rm) & (rm != 0)
        s.update(med_m=np.median(dm), p16_m=np.percentile(dm, 16),
                 p84_m=np.percentile(dm, 84),
                 rel_m=(100.0 * np.median(dm[ok] / rm[ok])) if ok.any() else np.nan)
    return s


_DSIG_HDR = (f"{'comparison':<34}{'N':>6}{'move':>15}{'median':>9}"
             f"{'p16..p84 (moving)':>21}{'min..max':>18}{'rel':>8}")


def _dsig_row(label, s):
    """One formatted stats line. 'median' and the p16..p84 band are over the
    MOVING cells; min..max spans all compared cells."""
    if s is None:
        return f"{label:<34}{'-- no overlapping viable cells --':>65}"
    if not s.get('n_move'):
        return (f"{label:<34}{s['n']:>6}{'0 (0.0%)':>13}"
                f"{'identical everywhere':>56}")
    return (f"{label:<34}{s['n']:>6}{s['n_move']:>6} ({s['frac']:5.1f}%)"
            f"{s['med_m']:>+9.2f}{s['p16_m']:>+10.2f}..{s['p84_m']:<+10.2f}"
            f"{s['lo']:>+9.2f}..{s['hi']:<+8.2f}{s['rel_m']:>+7.1f}%")


def _dsig_panel(ax, D, B4, D0, vlim, title, *, lost=None, gained=None):
    """One difference panel in the Fig-5 plane.

    Three things are drawn on top of the difference itself, because a bare
    diverging map hides the most interesting part of a sensitivity test:

    * **lost / gained cells** (black / white outline) -- parameter sets that are
      viable in one scan but not the other. A raised M_T0 pushes cells out of
      viability entirely, and those cells have NO Delta to colour: they read as
      grey no-data unless the boundary is drawn.
    * **the two paper parameter sets**, so the reader can see whether the shift
      matters where it matters.
    * **the median and p16-p84 spread**, printed in the corner. It is the number
      the caption wants and the eye cannot get off a colour scale.

    Returns the mesh, for a colorbar.
    """
    pcm = diverging_map(ax, B4, D0, D, vlim)
    if lost is not None and lost.any():
        ax.contour(B4, D0, lost.astype(float), levels=[0.5], colors='k',
                   linewidths=1.3, zorder=5)
    if gained is not None and gained.any():
        ax.contour(B4, D0, gained.astype(float), levels=[0.5], colors='white',
                   linewidths=1.3, zorder=5)
    for _p in quark_param_sets:                     # the sets the paper follows
        ax.plot(_p['B4'], _p['Delta0'], marker='*', ms=11, color='yellow',
                mec='k', mew=0.6, ls='none', zorder=7)
    _f = D[np.isfinite(D)]
    if _f.size:
        _md, _lo, _hi = np.median(_f), *np.percentile(_f, [16, 84])
        ax.text(0.03, 0.03,
                rf'median ${_md:+.1f}$' '\n' rf'p16-p84 ${_lo:+.1f}\,..\,{_hi:+.1f}$',
                transform=ax.transAxes, va='bottom', ha='left', fontsize=7.5,
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.8',
                          alpha=0.85), zorder=8)
    ax.set_xlabel(r'$B^{1/4}$ [MeV]'); ax.set_ylabel(r'$\Delta_0$ [MeV]')
    ax.set_xlim(B4.min(), B4.max()); ax.set_ylim(0.1, D0.max())
    ax.set_title(title)
    return pcm


# ---- (A) M_T0 sensitivity ---------------------------------------------------
_ref = load_scan(DSIG_REF_MT0, phase=DSIG_PHASE)
_dA = []
if _ref is None:
    print(f"(A) skipped: no baseline grid at M_T0={DSIG_REF_MT0:g}")
else:
    _B4d, _D0d, _ald = _ref['B4_grid'], _ref['Delta0_grid'], _ref['alpha_slices']
    _ia = min(DSIG_ALPHA, len(_ald) - 1)
    _missA = []
    for _mt in DSIG_MT0S:
        _g = load_scan(_mt, phase=DSIG_PHASE)
        if _g is None:
            _missA.append(_mt)
            continue
        _D, _R = _dsig_diff(_g, _ref)
        _dA.append((_mt, _D, _R))
    if _missA:
        print(f"(A) missing grids for M_T0 = {_missA}: add them to MT0_GRID and "
              f"re-run Part II.8, then re-run this cell.")

if _dA:
    _vA = symmetric_vlim([_D[_ia] for _, _D, _ in _dA], pct=DSIG_PCT,
                         override=DSIG_VLIM)
    # Layout follows the panel COUNT: two comparisons in a 2x2 left two blank
    # quadrants and shrank the maps for nothing.
    fig, _axA = paper_grid('1x2' if len(_dA) <= 2 else '2x2', mode='double',
                           placeholder=False, square=False,
                           **(PAPER_STYLE | {'aspect': 1.0}))
    _axAf = list(_axA.flat)
    _pcm = None
    for _k, (_mt, _D, _) in enumerate(_dA[:len(_axAf)]):
        # A cell viable at the baseline but not at this M_T0 has no difference
        # to colour: outline it, or the accept region silently shrinking looks
        # the same as a cell that was never viable.
        _gk = load_scan(_mt, phase=DSIG_PHASE)
        _lost = np.isfinite(_ref['sig_crit'][_ia]) & ~np.isfinite(_gk['sig_crit'][_ia])
        _gain = ~np.isfinite(_ref['sig_crit'][_ia]) & np.isfinite(_gk['sig_crit'][_ia])
        _pcm = _dsig_panel(_axAf[_k], _D[_ia], _B4d, _D0d, _vA,
                           rf"$M_{{T0}}={_mt:g}\,M_\odot$",
                           lost=_lost, gained=_gain)
        panel_label(_axAf[_k], f"({chr(97 + _k)})")
    for _ax in _axAf[len(_dA):]:                    # unused panels of a 2x2
        _ax.set_visible(False)
    fig.colorbar(_pcm, ax=_axA,
                 label=(rf'$\sigma_{{\rm crit}}(M_{{T0}}) - '
                        rf'\sigma_{{\rm crit}}({DSIG_REF_MT0:g})$'
                        r' [MeV/fm$^2$]'), fraction=0.046, pad=0.02)
    fig.suptitle(rf"$\alpha_s=\pi/2\times{_ald[_ia]/(np.pi/2):.1f}$, "
                 rf"{DSIG_PHASE} — black outline: viable at "
                 rf"$M_{{T0}}={DSIG_REF_MT0:g}$ but not here; "
                 rf"stars: the two paper sets", fontsize=8)
    save_paper_figure(fig, f'dsigma_MT0_{xsd_tag}_{DSIG_PHASE}', supp=True)
    plt.show()

# ---- (B) droplet-phase sensitivity at the baseline M_T0 ---------------------
_pA, _pB = DSIG_PHASES
_gA, _gB = load_scan(DSIG_REF_MT0, phase=_pA), load_scan(DSIG_REF_MT0, phase=_pB)
_dB = None
if _gA is None or _gB is None:
    print(f"(B) skipped: need both phase grids at M_T0={DSIG_REF_MT0:g} "
          f"({_pA}: {'ok' if _gA is not None else 'MISSING'}, "
          f"{_pB}: {'ok' if _gB is not None else 'MISSING'})")
else:
    _DB, _RB = _dsig_diff(_gA, _gB)
    _dB = (_DB, _RB)
    _B4b, _D0b, _alb = _gA['B4_grid'], _gA['Delta0_grid'], _gA['alpha_slices']
    _shw = [i for i in F8_SHOW if i < len(_alb)]     # same alpha panels as Fig 5
    _vB = symmetric_vlim([_DB[i] for i in _shw], pct=DSIG_PCT,
                         override=DSIG_VLIM)
    fig, _axB = paper_grid('1x2', mode='double', placeholder=False,
                           square=False, **(PAPER_STYLE | {'aspect': 1.0}))
    for _c, _i in enumerate(_shw[:2]):
        _lostB = np.isfinite(_gA['sig_crit'][_i]) & ~np.isfinite(_gB['sig_crit'][_i])
        _gainB = ~np.isfinite(_gA['sig_crit'][_i]) & np.isfinite(_gB['sig_crit'][_i])
        _pcmB = _dsig_panel(_axB[0, _c], _DB[_i], _B4b, _D0b, _vB,
                            rf"$\alpha_s=\pi/2\times{_alb[_i]/(np.pi/2):.1f}$",
                            lost=_lostB, gained=_gainB)
        panel_label(_axB[0, _c], f"({chr(97 + _c)})")
    fig.colorbar(_pcmB, ax=_axB,
                 label=(rf'$\sigma_{{\rm crit}}$({_pA}) $-\ '
                        rf'\sigma_{{\rm crit}}$({_pB}) [MeV/fm$^2$]'),
                 fraction=0.046, pad=0.02)
    save_paper_figure(fig, f'dsigma_phase_{xsd_tag}_MT0{DSIG_REF_MT0:.2f}',
                      supp=True)
    plt.show()

# ---- the typical differences, printed ---------------------------------------
print('\n' + _DSIG_HDR)
print('-' * len(_DSIG_HDR))
for _mt, _D, _R in _dA:
    print(_dsig_row(f'MT0 {_mt:g} - {DSIG_REF_MT0:g}  ({DSIG_PHASE}, all alpha)',
                    _dsig_stats(_D, _R)))
    print(_dsig_row(f'   ... alpha slice {_ia} only',
                    _dsig_stats(_D[_ia], _R[_ia])))
if _dB is not None:
    print(_dsig_row(f'{_pA} - {_pB}  (MT0 {DSIG_REF_MT0:g}, all alpha)',
                    _dsig_stats(*_dB)))
    for _i in _shw:
        print(_dsig_row(f'   ... alpha = pi/2 x {_alb[_i]/(np.pi/2):.1f}',
                        _dsig_stats(_dB[0][_i], _dB[1][_i])))

# %% [markdown]
# ## V.3 — Is $W_*/T$ an invariant over the accessible plane?
#
# Classical nucleation theory fixes the *rate*, not the barrier:
# $\tau = 1/(V\Gamma)$ with $\Gamma=(\kappa\Omega_0/2\pi)e^{-W_*/T}$, so
# demanding $\tau=\tau_{\rm target}$ pins
#
# $$W_*/T \;=\; \ln\!\big(V\,\kappa\,\Omega_0\,\tau_{\rm target}/2\pi\big),$$
#
# which depends on the parametrization **only logarithmically, through the
# prefactor**. So at each cell's *own* $\sigma_{\rm crit}$ this map should come
# out nearly flat — that flatness is the claim, and the printed spread is the
# number to quote. Set `WT_SIGMA` to a fixed $\sigma$ instead and the same map
# becomes a genuine barrier landscape, where large $W_*/T$ means hard to
# nucleate.
#
# $W_*$ is read at the **deciding shell**, the one with the shortest $\tau$,
# because $\sigma_{\rm crit}$ is itself star-wide and that shell is not always
# the centre.

# %%
# =============================================================================
#  W*/T over the (B^1/4, Delta_0) plane, for every accessible cell of the Fig 5
#  grid. Same plane, same alpha slices, same masking; the quantity differs.
# =============================================================================
set_paper_style()
WT_SHOW  = [0, 2]      # alpha_s slice indices to DRAW (two panels, as Fig 5)
WT_SHOW  = [i for i in WT_SHOW if i < len(_al8)]        # see the note in IV.5
# Slices to SUMMARIZE, deliberately decoupled from WT_SHOW: Fig 5 has room for
# two panels, but a claim about "the whole accepted island" must be computed
# over every slice the sigma_crit range is quoted over, or the printed statistic
# silently describes part of it. None -> all slices.
WT_STATS = None
WT_SIGMA = None        # None -> each cell's OWN sigma_crit; a float -> fixed
WT_SHELLS = N_SHELLS   # 1 = centre only (fast); N_SHELLS = star-wide as sigma_crit
WT_CMAP  = 'magma'     # sequential, one hue -- deliberately NOT Fig 5's viridis
WT_CLIP  = (2, 98)     # robust colour limits [percentiles]

# The method must match the grid the sigma_crit values came from, or W* would be
# evaluated for a different droplet than the one that set sigma_crit.
WT_FLAVOR, WT_CHARGE, WT_PHASE = (str(_F8['flavor']), str(_F8['charge']),
                                  str(_F8['phase']))
_WT_MT0 = float(_F8['MT0'])
# Shells are hadronic (star match + isentrope) -> quark-independent -> built ONCE.
_wt_shells = star_shell_states(_WT_MT0, star_scan, WT_SHELLS,
                               nB_min=NB_SHELL_MIN)
print(f"W*/T map: {WT_PHASE}/{WT_FLAVOR}/{WT_CHARGE}, MT0={_WT_MT0:.2f}, "
      f"{len(_wt_shells)} shell(s) "
      f"n_B={_wt_shells[0][0].n_B:.3f}..{_wt_shells[-1][0].n_B:.3f} fm^-3, "
      f"sigma={'sigma_crit(cell)' if WT_SIGMA is None else f'{WT_SIGMA:g}'}")

# Every slice that is drawn, plus every slice that is summarized.
_wt_slices = sorted(set(WT_SHOW) | set(range(len(_al8)) if WT_STATS is None
                                       else WT_STATS))
_WT_path = DIRS['sigma_crit'] / f'WoverT_{xsd_tag}_MT0{_WT_MT0:.2f}.npz'
if _WT_path.exists():
    with np.load(_WT_path) as _z:
        _WT8, _WTT = _z['W_over_T'], _z['T_dec']
    print(f"  loaded cached {_WT_path.name}")
else:
    _WT8, _WTT, _WTK = barrier_ratio_map(
        _SIG8, _al8, _B48, _D08, shells=_wt_shells, nuc=nuc_cfg,
        m_s=FILTER_KW['m_s'], sigma=WT_SIGMA, flavor=WT_FLAVOR,
        charge=WT_CHARGE, phase=WT_PHASE, slices=_wt_slices,
        n_jobs=SCAN['n_jobs'])
    np.savez(_WT_path, W_over_T=_WT8, T_dec=_WTT, shell_index=_WTK)
    print(f"  saved {_WT_path.name}")

# ---- map --------------------------------------------------------------------
_wtv = _WT8[np.isfinite(_WT8)]
_wlo, _whi = np.percentile(_wtv, WT_CLIP) if _wtv.size else (0.0, 1.0)
_wtcm = plt.get_cmap(WT_CMAP).copy()
_wtcm.set_bad('0.85')             # not accessible = grey, never a colour
fig, axes = paper_grid('1x2' if len(WT_SHOW) <= 2 else '2x2', mode='double',
                       placeholder=False, fontsize=11, labelsize=11,
                       legendsize=9, aspect=1.0)
for _ax in axes.flat[len(WT_SHOW):]:
    _ax.set_visible(False)
_wpcm = None
for _c, _ia in enumerate(WT_SHOW):
    ax = axes.flat[_c]
    _wpcm = ax.pcolormesh(_B48, _D08, np.ma.masked_invalid(_WT8[_ia]),
                          cmap=_wtcm, vmin=_wlo, vmax=_whi, shading='nearest',
                          zorder=2)
    # With the colour range this tight the eye cannot read gradients off the
    # fill alone, so put numbers on it. Levels at INTERIOR quartiles: a level
    # sitting on the clip bound would trace the clip, not an iso-line.
    _wl = np.unique(np.round(np.nanpercentile(_WT8[_ia], [25, 50, 75]), 1))
    if _wl.size:
        _wc = ax.contour(_B48, _D08, _WT8[_ia], levels=_wl, colors=['white'],
                         linewidths=0.8, alpha=0.9, zorder=3)
        ax.clabel(_wc, fmt='%.1f', fontsize=8, inline=True)
    ax.set_title(rf"$\alpha_s=\pi/2\times{_al8[_ia]/(np.pi/2):.1f}$")
    ax.set_xlabel(r'$B^{1/4}$ [MeV]'); ax.set_ylabel(r'$\Delta_0$ [MeV]')
    ax.set_xlim(_B48.min(), _B48.max()); ax.set_ylim(0.1, _D08.max())
    panel_label(ax, f"({chr(97 + _c)})")
if _wpcm is not None:
    fig.colorbar(_wpcm, cax=axes[0, -1].inset_axes([1.05, 0.0, 0.05, 1.0]),
                 label=r'$W_*/T$')
save_paper_figure(fig, f'WoverT_map_{xsd_tag}_MT0{_WT_MT0:.2f}', supp=True)
plt.show()

# ---- how constant is it? these numbers ARE the claim -------------------------
_stat_slices = list(range(len(_al8))) if WT_STATS is None else list(WT_STATS)
_wt_all = _WT8[_stat_slices][np.isfinite(_WT8[_stat_slices])]
if _wt_all.size:
    _p16, _p50, _p84 = np.percentile(_wt_all, [16, 50, 84])
    print(f"\nW*/T over {_wt_all.size} accessible cells "
          f"(alpha slices {_stat_slices}):")
    print(f"  median {_p50:.2f}   p16..p84 [{_p16:.2f}, {_p84:.2f}]   "
          f"min..max [{_wt_all.min():.2f}, {_wt_all.max():.2f}]")
    print(f"  relative spread (p84-p16)/median = "
          f"{(_p84 - _p16) / _p50:.3f}")
    pd.DataFrame({'W_over_T': _wt_all}).to_csv(
        FIG_D / f'WoverT_cells_{xsd_tag}.csv', index=False)

# %% [markdown]
# ## V.4 — Unpaired matter: $\sigma_{\rm crit}(B^{1/4},\alpha_s)$ at three $m_s$
#
# The control case. An unpaired droplet has no gap, so **everything here is
# $\Delta_0$-independent** and the natural plane is $(B^{1/4},\alpha_s)$, one
# panel per strange-quark mass. Comparing against Fig. 5 isolates how much of the
# answer is due to pairing rather than to the bulk equation of state.
#
# The layers mirror Fig. 5, but every quantity comes from the **unpaired**
# $\beta$-equilibrium quark EoS:
#
# * heatmap + black iso-$\sigma_{\rm crit}$ lines;
# * white dashed iso-$M_{\max}$ from the unpaired TOV;
# * coloured reject outlines — the unpaired gate has no Witten/2-flavour column
#   by design, so only $M_{\max}$, re-hadronization and solver failure appear;
# * the two stability bounds as **curves**, not gates: green = 3-flavour SQM
#   absolutely stable ($e/n_B|_{P=0}=930$ MeV, Witten), grey = 2-flavour matter
#   stable. The fill is shown only between them, which is the Witten window.
#
# Yellow stars mark the tabulated parameter sets at that $m_s$.

# %%
# =============================================================================
#  UNPAIRED sigma_crit(B^1/4, alpha_s), one panel per m_s, full layer set.
#  scan_unpaired_filters returns (viable, M_max, reason) for free -> iso-M_max
#  and reject outlines cost nothing extra. The two stability curves are the only
#  additional physics: e/n_B at P=0 against 930 MeV, root-found in B^1/4.
# =============================================================================
# ---- knobs ------------------------------------------------------------------
U3_MS_LIST    = [80.0, 100.0, 150.0]       # strange-quark masses [MeV]
U3_MT0        = MT0_REF                    # nucleation-threshold mass [M_sun]
U3_CHARGE     = MAIN_CHARGE
U3_ALPHA_GRID = np.linspace(np.pi / 2 * 0.0, np.pi / 2 * 0.3,
                            7 if REDUCED_GRID else 31)     # alpha_s (y-axis)
U3_B4_GRID    = np.linspace(130.0, 160.0,
                            11 if REDUCED_GRID else 51)    # B^1/4 (x-axis)
# ---- layer toggles ----------------------------------------------------------
U3_ISO_SIGMA   = True      # black iso-sigma_crit lines
U3_ISO_MMAX    = True      # white dashed iso-M_max lines (unpaired TOV)
U3_REJECT      = True      # coloured reject-zone outlines + labels
U3_WITTEN      = True      # green: 3-flavour SQM absolute stability
U3_TWOFLAV     = True      # grey: 2-flavour stability
U3_WINDOW_ONLY = True      # colour sigma_crit ONLY inside the stability window
U3_FILL_QUASI  = True      # also solve in quasi-re-hadr. cells, so they fill
U3_STAB_NALPHA = 13        # alpha samples for the stability curves (coarse=cheap)
# -----------------------------------------------------------------------------
# Only the reasons the UNPAIRED filter can emit; colours echo the Fig-5 set.
_C3 = dict(mmax=OKAB['vermillion'], rehadr=OKAB['blue'],
           rehad_quasi=OKAB['purple'], solve=OKAB['grey'],
           witten=OKAB['green'], twoflav='#8a8a8a')
_RSPEC3 = [(REASON_CODE['mmax'],        _C3['mmax'],
            r'$M_{\rm QS}^{\rm max}<2M_\odot$'),
           (REASON_CODE['rehadr'],      _C3['rehadr'],      're-hadr.'),
           (REASON_CODE['rehad_quasi'], _C3['rehad_quasi'], 'quasi-\nre-hadr.'),
           (REASON_CODE['solve'],       _C3['solve'],       'no solve')]

_unp_path = (DIRS['sigma_crit'] /
             f'sigma_crit_unpaired_{xsd_tag}_MT0{U3_MT0:.2f}.npz')
if _unp_path.exists():
    with np.load(_unp_path) as _z:
        _u3 = [(_z[f'sig_{k}'], _z[f'ok_{k}'], _z[f'mm_{k}'], _z[f'rs_{k}'])
               for k in range(len(U3_MS_LIST))]
        U3_ALPHA_GRID, U3_B4_GRID = _z['alpha_grid'], _z['B4_grid']
    print(f"  loaded cached {_unp_path.name}")
else:
    # Per m_s: unpaired viability + M_max + reason (free), then sigma_crit at
    # the PNS centre. The singleton Delta_0 axis is dropped straight away.
    _u3 = []
    for _ms in U3_MS_LIST:
        _cfg = dc_replace(filt_cfg, m_s=_ms)
        _ok, _mm, _rs = scan_unpaired_filters(U3_ALPHA_GRID, U3_B4_GRID, _cfg,
                                              n_jobs=SCAN['n_jobs'])
        # quasi-re-hadronization is a SOFT rejection, so sigma_crit is solved
        # there too and the fill can extend into it; the zone stays outlined.
        _ok_sig = (_ok | (_rs == REASON_CODE['rehad_quasi'])) if U3_FILL_QUASI else _ok
        _sig = compute_sigma_crit(
            _ok_sig, U3_MT0, 'saddlepoint', U3_CHARGE, 'unpaired',
            U3_ALPHA_GRID, U3_B4_GRID, [100.0], star_scan, nuc_cfg, m_s=_ms,
            n_jobs=SCAN['n_jobs'], n_shells=1)   # CENTRE-ONLY, unlike Fig. 5
        _u3.append((_sig[:, 0, :], _ok[:, 0, :], _mm[:, 0, :], _rs[:, 0, :]))
    np.savez(_unp_path, alpha_grid=U3_ALPHA_GRID, B4_grid=U3_B4_GRID,
             m_s_list=np.array(U3_MS_LIST), MT0=U3_MT0,
             **{f'{n}_{k}': a for k, t in enumerate(_u3)
                for n, a in zip(('sig', 'ok', 'mm', 'rs'), t)})
    print(f"  saved {_unp_path.name}")

# one colour scale across the three panels (finite sigma_crit only)
_allfin = np.concatenate([s[np.isfinite(s)] for s, _, _, _ in _u3])
_v3min, _v3max = ((float(_allfin.min()), float(_allfin.max()))
                  if _allfin.size else (0.0, 1.0))

# Stability curves: coarse alpha grid, root-found in B^1/4. The 2-flavour one is
# m_s-INDEPENDENT (no strange quarks) so it is computed once; Witten is per-m_s.
_stab_al = np.linspace(U3_ALPHA_GRID.min(), U3_ALPHA_GRID.max(), U3_STAB_NALPHA)
_lo3, _hi3 = float(U3_B4_GRID.min()), float(U3_B4_GRID.max())
_B2flav = (stability_curve(lambda a, b: two_flavour_energy_at_P0(a, b, filt_cfg),
                           _stab_al, _lo3, _hi3)
           if (U3_TWOFLAV or U3_WINDOW_ONLY) else None)

set_paper_style()
fig, axes = plt.subplots(1, len(U3_MS_LIST),
                         figsize=(5.25 * len(U3_MS_LIST), 4.8),
                         squeeze=False, constrained_layout=True)
pcm = None
for _c, (_ms, (_sig, _ok, _mm, _rs)) in enumerate(zip(U3_MS_LIST, _u3)):
    ax = axes[0, _c]; ax.set_box_aspect(1)             # square, as in Fig. 5
    _Bw = (stability_curve(
        lambda a, b, m=_ms: energy_per_baryon_at_P0(a, b, filt_cfg, m_s=m),
        _stab_al, _lo3, _hi3) if (U3_WITTEN or U3_WINDOW_ONLY) else None)
    if U3_WINDOW_ONLY:
        # Keep sigma_crit only where 2-flavour matter is UNBOUND (right of the
        # grey line) AND 3-flavour SQM is absolutely stable (left of green).
        # M_max and no-rehadronization are automatic: sigma_crit is already NaN
        # on those cells. quasi-re-hadr. does NOT enter the mask.
        _b2f = resample_curve(_B2flav, _stab_al, U3_ALPHA_GRID, U3_B4_GRID.min())
        _bwf = resample_curve(_Bw, _stab_al, U3_ALPHA_GRID, U3_B4_GRID.max())
        _win = ((U3_B4_GRID[None, :] >= _b2f[:, None]) &
                (U3_B4_GRID[None, :] <= _bwf[:, None]))
        _sigp = np.where(_win, _sig, np.nan)
    else:
        _sigp = _sig
    pcm = ax.pcolormesh(U3_B4_GRID, U3_ALPHA_GRID, np.ma.masked_invalid(_sigp),
                        cmap='viridis', vmin=_v3min, vmax=_v3max,
                        shading='nearest', zorder=2)
    if U3_ISO_SIGMA:
        _lv = [l for l in (50, 100, 150, 200, 250)
               if np.isfinite(_sigp).any()
               and np.nanmin(_sigp) <= l <= np.nanmax(_sigp)]
        if _lv:
            _cs = ax.contour(U3_B4_GRID, U3_ALPHA_GRID, _sigp, levels=_lv,
                             colors='k', linewidths=0.7, alpha=0.6, zorder=3)
            ax.clabel(_cs, fmt='%.0f', fontsize=11, inline=True)
    if U3_ISO_MMAX:
        _mmv = np.where(_ok | (_rs == REASON_CODE['mmax']), _mm, np.nan)
        _lvm = [round(x, 1) for x in np.arange(2.2, 4.001, 0.2)
                if np.isfinite(_mmv).any()
                and np.nanmin(_mmv) <= round(x, 1) <= np.nanmax(_mmv)]
        if _lvm:
            _cm = ax.contour(U3_B4_GRID, U3_ALPHA_GRID, _mmv, levels=_lvm,
                             colors='white', linewidths=0.9, linestyles='--',
                             alpha=0.9, zorder=3.5)
            ax.clabel(_cm, fmt='%.1f', fontsize=11, inline=True)
    if U3_REJECT:
        for _code, _col, _lab in _RSPEC3:
            _m = _rs == _code
            if not _m.any():
                continue
            ax.contour(U3_B4_GRID, U3_ALPHA_GRID, _m.astype(float),
                       levels=[0.5], colors=[_col], linewidths=2.2, zorder=4)
            _r, _cc = np.where(_m)
            ax.text(np.median(U3_B4_GRID[_cc]), np.median(U3_ALPHA_GRID[_r]),
                    _lab, color=_col, fontsize=8.5, fontweight='bold',
                    ha='center', va='center', zorder=7)
    if U3_TWOFLAV and _B2flav is not None and np.isfinite(_B2flav).any():
        ax.plot(_B2flav, _stab_al, color=_C3['twoflav'], lw=2.4, zorder=4.5)
        _k = np.isfinite(_B2flav)
        ax.text(_B2flav[_k][-1], _stab_al[_k][-1], '2 flav.\nstable ',
                color=_C3['twoflav'], fontsize=8.5, fontweight='bold',
                ha='right', va='center', zorder=7)
    if U3_WITTEN and _Bw is not None and np.isfinite(_Bw).any():
        ax.plot(_Bw, _stab_al, color=_C3['witten'], lw=2.4, zorder=4.5)
        _k = np.isfinite(_Bw)
        ax.text(_Bw[_k][0], _stab_al[_k][0], ' SQM not\n abs. stable',
                color=_C3['witten'], fontsize=8.5, fontweight='bold',
                ha='left', va='center', zorder=7)
    for p in quark_param_sets:                  # tabulated sets at this m_s
        if abs(p['m_s'] - _ms) < 1.0:
            ax.scatter([p['B4']], [p['alpha']], s=120, marker='*', c='yellow',
                       edgecolors='k', zorder=6)
    ax.set_title(rf"$m_s={_ms:.0f}$ MeV")
    ax.set_xlabel(r'$B^{1/4}$ [MeV]'); ax.set_ylabel(r'$\alpha_s$')
    # y-axis in units of pi/2, the convention the parameter sets are quoted in.
    ax.yaxis.set_major_locator(MultipleLocator(np.pi / 2 * 0.1))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: r'$0$' if abs(v) < 1e-9
        else rf'$\frac{{\pi}}{{2}}\,{v/(np.pi/2):.1f}$'))
    ax.set_xlim(U3_B4_GRID.min(), U3_B4_GRID.max())
    ax.set_ylim(U3_ALPHA_GRID.min(), U3_ALPHA_GRID.max())
    panel_label(ax, f"({chr(97 + _c)})")
fig.colorbar(pcm, ax=axes.ravel().tolist(),
             label=r'$\sigma_{\rm crit}$ [MeV/fm$^2$]', shrink=0.9)
fig.suptitle(rf"unpaired $\sigma_{{\rm crit}}$ — $M_{{T0}}={U3_MT0:g}\,M_\odot$, "
             rf"$Y_L={SCAN_SNAPSHOT['YLH']}$, $S={SCAN_SNAPSHOT['S']}$, "
             rf"$\tau={TAU_TARGET*1e3:g}$ ms — {xsd_tag}", y=1.02)
save_paper_figure(fig, f'sigcrit_unpaired_ms_panels_{xsd_tag}', supp=True)
plt.show()

# %% [markdown]
# ## V.5 — Star-wide vs centre-only $\sigma_{\rm crit}$: does the definition matter?
#
# Every $\sigma_{\rm crit}$ in this paper is **star-wide**: $\tau>\tau_{\rm target}$
# is demanded at `N_SHELLS` densities from `NB_SHELL_MIN` up to the centre, and
# the threshold is set by whichever shell nucleates fastest. The obvious
# alternative is to ask only about the centre.
#
# The reason to expect a difference: the driving force $|\Delta f|$ peaks around
# $2\,n_{\rm sat}$ and can *weaken* inward, so an off-centre shell may nucleate
# first and the centre-only number would then **underestimate** the threshold.
#
# **The measured answer, on this grid, is that it barely matters.** At
# $M_{T0}=1.4\,M_\odot$ the two definitions agree exactly over 97% of the viable
# plane; where they differ the star-wide value is larger by at most ~7 MeV/fm$^2$
# (~8%). The centre is the deciding shell almost everywhere, and the star-wide
# definition is the conservative choice rather than a necessary one. The cell
# prints the exact numbers, which are the ones to quote — do not carry over the
# "factor ~2" figure that circulated earlier, it is not what this grid shows.
#
# Red means star-wide is larger, i.e. centre-only would have called a star safe
# that is not. White means the centre is the deciding shell after all.

# %%
# =============================================================================
#  V.5.  sigma_crit(star-wide) - sigma_crit(centre-only), same plane as Fig. 5.
#  The centre-only pass reuses the SAVED viability mask, so no filter is re-run:
#  only the sigma_crit root-find is repeated at n_shells=1.
# =============================================================================
set_paper_style()
V5_SLICES = list(F8_SHOW)          # the alpha_s slices Fig. 5 draws

_v5_path = DIRS['sigma_crit'] / f'sigma_crit_centre_{xsd_tag}_MT0{SCAN_MT0:.2f}.npz'
if _v5_path.exists():
    with np.load(_v5_path) as _z:
        _SIG_C = _z['sig_crit']
    print(f"  loaded cached {_v5_path.name}")
else:
    # Restrict the work to the drawn slices: the other alpha_s planes would cost
    # the same again and no panel shows them.
    _mask = np.zeros_like(scan_main['cfl_ok'])
    _mask[V5_SLICES] = scan_main['cfl_ok'][V5_SLICES]
    _SIG_C = compute_sigma_crit(
        _mask, SCAN_MT0, MAIN_FLAVOR, MAIN_CHARGE, MAIN_PHASE,
        AL, B4G, D0G, star_scan, nuc_cfg, m_s=FILTER_KW['m_s'],
        n_jobs=SCAN['n_jobs'], n_shells=1)      # 1 = the star CENTRE only
    np.savez(_v5_path, sig_crit=_SIG_C, MT0=SCAN_MT0, slices=np.array(V5_SLICES))
    print(f"  saved {_v5_path.name}")

# Both defined -> difference; either missing -> no comparison exists.
_D5 = np.where(np.isfinite(SIG) & np.isfinite(_SIG_C), SIG - _SIG_C, np.nan)
_v5lim = symmetric_vlim([_D5[i] for i in V5_SLICES], pct=99)

fig, _ax5 = paper_grid('1x2' if len(V5_SLICES) <= 2 else '2x2', mode='double',
                       placeholder=False, square=False,
                       **(PAPER_STYLE | {'aspect': 1.0}))
_a5 = list(_ax5.flat)
for _k, _ia in enumerate(V5_SLICES[:len(_a5)]):
    _pcm5 = _dsig_panel(_a5[_k], _D5[_ia], B4G, D0G, _v5lim,
                        rf"$\alpha_s=\pi/2\times{AL[_ia]/(np.pi/2):.1f}$")
    panel_label(_a5[_k], f"({chr(97 + _k)})")
for _ax in _a5[len(V5_SLICES):]:
    _ax.set_visible(False)
fig.colorbar(_pcm5, ax=_ax5, fraction=0.046, pad=0.02,
             label=r'$\sigma_{\rm crit}^{\rm star} - \sigma_{\rm crit}^{\rm centre}$'
                   r' [MeV/fm$^2$]')
save_paper_figure(fig, f'supp_starwide_vs_centre_{xsd_tag}', supp=True)
plt.show()

# The numbers the caption should quote.
_f5 = _D5[np.isfinite(_D5)]
if _f5.size:
    _rel = 100.0 * _D5[np.isfinite(_D5)] / _SIG_C[np.isfinite(_D5)]
    print(f"\nstar-wide minus centre-only over {_f5.size} cells:")
    print(f"  median {np.median(_f5):+.1f}   p16-p84 "
          f"[{np.percentile(_f5, 16):+.1f}, {np.percentile(_f5, 84):+.1f}]   "
          f"max {_f5.max():+.1f} MeV/fm^2")
    print(f"  as a fraction of the centre-only value: median "
          f"{np.median(_rel):+.1f}%, max {np.nanmax(_rel):+.1f}%")
    print(f"  cells where the centre is NOT the deciding shell: "
          f"{100.0 * (np.abs(_f5) > 1.0).mean():.0f}%")
    pd.DataFrame({'delta_sigma_crit': _f5}).to_csv(
        FIG_D / f'supp_starwide_vs_centre_{xsd_tag}.csv', index=False)

# %% [markdown]
# ## V.6 — Anatomy of the barrier: which term sets $R_*$?
#
# Fig. 1 shows $W(R)$; this shows what it is *made of*. The work of formation is
#
# $$W(R) = \underbrace{\tfrac{4}{3}\pi R^3\,\Delta f}_{\text{bulk, } <0}
#          + \underbrace{4\pi R^2\sigma}_{\text{surface}}
#          + \underbrace{W_{\rm C}(R)}_{\text{Coulomb}},$$
#
# and the barrier exists only because the surface term wins at small $R$ while
# the bulk term wins at large $R$. **(a)** separates the three at the reference
# PNS centre — the peak sits where bulk and surface cross, and the Coulomb term
# decides how far that crossing moves.
#
# **(b)** the consequence: $R_*$ and $W_*$ against $\sigma$. Without Coulomb,
# classical nucleation theory gives $R_*\propto\sigma$ and $W_*\propto\sigma^3$
# exactly; the Coulomb term bends both, and the deviation is the reason
# $\sigma_{\rm crit}$ has to be solved for rather than scaled. The dashed guides
# are the pure-CNT power laws anchored at the smallest $\sigma$ drawn.

# %%
# =============================================================================
#  V.6 knobs.  Everything here is a per-point solve -- no tables involved.
# =============================================================================
V6_SET   = quark_param_sets[0]
V6_PHASE = 'unpaired'      # the single-phase droplet: one Delta_f, one clean split
V6_SIGMA = 100.0           # MeV/fm^2, for panel (a)
V6_SIGMAS = np.linspace(40., 250., 12 if REDUCED_GRID else 30)   # panel (b)
V6_R = np.linspace(0.02, 12.0, 500)

_v6_par = quark_params(alpha=V6_SET['alpha'], B4=V6_SET['B4'],
                       m_s=V6_SET['m_s'])
_v6_nB, _v6_T, _ = central_state(MT0_REF, star_scan)
print(f"V.6 at the reference PNS centre: n_B = {_v6_nB/n_sat:.2f} n_sat, "
      f"T = {_v6_T:.1f} MeV")

fig, ((axA, axB),) = paper_grid('1x2', mode='double', placeholder=False,
                                square=False,
                                **(PAPER_STYLE | {'aspect': 1.15}))

# --- (a) the three terms ------------------------------------------------------
_eb6 = compute_energy_barrier(
    H['trapped'], _v6_nB, _v6_T, V6_SIGMA, params=_v6_par,
    Y_L_H=SCAN_SNAPSHOT['YLH'], flavor_mode=MAIN_FLAVOR,
    electric_charge_mode=MAIN_CHARGE, quark_phase=V6_PHASE, R_values=V6_R)
for _y, _lbl, _col, _ls in (
        (_eb6.W_bulk,    r'bulk  $\frac{4}{3}\pi R^3\Delta f$', OKAB['blue'],  '--'),
        (_eb6.W_surface, r'surface  $4\pi R^2\sigma$',          OKAB['orange'], '--'),
        (_eb6.W_coulomb, r'Coulomb',                            OKAB['green'],  '--'),
        (_eb6.W,         r'total $W(R)$',                       'k',            '-')):
    axA.plot(V6_R, _y, color=_col, ls=_ls, lw=(1.9 if _ls == '-' else 1.3),
             label=_lbl)
_k6 = int(np.nanargmax(_eb6.W))
axA.plot(V6_R[_k6], _eb6.W[_k6], 'o', ms=6, color='k', mfc='white', zorder=6)
axA.axvline(V6_R[_k6], color='0.6', lw=0.8, ls=':', zorder=1)
axA.annotate(rf'$R_*={V6_R[_k6]:.2f}$ fm', (V6_R[_k6], _eb6.W[_k6]),
             textcoords='offset points', xytext=(8, 6), fontsize=8.5)
axA.axhline(0, color='0.6', lw=0.7, zorder=0)
axA.set_xlim(0, 8)
axA.set_ylim(-1.6 * abs(_eb6.W[_k6]), 2.2 * abs(_eb6.W[_k6]))
axA.set_xlabel(r'$R$ [fm]'); axA.set_ylabel(r'$W$ [MeV]')
axA.legend(loc='lower left', fontsize=PAPER_STYLE['legendsize'] - 1)
axA.set_title(rf'$\sigma={V6_SIGMA:.0f}$ MeV/fm$^2$, {V6_PHASE}',
              fontsize=PAPER_STYLE['legendsize'])
panel_label(axA, '(a)', corner='upper right')

# --- (b) how R_* and W_* scale with sigma ------------------------------------
_R6 = np.full(V6_SIGMAS.size, np.nan)
_W6 = np.full(V6_SIGMAS.size, np.nan)
_cache6 = {}
for _i, _sg in enumerate(V6_SIGMAS):
    _pt6 = nucleation_point(H['trapped'], _v6_nB, _v6_T, _sg, params=_v6_par,
                            Y_L_H=SCAN_SNAPSHOT['YLH'], quark_phase=V6_PHASE,
                            flavor_mode=MAIN_FLAVOR,
                            electric_charge_mode=MAIN_CHARGE, V=V_NUC,
                            cache=_cache6)
    _R6[_i], _W6[_i] = _pt6.R_star, _pt6.W_star
axB.plot(V6_SIGMAS, _R6, color=OKAB['blue'], lw=1.8, label=r'$R_*$ [fm]')
_axB2 = axB.twinx()
_axB2.plot(V6_SIGMAS, _W6, color=OKAB['vermillion'], lw=1.8,
           label=r'$W_*$ [MeV]')
# Pure-CNT scalings, anchored at the first finite point: R* ~ sigma, W* ~ sigma^3.
_j6 = int(np.flatnonzero(np.isfinite(_R6))[0])
axB.plot(V6_SIGMAS, _R6[_j6] * V6_SIGMAS / V6_SIGMAS[_j6], color=OKAB['blue'],
         lw=0.9, ls=':', alpha=0.8)
_axB2.plot(V6_SIGMAS, _W6[_j6] * (V6_SIGMAS / V6_SIGMAS[_j6])**3,
           color=OKAB['vermillion'], lw=0.9, ls=':', alpha=0.8)
_axB2.set_yscale('log')
axB.set_xlabel(r'$\sigma$ [MeV/fm$^2$]')
axB.set_ylabel(r'$R_*$ [fm]', color=OKAB['blue'])
_axB2.set_ylabel(r'$W_*$ [MeV]', color=OKAB['vermillion'])
axB.tick_params(axis='y', colors=OKAB['blue'])
_axB2.tick_params(axis='y', colors=OKAB['vermillion'])
axB.set_title(r'dotted: $R_*\propto\sigma$, $W_*\propto\sigma^3$ (no Coulomb)',
              fontsize=PAPER_STYLE['legendsize'] - 1)
panel_label(axB, '(b)')

pd.DataFrame({'R_fm': V6_R, 'W_total': _eb6.W, 'W_bulk': _eb6.W_bulk,
              'W_surface': _eb6.W_surface, 'W_coulomb': _eb6.W_coulomb}).to_csv(
    FIG_D / f'supp_barrier_anatomy_{xsd_tag}.csv', index=False)
pd.DataFrame({'sigma': V6_SIGMAS, 'R_star_fm': _R6, 'W_star_MeV': _W6}).to_csv(
    FIG_D / f'supp_barrier_scaling_{xsd_tag}.csv', index=False)
save_paper_figure(fig, f'supp_barrier_anatomy_{xsd_tag}', supp=True)
plt.show()

# %% [markdown]
# ## V.7 — Quantum tunnelling vs thermal activation
#
# The paper's rates are **thermal**: the droplet is carried over the barrier by a
# fluctuation, at a rate $\propto e^{-W_*/T}$. A droplet can also **tunnel**
# through it, and that channel does not care about $T$ at all — so as a star
# cools, tunnelling must eventually win. The question is whether it wins anywhere
# a proto-neutron star actually lives.
#
# The quantum time comes from the relativistic WKB pipeline: an effective inertia
# $M(R)$ for the hadronic flow the growing droplet displaces, a Bohr–Sommerfeld
# ground-state energy $E_0$ inside the well, and the tunnelling action $A(E_0)$
# between the two turning points, giving
# $\tau_{\rm qt} = 1/(N_c\,\nu_0\,e^{-A})$.
#
# The crossing of the two curves is the **crossover temperature**: above it
# thermal activation dominates, below it tunnelling does. Read against the shaded
# PNS band — the temperatures the star actually passes through — it says whether
# the quantum channel is relevant to this problem or a footnote.

# %%
# =============================================================================
#  V.7 knobs.  Per-point WKB -- no quantum grid table is needed, the barrier and
#  the inertia are rebuilt at each temperature from the same engine the tables
#  use, so the two rates are compared at literally the same droplet.
# =============================================================================
V7_SET   = quark_param_sets[0]
V7_PHASE = 'unpaired'          # single-phase: one Delta_f, hence one clean W(R)
V7_SIGMA = 100.0               # MeV/fm^2
V7_T = np.linspace(1.0, 60.0, 15 if REDUCED_GRID else 40)   # MeV
V7_NC = 1e48                   # attempt centres in the nucleation volume

_v7_par = quark_params(alpha=V7_SET['alpha'], B4=V7_SET['B4'],
                       m_s=V7_SET['m_s'])
_v7_nB = _v6_nB                # the same reference centre as V.6
M_NEUTRON = 939.565            # MeV, for the hadronic mass density rho_H

_tau_th = np.full(V7_T.size, np.nan)
_tau_qt = np.full(V7_T.size, np.nan)
for _i, _T in enumerate(V7_T):
    _pt = nucleation_point(H['trapped'], _v7_nB, _T, V7_SIGMA, params=_v7_par,
                           Y_L_H=SCAN_SNAPSHOT['YLH'], quark_phase=V7_PHASE,
                           flavor_mode=MAIN_FLAVOR,
                           electric_charge_mode=MAIN_CHARGE, V=V_NUC)
    _tau_th[_i] = _pt.tau
    if not (np.isfinite(_pt.R_star) and _pt.R_star > 0):
        continue                       # no critical droplet -> nothing to tunnel
    # Same W(R) the thermal path maximised, and the inertia of the hadronic
    # flow the droplet displaces -- the recipe tables/quantum.py uses per point.
    _rho_H = M_NEUTRON * _v7_nB
    _ratio = float(np.atleast_1d(_pt.Qs.n_B)[0]) / _v7_nB
    # lambda_D exists ONLY for the 'screening' charge mode; every other mode
    # leaves it NaN, and passing that NaN through makes W(R) NaN at every
    # radius -- which the WKB root-find then reports as "no turning points",
    # i.e. a silently empty curve rather than an error.
    _lamD = (float(_pt.lambda_D) if np.isfinite(_pt.lambda_D) else None)

    def _W(R, _df=float(_pt.Delta_f), _s=V7_SIGMA,
           _dnC=float(_pt.delta_n_C), _lam=_lamD):
        return work_of_formation(R, _df, _s, _dnC, lambda_D=_lam)

    def _M(R, _r=_rho_H, _q=_ratio):
        return effective_inertia(R, _r, _q)

    try:
        _tau_qt[_i] = quantum_nucleation_time(_W, _M, float(_pt.R_star),
                                              N_c=V7_NC).tau_qt
    except Exception:
        pass       # no two real turning points -> genuinely non-tunnelling

fig, ((axA, axB),) = paper_grid('1x2', mode='double', placeholder=False,
                                square=False,
                                **(PAPER_STYLE | {'aspect': 1.15}))

# --- (a) the two nucleation times --------------------------------------------
with np.errstate(divide='ignore', invalid='ignore'):
    axA.plot(V7_T, np.log10(_tau_th), color=OKAB['vermillion'], lw=1.9,
             label=r'thermal, $\tau=1/(V\Gamma)$')
    axA.plot(V7_T, np.log10(_tau_qt), color=OKAB['blue'], lw=1.9,
             label=r'quantum, $\tau_{\rm qt}=1/(N_c\nu_0 e^{-A})$')
axA.axhline(np.log10(TAU_TARGET), color='0.4', lw=1.0, ls=':',
            label=rf'$\tau_{{\rm target}}={TAU_TARGET*1e3:g}$ ms')
# The temperature band a PNS centre actually occupies, from the two snapshots.
_T_pns = [float(H['iso_trapped']['T'](_v7_nB, _s['YLH'], _s['S']))
          for _s in (PNS_T0, PNS_TMAX)]
axA.axvspan(min(_T_pns), max(_T_pns), color=OKAB['grey'], alpha=0.18, zorder=0)
axA.text(np.mean(_T_pns), 0.02, 'PNS\ncentre', transform=axA.get_xaxis_transform(),
         ha='center', va='bottom', fontsize=8, color='0.35',
         bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))
# The crossover: where tunnelling becomes the faster channel.
_ok7 = np.isfinite(_tau_th) & np.isfinite(_tau_qt)
if _ok7.any():
    _sgn = np.sign(np.log10(_tau_qt[_ok7]) - np.log10(_tau_th[_ok7]))
    _fl = np.flatnonzero(np.diff(_sgn) != 0)
    if _fl.size:
        _Tx = V7_T[_ok7][_fl[0]]
        axA.axvline(_Tx, color='k', lw=1.0, ls='--', zorder=3)
        axA.annotate(rf'$T_\times\simeq{_Tx:.0f}$ MeV', (_Tx, 0),
                     textcoords='offset points', xytext=(5, 10), fontsize=8.5)
        print(f"crossover temperature: T_x = {_Tx:.1f} MeV "
              f"(below it tunnelling is the faster channel)")
    else:
        print("no crossover in the sampled range: one channel dominates "
              "throughout (compare the two curves to see which)")
    # The comparison the figure exists to make, stated in numbers rather than
    # left to the eye: which channel is faster where the star actually is.
    _iT = int(np.argmin(np.abs(V7_T - np.mean(_T_pns))))
    if np.isfinite(_tau_th[_iT]) and np.isfinite(_tau_qt[_iT]):
        print(f"at the PNS centre (T = {V7_T[_iT]:.0f} MeV): "
              f"tau_thermal = {_tau_th[_iT]:.2e} s, "
              f"tau_quantum = {_tau_qt[_iT]:.2e} s "
              f"-> thermal faster by {np.log10(_tau_qt[_iT] / _tau_th[_iT]):.0f} "
              f"orders of magnitude")
        print("  tunnelling only becomes the faster channel at temperatures "
              "where BOTH times are astronomically long, so it never competes "
              "in the regime this paper is about.")
axA.set_xlabel(r'$T$ [MeV]'); axA.set_ylabel(r'$\log_{10}\,\tau$ [s]')
# Lower left: the top of the panel is where both curves and the band label are.
axA.legend(loc='lower left', fontsize=PAPER_STYLE['legendsize'] - 2,
           framealpha=0.92)
panel_label(axA, '(a)', corner='upper right')

# --- (b) why: the exponent of each channel -----------------------------------
# Thermal is suppressed by W_*/T, which diverges as T -> 0; the tunnelling
# action A is nearly T-independent. Plotting the two exponents together shows
# the crossover as the point where they cross, with no rate prefactors involved.
_WT7 = np.full(V7_T.size, np.nan)
for _i, _T in enumerate(V7_T):
    _p = nucleation_point(H['trapped'], _v7_nB, _T, V7_SIGMA, params=_v7_par,
                          Y_L_H=SCAN_SNAPSHOT['YLH'], quark_phase=V7_PHASE,
                          flavor_mode=MAIN_FLAVOR,
                          electric_charge_mode=MAIN_CHARGE, V=V_NUC)
    if np.isfinite(_p.W_star) and _T > 0:
        _WT7[_i] = _p.W_star / _T
axB.plot(V7_T, _WT7, color=OKAB['vermillion'], lw=1.9, label=r'$W_*/T$')
if not np.isfinite(_tau_qt).any():
    print("  !! no quantum point converged -- check that lambda_D is finite or "
          "None, and that W(R) is not NaN (that is how this fails silently)")
with np.errstate(divide='ignore', invalid='ignore'):
    # A = ln(N_c nu_0 tau_qt); the prefactor is slowly varying, so this reads
    # the action straight off the time without re-running the WKB integral.
    axB.plot(V7_T, np.log(np.clip(_tau_qt, 1e-300, None)) + np.log(V7_NC),
             color=OKAB['blue'], lw=1.9, label=r'$A$ (tunnelling action)')
axB.set_yscale('log')
axB.set_xlabel(r'$T$ [MeV]'); axB.set_ylabel('exponent')
axB.legend(loc='upper right', fontsize=PAPER_STYLE['legendsize'] - 1)
axB.set_title(rf'$n_B={_v7_nB/n_sat:.1f}\,n_{{\rm sat}}$, '
              rf'$\sigma={V7_SIGMA:.0f}$ MeV/fm$^2$',
              fontsize=PAPER_STYLE['legendsize'])
panel_label(axB, '(b)')

pd.DataFrame({'T_MeV': V7_T, 'tau_thermal_s': _tau_th, 'tau_quantum_s': _tau_qt,
              'W_over_T': _WT7}).to_csv(
    FIG_D / f'supp_quantum_vs_thermal_{xsd_tag}.csv', index=False)
save_paper_figure(fig, f'supp_quantum_vs_thermal_{xsd_tag}', supp=True)
plt.show()

# %%
print("\n" + "=" * 78)
print("Notebook complete.")
print(f"  paper figures -> {DIRS['fig_paper'].relative_to(OUT)}")
print(f"  supplementary -> {DIRS['fig_supp'].relative_to(OUT)}")
print(f"  figure data   -> {DIRS['fig_data'].relative_to(OUT)}")
if REDUCED_GRID:
    print("  NOTE: smoke-test output. Set REDUCED_GRID=False for paper values.")
print("=" * 78)
