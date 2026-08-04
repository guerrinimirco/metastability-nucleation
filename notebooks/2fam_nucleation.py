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
# # Coexistence of strange quark stars and neutron stars: metastability and nucleation in proto-neutron stars
#
# Thermal nucleation of a deconfined quark droplet inside hot, dense hadronic
# matter (proto-neutron star), for the 2-family scenario under the absolutely stability of strange-quark-matter hypothesis.
#
# The notebook is organised in four parts:
#
# - **Part I — Setup & parameters.** Imports and every physical/numerical knob
#   (hadronic EoS, quark parametrizations, nucleation). *Always run.*
# - **Part II — Table generation & diagnostics.** Compute and save the heavy
#   tables (EoS, TOV, Q\*, nucleation) with a diagnostic plot next to each. Run
#   once; **skip on later sessions** — the tables are reloaded from disk in Part III.
# - **Part III — Load produced tables.** Rebuild every in-memory object
#   (interpolators, TOV sequences, Q\*/nucleation tables) from disk.
# - **Part IV — Analysis & plots.** Quark-set validation (IV.0b), critical
#   surface tension at the PNS centre (IV.2–IV.3), the paper figures (IV.4) and
#   the cross-method comparisons (IV.5).
#
# **Parts I, III and IV run without Part II** (they read the saved tables). 
#
# Heavy numerics live in the installed packages. The notebook holds parameters, generation loops
# and figures only.

# %% [markdown]
# # Part I — Setup & parameters
#
# *Always run.* Imports and all tunable parameters; no heavy computation here.

# %% [markdown]
# ## I.1 — Imports & installs
#
# Single setup cell: re-running it from a fresh kernel restores every symbol used
# downstream. Keep the GitHub-install lines commented while developing the `eos` /
# `nucleation` packages locally (editable install) — repo edits then take effect
# after a kernel restart only.
#

# %%
import sys, os

# ─── Install the packages ─────────────────────────────────────────
# !{sys.executable} -m pip install --no-deps --force-reinstall git+https://github.com/guerrinimirco/eos.git --quiet
print("eos package loaded successfully!")
# !{sys.executable} -m pip install --no-deps --force-reinstall git+https://github.com/guerrinimirco/metastability-nucleation.git --quiet
print("nucleation package loaded successfully!")


# ─── Standard library ────────────────────────────────────────────────────────
import glob, re


# ─── Standard scientific Python ──────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import pandas as pd
from scipy.interpolate import interp1d


# ─── SFHo equation of state (hadronic phase) ─────────────────────────────────
from eos.sfho.compute_tables import (
    TableSettings, compute_table,
    load_eos_table as load_eos_table_sfho,
    build_interpolators as build_interpolators_sfho,
)
from eos.sfho.parameters import create_custom_parametrization


# ─── Alpha-bag equation of state (quark phase) ───────────────────────────────
from eos.alphabag.compute_tables import AlphaBagTableSettings, compute_alphabag_table
from eos.alphabag.parameters import get_alphabag_custom


# ─── TOV solver (neutron-star structure from an EoS) ─────────────────────────
from eos.tov.solver import (
    EOSTable_for_TOV, generate_ec_logspace,
    compute_tov_sequence, truncate_to_stable_branch,
)


# ─── Nucleation: energy barrier + thermal nucleation observables ─────────────
# Package layout (post-rewrite): barrier -> composition -> critical -> rates ->
# tables -> curves, all re-exported from the top-level `nucleation` namespace.
from nucleation import (
    compute_Qstar_table, load_Qstar_table, build_Qstar_interpolators,
    compute_thermal_nucleation_observables,
    build_thermal_nucleation_interpolators,
    load_thermal_nucleation_table,
    compute_nucleation_density, nucleation_curve,
    export_table, QstarTableData,
    compute_energy_barrier,
)
import nucleation.analysis as nuc_an               # sigma_crit scan engine

# ─── Figure styling + observational-constraint overlays (shared, in-package) ─
from nucleation.analysis.figure import (
    set_paper_style, paper_grid, panel_label, STANDARD_COLORS,
    add_observational_constraints,
)

# ─── Palette convention (whole notebook) ─────────────────────────────────────
# Categorical groups (a few discrete lines): Okabe-Ito colourblind-safe set.
# Continuous/sequential fields (heatmaps, σ-gradients): viridis. Diverging: RdBu.
OKAB = dict(orange='#E69F00', sky='#56B4E9', green='#009E73', blue='#0072B2',
            vermillion='#D55E00', purple='#CC79A7', grey='#9a9a9a')
# ordered picks for "colour = category" line groups: OKAB_CAT[:n] gives n
# well-separated hues (blue→orange→green→vermillion→purple→sky), no yellow
# (poor contrast on white) and no black/grey (reserved for reference lines).
OKAB_CAT = ('#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9')

# Phase line thickness — ONE source of truth so Fig 1 / σ-sweep / Fig 4 stay
# coherent (unpCFL solid+thick, CFL/unpaired thinner+equal). Edit here to rescale all.
PHASE_LW = {'unpCFL': 2.1, 'cfl': 1.4, 'unpaired': 1.4}
# Phase opacity — same single source. unpCFL fully opaque and drawn on top; CFL and
# unpaired share ONE fainter alpha so the composite reads as the emphasised phase.
PHASE_ALPHA = {'unpCFL': 1.0, 'cfl': 0.6, 'unpaired': 0.6}

# Paper-figure text sizes — ONE source, splatted as paper_grid(..., **PAPER_STYLE).
#   fontsize   : title / fallback (10 = PRD body at true column width; 12 if down-scaled)
#   labelsize  : axis names + tick numbers  (settable apart from the legend)
#   legendsize : legend text                (journals often run it a touch smaller)
# Legend handle lines and box/tick thickness are set once in set_paper_style().
PAPER_STYLE = dict(fontsize=11, labelsize=11, legendsize=9, aspect=1.1)

# Precomputed NICER/HESS contours live in the sibling `eos` project; the path
# is relative to the notebooks/ cwd. Regenerate offline if the samples change.
CONTOUR_DIR = "../../eos/plot/data/contours"

# ─── TOV backend (one switch for the whole notebook + σ_crit engine) ─────────
# 'fast'  → numba Dormand-Prince RK45 solver (eos.tov.solver_fast), ~100-1000x
#           faster; the right choice for the heavy Part-IV CFL scans/replays.
# 'scipy' → the trusted adaptive DOP853 reference (fall back here to cross-check
#           or if numba is unavailable). Both give the same columns & physics.
TOV_BACKEND = 'fast'


# ─── Output directories (created once, idempotent) ───────────────────────────
for d in ('../output/tables_Hphase', '../output/tables_tov',
          '../output/tables_Qphase', '../output/tables_Qstar',
          '../output/tables_nucleation', '../output/mc_cfl',
          '../output/figures', '../output/figure_data'):
    os.makedirs(d, exist_ok=True)


# ─── Hadronic-table loader (shared by Part II and Part III) ──────────────────
def load_hadronic_tables(xsd_tag):
    """Load the four saved hadronic EoS tables and build their interpolators.

    Returns (H, H_table): H[case][quantity](*coords) vectorized interpolators
    and the raw EOSTableData per case. Cases: 'betaeq' (n_B, T), 'trapped'
    (n_B, Y_L, T), 'iso_betaeq' (n_B, S), 'iso_trapped' (n_B, Y_L, S).
    """
    cases = {   # case label -> (filename stem, optional suffix, loader eq tag)
        'betaeq':      ('eos_hadronic_betaeq_sfho_2famphi',  '',            'beta_eq'),
        'trapped':     ('eos_hadronic_trapped_sfho_2famphi', '',            'trapped_neutrinos'),
        'iso_betaeq':  ('eos_hadronic_betaeq_sfho_2famphi',  '_isentropic', 'isentropic_beta_eq'),
        'iso_trapped': ('eos_hadronic_trapped_sfho_2famphi', '_isentropic', 'isentropic_trapped'),
    }
    H, H_table = {}, {}
    for key, (stem, suf, eq) in cases.items():
        path = f'../output/tables_Hphase/{stem}_{xsd_tag}{suf}.dat'
        H_table[key] = load_eos_table_sfho(path, eq)
        H[key] = build_interpolators_sfho(H_table[key])
    for key, t in H_table.items():   # grid extents, for sanity-checking
        rng = ', '.join(f"{ax} ∈ [{t.grids[ax][0]:.4g}, {t.grids[ax][-1]:.4g}]"
                        for ax in t.grids)
        print(f"  {key:12s}  {rng}")
    return H, H_table


# %% [markdown]
# ## I.2 — Hadronic phase: parameters
#
# SFHo relativistic mean-field EoS, `2fam_phi` variant (φ-meson channel ⇒ hyperons
# and Δ resonances). The knobs below are the couplings **not** pinned by symmetric
# nuclear-matter saturation: the hyperon potential depths $U_{HN}$ and the Δ–meson
# coupling ratios $x_{y\Delta}$.

# %%
# =============================================================================
#  Hadronic-phase EoS setup
# =============================================================================

# ── Nuclear-saturation density: physical reference + density-grid unit. ──────
n_sat = 0.1583   # fm^-3

# ── Independent-variable grids for the EoS tables ────────────────────────────
# n_B (baryon number density). 300 points, 0.1 → 12 × n_sat — covers from the crust edge up to ≳10 n_sat.
n_B_values = np.linspace(0.1, 12, 300) * n_sat                    # fm^-3

# T (temperature). Two near-zero anchors + uniform 2 MeV up to 100 MeV
#   (typical of proto-NS / merger-remnant regime).
T_values   = np.concatenate([[0.01, 0.1], np.arange(2, 101., 2)])  # MeV

# S = s/n_B (entropy per baryon, k_B units). Used for *isentropic* profiles
#   — appropriate when ν diffuse fast compared to the timescale of interest.
S_values   = np.arange(0.5, 4.01, 0.5)

# Y_L = (n_e + n_νe)/n_B (lepton fraction). Only meaningful when ν are trapped
#   (μ_ν ≠ 0): then Y_L is conserved and replaces β-equilibrium.
Y_L_values = np.arange(0.1, 0.401, 0.05)


# ── Hadronic parametrization & particle content ──────────────────────────────
# Built-in choices in `eos.sfho`: 'sfho' (nucleons only), 'sfhoy' (+ hyperons),
# 'sfhoy_star' (+ φ-meson channel), '2fam_phi' (+ hyperons + Δ resonances).
parametrization  = '2fam_phi'
particle_content = 'nucleons_hyperons_deltas'

# Subdominant contributions to the EoS:
include_photons             = True   # blackbody photons (T > 0)
include_pseudoscalar_mesons = True   # thermal π, K, η (relevant at high T)
include_thermal_neutrinos   = True   # thermal ν, ν̄ population (only when μ_ν=0)


# ── Custom couplings ─────────────────────────────────────────────────────────
# x_yΔ = g_yΔ / g_yN  (Δ–meson coupling ratios, relative to the nucleon).
#   The σ–Δ coupling is the most uncertain; literature range ≈ 1.0–1.25.
# U_HN  = depth of the H–N single-particle potential at n_sat in symmetric NM.
#   These pin the *scalar* RMF couplings of each hyperon family.
x_sigma_delta = 1.15      # σ–Δ ratio
x_omega_delta = 1.0       # ω–Δ ratio (default SU(6))
x_rho_delta   = 1.0       # ρ–Δ ratio (default SU(6))

U_Lambda_N = -28.0        # Λ potential at n_sat in symmetric NM  [MeV]
U_Sigma_N  = +30.0        # Σ potential (positive ⇒ strongly repulsive)
U_Xi_N     = -18.0        # Ξ potential

# Tag that uniquely identifies this coupling set in filenames & logs.
xsd_tag = f"xsd{int(round(x_sigma_delta * 100))}"   # 1.15 → 'xsd115'

params = create_custom_parametrization(
    U_Lambda_N=U_Lambda_N, U_Sigma_N=U_Sigma_N, U_Xi_N=U_Xi_N,
    x_sigma_delta=x_sigma_delta,
    x_omega_delta=x_omega_delta,
    x_rho_delta=x_rho_delta,
    name=f"2fam_phi_{xsd_tag}",       
)


# %% [markdown]
# ## I.3 — Quark phase: parametrizations
#
# Fix the quark parametrization set(s) to tabulate. Each set is
# $(\alpha_s, B^{1/4}, \Delta_0, m_s)$: $(\alpha_s, B^{1/4}, m_s)$ are shared by
# the unpaired and CFL phases, $\Delta_0$ is the CFL pairing gap (ignored by the
# unpaired phase). Each set is checked against the CFL acceptance filters in
# **IV.0b** (needs the loaded tables). Add dicts to `quark_param_sets` to handle
# several sets — every
# downstream cell loops over all of them, tagging files/keys via `q_tag_of`.

# %%
# =============================================================================
#  INPUT: quark parametrization set(s) + surface tensions.
#  Each set = (α_s, B^1/4, Δ_0, m_s); (α_s, B^1/4, m_s) are shared by unpaired
#  and CFL, Δ_0 is CFL-only. Add more dicts to handle several sets (a *list* of
#  sets, not a cartesian grid). Every downstream cell (validation, EoS, Q*,
#  nucleation) loops over all sets, tagging files/keys via q_tag_of(p).
# =============================================================================
quark_param_sets = [
    dict(alpha=0.1*np.pi/2, B4=145.0, Delta0=80.0, m_s=100.0),
    dict(alpha=0.08*np.pi/2, B4=158.0, Delta0=157.0, m_s=100.0),
    #dict(alpha=0.3*np.pi/2, B4=145.0, Delta0=80.0, m_s=100.0),
    #dict(alpha=0.3*np.pi/2, B4=165.0, Delta0=200.0, m_s=100.0),
    #dict(alpha=0.3*np.pi/2, B4=150.0, Delta0=100.0, m_s=100.0),
    #dict(alpha=0.3*np.pi/2, B4=160.0, Delta0=150.0, m_s=100.0),
    #dict(alpha=0.18*np.pi/2, B4=150.0, Delta0=126.0, m_s=100.0),  # marginalized posterior
    #dict(alpha=0.08*np.pi/2, B4=158.0, Delta0=157.0, m_s=100.0)   # maximum posterior
]


def q_tag_of(p):
    """Unique filename/key tag for one parameter set."""
    return (f"B4{int(round(p['B4']))}_D{int(round(p['Delta0']))}"
            f"_a{p['alpha']:.2f}_ms{int(round(p['m_s']))}")

print(f"{len(quark_param_sets)} quark set(s):")
for p in quark_param_sets:
    print(f"  {q_tag_of(p)}  (α_s={p['alpha']:.3f}, B¼={p['B4']}, "
          f"Δ_0={p['Delta0']}, m_s={p['m_s']})")


# %% [markdown]
# ## I.4 — Nucleation: parameters & method sets
#
# Global nucleation knobs and the method tables reused by generation (Part II.6)
# and the analysis plots (Part IV).

# %%
# =============================================================================
#  Nucleation — global parameters and method sets.  (all tunable knobs)
# =============================================================================

# System volume entering the nucleation time  τ = 1 / (Γ · V):
# a sphere of radius 100 m (the central region of the star).
V_nuc = 4.18879e51        # fm^3

# Target nucleation timescale.  σ_crit, T_nuc and the n_B(T) curves are all
# defined as "the value at which τ = tau_target".
tau_target = 1e-3         # s  (1 ms)

# Surface tensions σ of the quark–hadron interface to scan [MeV/fm^2].
# Used by the Q* tables (Part II.5) and every nucleation observable downstream.
sigma_list = [50.0, 80, 100.0, 150, 200]

# ── unpCFL crossover radius ──────────────────────────────────────────────────
# unpCFL droplet = CFL core + unpaired mantle, switching at the coherence radius
# R_x(T) = ħc/Δ(T), BCS gap Δ(T)=Δ0·√(1−(T/T_c)²), T_c = 0.57·Δ0.  Use the engine
# implementation so notebook and nucleation.analysis agree:
#     nuc_an.crossover_radius(T, Delta0)   →  R_x [fm]  (∞ above T_c)

# ── Method set for GENERATION (Part II.6) ────────────────────────────────────
# (flavor_mode, electric_charge_mode, [phases]).
#   flavor_mode : 'saddlepoint' | 'frozen'
#   charge mode : 'lcn' (local/Maxwell) | 'gcn' (global/Gibbs) | 'coulomb_minimize'
#   phases      : 'unpaired' | 'cfl' | 'unpCFL'   (frozen supports 'unpaired' only)
_nuc_modes = [
    ('saddlepoint', 'lcn',              ['unpaired', 'cfl', 'unpCFL']),
    ('saddlepoint', 'gcn',              ['unpaired', 'cfl', 'unpCFL']),
    ('saddlepoint', 'coulomb_minimize', ['unpaired', 'cfl', 'unpCFL']),
    ('frozen',      'lcn',              ['unpaired']),
]

# ── Method set for DIAGNOSTIC OVERLAYS (Part II.6 plots) ──────────────────────
# (flavor, charge, colour, linestyle, label) — the four methods drawn together.
_methods = [
    ('frozen',      'lcn',              'tab:gray',   ':',  'frozen LCN'),
    ('saddlepoint', 'lcn',              'tab:orange', '--', 'saddle LCN'),
    ('saddlepoint', 'gcn',              'tab:red',    '-',  'saddle GCN'),
    ('saddlepoint', 'coulomb_minimize', 'tab:blue',   '-.', 'Coulomb min'),
]

print(f"nucleation params: V={V_nuc:.3e} fm^3, tau_target={tau_target:g} s, "
      f"sigma_list={sigma_list} MeV/fm^2")


# %% [markdown]
# # Part II — Table generation & diagnostics
#
# Compute and **save** the heavy tables; each generation cell is followed by a
# diagnostic plot built from the in-memory result. **Run once** — later sessions
# skip this part and reload from disk in Part III.
#
# > Generation needs the hadronic interpolators `H` as input. Part II.1 builds
# > them right after computing the hadronic tables, so run Part II top-to-bottom.

# %% [markdown]
# ## II.1 — Hadronic EoS tables
#
# Compute the four equilibrium scenarios (they span the thermal/leptonic states of
# hot dense matter), then build the interpolators used by the rest of Part II.
#
# | Tag           | Equilibrium                       | Grid axes         |
# |---------------|-----------------------------------|-------------------|
# | `betaeq`      | β-eq, free-streaming ν            | $(n_B, T)$        |
# | `iso_betaeq`  | β-eq, isentropic                  | $(n_B, S)$        |
# | `trapped`     | fixed $Y_L$, trapped ν            | $(n_B, Y_L, T)$   |
# | `iso_trapped` | fixed $Y_L$, trapped ν, isentropic| $(n_B, Y_L, S)$   |

# %%
# =============================================================================
#  Compute the four hadronic EoS tables in one loop.
#  Each row in `jobs` is one independent table written to disk.
# =============================================================================

# Settings that are identical for all four jobs.
common = dict(
    parametrization  = parametrization,
    custom_params    = params,                  # actual physics; parametrization is just a label
    particle_content = particle_content,
    n_B_values       = n_B_values,
    include_photons             = include_photons,
    include_pseudoscalar_mesons = include_pseudoscalar_mesons,
    include_thermal_neutrinos   = include_thermal_neutrinos,
    print_results = True, print_first_n = 1,
    print_errors  = True, print_timing  = True,
    save_to_file  = True,
)

# (label, equilibrium tag, extra grid axes for THIS equilibrium, filename)
jobs = [
    # β-equilibrium, free-streaming ν (μ_ν=0)  — grid: (n_B, T)
    ('betaeq',
        'beta_eq',
        dict(T_values=T_values),
        f'eos_hadronic_betaeq_sfho_2famphi_{xsd_tag}.dat'),

    # Same physics, isentropic — grid: (n_B, S)
    ('iso_betaeq',
        'isentropic_beta_eq',
        dict(S_values=S_values),
        f'eos_hadronic_betaeq_sfho_2famphi_{xsd_tag}_isentropic.dat'),

    # Fixed Y_L, trapped ν — grid: (n_B, Y_L, T)
    ('trapped',
        'trapped_neutrinos',
        dict(Y_L_values=Y_L_values, T_values=T_values),
        f'eos_hadronic_trapped_sfho_2famphi_{xsd_tag}.dat'),

    # Fixed Y_L, trapped ν, isentropic — grid: (n_B, Y_L, S)
    ('iso_trapped',
        'isentropic_trapped',
        dict(Y_L_values=Y_L_values, S_values=S_values),
        f'eos_hadronic_trapped_sfho_2famphi_{xsd_tag}_isentropic.dat'),
]

results_H = {}   # in-memory results: results_H['betaeq'], results_H['trapped'], ...
for label, eq, extras, fname in jobs:
    print(f"\n── Computing hadronic table: {label}  ({eq}) ──")
    settings = TableSettings(
        **common, equilibrium=eq, **extras,
        output_filename=f'../output/tables_Hphase/{fname}',
    )
    results_H[label] = compute_table(settings)


# ── Build the interpolators from the tables just written ─────────────────────
# (so the rest of Part II has H available; Part III.1 runs the same helper for
#  the skip-Part-II workflow).
H, H_table = load_hadronic_tables(xsd_tag)


# %% [markdown]
# ## II.2 — Hadronic TOV sequences
#
# Integrate the TOV equations for (a) the cold ($T=0$) β-equilibrium star and
# (b) trapped, isentropic profiles over a $(Y_L, S)$ grid. An SFHo crust is spliced
# below $n_{\rm transition}$ via a tanh blend; the stable branch is kept up to
# $M_{\max}$. Results: `tov_cold` (array) and `tov_trapped[(Y_L, S)]` (dict).

# %%
# =============================================================================
#  TOV: cold β-equilibrium neutron star (T ≈ 0)
# -----------------------------------------------------------------------------
#  - Build P(n_B), e(n_B) at T = 0.01 MeV (our cold-limit anchor).
#  - Splice the SFHo crust at n < n_transition via a tanh blend of width delta_n.
#  - Integrate TOV over a logspace grid of central energy densities.
# =============================================================================

T_cold = 0.01   # MeV — lowest T on the table grid (effective T = 0)

# Cold β-eq slice; interpolators are vectorized → no list comp needed.
P_T0 = H['betaeq']['P'](n_B_values,   T_cold)
e_T0 = H['betaeq']['eps'](n_B_values, T_cold)

eos_T0 = EOSTable_for_TOV(P=P_T0, epsilon=e_T0, nB=n_B_values)

# Density at which we splice our high-density EoS to the SFHo crust:
#   above n_transition → our table; below → SFHo crust at T=0, β-eq.
# delta_n is the *width* of the tanh blend used by compute_tov_sequence.
n_transition = 0.5 * n_sat       # ≈ 0.08 fm^-3 — well below the inner-crust scale
delta_n      = 0.05 * n_sat      # smooth blend width

# Central energy densities to scan (log-spaced, covers low-mass → M_max → past).
e_c_vec = generate_ec_logspace(e_min=150, e_max=2500, n_points=100)

results_tov_full = compute_tov_sequence(
    eos_T0, e_c_vec=e_c_vec,
    add_crust_table='compose_sfho_nT0_beta', add_crust_mode='interpolate',
    n_transition=n_transition, delta_n=delta_n,
    compute_baryonic_mass=True, compute_tidal=False,
    backend=TOV_BACKEND, verbose=True,
)

# Keep only the stable branch (everything up to M_max), save to disk.
results_tov, M_max_H, e_c_max_H = truncate_to_stable_branch(
    results_tov_full,
    output_file=f'../output/tables_tov/tov_hadronic_betaeq_2famphi_{xsd_tag}_T0.dat',
    header_info=f"Hadronic EOS, T=0, β-eq + crust, {params.name}",
    verbose=True,
)

tov_cold = results_tov   # canonical name used downstream


# %%
# =============================================================================
#  TOV: trapped neutrinos, isentropic profile — sweep over (Y_L, S)
# =============================================================================

# Use the Y_L grid the table was actually computed on (avoids silent
# extrapolation if a hardcoded value drifts out of bounds).
YL_values_tov = list(H_table['iso_trapped'].grids['Y_L'])

# S is expensive in the TOV sweep (one TOV sequence per pair) → take a subset.
S_values_tov  = [1.0, 1.5, 2.0, 2.5, 3.0]

# Lower n_transition than the cold case: hot crusts extend higher in density
# and the COMPOSE trapped-ν tables reach further up. The tanh-blend width is
# kept the same as the T = 0 case for consistency.
n_transition = 0.3 * n_sat
delta_n      = 0.05 * n_sat

e_c_vec = generate_ec_logspace(e_min=150, e_max=2500, n_points=100)

tov_results_trapped = {}   # keyed by (Y_L, S)
for YL in YL_values_tov:
    for S in S_values_tov:
        print(f"\n── TOV  Y_L = {YL:.2f}  S = {S:.1f} ──")

        P_YLS = H['iso_trapped']['P'](n_B_values,   YL, S)
        e_YLS = H['iso_trapped']['eps'](n_B_values, YL, S)
        eos_YLS = EOSTable_for_TOV(P=P_YLS, epsilon=e_YLS, nB=n_B_values)

        results_full = compute_tov_sequence(
            eos_YLS, e_c_vec=e_c_vec,
            add_crust_table='compose_sfho_nYLS_trap', add_crust_mode='interpolate',
            n_transition=n_transition, delta_n=delta_n,
            crust_YL=YL, crust_S=S,
            compute_baryonic_mass=True, compute_tidal=False,
            backend=TOV_BACKEND, verbose=False,
        )

        # Drop TOV points whose integration returned a non-finite mass: high-e_c
        # central states past the isentropic EoS table's validity give NaN M,
        # which would crash the M_max spline (PchipInterpolator) inside
        # truncate_to_stable_branch. Keep only the physical, finite branch.
        _finite = np.isfinite(results_full[:, 4])
        if _finite.sum() < 5:
            print(f"  ⚠ only {int(_finite.sum())} finite-M points — skipping this (Y_L, S)")
            continue
        if not _finite.all():
            print(f"  dropped {int((~_finite).sum())} non-finite-M TOV points")
        results_full = results_full[_finite]

        results, M_max, _ = truncate_to_stable_branch(
            results_full,
            output_file=(f'../output/tables_tov/'
                         f'tov_hadronic_trapped_2famphi_{xsd_tag}_'
                         f'YL{YL:.2f}_S{S:.1f}.dat'),
            header_info=f"Hadronic trapped, {params.name}, YL={YL}, S={S}",
        )
        tov_results_trapped[(YL, S)] = results

print(f"\nAll TOV sequences done. {len(tov_results_trapped)} entries in "
      f"tov_results_trapped[(Y_L, S)].")

tov_trapped = tov_results_trapped   # canonical name used downstream


# %% [markdown]
# ## II.4 — Quark EoS tables (unpaired & CFL)
#
# One αBag table per phase, β-equilibrium, over the full $(n_B, T)$ grid, for every
# set in `quark_param_sets`. `compute_alphabag_table` dispatches on `phase`
# (the `Delta0_values` entry is unused by the unpaired table). Files tagged per set.

# %%
# =============================================================================
#  Compute the unpaired + CFL αBag tables for EVERY set in quark_param_sets.
#  results_Q keyed by (tag, label); files tagged per set.
# =============================================================================
os.makedirs('../output/tables_Qphase', exist_ok=True)

# Denser n_B grid than the hadronic table — quark matter only matters at high n_B,
# and the EoS is smooth so this stays cheap.  Same T grid as the hadronic tables.
n_B_grid_quark = np.linspace(0.5, 15, 400) * n_sat     # fm^-3

results_Q = {}
for p in quark_param_sets:
    stag = q_tag_of(p)
    # (label, phase-specific settings) — Δ_0 is per-set; unused by unpaired.
    quark_jobs = [
        ('unpaired', dict(phase='unpaired', equilibrium='beta_eq')),
        ('cfl',      dict(phase='cfl',      Delta0_values=[p['Delta0']])),
    ]
    for label, extra in quark_jobs:
        print(f"\n── quark EoS [{stag}]  {label} ──")
        settings = AlphaBagTableSettings(
            alpha=p['alpha'], B4=p['B4'], m_s=p['m_s'],
            n_B_values=n_B_grid_quark, T_values=T_values,
            include_photons=True, include_gluons=True, include_thermal_neutrinos=True,
            print_results=True, print_first_n=1, print_errors=True, print_timing=True,
            save_to_file=True,
            output_filename=f'../output/tables_Qphase/eos_quark_{label}_{stag}.dat',
            **extra,
        )
        results_Q[(stag, label)] = compute_alphabag_table(settings)


# %% [markdown]
# ## II.5 — Q\* (critical-droplet) tables
#
# For every set × hadronic background × charge mode × quark phase × σ, compute the
# $Q^*$ tables in the **saddlepoint** flavour mode. LCN/GCN are σ-independent
# (only $R_c=-2\sigma/\Delta f$ scales) — solved once and rescaled per σ;
# `coulomb_minimize` couples σ into the solve, so it is re-solved at every σ.

# %%
# =============================================================================
#  Compute Q* over quark_param_sets × {β-eq, trapped} × {lcn, gcn,
#  coulomb_minimize} × {unpaired, cfl} × σ.  Stored in Qstar_sets by stem key.
# =============================================================================
os.makedirs('../output/tables_Qstar', exist_ok=True)

_bg_tab  = {'Htrapped': H_table['trapped']} #'Hbetaeq': H_table['betaeq'], 
_charges = ['lcn', 'gcn', 'coulomb_minimize']

def qs_stem(stag, bg, charge, ph, sg):
    """Filename stem / dict key for one Q* table."""
    return f"{bg}_saddlepoint_{charge}_{ph}_{stag}_s{int(sg)}"

Qstar_sets = {}
for p in quark_param_sets:
    stag = q_tag_of(p)
    pars = get_alphabag_custom(alpha=p['alpha'], B4=p['B4'], m_s=p['m_s'])
    for bg, htab in _bg_tab.items():
        for charge in _charges:
            cn = 'local' if charge == 'lcn' else 'global'
            sigma_in_solve = (charge == 'coulomb_minimize')
            for ph, D0 in (('unpaired', None), ('cfl', p['Delta0'])):
                if sigma_in_solve:
                    # σ enters the solve → re-solve at each σ (no rescale).
                    for sg in sigma_list:
                        stem = qs_stem(stag, bg, charge, ph, sg)
                        t = compute_Qstar_table(
                            htab, flavor_mode='saddlepoint', electric_charge_mode=charge,
                            params=pars, sigma=sg, quark_phase=ph, Delta0=D0,
                            verbose=False, save_table=True,
                            output_file=f'../output/tables_Qstar/Qstar_{stem}.dat')
                        Qstar_sets[stem] = dict(table=t, interp=build_Qstar_interpolators(t))
                    nconv, ntot = int(np.sum(t.data['converged'])), t.data['converged'].size
                else:
                    # σ-independent solve → compute once @σ_ref, rescale R_c ∝ σ.
                    base = compute_Qstar_table(
                        htab, flavor_mode='saddlepoint', electric_charge_mode=charge,
                        params=pars, sigma=sigma_list[0], quark_phase=ph, Delta0=D0,
                        verbose=False, save_table=False)
                    nconv, ntot = int(np.sum(base.data['converged'])), base.data['converged'].size
                    for sg in sigma_list:
                        data_sg = {**base.data, 'R_c': base.data['R_c'] * (sg / sigma_list[0])}
                        t = QstarTableData(eq_type=base.eq_type,
                                           hadronic_grids=base.hadronic_grids, data=data_sg)
                        stem = qs_stem(stag, bg, charge, ph, sg)
                        export_table(t, pars, output_file=f'../output/tables_Qstar/Qstar_{stem}.dat',
                                     charge_neutrality=cn, sigma=sg)
                        Qstar_sets[stem] = dict(table=t, interp=build_Qstar_interpolators(t))
                print(f"  [{stag}] {bg:9s} {charge:16s} {ph:8s}: {nconv}/{ntot} converged")

print(f"\nComputed {len(Qstar_sets)} Q* tables "
      f"({len(quark_param_sets)} set(s) × {len(_bg_tab)} bg × {len(_charges)} charge × 2 phase × {len(sigma_list)} σ).")


# %% [markdown]
# ## II.6 — Thermal nucleation observables (trapped background) & diagnostics
#
# Compute $R_c, W_c, \Gamma, \tau$ over $(n_B, Y_L, T)$ for each set × σ, for the
# methods in `_nuc_modes` (saddlepoint LCN/GCN/coulomb_minimize over unpaired / CFL
# / unpCFL, plus frozen LCN unpaired). Reuses the Q\* tables from Part II.5. The
# diagnostic cells below plot the observables and the $\tau=\tau_{\rm target}$
# nucleation curves (computed on-the-fly — no separate saved table).

# %%
# =============================================================================
#  Compute trapped nucleation observables. Reuses Q* tables where available.
# =============================================================================
os.makedirs('../output/tables_nucleation', exist_ok=True)

# unpCFL crossover radius R_x(T): use the engine implementation (see Part I.4).

# (flavor, charge, phases). 'unpCFL' = CFL core + unpaired mantle (saddlepoint
# only); it reuses the unpaired & CFL Q* tables. frozen has no CFL/unpCFL.
_nuc_modes = [
    ('saddlepoint', 'lcn',              ['unpaired', 'cfl', 'unpCFL']),
    ('saddlepoint', 'gcn',              ['unpaired', 'cfl', 'unpCFL']),
    ('saddlepoint', 'coulomb_minimize', ['unpaired', 'cfl', 'unpCFL']),
    ('frozen',      'lcn',              ['unpaired']),
]

htab_tr = H_table['trapped']
Rx_T    = htab_tr.grids['T']          # R_x evaluated on the hadronic T grid
nuc_sets = {}
for p in quark_param_sets:
    stag = q_tag_of(p)
    pars = get_alphabag_custom(alpha=p['alpha'], B4=p['B4'], m_s=p['m_s'])
    Rx   = nuc_an.crossover_radius(Rx_T, p['Delta0'])
    for sg in sigma_list:
        for flavor, charge, phases in _nuc_modes:
            for ph in phases:
                stem = f"Htrapped_{flavor}_{charge}_{ph}_{stag}_s{int(sg)}"
                outfile = f'../output/tables_nucleation/nucleation_{stem}.dat'
                if ph == 'unpCFL':
                    # CFL core + unpaired mantle: reuse both saddlepoint Q* tables.
                    k_u = qs_stem(stag, 'Htrapped', charge, 'unpaired', sg)
                    k_c = qs_stem(stag, 'Htrapped', charge, 'cfl', sg)
                    obs = compute_thermal_nucleation_observables(
                        hadronic_table=htab_tr, sigma=sg,
                        Qstar_table_unp=Qstar_sets[k_u]['table'],
                        Qstar_table_cfl=Qstar_sets[k_c]['table'],
                        params_unp=pars, params_cfl=pars, V=V_nuc,
                        quark_phase='unpCFL', Delta0=p['Delta0'],
                        flavor_mode='saddlepoint', electric_charge_mode=charge,
                        switching_mode='step', Rx=Rx,
                        save_table=True, output_file=outfile)
                else:
                    D0 = p['Delta0'] if ph == 'cfl' else None
                    # Reuse the pre-computed Q* table for saddlepoint; frozen recomputes.
                    qkey = qs_stem(stag, 'Htrapped', charge, ph, sg)
                    Qtab = (Qstar_sets[qkey]['table'] if (flavor == 'saddlepoint'
                                                          and qkey in Qstar_sets) else None)
                    obs = compute_thermal_nucleation_observables(
                        hadronic_table=htab_tr, sigma=sg,
                        Qstar_table=Qtab, params=pars, V=V_nuc,
                        quark_phase=ph, Delta0=D0,
                        flavor_mode=flavor, electric_charge_mode=charge,
                        save_table=True, output_file=outfile)
                nuc_sets[stem] = obs
                print(f"  {stem}: {int(np.sum(obs.converged))}/{obs.converged.size} valid")

print(f"\nComputed {len(nuc_sets)} trapped nucleation tables.")


# %% [markdown]
# ### Diagnostics — shared selectors & 4-panel helper
#
# $R_c$, $W_c/T$, $\log_{10}\Gamma$, $\log_{10}\tau$ vs $n_B^H$ at a fixed $Y_L$
# slice. `set_sel` / `YL_sel` choose the parametrization and lepton fraction.

# %%
# =============================================================================
#  Shared selectors + a 4-row plotting helper for the trapped nucleation views.
# =============================================================================
set_sel  = quark_param_sets[0]
stag_sel = q_tag_of(set_sel)
YL_sel   = 0.25                     # nearest grid value used

# Reference grids (any table for this set).
_ref = next(o for k, o in nuc_sets.items() if k.endswith(f"_{stag_sel}_s{int(sigma_list[0])}"))
nBg  = _ref.hadronic_grids['n_B_H']
Tg   = _ref.hadronic_grids['T']
iYL  = int(np.argmin(np.abs(_ref.hadronic_grids['Y_L_H'] - YL_sel)))
YL_used = _ref.hadronic_grids['Y_L_H'][iYL]

def _iT(T):   return int(np.argmin(np.abs(Tg - T)))
def _get(flavor, charge, phase, sg):
    return nuc_sets.get(f"Htrapped_{flavor}_{charge}_{phase}_{stag_sel}_s{int(sg)}")

def nuc_panels(curves, suptitle):
    """curves: list of (obs, iT, color, ls, label). 4 rows vs n_B^H at iYL."""
    fig, axes = plt.subplots(4, 1, figsize=(7, 15), sharex=True)
    for obs, iT, color, ls, lbl in curves:
        if obs is None:
            continue
        sl = (slice(None), iYL, iT)
        with np.errstate(divide='ignore', invalid='ignore'):
            axes[0].plot(nBg, obs.R_c[sl],             color=color, ls=ls, label=lbl)
            axes[1].plot(nBg, obs.W_c[sl] / Tg[iT],    color=color, ls=ls, label=lbl)
            axes[2].plot(nBg, np.log10(obs.Gamma[sl]), color=color, ls=ls, label=lbl)
            axes[3].plot(nBg, np.log10(obs.tau[sl]),   color=color, ls=ls, label=lbl)
    axes[0].set_ylabel(r'$R_c$ [fm]');                            axes[0].set_ylim(0, 10)
    axes[1].set_ylabel(r'$W_c / T$'); axes[1].set_yscale('log');  axes[1].set_ylim(0.1, 500)
    axes[2].set_ylabel(r'$\log_{10}\,\Gamma$ [fm$^{-3}$s$^{-1}$]'); axes[2].set_ylim(-200, 20)
    axes[3].set_ylabel(r'$\log_{10}\,\tau$ [s]');                 axes[3].set_ylim(-70, 100)
    axes[3].axhline(np.log10(1e-3), color='k', ls=':', lw=0.8, alpha=0.6)
    axes[3].set_xlabel(r'$n_B^H$ [fm$^{-3}$]'); axes[0].legend(fontsize=8)
    for ax in axes:
        ax.set_xlim(nBg[0], 10 * n_sat); ax.grid(alpha=0.3)
    fig.suptitle(suptitle, y=1.005); fig.tight_layout(); plt.show()

# (_methods — the four-method overlay table — is defined in Part I.4.)


# %% [markdown]
# ### Diagnostic (1) — fixed $T,\sigma$: overlay methods

# %%
# =============================================================================
#  (1) Fixed T, Y_L, σ — overlay methods (phase = unpaired → all four).
# =============================================================================
T_sel, sigma_sel, phase_sel = 30.0, sigma_list[0], 'unpaired'

# _methods now lives in the shared-selectors cell above.
curves = [(_get(fl, ch, phase_sel, sigma_sel), _iT(T_sel), c, ls, lbl)
          for fl, ch, c, ls, lbl in _methods]
nuc_panels(curves, rf"trapped {phase_sel} — $T\approx{Tg[_iT(T_sel)]:.0f}$ MeV, "
                   rf"$Y_L\approx{YL_used:.2f}$, σ={int(sigma_sel)} — {stag_sel}")


# %% [markdown]
# ### Diagnostic (2) — fixed $T$, one method+phase: overlay $\sigma$

# %%
# =============================================================================
#  (2) Fixed T, Y_L, one method+phase — overlay σ.
# =============================================================================
T_sel2, phase2, flavor2, charge2 = 30.0, 'unpCFL', 'saddlepoint', 'coulomb_minimize'

_cmap = plt.cm.viridis(np.linspace(0, 1, len(sigma_list)))
curves = [(_get(flavor2, charge2, phase2, sg), _iT(T_sel2), _cmap[k], '-', f"σ={int(sg)}")
          for k, sg in enumerate(sigma_list)]
nuc_panels(curves, rf"trapped {phase2} {flavor2}/{charge2} — $T\approx{Tg[_iT(T_sel2)]:.0f}$ MeV, "
                   rf"$Y_L\approx{YL_used:.2f}$ — {stag_sel}  (vary σ)")


# %% [markdown]
# ### Diagnostic (3) — fixed $\sigma$, one method+phase: overlay $T$

# %%
# =============================================================================
#  (3) Fixed σ, Y_L, one method+phase — overlay T.
# =============================================================================
sigma_sel3, phase3, flavor3, charge3 = sigma_list[0], 'unpCFL', 'saddlepoint', 'gcn'
T_list3 = [10.0, 30.0, 50.0, 70.0]

obs3  = _get(flavor3, charge3, phase3, sigma_sel3)
_cmap = plt.cm.plasma(np.linspace(0, 1, len(T_list3)))
curves = [(obs3, _iT(T), _cmap[k], '-', f"T={Tg[_iT(T)]:.0f} MeV")
          for k, T in enumerate(T_list3)]
nuc_panels(curves, rf"trapped {phase3} {flavor3}/{charge3} — σ={int(sigma_sel3)}, "
                   rf"$Y_L\approx{YL_used:.2f}$ — {stag_sel}  (vary T)")

# log10(tau) vs T, one line per n_B^H/n_sat (transpose of the panels above).
nB_over_nsat3 = [2.0, 3.0, 4.0, 5.0, 6.0]
_cmap = plt.cm.viridis(np.linspace(0, 1, len(nB_over_nsat3)))
fig, ax = plt.subplots(figsize=(7, 5))
for k, x in enumerate(nB_over_nsat3):
    i_nB = int(np.argmin(np.abs(nBg / n_sat - x)))      # nearest n_B^H grid point
    with np.errstate(divide='ignore', invalid='ignore'):
        ax.plot(Tg, np.log10(obs3.tau[i_nB, iYL, :]), color=_cmap[k], lw=1.5,
                label=rf"$n_B^H/n_{{sat}}\approx{nBg[i_nB]/n_sat:.1f}$")
ax.axhline(np.log10(1e-3), color='k', ls=':', lw=0.8, alpha=0.6)  # τ = 1 ms
ax.set_xlabel(r'$T$ [MeV]'); ax.set_ylabel(r'$\log_{10}\,\tau$ [s]')
ax.set_ylim(-70, 100); ax.grid(alpha=0.3); ax.legend(fontsize=8)
ax.set_title(rf"trapped {phase3} {flavor3}/{charge3} — σ={int(sigma_sel3)}, "
             rf"$Y_L\approx{YL_used:.2f}$ — {stag_sel}")
fig.tight_layout(); plt.show()


# %% [markdown]
# ### Diagnostic (4) — fixed $T,\sigma$, one method: overlay phase

# %%
# =============================================================================
#  (4) Fixed T, Y_L, σ, one method — overlay phase (unpaired / CFL / unpCFL).
# =============================================================================
T_sel4, sigma_sel4, flavor4, charge4 = 30.0, sigma_list[2], 'saddlepoint', 'coulomb_minimize'

_phases4 = [
    ('unpaired', 'tab:orange', '--', 'unpaired'),
    ('cfl',      'tab:green',  '-.', 'CFL'),
    ('unpCFL',   'tab:blue',   '-',  'unpCFL'),
]
curves = [(_get(flavor4, charge4, ph, sigma_sel4), _iT(T_sel4), c, ls, lbl)
          for ph, c, ls, lbl in _phases4]
nuc_panels(curves, rf"trapped {flavor4}/{charge4} — $T\approx{Tg[_iT(T_sel4)]:.0f}$ MeV, "
                   rf"$Y_L\approx{YL_used:.2f}$, σ={int(sigma_sel4)} — {stag_sel}  (vary phase)")


# %% [markdown]
# # Part III — Load produced tables
#
# Rebuild every in-memory object from the saved tables. Run this (after Part I) to
# work with previously generated data **without** re-running Part II.

# %% [markdown]
# ## III.1 — Hadronic interpolators & TOV
#
# Build the four interpolator sets `H` (+ tables `H_table`) and the
# $P_H(\mu_B)$ comparator, then load the TOV sequences.
#
# ```python
# # usage examples
# H['betaeq']['P'](n_B, T)            # pressure, β-equilibrium          [MeV/fm^3]
# H['trapped']['mu_e'](n_B, Y_L, T)   # electron chem. pot., trapped ν   [MeV]
# H['iso_trapped']['T'](n_B, Y_L, S)  # temperature on an isentrope      [MeV]
# # TOV arrays — columns: 0=e_c 1=P_c 2=n_Bc 3=R[km] 4=M[Msun] 5=Mb[Msun]
# tov_cold[:, 3], tov_cold[:, 4]      # R, M of the cold hadronic branch
# tov_trapped[(0.25, 2.0)]            # trapped sequence at (Y_L, S)
# ```

# %%
# =============================================================================
#  Hadronic interpolators (from disk) + the cold P_H(mu_B) comparator.
# =============================================================================
H, H_table = load_hadronic_tables(xsd_tag)

# P_H(mu_B) at T≈0 — the hadronic side of the CFL no-rehadronization filter.
# P(mu_B) is single-valued because mu_B(n_B) is monotone at fixed T; the engine
# helper sorts by mu_B and returns a linear interpolator (NaN outside range).
P_H_of_muB, mu_B_H_sorted, P_H_sorted = nuc_an.build_PH_of_muB(
    H['betaeq'], H_table['betaeq'].grids['n_B'], H_table['betaeq'].grids['T'][0])


# ── TOV sequences from disk ───────────────────────────────────────────────────
tov_cold = np.loadtxt(f'../output/tables_tov/tov_hadronic_betaeq_2famphi_{xsd_tag}_T0.dat')
tov_trapped = {}
for _p in sorted(glob.glob(f'../output/tables_tov/tov_hadronic_trapped_2famphi_{xsd_tag}_YL*_S*.dat')):
    _m = re.search(r'YL([0-9.]+)_S([0-9.]+)\.dat', _p)
    tov_trapped[(float(_m.group(1)), float(_m.group(2)))] = np.loadtxt(_p)
print(f'loaded TOV: cold + {len(tov_trapped)} trapped (Y_L, S) sequences')


# %% [markdown]
# ## III.2 — Quark-phase quantities (Q\* tables)
#
# Load the $Q^*$ tables into `Qstar_sets` (keyed by filename stem).
#
# The quark-star $M$–$R$ and the CFL $P(\mu_B)$ are **recomputed on demand** from
# `solve_cfl` (cheap, not stored). The saved quark EoS tables in `tables_Qphase/`
# are inputs to $Q^*$ generation only; to interpolate one directly, e.g.:
#
# ```python
# # from eos.alphabag.compute_tables import load_eos_tables_multi, build_interpolators
# # t = load_eos_tables_multi('../output/tables_Qphase/eos_quark_cfl_<tag>.dat')
# # Q = build_interpolators(t);  Q['P'](n_B, T)
# ```

# %%
# =============================================================================
#  Load all Q* tables → Qstar_sets, keyed by filename stem.
# =============================================================================
# Key builder for a Q* table (same convention the Part II compute cell writes
# files with); defined here too so Part III works without running Part II.
def qs_stem(stag, bg, charge, ph, sg):
    """Filename stem / dict key for one Q* table."""
    return f"{bg}_saddlepoint_{charge}_{ph}_{stag}_s{int(sg)}"

Qstar_sets = {}
for path in sorted(glob.glob('../output/tables_Qstar/Qstar_*.dat')):
    stem = os.path.splitext(os.path.basename(path))[0].removeprefix('Qstar_')
    t = load_Qstar_table(path)
    Qstar_sets[stem] = dict(table=t, interp=build_Qstar_interpolators(t))

print(f"Loaded {len(Qstar_sets)} Q* tables.")


# %% [markdown]
# ## III.3 — Nucleation observables
#
# Load the trapped nucleation tables into `nuc_sets` (keyed by filename stem
# `Htrapped_{flavor}_{charge}_{phase}_{tag}_s{σ}`).

# %%
# =============================================================================
#  Load trapped nucleation observables → nuc_sets, keyed by filename stem.
# =============================================================================
nuc_sets = {}
for path in sorted(glob.glob('../output/tables_nucleation/nucleation_Htrapped_*.dat')):
    stem = os.path.splitext(os.path.basename(path))[0].removeprefix('nucleation_')
    nuc_sets[stem] = load_thermal_nucleation_table(path)

print(f"Loaded {len(nuc_sets)} trapped nucleation tables.")


# %% [markdown]
# # Part IV — Analysis & plots
#
# Critical surface tension at the PNS centre (IV.2–IV.3) and the paper figures
# (IV.4+). Needs Part I + Part III (or Part II).

# %% [markdown]
# ## IV.0 — $\sigma_{\rm crit}$ engine setup
#
# Bind the `nucleation.analysis.sigma_crit` engine (`nuc_an`) to this notebook's
# state: the CFL-filter config `_filt_cfg`, the nucleation config `_nuc_cfg`, and
# the star-match `_star` (cold + trapped TOV → central conditions). Two small
# presentation helpers (`_run_scan`, `_plot_sig_grid`) wrap the scan/plot with the
# file-naming and title conventions. Everything else calls `nuc_an.*` directly.

# %%
# =============================================================================
#  sigma_crit engine binding.  (cheap — run once; required by IV.2–IV.3)
# =============================================================================

# ---- user-configurable inputs ----------------------------------------------
# PNS evolution snapshots the paper is evaluated at (reused by Figs 2, 6, 7).
# Two representative times along the proto-NS deleptonization/cooling track:
#   t_0     (just after formation): Y_L^H = 0.35, S = 1.5  (lepton-rich, cooler)
#   t_Tmax  (temperature peak):     Y_L^H = 0.25, S = 2.0  (deleptonized, hottest)
# Colours: orange = t_0, red = t_Tmax.  Single source of truth — change here only.
PNS_T0   = dict(YLH=0.35, S=1.5, color='#fd8d3c', lbl=r'$t_0$')
PNS_TMAX = dict(YLH=0.25, S=2.0, color='#e31a1c', lbl=r'$t_{T_\mathrm{max}}$')

# Star match (Y_L, S) and CFL-filter acceptance constants.  sigma_crit is
# evaluated at t_Tmax (hottest -> hardest to nucleate -> most conservative).
YLH, S       = PNS_TMAX['YLH'], PNS_TMAX['S']   # trapped/isentropic PNS the σ_crit is evaluated at
sig_lo, sig_hi = 0.001, 300.0     # σ scan range for the brentq root-find [MeV/fm^2]
m_s_fixed    = 100.0            # strange-quark mass in the CFL filter [MeV]
n_B_grid_cfl = np.linspace(0.05, 2.5, 250)                      # fm^-3
e_c_vec_tov  = generate_ec_logspace(e_min=100, e_max=2000, n_points=80)
M_max_window  = (2.0, np.inf)   # accept iff M_max ≥ 2.0 M_⊙
e_over_nB_max = 930.0           # Witten bound [MeV]

# Methods for the σ_crit table (IV.2): (label, flavor, charge, phase).
_sig_methods = [
   # ('frozen LCN unp',     'frozen',      'lcn',              'unpaired'),
    ('saddle Cmin CFL',    'saddlepoint', 'coulomb_minimize', 'cfl'),
    ('saddle Cmin unpCFL', 'saddlepoint', 'coulomb_minimize', 'unpCFL'),
]

# Config objects consumed by the engine (P_H arrays come from Part III.1).
# tov_backend=TOV_BACKEND routes the CFL M_max filter and the M-R replay through
# the fast solver too — that (not Part II) is where most TOV time is spent.
# No-rehadronization filter = the bulk ΔP(mu_B)=P_CFL-P_H test at T=0 (CFL vs cold
# neutrinoless hadronic), using the P_H_of_muB comparator built above. A set must
# pass no_rehad (ΔP stays >0 past its crossing -> else 're-hadr.') AND
# no_rehad_strong (ΔP monotonically increasing in mu_B -> else 'quasi-re-hadr.').
# Both are acceptance criteria. merge_rehad_labels=True folds both into one
# 're-hadr.' zone.
_filt_cfg = nuc_an.FilterConfig(
    P_H_of_muB=P_H_of_muB, mu_B_H_sorted=mu_B_H_sorted, P_H_sorted=P_H_sorted,
    m_s=m_s_fixed, n_B_grid=n_B_grid_cfl, e_c_vec_tov=e_c_vec_tov,
    M_max_window=M_max_window, e_over_nB_max=e_over_nB_max,
    tov_backend=TOV_BACKEND, merge_rehad_labels=False)
_nuc_cfg = nuc_an.NucConfig(sig_lo=sig_lo, sig_hi=sig_hi,
                            tau_target=tau_target, V=V_nuc)

# Cold-β-eq + trapped TOV → (MT0 → central state) map, cached per (Y_L, S).
_star_cache = {}
def _star_for(yl, s):
    k = (round(yl, 3), round(s, 3))
    if k not in _star_cache:
        _star_cache[k] = nuc_an.make_star_match(
            H, yl, s,
            f'../output/tables_tov/tov_hadronic_betaeq_2famphi_{xsd_tag}_T0.dat',
            f'../output/tables_tov/tov_hadronic_trapped_2famphi_{xsd_tag}_YL{yl:.2f}_S{s:.1f}.dat')
    return _star_cache[k]
_star = _star_for(YLH, S)

# ── Presentation glue for the grid scan (IV.3) ───────────────────────────────
def _plot_sig_grid(d, MT0, fl, ch, ph):
    """Draw one saved/scanned σ_crit grid with the notebook's title + overlays."""
    nuc_an.plot_sigma_crit_grid(
        B4_grid_scan, Delta0_grid_scan, np.array(alpha_slices),
        d['sig_crit'], d['cfl_ok'], d['M_max'], d['reason'],
        scan_label=f"{fl}/{ch}/{ph}", mass_window=M_max_window,
        title_extra=(rf", $M_{{T0}}$={MT0:.1f} $M_\odot$, $Y_L$={YLH}, S={S}, "
                     rf"$m_s$={m_s_fixed:.0f}, $\tau$={tau_target*1e3:g} ms — {xsd_tag}"),
        show_split_grey=True, show_cfl_boundary=True, show_filter_lines=True,
        show_mmax_lines=True, show_param_sets=True, param_sets=quark_param_sets,
        mc_csv=f'../output/mc_cfl/cfl_accepted_{xsd_tag}.csv')

def _regime_grid(sig_crit, alpha_slices, B4_grid, Delta0_grid, MT0, slices=None, n_jobs=-1):
    """Classify the unpCFL critical droplet R* over the (B4, Delta0) grid at each
    cell's sigma_crit: 0=R_unp, 1=R_Delta (crossover kink), 2=R_CFL, -1=no droplet.

    R* (the barrier peak, from the two-layer unpCFL solve) is one of three radii:
    the pure-unpaired R_unp, the pure-CFL R_CFL, or the pairing-coherence radius
    R_Delta=R_x(T)=hc/Delta(T) when the peak is pinned at the switching kink. We
    recompute all three (the saved grid stores only sigma_crit) and label R* by the
    nearest. The star centre (nBHc, T_c) is MT0-only -> computed once for the grid.
    ``slices`` selects which alpha indices to fill (default all)."""
    from joblib import Parallel, delayed
    _, Tc, Hpt = nuc_an.central_state(float(MT0), _star)
    base = (Hpt, Tc, 'saddlepoint', 'coulomb_minimize')
    def code(al, b4, dl, sc):
        if not np.isfinite(sc):
            return -1
        with np.errstate(divide='ignore', invalid='ignore'):     # crossover_radius @ Delta0=0
            p = get_alphabag_custom(alpha=al, B4=b4, m_s=m_s_fixed)
            Rs = nuc_an.critical_droplet_pt(sc, *base, 'unpCFL', {}, p, dl, _nuc_cfg)[0]
            if not np.isfinite(Rs):
                return -1
            Rc = nuc_an.critical_droplet_pt(sc, *base, 'cfl',      {}, p, dl, _nuc_cfg)[0]
            Ru = nuc_an.critical_droplet_pt(sc, *base, 'unpaired', {}, p, dl, _nuc_cfg)[0]
            Rx = float(nuc_an.crossover_radius(Tc, dl))
        cand = [(v, c) for v, c in ((Ru, 0), (Rx, 1), (Rc, 2)) if np.isfinite(v)]
        return min(cand, key=lambda t: abs(Rs - t[0]))[1]
    NA, ND, NB = sig_crit.shape
    reg = np.full(sig_crit.shape, -1, dtype=int)
    for ia in (range(NA) if slices is None else slices):
        res = Parallel(n_jobs=n_jobs)(
            delayed(code)(alpha_slices[ia], B4_grid[j], Delta0_grid[i], sig_crit[ia, i, j])
            for i in range(ND) for j in range(NB))
        reg[ia] = np.array(res).reshape(ND, NB)
    return reg

# STAR-WIDE σ_crit: τ > τ_target is demanded at N_SHELLS densities from
# NB_SHELL_MIN up to the star centre, not just at r=0 — the centre is NOT always
# the easiest nucleation site (|Δf| peaks near ~2 n_sat close to the SQM-stability
# corner, so outer shells can nucleate at σ where the centre is already quiet).
# N_SHELLS=1 recovers the old centre-only definition. 6 shells is CONVERGED
# (σ_crit matches 12 shells to <0.5%; the controlling shell sits at the broad
# Δf peak ~2 n_sat) at ~2.9× centre-only cost (early-exit keeps nucleating-σ
# points ~1 shell; 12 shells is ~5.7× for no accuracy gain).
N_SHELLS     = 6
NB_SHELL_MIN = 0.25          # fm^-3 (~1.6 n_sat) — lowest shell checked

def _run_scan(MT0s, fl, ch, ph, do_plot=True):
    """CFL filter (cached) + σ_crit per MT0; save each grid to .npz; optional plot."""
    res = nuc_an.run_sigma_crit_scan(
        MT0s, fl, ch, ph, alpha_slices, B4_grid_scan, Delta0_grid_scan,
        _filt_cfg, _nuc_cfg, _star, n_jobs=scan_n_jobs, reuse_filter=True,
        xsd_tag=xsd_tag, extra_save=dict(YLH=YLH, S=S),
        n_shells=N_SHELLS, nB_shell_min=NB_SHELL_MIN,
        save_path_fmt=('../output/mc_cfl/sigma_crit_grid_{xsd}_MT0{MT0:.2f}_'
                       '{flavor}-{charge}-{phase}.npz'))
    if do_plot:
        for _MT0, _d in res.items():
            _plot_sig_grid(_d, _MT0, fl, ch, ph)
    return res

print(f"engine bound: star (Y_L={YLH}, S={S}), σ∈[{sig_lo},{sig_hi}], "
      f"τ_target={tau_target:g} s, joblib={nuc_an._HAVE_JOBLIB}")


# %% [markdown]
# ### IV.0b — Validate the configured quark sets at $T=0$
#
# Apply the CFL acceptance filters (Witten bound, no re-hadronization,
# $M_{\max}$ window, 2-flavour stability — same physics as the IV.3 grid scan)
# to every set in `quark_param_sets`, then overlay their cold quark-star
# $M$–$R$ curves on the hadronic reference and the observational constraints.

# %%
# =============================================================================
#  Validate EVERY set in quark_param_sets with the engine filters + M-R plot.
# =============================================================================
_REASON_TXT = {'solve': 'CFL solve failed', 'witten': 'Witten bound',
               'rehadr': 're-hadronizes', 'rehad_quasi': 'quasi-re-hadronizes',
               'mmax': 'M_max outside window',
               'OK': 'ALL PASS', 'twoflavor': '2-flavor matter bound'}

val_curves = []                       # (tag, curve dict, all-pass flag)
for p in quark_param_sets:
    stag = q_tag_of(p)
    ok, M_max_v, reason = nuc_an.passes_cfl_filters(
        p['alpha'], p['B4'], p['Delta0'], _filt_cfg)
    c = nuc_an.replay_cfl(p['alpha'], p['B4'], p['Delta0'], _filt_cfg)
    print(f"── {stag}:  M_max={M_max_v if np.isfinite(M_max_v) else float('nan'):.3f} "
          f"M_sun  →  {_REASON_TXT[reason]}")
    if c is not None:
        val_curves.append((stag, c, ok))

fig, ax = plt.subplots(figsize=(6.5, 5))
ax.plot(tov_cold[:, 3], tov_cold[:, 4], 'k-', lw=2, label='Hadronic (T=0)')
for (stag, c, allp), col in zip(val_curves,
                                plt.cm.viridis(np.linspace(0, 1, max(len(val_curves), 1)))):
    ax.plot(c['R'], c['M'], color=col, lw=2, label=f"{stag} {'✓' if allp else '✗'}")
ax.axhline(M_max_window[0], color='red', ls=':', lw=1,
           label=rf'${M_max_window[0]:g}\,M_\odot$')
ax.set_xlim(7, 16); ax.set_ylim(0, 3.0)
add_observational_constraints(ax, CONTOUR_DIR)   # NICER/HESS behind the curves
ax.set_xlabel(r'$R$ [km]'); ax.set_ylabel(r'$M$ [$M_\odot$]')
ax.set_title('CFL quark stars at T=0 (✓ = all filters pass)')
ax.legend(fontsize=7); ax.grid(alpha=0.3)
fig.tight_layout(); plt.show()


# %% [markdown]
# ## IV.2 — $\sigma_{\rm crit}$ table: $M_{T0}$ × method, per quark set
#
# $\sigma_{\rm crit}$ for a few $M_{T0}$ across the methods in `_sig_methods`
# (including the two-layer unpCFL droplet), for every set in `quark_param_sets`.
# Each $\sigma_{\rm crit}$ is a `nuc_an.sigma_target_pt` root-find at the
# trapped/isentropic star centre; the central states are $M_{T0}$-only (hadronic
# EoS is fixed) so they are computed once.

# %%
# =============================================================================
# Critical surface tension table: sigma_target over MT0 x method, for each quark
# parametrization, at the trapped/isentropic star centre (YLH, S, tau from IV.0).
# HEAVY: loops quark_param_sets x MT0 x methods (each a quark solve + brentq).
# Needs the IV.0 engine cell (nuc_an, _nuc_cfg, _star).
# =============================================================================
# Central states (quark-independent) computed once.
MT0_list = [1.0, 1.4]
_states = {MT0: nuc_an.central_state(MT0, _star) for MT0 in MT0_list}
for MT0 in MT0_list:
    nBHc, T_c, _ = _states[MT0]
    print(f"MT0={MT0:.2f}:  nBHc={nBHc:.4f} fm^-3,  T_c={T_c:.2f} MeV")

# One sigma_target table per quark parametrization.
for p in quark_param_sets:
    params_set = get_alphabag_custom(alpha=p['alpha'], B4=p['B4'], m_s=p['m_s'])
    Delta0_set = p['Delta0']
    tbl = pd.DataFrame(index=[lbl for lbl, *_ in _sig_methods])
    for MT0 in MT0_list:
        _, T_c, H_pt = _states[MT0]
        tbl[f"MT0={MT0:.1f}"] = [
            nuc_an.sigma_target_pt(H_pt, T_c, fl, ch, ph, params_set, Delta0_set, _nuc_cfg)
            for _, fl, ch, ph in _sig_methods]
    print(f"\n=== {q_tag_of(p)} ===  sigma_target [MeV/fm^2] "
          f"(YL={YLH}, S={S}, tau={tau_target:g}s)")
    print(tbl.round(3).to_string())


# %% [markdown]
# ## IV.3 — $\sigma_{\rm crit}$ grid scan over $(\alpha_s, B^{1/4}, \Delta_0)$
#
# Scan the quark parameters and, per cell, apply the four CFL filters (Witten,
# no-rehadronization, $M_{\max}$, 2-flavour stability) and compute
# $\sigma_{\rm crit}$. Output: $\sigma_{\rm crit}$ heatmaps over $(B^{1/4},\Delta_0)$,
# one panel per $\alpha_s$; CFL-rejected cells are blank. Grids are small here for a
# first pass — enlarge them once the pipeline looks right.

# %%
# =============================================================================
#  IV.3 grid scan: sigma_crit over (alpha_s, B^1/4, Delta_0), one run per method.
#  Each (flavor, charge, phase) case = CFL filter (cached, phase-independent) +
#  sigma_crit, saved to its own .npz, plotted, and summarised. Heavy —
#  coulomb_minimize re-solves per σ (unpCFL twice). Trim scan_cases / shrink the
#  grids for a quick first pass.
# =============================================================================
# ---- grid knobs -------------------------------------------------------------
MT0_grid_thr     = [1.4]                                  # nucleation threshold(s) [M_sun]
alpha_slices     = [0, 0.1 * np.pi / 2, 0.2 * np.pi / 2, 0.3 * np.pi / 2]   # α_s panels
B4_grid_scan     = np.linspace(130.0, 180.0, 51)         # B^1/4 [MeV]
Delta0_grid_scan = np.linspace(0.0, 200.0, 51)           # Δ_0  [MeV]
scan_n_jobs      = -1                                     # joblib workers (-1 = all cores)
# ---- methods to run: (flavor, charge, phase). The FIRST is the "primary" one,
#      it drives results_scan + the reload / M-R cells below. Trim to taste.
scan_cases = [
    ('saddlepoint', 'coulomb_minimize', 'unpCFL'),
    #('saddlepoint', 'coulomb_minimize', 'cfl'),
]
# -----------------------------------------------------------------------------
scan_flavor, scan_charge, scan_phase = scan_cases[0]    

# %%
# =============================================================================
#  IV.3 grid scan: sigma_crit over (alpha_s, B^1/4, Delta_0), one run per method.
#  Each (flavor, charge, phase) case = CFL filter (cached, phase-independent) +
#  sigma_crit, saved to its own .npz, plotted, and summarised. Heavy —
#  coulomb_minimize re-solves per σ (unpCFL twice). Trim scan_cases / shrink the
#  grids for a quick first pass.
# =============================================================================
# ---- grid knobs -------------------------------------------------------------
MT0_grid_thr     = [1.4]                                  # nucleation threshold(s) [M_sun]
alpha_slices     = [0, 0.1 * np.pi / 2, 0.2 * np.pi / 2, 0.3 * np.pi / 2]   # α_s panels
B4_grid_scan     = np.linspace(130.0, 180.0, 51)         # B^1/4 [MeV]
Delta0_grid_scan = np.linspace(0.0, 200.0, 51)           # Δ_0  [MeV]
scan_n_jobs      = -1                                     # joblib workers (-1 = all cores)
# ---- methods to run: (flavor, charge, phase). The FIRST is the "primary" one,
#      it drives results_scan + the reload / M-R cells below. Trim to taste.
scan_cases = [
    ('saddlepoint', 'coulomb_minimize', 'unpCFL'),
    #('saddlepoint', 'coulomb_minimize', 'cfl'),
]
# -----------------------------------------------------------------------------
scan_flavor, scan_charge, scan_phase = scan_cases[0]      # primary (downstream default)

results_all = {}
for fl, ch, ph in scan_cases:
    print(f"\n=== {fl}/{ch}/{ph}  @ MT0={MT0_grid_thr} ===", flush=True)
    results_all[(fl, ch, ph)] = _run_scan(MT0_grid_thr, fl, ch, ph, do_plot=True)
results_scan = results_all[(scan_flavor, scan_charge, scan_phase)]   # primary, for the M-R replay
print("\nAll cases done. Saved sigma_crit_grid_*.npz in ../output/mc_cfl/.")



# %% [markdown]
# ### Re-plot a saved scan (no rerun)
#
# Load any `sigma_crit_grid_*.npz` and redraw with whatever overlays you like.

# %%
# =============================================================================
#  Reload a saved sigma_crit grid and re-plot (no scanning).
# =============================================================================
reload_npz = (f'../output/mc_cfl/sigma_crit_grid_{xsd_tag}_MT0{MT0_grid_thr[0]:.2f}_'
              f'{scan_flavor}-{scan_charge}-{scan_phase}.npz')
_d = np.load(reload_npz, allow_pickle=False)
nuc_an.plot_sigma_crit_grid(
    _d['B4_grid'], _d['Delta0_grid'], _d['alpha_slices'],
    _d['sig_crit'], _d['cfl_ok'], _d['M_max'], _d['reason'],
    scan_label=f"{_d['flavor'].item()}/{_d['charge'].item()}/{_d['phase'].item()}",
    mass_window=tuple(_d['mass_window']),
    title_extra=(rf", $M_{{T0}}$={float(_d['MT0']):.1f} $M_\odot$, $Y_L$={float(_d['YLH'])}, "
                 rf"S={float(_d['S'])}, $m_s$={float(_d['m_s']):.0f}, "
                 rf"$\tau$={float(_d['tau'])*1e3:g} ms — {xsd_tag}"),
    show_split_grey=True, show_cfl_boundary=True, show_filter_lines=True,
    show_mmax_lines=True, show_sigcrit_lines=False, show_param_sets=True,
    param_sets=quark_param_sets,
    mc_csv=f'../output/mc_cfl/cfl_accepted_{xsd_tag}.csv')
print(f"re-plotted from {reload_npz}")


# %% [markdown]
# ### $M$–$R$ and $P_{\rm CFL}(\mu_B)$ for the acceptable sets, coloured by $\sigma_{\rm crit}$
#
# Replay every $\sigma_{\rm crit}$-acceptable set (CFL-pass **and** nucleating at
# `mr_MT0`): re-solve the cold CFL EoS + TOV and overlay $M$–$R$ and $P(\mu_B)$,
# coloured by $\sigma_{\rm crit}$ on the same viridis scale as the heatmap. The
# hadronic reference is black.

# %%
# =============================================================================
# M-R and P_CFL(mu_B) for the acceptable sets, coloured by sigma_crit.
# Heavy lifting lives in nuc_an.replay_accepted (parallel EoS + TOV replay).
# =============================================================================
# ---- user-configurable inputs ----------------------------------------------
mr_source     = results_scan           # {MT0: {sig_crit, ...}}; point elsewhere,
                                       # e.g. results_all[('saddlepoint','coulomb_minimize','cfl')]
mr_MT0        = float(MT0_grid_thr[0])  # which threshold's acceptable set to draw
mr_max_curves = 400                    # even subsample for the overlay (None = all;
                                       # a few hundred curves are visually identical
                                       # to thousands at a fraction of the cost)
# ---- computation ------------------------------------------------------------
_sig = mr_source[mr_MT0]['sig_crit']
_fin = _sig[np.isfinite(_sig)]
norm = plt.Normalize(vmin=float(_fin.min()), vmax=float(_fin.max()))  # == heatmap scale
cmap = plt.cm.viridis

mr_curves = nuc_an.replay_accepted(_sig, alpha_slices, B4_grid_scan,
                                   Delta0_grid_scan, _filt_cfg,
                                   max_curves=mr_max_curves, n_jobs=scan_n_jobs)

# ---- layout ------------------------------------------------------------------
fig, (axMR, axP) = plt.subplots(1, 2, figsize=(12, 4.8))
# (a) Mass-radius (stable branch); hadronic reference + 2 Msun line.
axMR.plot(tov_cold[:, 3], tov_cold[:, 4], 'k-', lw=2, label='Hadronic (T=0)')
axMR.axhline(2.0, color='red', ls=':', lw=1, label=r'$2\,M_\odot$')
for c, sc in mr_curves:
    axMR.plot(c['R'], c['M'], color=cmap(norm(sc)), lw=0.7, alpha=0.8)
axMR.set_xlim(7, 16); axMR.set_ylim(0, 3.0)
axMR.set_xlabel(r'$R$ [km]'); axMR.set_ylabel(r'$M$ [$M_\odot$]')
axMR.legend(fontsize=8); axMR.grid(alpha=0.3)
axMR.set_title('Mass–radius (acceptable CFL sets)')
# (b) P_CFL(mu_B); hadronic P_H(mu_B) in black.
axP.plot(mu_B_H_sorted, P_H_sorted, 'k-', lw=2, label=r'$P_H$ (T≈0)')
for c, sc in mr_curves:
    axP.plot(c['mu'], c['P'], color=cmap(norm(sc)), lw=0.7, alpha=0.8)
axP.set_xlabel(r'$\mu_B$ [MeV]'); axP.set_ylabel(r'$P$ [MeV/fm$^3$]')
axP.legend(fontsize=8); axP.grid(alpha=0.3)
axP.set_title(r'$P_{\rm CFL}(\mu_B)$ vs $P_H$')

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cb = fig.colorbar(sm, ax=[axMR, axP], shrink=0.85, pad=0.02)
cb.set_label(r'$\sigma_{\rm crit}$ [MeV/fm$^2$]')
fig.suptitle(rf"Acceptable sets (CFL-pass & nucleating) — $M_{{T0}}$={mr_MT0:.1f} "
             rf"$M_\odot$, {scan_flavor}/{scan_charge}/{scan_phase}, {xsd_tag}", y=1.02)
plt.show()


# %% [markdown]
# ## IV.4 — Paper figures
#
# Publication plots. `set_paper_style()` / `panel_label()` come from
# `nucleation.analysis.figure` (imported in I.1): CMU Serif at LaTeX-matched
# sizes, inward ticks, frame-less legends. Run the style cell once, then any
# figure.

# %%
set_paper_style()      # publication rcParams (nucleation.analysis.figure.style)


# %% [markdown]
# ### Paper Figure 1 — nucleation barrier & critical quantities
#
# Four panels at fixed $Y_L, \sigma$, one set and method. **(a)** barrier $W(R)$ at
# $T=30$ MeV for $n_B^H/n_0=1,3,5$ (colour) and the three phases (line style), dot
# = critical point. **(b,c,d)** $R_*$, $W_*/T$, $\log_{10}\tau$ vs $n_B^H/n_0$,
# coloured by $T$. $W(R)$ is recomputed with `compute_energy_barrier`.

# %%
# ============================================================================
#  Figure 1.  Knobs -> which set / method / fixed values.
# ============================================================================
# One figure per quark parametrization. Reverse order so set[0]'s helper
# bindings (_params, _f1_get, _ref, ...) persist for the sigma-sweep cell below.
for F1_SET in [quark_param_sets[0]]:
    F1_FLAVOR = 'saddlepoint'                        # 'frozen' | 'saddlepoint'
    F1_CHARGE = 'coulomb_minimize'                   # 'lcn' | 'gcn' | 'coulomb_minimize'
    F1_YLH    = 0.25
    F1_SIGMA  = 100.0
    F1_TW     = 30.0                                 # panel (a) temperature [MeV]
    F1_DENS   = [1.0, 4, 8]                      # panel (a) n_B^H / n_0
    F1_TEMPS  = [20.0, 30.0, 40.0, 50.0, 60.0]       # panels (b,c,d) temperatures [MeV]

    _PHASE_LS    = {'unpCFL': '-', 'cfl': '--', 'unpaired': ':'}   # line style per phase
    _PHASE_FILL  = {'unpCFL': True, 'cfl': False, 'unpaired': False}
    _PHASE_LBL   = {'unpCFL': 'unpCFL', 'cfl': 'CFL', 'unpaired': 'unpaired'}
    _PHASE_LW    = PHASE_LW                                        # shared (solid unpCFL thicker)
    _PHASE_ALPHA = PHASE_ALPHA                                     # shared opacity (one source)

    _stag   = q_tag_of(F1_SET)
    _params = get_alphabag_custom(alpha=F1_SET['alpha'], B4=F1_SET['B4'], m_s=F1_SET['m_s'])


    def _f1_get(phase, sg=F1_SIGMA):
        """nuc_sets table for this method/phase/set/sigma (or None)."""
        return nuc_sets.get(f"Htrapped_{F1_FLAVOR}_{F1_CHARGE}_{phase}_{_stag}_s{int(sg)}")


    def _f1_Rx(T):
        """unpCFL crossover radius R_x(T) = hc/Delta(T) — same CFL gap as the EoS
        (Tc = T_critical(Delta0)); delegate to the engine so they never diverge."""
        return float(nuc_an.crossover_radius(T, F1_SET['Delta0']))


    _ref = _f1_get('unpCFL')
    _nBg = _ref.hadronic_grids['n_B_H']
    _Tg  = _ref.hadronic_grids['T']
    _iYL = int(np.argmin(np.abs(_ref.hadronic_grids['Y_L_H'] - F1_YLH)))
    _YLused = _ref.hadronic_grids['Y_L_H'][_iYL]


    def _iT(T):
        return int(np.argmin(np.abs(_Tg - T)))


    # PRD size: mode='double' (4.75" square). Change to 'single'/'double' to resize.
    fig, ((axA, axB), (axC, axD)) = paper_grid('2x2', mode='double', placeholder=False, **PAPER_STYLE)

    # ---- (a) W(R): density = colour, phase = line style ----
    _cA = plt.cm.viridis(np.linspace(0.12, 0.85, len(F1_DENS)))
    _Rg = np.linspace(0.01, 14.0, 400)
    _Wmax = 0.0
    for _ci, _x in enumerate(F1_DENS):
        for _ph, _ls in _PHASE_LS.items():
            _eb = compute_energy_barrier(
                H['trapped'], _x * n_sat, F1_TW, F1_SIGMA,
                electric_charge_mode=F1_CHARGE, params=_params, flavor_mode=F1_FLAVOR,
                quark_phase=_ph, Delta0=F1_SET['Delta0'], Y_L_H=F1_YLH, R_values=_Rg,
                switching_mode='step', Rx=(_f1_Rx(F1_TW) if _ph == 'unpCFL' else None))
            axA.plot(_Rg, _eb.W, color=_cA[_ci], ls=_ls,
                     lw=_PHASE_LW[_ph], alpha=_PHASE_ALPHA[_ph])
            if np.isfinite(_eb.W).any():
                _k = int(np.nanargmax(_eb.W))
                if _ph != 'unpaired':                                   # no marker on unpaired
                    axA.plot(_Rg[_k], _eb.W[_k], 'o', ms=6, color=_cA[_ci], mec=_cA[_ci],
                             mfc=(_cA[_ci] if _PHASE_FILL[_ph] else 'white'),
                             zorder=(6 if _PHASE_FILL[_ph] else 5))  # solid dots on top of empty
                _Wmax = max(_Wmax, float(_eb.W[_k]))
    # shade R <= R_Delta (unpCFL->CFL crossover radius R_x at this T)
    _Rdelta = _f1_Rx(F1_TW)
    if np.isfinite(_Rdelta):
        axA.axvspan(0, _Rdelta, color='tab:blue', alpha=0.07, zorder=0)
    axA.axhline(0, color='0.6', lw=0.7, zorder=0)
    axA.set_xlim(0, 7); axA.set_ylim(0, 10000)
    axA.set_xlabel(r'$R$ [fm]'); axA.set_ylabel(r'$W$ [MeV]')
    axA.set_title(rf'$Y_L^H={F1_YLH}$, $T={F1_TW:.0f}$ MeV, '
                  rf'$\sigma={F1_SIGMA:.0f}$ MeV/fm$^2$')
    panel_label(axA, '(a)', corner='lower')
    _lgA = axA.legend([Line2D([], [], color=_cA[i], lw=2) for i in range(len(F1_DENS))],
                      [rf'$n_B^H/n_\mathrm{{sat}}={int(x)}$' for x in F1_DENS], loc='upper right',labelspacing=0.25)
    axA.add_artist(_lgA)
    # phase legend in the opposite (also empty) top corner -> no overlap with density legend or peak
    axA.legend([Line2D([], [], color='0.3', ls=_PHASE_LS[p]) for p in _PHASE_LS],
               [_PHASE_LBL[p] for p in _PHASE_LS], loc='upper left')

    # ---- (b,c,d) vs n_B^H/n_0: temperature = colour, phase = line style ----
    _cT = plt.cm.plasma(np.linspace(0.05, 0.85, len(F1_TEMPS)))


    def _f1_vs_nBH(ax, getter, ylabel, logy=False):
        for _ti, _T in enumerate(F1_TEMPS):
            for _ph, _ls in _PHASE_LS.items():
                _o = _f1_get(_ph)
                if _o is None:
                    continue
                with np.errstate(divide='ignore', invalid='ignore'):
                    ax.plot(_nBg / n_sat, getter(_o, _iT(_T)), color=_cT[_ti], ls=_ls,
                            lw=_PHASE_LW[_ph], alpha=_PHASE_ALPHA[_ph])
        ax.set_xlabel(r'$n_B^H/n_\mathrm{sat}$'); ax.set_ylabel(ylabel); ax.set_xlim(0.5, 10)
        if logy:
            ax.set_yscale('log')


    _f1_vs_nBH(axB, lambda o, it: o.R_c[:, _iYL, it], r'$R_*$ [fm]')
    axB.set_ylim(1, 7); axB.set_title(rf'$Y_L^H={F1_YLH}$, $\sigma={F1_SIGMA:.0f}$ MeV/fm$^2$')
    panel_label(axB, '(b)', corner='lower')
    axB.legend([Line2D([], [], color=_cT[i], lw=2) for i in range(len(F1_TEMPS))],
               [rf'$T={int(T)}\,\,\rm MeV$' for T in F1_TEMPS], loc='upper right')

    _f1_vs_nBH(axC, lambda o, it: o.W_c[:, _iYL, it] / _Tg[it], r'$W_*/T$')
    axC.set_ylim(0, 500); axC.set_title(rf'$Y_L^H={F1_YLH}$, $\sigma={F1_SIGMA:.0f}$ MeV/fm$^2$')
    panel_label(axC, '(c)', corner='lower')

    _f1_vs_nBH(axD, lambda o, it: np.log10(o.tau[:, _iYL, it]), r'$\log_{10}\,\tau$ [s]')
    axD.axhline(np.log10(1e-3), color='k', ls=(0, (1, 1)), lw=0.9)   # tau = 1 ms
    axD.set_ylim(-60, 60); axD.set_title(rf'$Y_L^H={F1_YLH}$, $\sigma={F1_SIGMA:.0f}$ MeV/fm$^2$')
    panel_label(axD, '(d)', corner='lower')

    if F1_SET is quark_param_sets[1]:                 # export (b,c,d) data for param [1]
        _d1 = []
        for _T in F1_TEMPS:
            _it = _iT(_T)
            for _ph in _PHASE_LS:
                _o = _f1_get(_ph)
                if _o is None:
                    continue
                with np.errstate(divide='ignore', invalid='ignore'):
                    _R = _o.R_c[:, _iYL, _it]; _WT = _o.W_c[:, _iYL, _it] / _Tg[_it]
                    _LT = np.log10(_o.tau[:, _iYL, _it])
                for _nb, _r, _w, _lt in zip(_nBg / n_sat, _R, _WT, _LT):
                    _d1.append((_T, _ph, float(_nb), float(_r), float(_w), float(_lt)))
        pd.DataFrame(_d1, columns=['T_MeV', 'phase', 'nBH_over_n0', 'R_star_fm',
                                   'W_over_T', 'log10_tau_s']).to_csv(
            f'../output/figure_data/fig1_bcd_{xsd_tag}_{_stag}.csv', index=False)
        print(f'wrote fig1_bcd CSV ({_stag})')
    fig.savefig(f'../output/figures/paper_fig1_barrier_{xsd_tag}_{_stag}.pdf', bbox_inches='tight')
    plt.show()

   


# %% [markdown]
# ### Paper Figure 1b (σ-sweep) — same panels but T fixed, coloured by σ
#
# Mirror of Fig 1 with the roles of $T$ and $\sigma$ swapped: fix $T$, sweep
# $\sigma$ (colour). **(a)** $W(R)$ at one $n_B^H$; **(b,c,d)** $R_*$, $W_*/T$,
# $\log_{10}\tau$ vs $n_B^H/n_\mathrm{sat}$.  Self-contained like Fig 1: pick the
# quark parametrization via `F1_SET` in the cell.  Tables must exist for each σ
# in `F1S_SIGMAS`.

# %%
# ============================================================================
#  Figure 1 variant: fix T, sweep σ (colour = σ).  Panels mirror Fig 1(a-d).
#  Self-contained (mirrors Fig 1): pick the quark parametrization here — no
#  longer inherits the Fig-1 loop's leaked bindings.
# ============================================================================
F1_SET     = quark_param_sets[0]       # <-- SELECT quark parametrization (also sets Delta0)
F1_FLAVOR  = 'saddlepoint'             # 'frozen' | 'saddlepoint'
F1_CHARGE  = 'coulomb_minimize'        # 'lcn' | 'gcn' | 'coulomb_minimize'
F1_YLH     = 0.25

F1S_T      = 30.0                      # fixed temperature [MeV]
F1S_SIGMAS = sigma_list                # σ values to sweep (colour); tables must exist
F1S_DENS_A = 3.0                       # panel (a): single n_B^H/n_sat for the W(R) sweep

# phase styling — identical to Fig 1 (unpCFL solid+thick, CFL dashed, unpaired dotted)
_PHASE_LS    = {'unpCFL': '-', 'cfl': '--', 'unpaired': ':'}
_PHASE_FILL  = {'unpCFL': True, 'cfl': False, 'unpaired': False}
_PHASE_LBL   = {'unpCFL': 'unpCFL', 'cfl': 'CFL', 'unpaired': 'unpaired'}
_PHASE_LW    = PHASE_LW                                        # shared thickness
_PHASE_ALPHA = PHASE_ALPHA                                     # shared opacity (one source)

_stag   = q_tag_of(F1_SET)
_params = get_alphabag_custom(alpha=F1_SET['alpha'], B4=F1_SET['B4'], m_s=F1_SET['m_s'])


def _f1_get(phase, sg):
    """nuc_sets table for this method/phase/set/σ (or None)."""
    return nuc_sets.get(f"Htrapped_{F1_FLAVOR}_{F1_CHARGE}_{phase}_{_stag}_s{int(sg)}")


def _f1_Rx(T):
    """unpCFL crossover radius R_x(T)=hc/Δ(T) — same CFL gap as the EoS."""
    return float(nuc_an.crossover_radius(T, F1_SET['Delta0']))


_ref = _f1_get('unpCFL', F1S_SIGMAS[0])
_nBg = _ref.hadronic_grids['n_B_H']
_Tg  = _ref.hadronic_grids['T']
_iYL = int(np.argmin(np.abs(_ref.hadronic_grids['Y_L_H'] - F1_YLH)))


def _iT(T):
    return int(np.argmin(np.abs(_Tg - T)))


_cS = plt.cm.viridis(np.linspace(0.12, 0.85, len(F1S_SIGMAS)))

# PRD size: mode='double' (4.75" square). Change to 'single'/'double' to resize.
fig, ((axA, axB), (axC, axD)) = paper_grid('2x2', mode='double', placeholder=False, **PAPER_STYLE)

# ---- (a) W(R) at fixed (T, n_B): colour = σ, line style = phase ----
_Rg = np.linspace(0.01, 14.0, 400)
_Wmax = 0.0
for _si, _sig in enumerate(F1S_SIGMAS):
    for _ph, _ls in _PHASE_LS.items():
        _eb = compute_energy_barrier(
            H['trapped'], F1S_DENS_A * n_sat, F1S_T, _sig,
            electric_charge_mode=F1_CHARGE, params=_params, flavor_mode=F1_FLAVOR,
            quark_phase=_ph, Delta0=F1_SET['Delta0'], Y_L_H=F1_YLH, R_values=_Rg,
            switching_mode='step', Rx=(_f1_Rx(F1S_T) if _ph == 'unpCFL' else None))
        axA.plot(_Rg, _eb.W, color=_cS[_si], ls=_ls,
                 lw=_PHASE_LW[_ph], alpha=_PHASE_ALPHA[_ph])
        if np.isfinite(_eb.W).any():
            _k = int(np.nanargmax(_eb.W))
            if _ph != 'unpaired':
                axA.plot(_Rg[_k], _eb.W[_k], 'o', ms=6, color=_cS[_si], mec=_cS[_si],
                         mfc=(_cS[_si] if _PHASE_FILL[_ph] else 'white'), zorder=5)
            _Wmax = max(_Wmax, float(_eb.W[_k]))
_Rdelta = _f1_Rx(F1S_T)                                     # R_Delta(T) — same for all σ
if np.isfinite(_Rdelta):
    axA.axvspan(0, _Rdelta, color='tab:blue', alpha=0.07, zorder=0)
axA.axhline(0, color='0.6', lw=0.7, zorder=0)
axA.set_xlim(0, 8); axA.set_ylim(-0.2 * _Wmax, 1.18 * _Wmax)
axA.set_xlabel(r'$R$ [fm]'); axA.set_ylabel(r'$W$ [MeV]')
axA.set_title(rf'$Y_L^H={F1_YLH}$, $T={F1S_T:.0f}$ MeV, '
              rf'$n_B^H/n_\mathrm{{sat}}={F1S_DENS_A:g}$')
panel_label(axA, '(a)', corner='lower')
_lgA = axA.legend([Line2D([], [], color=_cS[i], lw=2) for i in range(len(F1S_SIGMAS))],
                  [rf'$\sigma={int(s)}$' for s in F1S_SIGMAS], loc='upper left')
axA.add_artist(_lgA)
axA.legend([Line2D([], [], color='0.3', ls=_PHASE_LS[p]) for p in _PHASE_LS],
           [_PHASE_LBL[p] for p in _PHASE_LS], loc='upper right')


# ---- (b,c,d) vs n_B^H/n_sat at fixed T: colour = σ, line style = phase ----
def _f1s_vs_nBH(ax, getter, ylabel):
    for _si, _sig in enumerate(F1S_SIGMAS):
        for _ph, _ls in _PHASE_LS.items():
            _o = _f1_get(_ph, _sig)
            if _o is None:
                continue
            with np.errstate(divide='ignore', invalid='ignore'):
                ax.plot(_nBg / n_sat, getter(_o, _iT(F1S_T)), color=_cS[_si], ls=_ls,
                        lw=_PHASE_LW[_ph], alpha=_PHASE_ALPHA[_ph])
    ax.set_xlabel(r'$n_B^H/n_\mathrm{sat}$'); ax.set_ylabel(ylabel); ax.set_xlim(0.5, 10)


_f1s_vs_nBH(axB, lambda o, it: o.R_c[:, _iYL, it], r'$R_*$ [fm]')
axB.set_ylim(1, 15); axB.set_title(rf'$Y_L^H={F1_YLH}$, $T={F1S_T:.0f}$ MeV')
panel_label(axB, '(b)', corner='lower')
axB.legend([Line2D([], [], color=_cS[i], lw=2) for i in range(len(F1S_SIGMAS))],
           [rf'$\sigma={int(s)}$' for s in F1S_SIGMAS], loc='upper left')

_f1s_vs_nBH(axC, lambda o, it: o.W_c[:, _iYL, it] / _Tg[it], r'$W_*/T$')
axC.set_yscale('log')                                       # W_*/T ∝ σ^3 -> log spans σ better
axC.set_title(rf'$Y_L^H={F1_YLH}$, $T={F1S_T:.0f}$ MeV')
panel_label(axC, '(c)', corner='lower')

_f1s_vs_nBH(axD, lambda o, it: np.log10(o.tau[:, _iYL, it]), r'$\log_{10}\,\tau$ [s]')
axD.axhline(np.log10(1e-3), color='k', ls=(0, (1, 1)), lw=0.9)   # τ = 1 ms
axD.set_ylim(-60, 60); axD.set_title(rf'$Y_L^H={F1_YLH}$, $T={F1S_T:.0f}$ MeV')
panel_label(axD, '(d)', corner='lower')

fig.savefig(f'../output/figures/paper_fig1b_sigma_sweep_{xsd_tag}_{_stag}.pdf', bbox_inches='tight')
plt.show()


# %% [markdown]
# ### Paper Figure 2 — stellar sequences, EoS & central density
#
# Sequences (styled like Fig 1): **NS** (cold hadronic), **QS** (cold CFL, one
# green curve per quark set), **PNS $t_0$** = $(Y_L,S)=(0.35,1.5)$,
# **PNS $t_{T_\mathrm{max}}$** = $(0.25,2.0)$. Curves labelled inline (no legend).
# **(a)** $M$–$R$ with NICER/HESS/GW170817 68%/95% regions + J0740/J0952 mass
# bands. **(b)** $M$ vs $M_b$. **(c)** central temperature $T_c$ vs $M$ (NS/QS
# cold; PNS from their isentrope). **(d)** central $n_B^c/n_\mathrm{sat}$ vs $M$
# up to $M_\mathrm{max}$.

# %%
# ============================================================================
#  Figure 2.  Knobs.
# ============================================================================
F2_SET   = quark_param_sets[0]                   # quark parametrization (ref plot + Delta0)
_QS_GREENS = [ '#31a354', '#006d2c']   # QS star colours, per quark set (light->dark)

# fonts come from set_paper_style() (10 pt = PRD body), same as every other paper
# figure — do NOT override here or Fig 2 desyncs (this made PSR/HESS look tiny).

_f2params = get_alphabag_custom(alpha=F2_SET['alpha'], B4=F2_SET['B4'], m_s=F2_SET['m_s'])


def _cold_cfl_stable(pset):
    """Cold (T=0) CFL EoS -> crust-less TOV -> stable branch, for one quark set.

    Needs the baryonic-mass column (panels b/c/d plot against M_B), so it runs
    the full compute_tov_sequence + truncate rather than nuc_an.replay_cfl.
    The EoS solve itself is the engine's (warm-started solve_cfl loop)."""
    P, e, mu, ok = nuc_an.cfl_eos_at_params(pset['alpha'], pset['B4'],
                                            pset['Delta0'], _filt_cfg)
    pos = ok & (P > 0)
    tov = compute_tov_sequence(
        EOSTable_for_TOV(P=P[pos], epsilon=e[pos], nB=_filt_cfg.n_B_grid[pos]),
        e_c_vec=generate_ec_logspace(e_min=100, e_max=2000, n_points=80),
        add_crust_table='No', compute_baryonic_mass=True, compute_tidal=False,
        backend=TOV_BACKEND, verbose=False)
    st, Mmax, _ = truncate_to_stable_branch(tov, verbose=False)
    return st, Mmax


# QS = cold CFL star, one curve PER quark parametrization (both sets shown).
# Set F2_QS_SETS = [F2_SET] to fall back to a single QS branch.
F2_QS_SETS = list([quark_param_sets[0]])
_qs_tov = []
for _p in F2_QS_SETS:
    _st, _mmax = _cold_cfl_stable(_p)
    _qs_tov.append((_p, _st))
    print(f"CFL star ({q_tag_of(_p)}): M_max = {_mmax:.3f} M_sun")

# Sequence config. TOV cols: 0=e_c 1=P_c 2=n_Bc 3=R 4=M 5=Mb.
# Colours: NS near-black, QS green, PNS hot (orange=t_0, red=t_Tmax).
# Per-panel inline-label anchors = (frac_along_stable_branch, dx, dy, ha, va) -- tune to taste.
# Mass-marker label offsets in panels (c)/(d): cdot/ddot = M-dot value labels,
# cstar/dstar = the M_max star label, each (dx_pt, dy_pt, ha, va) in *points* so
# text clears the curves (sitting over the contours is fine). Tune vs the render.
_F2SEQ = [
    dict(arr=tov_cold, c='#000000', lbl='NS',
         a=(0.80, -0.2, 0.03, 'right', 'bottom'),
         b=(0.6,  -0.04, 0.12, 'center', 'bottom'),
         d=(0.78, -0.06, 0.4,  'right', 'bottom'),
         cdot=(-9, 26, 'center', 'bottom'), cstar=(-9, 34, 'center', 'bottom'),  # tilted slightly left; M_max arrow longer
         ddot=(-8, 2, 'right', 'bottom'),  dstar=(-6, -2, 'right', 'top')),
]
for _i, (_p, _st) in enumerate(_qs_tov):
    _F2SEQ.append(dict(
        arr=_st, c=_QS_GREENS[_i % len(_QS_GREENS)],       # one green per set
        lbl=(f'QS ({q_tag_of(_p)})' if len(_qs_tov) > 1 else 'QS'),
        a=(0.85, 0.15, 0.08, 'left', 'bottom'),
        b=(0.9, 0.05, -0.04, 'left', 'top'),
        d=(0.92, 0.03, 0.2, 'left', 'bottom'),
        cdot=(-13, 13, 'center', 'bottom'), cstar=(-13, 13, 'center', 'bottom'),  # ~45 deg CCW arrows
        ddot=(0, -9, 'center', 'top'),   dstar=(-7, 0, 'right', 'center')))       # M_max label on the LEFT of the point
_F2SEQ += [
    dict(arr=tov_trapped[(PNS_T0['YLH'], PNS_T0['S'])], c=PNS_T0['color'],
         lbl=r'PNS ($t_0$)', yls=(PNS_T0['YLH'], PNS_T0['S']),
         a=(0.92,  0.35, -0.02, 'left',  'top'),
         b=(0.92,  0.05, -0.10, 'left',  'top'),
         pc=(0.6,  0.02, -1.8,  'left',  'top'),
         d=(0.5,   0.03, -0.5,  'left',  'top'),
         cdot=(0, -9, 'center', 'top'),   cstar=(8, -3, 'left', 'top'),
         ddot=(8, -1, 'left', 'top'),     dstar=(7, -2, 'left', 'top')),
    dict(arr=tov_trapped[(PNS_TMAX['YLH'], PNS_TMAX['S'])], c=PNS_TMAX['color'],
         lbl=r'PNS ($t_{T_\mathrm{max}}$)', yls=(PNS_TMAX['YLH'], PNS_TMAX['S']),
         a=(0.85, -0.35, 0.10, 'right', 'bottom'),
         b=(0.72, -0.05, 0.06, 'right', 'bottom'),
         pc=(0.55, 0.02, 1.5,  'left',  'bottom'),
         d=(0.72,  0.03, 0.4,  'left',  'bottom'),
         cdot=(0, 8, 'center', 'bottom'), cstar=(8, -4, 'left', 'top'),
         ddot=(-8, 0, 'right', 'center'), dstar=(-4, 7, 'right', 'bottom')),
]


def _stable(arr):
    """Slice a TOV sequence to its stable branch (up to the first M_max)."""
    return arr[:int(np.argmax(arr[:, 4])) + 1]


def _inline(ax, xs, ys, text, color, anchor):
    """Colour-matched inline curve label at a fraction along the (finite) curve."""
    if not text:
        return
    frac, dx, dy, ha, va = anchor
    m = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = np.asarray(xs)[m], np.asarray(ys)[m]
    if xs.size == 0:
        return
    j = int(np.clip(round(frac * (xs.size - 1)), 0, xs.size - 1))
    ax.text(xs[j] + dx, ys[j] + dy, text, color=color, fontsize=10, fontweight='bold',
            ha=ha, va=va, zorder=6, clip_on=True)


_MDOTS = (1.0, 1.2, 1.4, 1.6)          # gravitational masses [M_sun] to mark (dots)


def _mass_marks(ax, arr, yvals, color, dot_off=(0, 7, 'center', 'bottom'),
                star_off=(6, 0, 'left', 'center'), label=True, dot_arrow=False):
    """Mark a sequence at (M_B, yvals): a dot at each M_grav in _MDOTS plus a star
    at M_max (the stable-branch tip) — for (c)/(d) whose x-axis is baryonic mass.
    M_grav (col 4) is monotone on the branch, so np.interp keys off it.  dot_off /
    star_off = (dx_pt, dy_pt, ha, va) nudge the label text off the curves (over the
    contours is fine).  label=False draws markers only."""
    Mg, Mb = arr[:, 4], arr[:, 5]
    m = np.isfinite(Mg) & np.isfinite(Mb) & np.isfinite(yvals)
    Mg, Mb, yv = Mg[m], Mb[m], np.asarray(yvals)[m]
    _dx, _dy, _dha, _dva = dot_off
    for mt in _MDOTS:
        if Mg.min() <= mt <= Mg.max():
            xb, yb = np.interp(mt, Mg, Mb), np.interp(mt, Mg, yv)
            ax.plot(xb, yb, 'o', ms=5.5, color=color, mec='white', mew=0.7, zorder=7)
            if label:
                ax.annotate(f"{mt:g}", (xb, yb), textcoords='offset points',
                            xytext=(_dx, _dy), ha=_dha, va=_dva, fontsize=8.5,
                            color=color, zorder=8,
                            arrowprops=(dict(arrowstyle='->', color=color, lw=0.7,
                                             shrinkA=1, shrinkB=2) if dot_arrow else None))
    ax.plot(Mb[-1], yv[-1], '*', ms=13, color=color, mec='white', mew=0.6, zorder=7)  # M_max
    if label:
        _sx, _sy, _sha, _sva = star_off
        ax.annotate(r"$M_{\max}$", (Mb[-1], yv[-1]), textcoords='offset points',
                    xytext=(_sx, _sy), ha=_sha, va=_sva, fontsize=8.5, color=color, zorder=8,
                    arrowprops=(dict(arrowstyle='->', color=color, lw=0.7,
                                     shrinkA=1, shrinkB=2) if dot_arrow else None))


def _grid(ax):
    """Gridlines disabled (uncomment the body to restore light gray gridlines)."""
    # ax.grid(True, color='gray', alpha=0.25, lw=0.6, zorder=0)
    pass



# PRD size: mode='double' (4.75" square). Change to 'single'/'double' to resize.
fig, ((axA, axB), (axC, axD)) = paper_grid('2x2', mode='double', placeholder=False, **PAPER_STYLE)

# (a) M-R.  Fix the view FIRST so the constraint fills (some reach R~21) can't
#     auto-expand the axis and fling the labels into the margin.
axA.set_xlim(8, 16); axA.set_ylim(0.3, 2.8)
for _s in _F2SEQ:
    _a = _stable(_s['arr'])
    axA.plot(_a[:, 3], _a[:, 4], color=_s['c'], lw=2.4, zorder=4)      # curve labels -> legend in (b)
add_observational_constraints(axA, CONTOUR_DIR, show_mass_bands=True,
                              inline_labels=True)  # NICER/HESS + mass bands
axA.set_xlim(8, 16); axA.set_ylim(0.3, 2.8)                 # re-assert after fills
axA.set_xlabel(r'$R$ [km]'); axA.set_ylabel(r'$M$ [$M_\odot$]')
_grid(axA); panel_label(axA, '(a)', corner='lower right')

# (b) M vs M_baryonic.  Carries the shared curve legend (NS / QS / PNS) for all panels.
for _s in _F2SEQ:
    _a = _stable(_s['arr'])
    axB.plot(_a[:, 5], _a[:, 4], color=_s['c'], lw=2.4, label=_s['lbl'])
axB.set_xlabel(r'$M_B$ [$M_\odot$]'); axB.set_ylabel(r'$M$ [$M_\odot$]')
axB.set_xlim(0.6, 3.4); axB.set_ylim(0.6, 2.6)
axB.legend(loc='upper left', fontsize=10, frameon=False)
_grid(axB); panel_label(axB, '(b)', corner='lower right')

# (c) central temperature T_c vs baryonic mass M_B.  NS/QS are cold: drawn as a
#     coloured T_c=0 horizontal line over their M_B span; the two PNS sequences
#     carry T_c(n_B^c) from their isentrope.  Every curve gets M-dot + M_max-star.
for _s in _F2SEQ:
    _a = _stable(_s['arr'])
    if _s.get('yls') is None:                       # cold NS/QS -> T_c=0 line over its M_B span
        _mb = _a[:, 5]
        axC.plot([_mb.min(), _mb.max()], [0.0, 0.0], color=_s['c'], lw=2.4)
        # NS/QS mass dots + M_max crowd the T_c=0 line -> arrows lift labels off it.
        _mass_marks(axC, _a, np.zeros(len(_a)), _s['c'], _s['cdot'], _s['cstar'],
                    dot_arrow=True)
        continue
    _yl, _S = _s['yls']
    _Tc = np.asarray(H['iso_trapped']['T'](_a[:, 2], _yl, _S))    # central T on the isentrope
    axC.plot(_a[:, 5], _Tc, color=_s['c'], lw=2.4)
    _mass_marks(axC, _a, _Tc, _s['c'], _s['cdot'], _s['cstar'])
axC.set_xlabel(r'$M_B$ [$M_\odot$]'); axC.set_ylabel(r'$T_c$ [MeV]')
axC.set_xlim(0.6, 3.4); axC.set_ylim(bottom=-1)
_grid(axC); panel_label(axC, '(c)', corner='upper right')

# (d) central density vs baryonic mass; every curve gets M-dot + M_max-star labels.
for _s in _F2SEQ:
    _a = _stable(_s['arr'])
    _y = _a[:, 2] / n_sat
    axD.plot(_a[:, 5], _y, color=_s['c'], lw=2.4)
    _mass_marks(axD, _a, _y, _s['c'], _s['ddot'], _s['dstar'])
axD.set_xlabel(r'$M_B$ [$M_\odot$]'); axD.set_ylabel(r'$n_B^c/n_\mathrm{sat}$')
axD.set_xlim(0.6, 3.4); _grid(axD); panel_label(axD, '(d)', corner='lower right')

_d2 = []
for _s in _F2SEQ:
    _a = _stable(_s['arr'])
    _yls = _s.get('yls')
    _Tc = (np.zeros(len(_a)) if _yls is None
           else np.asarray(H['iso_trapped']['T'](_a[:, 2], _yls[0], _yls[1])))
    for _row, _tc in zip(_a, _Tc):
        _d2.append((_s['lbl'], float(_row[5]), float(_row[4]), float(_row[3]),
                    float(_row[2]) / n_sat, float(_tc)))
pd.DataFrame(_d2, columns=['sequence', 'M_B_Msun', 'M_Msun', 'R_km', 'nBc_over_n0', 'Tc_MeV']
             ).to_csv(f'../output/figure_data/fig2_sequences_{xsd_tag}.csv', index=False)
print('wrote fig2_sequences CSV')
fig.savefig(f'../output/figures/paper_fig2_stellar_sequences_{xsd_tag}.pdf', bbox_inches='tight')
plt.show()

# %% [markdown]
# ### Paper Figure 3 — $R_*$, $W_*/T$, $\log_{10}\tau$, $\sigma_{\rm crit}$ at the PNS centre vs $M_{\rm PNS}$
#
# One figure for the SELECTED quark set (`FN2_SET`) and σ set (`FN2_SIGMAS`),
# 2×2 like Figs 1–2. **(a,b,c)** $R_*$, $W_*/T$, $\log_{10}\tau$ along the
# trapped/isentropic sequence at fixed $(Y_L^H, S_H)=$ t_Tmax; colour = $\sigma$,
# line style = phase (unpaired ':' , CFL '--', unpCFL '-'). **(d)** the critical
# surface tension $\sigma_{\rm crit}$ at which the central $\tau$ equals a target
# (a `nuc_an.sigma_target_pt` root-find at each $M_{\rm PNS}$' centre $n_B^H$);
# line style = phase, colour = quark set (`FN2_D_MODE='quarks'`, $\tau$=`FN2_TAU`)
# or $\tau$ target (`'tau'`, $\tau\in$`FN2_D_TAUS`). Red vertical guides (dashed =
# fixed $M_{\rm PNS}$, dotted = $M_b$-matched to a cold star) as in the caption.
# No gridlines, no in-figure title.

# %%
# ============================================================================
#  Figure 3.  R*, W*/T and log10(tau) at the PNS centre vs M_PNS.
#    rows = observable;  columns = quark parametrization.
#    colour = sigma;  line style: unpaired ':', CFL '--', unpCFL '-'.
#  Vertical guides: dashed = M_PNS = 1, 1.4;  dotted = the M_PNS whose baryonic
#  mass equals Mb of a COLD (T=0) star with M(T0) = 1, 1.4 (same star, cooled).
#  The (M_PNS -> n_Bc -> T_c) map is hadronic-only: built ONCE, reused.
# ============================================================================
set_paper_style()

# ---- knobs ----
FN2_YL, FN2_S       = PNS_TMAX['YLH'], PNS_TMAX['S']   # trapped-PNS (Y_L, S) = t_Tmax
FN2_FLAVOR          = 'saddlepoint'
FN2_CHARGE          = 'coulomb_minimize'
FN2_TAU             = None               # s (target line on the log10 tau panel)
FN2_MVERT           = []         # masses for the vertical (red) guides
# ---- SELECT what to plot ----
FN2_SET             = quark_param_sets[0]                 # <-- quark parametrization (panels a–c)
FN2_SIGMAS          = list([80,100,150])                    # <-- σ SET drawn in a–c (colour); edit freely, e.g. [50, 100, 200]
FN2_VGUIDE_COL      = STANDARD_COLORS['Red']              # vertical mass-guide colour
# ---- panel (d): σ_crit(τ=target) vs M_PNS ----
FN2_D_MODE          = 'tau'            # (d) colour: 'quarks' (all quark_param_sets, τ=FN2_TAU)
                                          #             or 'tau' (FN2_SET only, τ ∈ FN2_D_TAUS)
FN2_D_TAUS          = [1e-3, 1.0, 100.0]  # s — τ targets when FN2_D_MODE=='tau'
FN2_D_NM            = 100                   # M_PNS samples in (d); each is a σ root-find → keep modest
_phase_ls = [('unpaired', ':'), ('cfl', '--'), ('unpCFL', '-')]
_sig_col  = {sg: mpl.cm.viridis(t) for sg, t in
             zip(FN2_SIGMAS, np.linspace(0.15, 0.85, len(FN2_SIGMAS)))}
from dataclasses import replace as _dc_replace              # vary τ_target per curve in (d)
from eos.alphabag.thermodynamics_quarks import T_critical   # CFL Tc (mask CFL above it)

# ---- PNS sequence (quark-independent): M_PNS -> (n_Bc, T_c) ----
def _tov_trapped_seq(YL, S):
    key = min(tov_trapped, key=lambda k: abs(k[0] - YL) + abs(k[1] - S))
    return tov_trapped[key]

def _binterp(arr, xcol, ycol):
    """interp ycol(xcol) on a TOV sequence's stable branch (up to M_max)."""
    k = np.argmax(arr[:, 4]) + 1
    return interp1d(arr[:k, xcol], arr[:k, ycol], kind='cubic',
                    bounds_error=False, fill_value=np.nan)

_tov = _tov_trapped_seq(FN2_YL, FN2_S)        # cols: 2=n_Bc 4=M 5=Mb
_i   = np.argmax(_tov[:, 4]) + 1              # stable branch up to M_max
_msk = _tov[:_i, 4] >= 0.6                    # show from 0.6 M_sun up
M_seq   = _tov[:_i, 4][_msk]
nBc_seq = _tov[:_i, 2][_msk]
T_seq   = np.asarray(H['iso_trapped']['T'](nBc_seq, FN2_YL, FN2_S))
M_max   = float(_tov[:_i, 4].max())

# Vertical guides: dashed at fixed M_PNS; dotted at the M_PNS baryon-matched to
# the cold (T=0) star of the same gravitational mass.
_cold_M_to_Mb   = _binterp(tov_cold, 4, 5)    # cold:    M    -> Mb
_trap_Mb_to_M   = _binterp(_tov,     5, 4)    # trapped: Mb   -> M
M_dot = [float(_trap_Mb_to_M(_cold_M_to_Mb(m0))) for m0 in FN2_MVERT]

# panels (a,b,c): (ylabel, getter(itp), yscale, ylim) — from the stored tables.
# Labels use subscript-star R_*, W_* per the paper convention.
_abc = [
    (r'$R_*$ [fm]',
     lambda itp: np.array([itp['R_c'](nb, _YL, T) for nb, T in zip(nBc_seq, T_seq)]),
     'linear', (1, 7)),
    (r'$W_*/T$',
     lambda itp: np.array([itp['W_c'](nb, _YL, T) for nb, T in zip(nBc_seq, T_seq)]) / T_seq,
     'linear', (0, 500)),
    (r'$\log_{10}\,\tau$ [s]',
     lambda itp: np.array([itp['log10_tau'](nb, _YL, T) for nb, T in zip(nBc_seq, T_seq)]),
     'linear', (-40, 80)),
]

# ---- ONE figure for the selected parametrization (2×2: a=R_*, b=W_*/T, c=log10 τ, d=σ_crit) ----
_tag     = q_tag_of(FN2_SET)
_params  = get_alphabag_custom(alpha=FN2_SET['alpha'], B4=FN2_SET['B4'], m_s=FN2_SET['m_s'])
_D0      = FN2_SET['Delta0']
_T_CFL   = float(T_critical(_D0))                     # CFL undefined above this

# PRD size knob: mode='single'(3.375") / 'centered'(4.75") / 'double'(7.0"), all square.
# No sharex (new mode) — every panel carries its own x-axis + label.
fig, ((axA, axB), (axC, axD)) = paper_grid('2x2', mode='double', placeholder=False, **PAPER_STYLE)
_abc_axes = [axA, axB, axC]

# nearest table Y_L to FN2_YL (the getters read the interpolators at _YL)
_a0 = next((nuc_sets[k] for k in nuc_sets
            if k.endswith(f"_{_tag}_s{int(FN2_SIGMAS[0])}")), None)
_YL = (float(_a0.hadronic_grids['Y_L_H'][
           np.argmin(np.abs(_a0.hadronic_grids['Y_L_H'] - FN2_YL))])
       if _a0 is not None else FN2_YL)

# (a,b,c) R_*, W_*/T, log10 τ vs M — colour = σ, line style = phase (from stored tables)
for sg in FN2_SIGMAS:
    for ph, ls in _phase_ls:
        stem = f"Htrapped_{FN2_FLAVOR}_{FN2_CHARGE}_{ph}_{_tag}_s{int(sg)}"
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
#axC.axhline(np.log10(FN2_TAU), color='0.5', ls='--', lw=1.0)   # τ_target on (c)

# ---- (d) σ_crit(τ=target) vs M_PNS ----
# line style = phase; colour = quark set (FN2_D_MODE=='quarks', τ=FN2_TAU) OR τ target
# ('tau', FN2_SET). Each point is a nuc_an.sigma_target_pt root-find, so use a coarser
# M grid (FN2_D_NM) sampled from the SAME t_Tmax central (n_B, T) sequence as (a–c).
_nBc_of_M = interp1d(M_seq, nBc_seq, kind='cubic', bounds_error=False, fill_value=np.nan)
_T_of_M   = interp1d(M_seq, T_seq,   kind='cubic', bounds_error=False, fill_value=np.nan)
_Md  = np.linspace(M_seq.min(), M_seq.max(), FN2_D_NM)
_nBd, _Td = _nBc_of_M(_Md), _T_of_M(_Md)


def _sigma_crit_vs_M(params, D0, phase, nuc_cfg):
    """σ_crit(M_PNS) [MeV/fm²]: the σ at which the CENTRAL τ = nuc_cfg.tau_target, at
    each M_PNS' centre (n_B, T) on the t_Tmax isentrope (the user's 'centre n_B^H').
    NaN where no τ=target crossing exists, or where CFL is above its melting T_c."""
    Tc_cfl = float(T_critical(D0))
    out = np.full(len(_Md), np.nan)
    for k, (nb, T) in enumerate(zip(_nBd, _Td)):
        if not (np.isfinite(nb) and np.isfinite(T)):
            continue
        if phase == 'cfl' and T > Tc_cfl:
            continue                                   # CFL gap melted → no CFL droplet
        H_pt = nuc_an.hadronic_point(H['trapped'], nb, FN2_YL, T)
        s = nuc_an.sigma_target_pt(H_pt, T, FN2_FLAVOR, FN2_CHARGE, phase,
                                   params, D0, nuc_cfg)
        out[k] = s if np.isfinite(s) else np.nan       # ±inf (σ_crit outside scan) → gap
    return out


def _tau_lbl(t):
    return f'{t*1e3:g} ms' if t < 1 else f'{t:g} s'


if FN2_D_MODE == 'quarks':                             # colour = quark parametrization, τ = FN2_TAU
    _dcol = {q_tag_of(p): mpl.cm.plasma(t) for p, t in
             zip(quark_param_sets, np.linspace(0.15, 0.8, max(len(quark_param_sets), 2)))}
    for p in quark_param_sets:
        _pp = get_alphabag_custom(alpha=p['alpha'], B4=p['B4'], m_s=p['m_s'])
        for ph, ls in _phase_ls:
            axD.plot(_Md, _sigma_crit_vs_M(_pp, p['Delta0'], ph, _nuc_cfg),
                     color=_dcol[q_tag_of(p)], ls=ls, lw=PHASE_LW[ph], alpha=PHASE_ALPHA[ph])
    _d_handles = [Line2D([], [], color=_dcol[q_tag_of(p)], label=q_tag_of(p))
                  for p in quark_param_sets]
    _d_title = 'quark set'
else:                                                  # colour = τ target, one set (FN2_SET)
    _dcol = {t: mpl.cm.plasma(u) for t, u in
             zip(FN2_D_TAUS, np.linspace(0.15, 0.8, max(len(FN2_D_TAUS), 2)))}
    for tau in FN2_D_TAUS:
        for ph, ls in _phase_ls:
            axD.plot(_Md, _sigma_crit_vs_M(_params, _D0, ph,
                                           _dc_replace(_nuc_cfg, tau_target=tau)),
                     color=_dcol[tau], ls=ls, lw=PHASE_LW[ph], alpha=PHASE_ALPHA[ph])
    _d_handles = [Line2D([], [], color=_dcol[t], label=_tau_lbl(t)) for t in FN2_D_TAUS]
    _d_title = r'$\tau$'
axD.set_ylabel(r'$\sigma_{\rm crit}$ [MeV/fm$^2$]')
axD.set_ylim(50, 200)                                  # (d) σ_crit range — edit here
axD.legend(handles=_d_handles, loc='best', title=_d_title, fontsize=8)

# τ reference levels on panel (c): horizontal lines at log10(τ) for τ = 1 ms / 1 s /
# 100 s, coloured to MATCH panel (d)'s τ colouring (same plasma-over-FN2_D_TAUS
# recipe), dash-dot + faded so they read as guides, not data.
_tau_ref_col = {t: mpl.cm.plasma(u) for t, u in
                zip(FN2_D_TAUS, np.linspace(0.15, 0.8, max(len(FN2_D_TAUS), 2)))}
for _t in FN2_D_TAUS:
    axC.axhline(np.log10(_t), color=_tau_ref_col[_t], ls='-', lw=1.2,
                alpha=0.4, zorder=1)

# common cosmetics: red vertical mass guides, xlim, per-panel x-label, panel tag (no grid)
for _lab, ax in zip(('(a)', '(b)', '(c)', '(d)'), (axA, axB, axC, axD)):
    ax.set_xlim(0.7, M_max)
    ax.set_xlabel(r'$M_{\rm PNS}\ [M_\odot]$')         # every panel labelled (no sharex)
    for mv in FN2_MVERT:                       # dashed: fixed M_PNS
        ax.axvline(mv, color=FN2_VGUIDE_COL, ls='-', lw=1.0, zorder=1)
    for md in M_dot:                           # dotted: Mb-matched to cold
        if np.isfinite(md):
            ax.axvline(md, color=FN2_VGUIDE_COL, ls=':', lw=1.0, zorder=1)
    panel_label(ax, _lab, corner='upper left')

# Legends: σ colour on (a), phase line style on (b) — they used to be two
# axA.legend() calls, and matplotlib keeps only the last legend per axes, so the
# phase key was silently dropped. Panel (d) carries its own.
_ph = [Line2D([], [], color='k', ls=ls, lw=PHASE_LW[ph], label=lbl)
       for ph, lbl, ls in [('unpaired', 'unpaired', ':'), ('cfl', 'CFL', '--'),
                           ('unpCFL', 'unpCFL', '-')]]
_sg = [Line2D([], [], color=_sig_col[sg], ls='-', label=rf'$\sigma={int(sg)}$')
       for sg in FN2_SIGMAS]
axA.legend(handles=_sg, loc='upper right', title=r'$\sigma$ [MeV/fm$^2$]')
axB.legend(handles=_ph, loc='best')

_d3 = []                                              # (a,b,c) vs M_PNS
for sg in FN2_SIGMAS:
    for ph, ls in _phase_ls:
        stem = f"Htrapped_{FN2_FLAVOR}_{FN2_CHARGE}_{ph}_{_tag}_s{int(sg)}"
        if stem not in nuc_sets:
            continue
        itp = build_thermal_nucleation_interpolators(nuc_sets[stem])
        _c3 = [getter(itp) for _, getter, _, _ in _abc]   # R_c, W/T, log10 tau
        for _m, _r, _w, _lt in zip(M_seq, *_c3):
            _d3.append((sg, ph, float(_m), float(_r), float(_w), float(_lt)))
pd.DataFrame(_d3, columns=['sigma', 'phase', 'M_PNS_Msun', 'R_star_fm', 'W_over_T',
                           'log10_tau_s']).to_csv(
    f'../output/figure_data/fig3abc_vs_MPNS_{xsd_tag}_{_tag}.csv', index=False)
_d3d = []                                             # (d) sigma_crit vs M_PNS (recomputed)
_dc3 = 'quark_set' if FN2_D_MODE == 'quarks' else 'tau_s'
if FN2_D_MODE == 'quarks':
    for p in quark_param_sets:
        _pp = get_alphabag_custom(alpha=p['alpha'], B4=p['B4'], m_s=p['m_s'])
        for ph, ls in _phase_ls:
            for _m, _s in zip(_Md, _sigma_crit_vs_M(_pp, p['Delta0'], ph, _nuc_cfg)):
                _d3d.append((q_tag_of(p), ph, float(_m), float(_s)))
else:
    for tau in FN2_D_TAUS:
        for ph, ls in _phase_ls:
            for _m, _s in zip(_Md, _sigma_crit_vs_M(_params, _D0, ph,
                                                    _dc_replace(_nuc_cfg, tau_target=tau))):
                _d3d.append((f'{tau:g}', ph, float(_m), float(_s)))
pd.DataFrame(_d3d, columns=[_dc3, 'phase', 'M_PNS_Msun', 'sigma_crit']).to_csv(
    f'../output/figure_data/fig3d_sigmacrit_{xsd_tag}_{_tag}.csv', index=False)
print('wrote fig3 CSVs (abc + d)')
fig.savefig(f'../output/figures/paper_fig3_Rstar_Wc_tau_sigmacrit_{xsd_tag}_{_tag}.pdf',
            bbox_inches='tight')
plt.show()


# %% [markdown]
# ### Paper Figure 4 — $T_{\rm nuc}(n_B^H)$ nucleation conditions
#
# One 2×2 figure comparing the TWO quark parametrizations: **rows = set**
# (Set A top, Set B bottom), **columns = snapshot** ($t_0$ left, $t_{T_\mathrm{max}}$
# right). $\tau=\tau_{\rm target}$ curves in $(n_B^H, T)$, one line per $\sigma$
# (colour) and phase (line style). The stellar isentrope is drawn in the snapshot
# colour (orange $t_0$, red $t_{T_\mathrm{max}}$); markers on it mark the PNS
# central density (filled + labelled: fixed $M=1,1.2,1.4,1.6$, star = $M_{\max}$;
# open: $M_b$-matched to a cold star). **(a,b)** Set A, **(c,d)** Set B; snapshots
# $t_0$: $Y_L^H=0.35,\,S_H=1.5$ and $t_{T_\mathrm{max}}$: $Y_L^H=0.25,\,S_H=2$.

# %%
# ============================================================================
#  Figure 4.  T_nuc(n_B^H) nucleation-condition curves — ONE 2×2, two quark sets.
#    rows = set (A top, B bottom); cols = snapshot: t_0 (Y_L^H=0.35, S_H=1.5) and
#    t_Tmax (Y_L^H=0.25, S_H=2.0). Panels (a,b)=Set A, (c,d)=Set B.
#    colour = sigma;  line style: unpaired ':', CFL '--', unpCFL '-'.
#    isentrope T(n_B^H) dash-dot in the snapshot colour (orange t_0, red t_Tmax).
#    Markers ON the isentrope mark the PNS central density n_Bc (Fig-2 c/d style):
#      filled dot + label = PNS at grav. mass M = 1,1.2,1.4,1.6 (+ filled star = M_max);
#      open   dot         = PNS whose baryonic mass = Mb of a COLD (T=0) star with
#                           the same M(T0)          (+ open star = cold M_max, if on branch).
# ============================================================================
set_paper_style()

# ---- knobs ----
FN_FLAVOR = 'saddlepoint'
FN_CHARGE = 'coulomb_minimize'
FN_TAU    = 1e-3                         # s  (tau_target)
# σ to draw, SELECTED PER quark parametrization (key = q_tag_of(set)). All sets
# start at F6_SIGMAS_DEFAULT; edit the per-set entries below to taste. Each must
# be a subset of sigma_list (tables must exist). Colours are viridis over each
# set's own σ list, rebuilt inside the loop.
F6_SIGMAS_DEFAULT = [50, 100, 150]
F6_SIGMAS_BY_SET  = {q_tag_of(p): list(F6_SIGMAS_DEFAULT) for p in quark_param_sets}
F6_SIGMAS_BY_SET[q_tag_of(quark_param_sets[1])] = [150, 200, 250]
F6_SIGMAS_BY_SET[q_tag_of(quark_param_sets[0])] = [50, 100, 150]
FN_CUT_CFL_ABOVE_TC = True               # mask the CFL curve where T_nuc > T_CFL (gap vanishes above Tc)
from eos.alphabag.thermodynamics_quarks import T_critical   # same Tc as the CFL EoS gap

# Phase line styling — SAME convention as Figure 1: unpCFL solid+thick, the other
# two thinner and faded (unpCFL drawn last -> on top).  Order = draw order.
_PHASE_LS    = {'unpaired': ':', 'cfl': '--', 'unpCFL': '-'}
_PHASE_LW    = PHASE_LW                                        # shared thickness
_PHASE_ALPHA = PHASE_ALPHA                                     # shared opacity (one source)
_PHASE_LBL   = {'unpaired': 'unpaired', 'cfl': 'CFL', 'unpCFL': 'unpCFL'}
# σ -> colour is built PER set inside the loop (each set has its own F6_SIGMAS).

# TOV columns: 0=e_c 1=P_c 2=n_Bc 3=R 4=M 5=Mb.
_NBC, _M, _MB = 2, 4, 5
_MDOTS_F6 = (1.0, 1.2, 1.4, 1.6)         # grav. masses [M_sun] marked on the isentrope

def _branch_interp(arr, xcol, ycol):
    """interp ycol(xcol) along a TOV sequence's stable branch (up to M_max)."""
    i = np.argmax(arr[:, _M]) + 1
    return interp1d(arr[:i, xcol], arr[:i, ycol], kind='cubic',
                    bounds_error=False, fill_value=np.nan)

def _tov_trapped_seq(YL, S):
    """Nearest trapped/isentropic TOV sequence to (YL, S)."""
    key = min(tov_trapped, key=lambda k: abs(k[0] - YL) + abs(k[1] - S))
    return tov_trapped[key]

def _mark_pair(ax, pts, label, color, dy=5.0):
    """One value label plus a short arrow to EACH marker in pts — the solid
    (fixed-M) and white-filled (M_b-matched) rendering of the same star. Both
    arrows start at the SAME anchor (xt, yt), `dy` MeV above the higher marker;
    the text is drawn separately so it never shifts an arrow's tail. nan skipped."""
    pts = [(x, y) for x, y in pts if np.isfinite(x) and np.isfinite(y)]
    if not pts:
        return
    xt = float(np.mean([p[0] for p in pts]))
    yt = max(p[1] for p in pts) + dy
    ax.text(xt, yt, label, ha='center', va='bottom', fontsize=8.5,
            color=color, zorder=8)
    _arw = dict(arrowstyle='->', color=color, lw=0.6, shrinkA=2, shrinkB=3,
                mutation_scale=6)                     # small heads, common tail
    for p in pts:
        ax.annotate('', xy=p, xytext=(xt, yt), textcoords='data', arrowprops=_arw)


def _iso_markers(ax, tov_tr, T_iso, color):
    """PNS central-density markers on the isentrope, Fig-2 (c/d) styling (ms).
    For each grav mass M in _MDOTS_F6: a SOLID dot at the fixed-M PNS density and a
    WHITE-filled dot (same size) at the density where the PNS baryonic mass equals
    M_b of the COLD (T=0) star of the same M(T0); one value label arrows to both.
    Stars mark M_max: solid = trapped tip, white = cold M_max (M_b-matched)."""
    nBc_M     = _branch_interp(tov_tr, _M, _NBC)      # trapped: M_grav -> n_Bc
    Mb_of_M   = _branch_interp(tov_cold, _M, _MB)     # cold:    M_grav -> M_b
    nBc_of_Mb = _branch_interp(tov_tr,  _MB, _NBC)     # trapped: M_b    -> n_Bc

    def _pt(nb):                                       # density -> (n_B/n_sat, T)
        return (nb / n_sat, float(T_iso(nb))) if np.isfinite(nb) else (np.nan, np.nan)

    for mt in _MDOTS_F6:
        pf = _pt(float(nBc_M(mt)))                    # solid: fixed grav mass
        pw = _pt(float(nBc_of_Mb(Mb_of_M(mt))))       # white: M_b-matched to cold star
        if np.isfinite(pf[0]):
            ax.plot(*pf, 'o', ms=5.5, color=color, mec='white', mew=0.7, zorder=7)
        if np.isfinite(pw[0]):
            ax.plot(*pw, 'o', ms=5.5, color='white', mec=color, mew=1.3, zorder=7)
        _mark_pair(ax, [pf, pw], f"{mt:g}", color)

    sf = _pt(tov_tr[int(np.argmax(tov_tr[:, _M])), _NBC])              # trapped tip
    sw = _pt(float(nBc_of_Mb(tov_cold[int(np.argmax(tov_cold[:, _M])), _MB])))  # cold M_max
    if np.isfinite(sf[0]):
        ax.plot(*sf, '*', ms=13, color=color, mec='white', mew=0.6, zorder=7)
    if np.isfinite(sw[0]):
        ax.plot(*sw, '*', ms=13, color='white', mec=color, mew=1.1, zorder=7)
    _mark_pair(ax, [sf, sw], r"$M_{\max}$", color)

# ONE 2×2 figure: rows = quark parametrization (Set A top, Set B bottom),
# columns = snapshot (t_0 left, t_Tmax right). colour = σ (per-set list/palette),
# line style = phase; each panel titled with its set.
_F4_SETS = [('Set A', quark_param_sets[0]), ('Set B', quark_param_sets[1])]


def _draw_f4_panel(ax, FN_SET, pan, panel_lab, set_label):
    """Draw one T_nuc(n_B^H) panel for (quark set, snapshot); return (σ→colour, σ list)
    so the caller can build that row's σ legend (the σ palette differs per set)."""
    _tag      = q_tag_of(FN_SET)
    F6_SIGMAS = F6_SIGMAS_BY_SET.get(_tag, F6_SIGMAS_DEFAULT)      # per-set σ selection
    _sig_col  = {sg: mpl.cm.viridis(t) for sg, t in               # colour = σ, per set
                 zip(F6_SIGMAS, np.linspace(0.15, 0.85, len(F6_SIGMAS)))}
    _T_CFL    = float(T_critical(FN_SET['Delta0']))               # CFL Tc [MeV]
    YL, S, col = pan['YLH'], pan['S'], pan['color']
    YL_used = None
    for sg in F6_SIGMAS:
        for ph, ls in _PHASE_LS.items():              # unpCFL drawn last -> on top
            stem = f"Htrapped_{FN_FLAVOR}_{FN_CHARGE}_{ph}_{_tag}_s{int(sg)}"
            if stem not in nuc_sets:
                continue
            grids   = nuc_sets[stem].hadronic_grids
            iYL     = int(np.argmin(np.abs(grids['Y_L_H'] - YL)))
            YL_used = grids['Y_L_H'][iYL]
            res = compute_nucleation_density(nuc_sets[stem], tau_target=FN_TAU, scan='n_B')
            nB, T = nucleation_curve(res, iYL)
            m = np.isfinite(nB) & np.isfinite(T)
            if FN_CUT_CFL_ABOVE_TC and ph == 'cfl':
                m &= (T <= _T_CFL)                    # CFL undefined above Tc
            if m.any():
                ax.plot(nB[m] / n_sat, T[m], color=_sig_col[sg], ls=ls,
                        lw=_PHASE_LW[ph], alpha=_PHASE_ALPHA[ph])
    ax.set_xlim(0.5, 12); ax.set_ylim(1, 80)          # T [MeV], n_B^H/n_sat

    # isentrope T(n_B^H) at (Y_L, S), solid in the snapshot colour + PNS markers.
    YL_iso = YL if YL_used is None else float(YL_used)
    T_iso  = lambda nBc: H['iso_trapped']['T'](nBc, YL_iso, S)
    tov_tr = _tov_trapped_seq(YL_iso, S)
    nBc_mx = tov_tr[int(np.argmax(tov_tr[:, _M])), _NBC]          # M_max central density
    nB_iso = np.linspace(0.5 * n_sat, nBc_mx, 200)
    ax.plot(nB_iso / n_sat, T_iso(nB_iso), color=col, ls='-', lw=1.8, zorder=3)
    _iso_markers(ax, tov_tr, T_iso, col)

    ax.set_xlabel(r'$n_B^H / n_{\rm sat}$')
    ax.set_ylabel(r'$T_{\rm nuc}$ [MeV]')
    ax.set_title(rf"{pan['lbl']},  $Y_L^H={YL_iso:.2f}$,  {set_label}")
    # NOTE deliberately no set_box_aspect here: panel shape is paper_grid's job
    # (see the call below). Pinning it here fought PAPER_STYLE's aspect and the
    # width budgeted for the wider shape turned into a gutter between the panels.
    panel_label(ax, panel_lab)
    return _sig_col, F6_SIGMAS


# PRD size: mode='double' (7.0" wide, 2×2). rows = set, cols = snapshot.
# square=False: don't pin each panel's W/H, let it fill its slot, so leftover width
# becomes panel instead of gutter. aspect=1.0 makes the figure 7.0x7.0", which lands
# the unpinned panels at W/H ~ 1.06 -- square to the eye, with no empty slack. The
# ~0.6" left between the columns is the right column's y-label + tick numbers (ink,
# not waste); dropping those duplicated inner labels would take it to ~0.15".
# To go back to exactly-square panels: drop square=False (costs ~0.2" of gutter).
fig, ((axA, axB), (axC, axD)) = paper_grid('2x2', mode='double', placeholder=False,
                                           square=False,
                                           **(PAPER_STYLE | {'aspect': 1.0}))
(_nameA, _setA), (_nameB, _setB) = _F4_SETS
_scA, _sgsA = _draw_f4_panel(axA, _setA, PNS_T0,   '(a)', _nameA)   # Set A, t_0
_draw_f4_panel(axB, _setA, PNS_TMAX, '(b)', _nameA)                 # Set A, t_Tmax
_scB, _sgsB = _draw_f4_panel(axC, _setB, PNS_T0,   '(c)', _nameB)   # Set B, t_0
_draw_f4_panel(axD, _setB, PNS_TMAX, '(d)', _nameB)                 # Set B, t_Tmax

# Legends: phase (line style) on (a); per-row σ legend on the right panel of each
# row (b, d) — the σ palette differs per set.
_ph_handles = [Line2D([], [], color='k', ls=_PHASE_LS[p], lw=_PHASE_LW[p],
                      alpha=_PHASE_ALPHA[p], label=_PHASE_LBL[p]) for p in _PHASE_LS]
axA.legend(handles=_ph_handles, loc='upper right')
for _ax, _sc, _sgs in ((axB, _scA, _sgsA), (axD, _scB, _sgsB)):
    _sig_handles = [Line2D([], [], color=_sc[sg], ls='-', label=f'{int(sg)}') for sg in _sgs]
    _ax.legend(handles=_sig_handles, loc='upper right',
               title=r'$\sigma\;[\mathrm{MeV\,fm^{-2}}]$')

# CSV: T_nuc(n_B^H) for both sets × snapshots × σ × phase (one combined file).
_d4 = []
for _setname, _pset in _F4_SETS:
    _tag = q_tag_of(_pset)
    _F6  = F6_SIGMAS_BY_SET.get(_tag, F6_SIGMAS_DEFAULT)
    for _pan in (PNS_T0, PNS_TMAX):
        _YLp = _pan['YLH']
        for sg in _F6:
            for ph, ls in _PHASE_LS.items():
                stem = f"Htrapped_{FN_FLAVOR}_{FN_CHARGE}_{ph}_{_tag}_s{int(sg)}"
                if stem not in nuc_sets:
                    continue
                _grids = nuc_sets[stem].hadronic_grids
                _iYLp = int(np.argmin(np.abs(_grids['Y_L_H'] - _YLp)))
                _res = compute_nucleation_density(nuc_sets[stem], tau_target=FN_TAU, scan='n_B')
                _nB, _Tn = nucleation_curve(_res, _iYLp)
                _mk = np.isfinite(_nB) & np.isfinite(_Tn)
                for _n, _t in zip(_nB[_mk] / n_sat, _Tn[_mk]):
                    _d4.append((_setname, _tag, _pan['lbl'], sg, ph, float(_n), float(_t)))
pd.DataFrame(_d4, columns=['set', 'quark_tag', 'snapshot', 'sigma', 'phase',
                           'nBH_over_n0', 'T_nuc_MeV']
             ).to_csv(f'../output/figure_data/fig4_Tnuc_{xsd_tag}.csv', index=False)
print('wrote fig4_Tnuc CSV (both sets)')
fig.savefig(f'../output/figures/paper_fig4_Tnuc_{xsd_tag}.pdf', bbox_inches='tight')
plt.show()


# %% [markdown]
# ### Paper Figure 5 — $\sigma_{\rm crit}(B^{1/4},\Delta_0)$: iso-$\sigma_{\rm crit}$/-$M_{\max}$ + rejection + droplet-regime map
#
# The saved IV.3 unpCFL grid ($\tau$=1 ms, $M_{T0}$=1.4 $M_\odot$), two $\alpha_s$
# panels, combining several toggleable layers (see the `F8_*` flags at the top of
# the cell — set any to `False` to drop that layer):
# - **fill** — $\sigma_{\rm crit}$ heatmap with iso-$\sigma_{\rm crit}$ contour lines
#   (50–250 MeV/fm²);
# - **iso-$M_{\max}$** — crimson dashed lines (every 0.2 $M_\odot$; the $2\,M_\odot$
#   acceptance edge solid), over the mass-relevant cells only;
# - **hatched** — WHY each non-viable cell is rejected: red = $M_{\max}<2M_\odot$,
#   green = unbound 3-flavour matter (Witten), grey = 2-flavour bound, blue =
#   re-hadronizes;
# - **white outlines + labels** — which radius the critical droplet $R_*$ equals:
#   $R_{\rm CFL}$, $R_\Delta$ (pairing-coherence kink), or $R_{\rm unp}$.
# $R_*$ is recomputed on demand (`_regime_grid`) at each cell's $\sigma_{\rm crit}$.

# %%
# ============================================================================
#  Paper Fig 5: sigma_crit(B^1/4, Delta_0) for two alpha_s. Excluded regions are
#  drawn as coloured BOUNDARY outlines (not filled) + labels; the viable region
#  carries the sigma_crit heatmap with iso-sigma_crit (white) and iso-M_max (dark
#  dashed) contour lines, and the R* droplet-regime zones. Regime via _regime_grid.
# ============================================================================
# ---- layer toggles: set any False to drop that layer -----------------------
F8_SHOW          = [0,1,2,3]  # alpha_s slice indices to draw; grid saved 4 (pi/2 x 0, .1, .2, .3)
F8_HEATMAP       = True       # sigma_crit colour fill + colorbar
F8_ISO_SIGMA     = True       # iso-sigma_crit contour lines (white)
F8_ISO_MMAX      = True       # iso-M_max contour lines (dark dashed)
F8_REJECT        = True       # excluded-region boundary outlines
F8_REJECT_LABELS = True       # labels on the excluded regions
F8_REGIME        = False       # R* droplet-regime zone boundaries
F8_REGIME_LABELS = False       # inline R_CFL / R_Delta / R_unp labels
F8_UNP_STABLE    = False       # vertical line: B^1/4 where unpaired SQM turns abs. stable
# ----------------------------------------------------------------------------
set_paper_style()
_F8 = np.load(f'../output/mc_cfl/sigma_crit_grid_{xsd_tag}_MT0{MT0_grid_thr[0]:.2f}_'
              f'saddlepoint-coulomb_minimize-unpCFL.npz', allow_pickle=False)
_al8, _B48, _D08 = _F8['alpha_slices'], _F8['B4_grid'], _F8['Delta0_grid']
_SIG8, _RS8, _MM8, _OK8 = _F8['sig_crit'], _F8['reason'], _F8['M_max'], _F8['cfl_ok']
# regime layer is the expensive part -> only recompute R* if it will be drawn.
_reg8 = (_regime_grid(_SIG8, _al8, _B48, _D08, float(_F8['MT0']), slices=F8_SHOW)
         if F8_REGIME else np.full(_SIG8.shape, -1, dtype=int))

# excluded-region reason -> (reason int, boundary colour, label, label rotation,
# label-on-left-panel-only). Codes pulled from nuc_an.REASON_CODE by name so this
# never drifts when codes are renumbered. The no-rehadronization filter is the
# droplet test: 'rehadr' (no_rehad fails) -> "re-hadr.", 'rehad_quasi' (no_rehad
# passes but not monotone) -> "quasi-re-hadr." (drop this row + set
# merge_rehad_labels=True on _filt_cfg to fold both into one "re-hadr." zone).
_RC = nuc_an.REASON_CODE
_HSPEC = [(_RC['mmax'],        OKAB['vermillion'], r'$M_{\rm QS}^{\rm max}<2M_\odot$', 45,  False),
          (_RC['witten'],      OKAB['green'],      'SQM not abs.\nstable',             0,  False),
          (_RC['twoflavor'],   OKAB['grey'],       '2 flav.\nstability',               90, True),  # left only
          (_RC['rehadr'],      OKAB['blue'],       're-hadr.',                         0,  False),
          (_RC['rehad_quasi'], OKAB['purple'],     'quasi-\nre-hadr.',                 0,  False)]
_REGCOL = [OKAB['orange'], OKAB['purple'], OKAB['sky']]           # codes 0, 1, 2
_REGLAB = [r'$R_*=R^{\rm unp}_*$', r'$R_*=R_{\Delta}$', r'$R_*=R^{\rm CFL}_*$']     # codes 0, 1, 2
_vmin8, _vmax8 = np.nanmin(_SIG8), np.nanmax(_SIG8[np.isfinite(_SIG8)])
# Contour colours over viridis: only warm/magenta hues pop (blue/green/teal blend
# into the map, orange clashes with the vermillion M<2 edge). iso-sigma = white;
# iso-M = deep magenta. Alternatives to try: '#ff7fb0' (pink), STANDARD_COLORS['Brown'].
_ISO_SIG_COL = 'white'
_ISO_M_COL   = 'lightgray'                # mediumvioletred

def _draw_reject(ax, ra, cpanel, labels=True):
    """Outline each excluded reason-zone (no fill) in its colour + inline label."""
    for _code, _col, _lab, _rot, _left_only in _HSPEC:
        _m = ra == _code
        if not _m.any():
            continue
        ax.contour(_B48, _D08, _m.astype(float), levels=[0.5], colors=[_col],
                   linewidths=1.8, zorder=4)
        if labels and not (_left_only and cpanel > 0):
            _r, _cc = np.where(_m)
            _bc, _dc = np.median(_B48[_cc]), np.median(_D08[_r])
            _fs = 9.5 if _code == _RC['mmax'] else 8.5      # all labels same size (M_QS^max +1)
            if _code == _RC['rehadr']:
                # re-hadr. zone is thin and sits under its own blue boundary -> pull
                # the label into open space with an arrow back to the zone. Tweak the
                # xytext (axes-fraction) anchor if it lands on another label.
                ax.annotate(_lab, xy=(_bc, _dc), xycoords='data',
                            xytext=(42, 0), textcoords='offset points',  # horizontal arrow (same y as zone); flip sign to point left
                            color=_col, fontsize=_fs, fontweight='bold',
                            ha='left', va='center', zorder=8,
                            arrowprops=dict(arrowstyle='->', color=_col, lw=0.8,
                                            mutation_scale=6, shrinkB=2))  # smaller arrowhead
            else:
                ax.text(_bc, _dc, _lab, color=_col, fontsize=_fs, fontweight='bold',
                        ha='center', va='center', rotation=_rot, zorder=7)

def _mu_unp_P0(alpha, B4):
    """mu_B of UNPAIRED beta-eq SQM at P=0, T=0 [MeV]. At P=0, mu_B == e/n_B, so
    this is the unpaired-matter absolute-stability energy per baryon."""
    P, e, mu, ok = nuc_an.unpaired_eos_at_params(alpha, B4, _filt_cfg)
    n, P, mu = _filt_cfg.n_B_grid[ok], P[ok], mu[ok]
    if n.size < 5:
        return np.nan
    cr = nuc_an.zero_crossing(n, P)
    if cr is None:
        return np.nan
    j, fr = cr
    return float(mu[j] + fr * (mu[j + 1] - mu[j]))

def _B_unp_absstable(alpha, lo, hi):
    """B^1/4 where mu_unp(P=0,T=0)=930 MeV (mu rises with B, so below it unpaired
    SQM is absolutely stable). nan if no crossing brackets in [lo,hi]."""
    from scipy.optimize import brentq
    f = lambda b: _mu_unp_P0(alpha, b) - 930.0
    flo, fhi = f(lo), f(hi)
    if not (np.isfinite(flo) and np.isfinite(fhi)) or flo * fhi > 0:
        return np.nan
    return brentq(f, lo, hi, xtol=0.05)

# PRD size: mode='double' (4.75" wide). Change to 'single'/'double' to resize.
# The grid SHAPE follows len(F8_SHOW) and panels are taken with axes.flat, so adding
# or removing an alpha_s slice needs no edit here: 1 or 2 slices -> one row, 3 or 4
# -> two rows. (Hard-coding '2x2' while indexing axes[0, _c] is what broke on
# F8_SHOW = [0,1,2,3]: a 2x2 has only 2 columns, so _c = 2 ran off the row.)
fig, axes = paper_grid('1x2' if len(F8_SHOW) <= 2 else '2x2', mode='double',
                       placeholder=False, fontsize=11, labelsize=11, legendsize=9,
                       aspect=1)                    # square: it's a plane map
for _ax in axes.flat[len(F8_SHOW):]:                # 3 slices on a 2x2 → blank the 4th
    _ax.set_visible(False)
pcm = None
for _c, _ia in enumerate(F8_SHOW):
    ax = axes.flat[_c]
    _sa, _ra, _rg, _ma, _ok = _SIG8[_ia], _RS8[_ia], _reg8[_ia], _MM8[_ia], _OK8[_ia]
    # (1) sigma_crit heatmap (excluded cells are NaN -> left white)
    if F8_HEATMAP:
        pcm = ax.pcolormesh(_B48, _D08, np.ma.masked_invalid(_sa), cmap='viridis',
                            vmin=_vmin8, vmax=_vmax8, shading='nearest', zorder=2)
    # (3) iso-M_max lines (white dashed, same label font as iso-sigma) on the
    #     mass-relevant cells (CFL-viable + the M<2 band); levels above 2 Msun --
    #     the 2 Msun edge itself is the vermillion M_max reject boundary.
    if F8_ISO_MMAX:
        _mm = np.where(_ok | (_ra == _RC['mmax']), _ma, np.nan)
        _lv = [round(x, 1) for x in np.arange(2.2, 4.001, 0.2)
               if np.isfinite(_mm).any() and np.nanmin(_mm) <= round(x, 1) <= np.nanmax(_mm)]
        if _lv:
            _cm = ax.contour(_B48, _D08, _mm, levels=_lv, colors=[_ISO_M_COL],
                             linewidths=0.9, linestyles='--', alpha=1.0, zorder=2.5)  # below iso-sigma (3)
            ax.clabel(_cm, fmt='%.1f', fontsize=7.5, inline=True)
    # (2) iso-sigma_crit contour lines (black)
    if F8_ISO_SIGMA:
        _cs = ax.contour(_B48, _D08, _sa, levels=[50, 100, 150, 200, 250],
                         colors=[_ISO_SIG_COL], linewidths=0.8, alpha=0.9, zorder=3)
        ax.clabel(_cs, fmt='%.0f', fontsize=9, inline=True)
    # (4) excluded-region boundary outlines + labels
    if F8_REJECT:
        _draw_reject(ax, _ra, _c, labels=F8_REJECT_LABELS)
    # (5) R* droplet-regime zones: white boundaries + pill labels (no outline)
    if F8_REGIME:
        ax.contour(_B48, _D08, np.where(_rg >= 0, _rg, np.nan), levels=[0.5, 1.5],
                   colors='white', linewidths=0.9, zorder=5)
        if F8_REGIME_LABELS:
            for _k in (0, 1, 2):
                _m = _rg == _k
                if _m.sum() > 15:
                    _r, _cc = np.where(_m)
                    ax.text(np.median(_B48[_cc]), np.median(_D08[_r]), _REGLAB[_k],
                            color='k', fontsize=11, fontweight='bold', ha='center',
                            va='center', zorder=6,
                            bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                      ec='none', alpha=0.72))
    # (6) unpaired-SQM absolute-stability threshold: mu_unp(P=0,T=0)=930 MeV.
    #     Left of the line unpaired matter is absolutely stable.
    if F8_UNP_STABLE:
        _bstab = _B_unp_absstable(_al8[_ia], _B48.min(), _B48.max())
        if np.isfinite(_bstab):
            ax.axvline(_bstab, color='k', lw=1.6, ls=(0, (5, 2)), zorder=8)
            ax.text(_bstab, _D08.max(), r'unp. SQM abs. stable ', color='k',
                    fontsize=8.5, ha='right', va='top', rotation=90, zorder=8)
    ax.set_title(rf"$\alpha_s=\pi/2\times{_al8[_ia]/(np.pi/2):.1f}$")
    ax.set_xlabel(r'$B^{1/4}$ [MeV]')
    ax.set_xlim(_B48.min(), _B48.max()); ax.set_ylim(0.1, _D08.max())
    panel_label(ax, f"({chr(97 + _c)})")
    ax.set_ylabel(r'$\Delta_0$ [MeV]')          # y-axis on every panel (sharey off)
if F8_HEATMAP and pcm is not None:
    # colorbar exactly as tall as the (square) panels: inset on the right panel at
    # full axes height (set_box_aspect makes transAxes 0..1 span the square box).
    _cax = axes[0, -1].inset_axes([1.05, 0.0, 0.05, 1.0])
    fig.colorbar(pcm, cax=_cax, label=r'$\sigma_{\rm crit}$ [MeV/fm$^2$]')
fig.savefig(f'../output/figures/paper_fig5_sigcrit_map_isolines_{xsd_tag}.pdf', bbox_inches='tight')
plt.show()


# %% [markdown]
# ### Paper Figure 5a — $M$–$R$ and $P_{\rm CFL}(\mu_B)$ of the viable cells, coloured by $\sigma_{\rm crit}$
#
# Companion to Fig 5: for the $\alpha_s$ slices in `F8B_ALPHAS` (default [0,1,2]),
# replay the cold CFL EoS + TOV of every $\sigma_{\rm crit}$-viable
# $(B^{1/4},\Delta_0)$ cell and overlay **(a)** $M$–$R$ and **(b)**
# $P_{\rm CFL}(\mu_B)$, each curve coloured by its $\sigma_{\rm crit}$ on the SAME
# viridis scale as the Fig 5 heatmap. The hadronic reference (cold TOV /
# $P_H(\mu_B)$) is black.

# %%
# ============================================================================
#  Paper Fig 5a: M-R and P_CFL(mu_B) for the σ_crit-viable cells at α slices
#  F8B_ALPHAS, coloured by σ_crit (SAME scale as the Fig 5 heatmap, _vmin8/_vmax8).
#  Reuses nuc_an.replay_accepted (parallel CFL EoS + TOV). Hadronic curve = black.
#  Needs the Fig 5 cell above to have run (_SIG8, _al8, _B48, _D08, _vmin8/_vmax8).
# ============================================================================
set_paper_style()
F8B_ALPHAS     = [0,1,2]              # α_s slice indices to include (into _al8)
F8B_MAX_CURVES = None                    # even subsample of viable cells (None = all)
# ---- which cells to draw: keep only those whose σ_crit AND M_max both fall in
# these windows; (-inf, inf) means "no cut". In the saved grid the viable cells run
# σ_crit 63..286 MeV/fm² and M_max 2.00..4.63 M_sun, so e.g. F8B_SIG_RANGE=(100,150)
# isolates the mid-σ band and F8B_MMAX_RANGE=(2.0,2.5) the observationally tight
# stars. Both ends are inclusive.
F8B_SIG_RANGE  = (-np.inf, 150)      # σ_crit window [MeV/fm²]
F8B_MMAX_RANGE = (2, 2.6)      # M_max  window [M_sun]
# viable (finite-σ_crit) cells across the chosen α slices — same grid as Fig 5
_al_idx  = [i for i in F8B_ALPHAS if i < len(_al8)]
# Both cuts are applied BEFORE the replay: replay_accepted treats a non-finite
# σ_crit as "not accepted", so blanking the out-of-window cells means their
# (CFL EoS + TOV) is never solved — a narrow window is proportionally CHEAPER, not
# just sparser. M_max is read from the SAME saved grid as σ_crit (_MM8, identical
# cell indexing), so nothing has to be replayed to test it. Every finite-σ_crit cell
# in the grid carries a finite M_max, so a window never drops a cell for a missing
# M_max; the isfinite() below only excludes the +inf ("nucleates at every tested σ")
# cells, which replay_accepted rejects anyway — it keeps the printed count honest.
_s8b, _m8b = _SIG8[_al_idx], _MM8[_al_idx]
_keep8b = (np.isfinite(_s8b) &
           (_s8b >= F8B_SIG_RANGE[0])  & (_s8b <= F8B_SIG_RANGE[1]) &
           (_m8b >= F8B_MMAX_RANGE[0]) & (_m8b <= F8B_MMAX_RANGE[1]))
print(f"Fig 5a: {_keep8b.sum()}/{np.isfinite(_s8b).sum()} viable cells kept "
      f"(σ_crit ∈ {F8B_SIG_RANGE}, M_max ∈ {F8B_MMAX_RANGE})")
_f8b_curves = nuc_an.replay_accepted(np.where(_keep8b, _s8b, np.nan),
                                     _al8[_al_idx], _B48, _D08,
                                     _filt_cfg, max_curves=F8B_MAX_CURVES,
                                     n_jobs=scan_n_jobs)
_norm8 = plt.Normalize(vmin=_vmin8, vmax=_vmax8)     # == Fig 5 heatmap colour scale
# NOTE deliberately the FULL Fig-5 range, so a colour means the same σ_crit in both
# figures. If a narrow F8B_SIG_RANGE leaves the curves near-monochrome, swap in
# plt.Normalize(*F8B_SIG_RANGE) to stretch the colours across the window instead.
_cmap8 = plt.cm.viridis

# ---- how to render the bundle ----------------------------------------------
# 'curves' = one thin line per parametrization (the original). Honest but it is
#   thousands of lines on top of each other: the colour you SEE at any pixel is
#   just whichever cell happened to be drawn last, so you cannot read where
#   σ_crit is large or small. Sorting (below) at least makes that deterministic.
# 'bands'  = split the cells into F8B_NBIN equal-count σ_crit bins and draw each
#   bin as its median curve + a p16..p84 envelope. Same data, ~5 readable objects
#   instead of ~10^3 overlapping ones, and the σ ordering becomes visible because
#   the bands separate. This is the one that answers "where is σ_crit large".
F8B_MODE  = 'curves'                      # 'bands' | 'curves'
F8B_NBIN  = 10                            # σ_crit bins when F8B_MODE == 'bands'
F8C_RDEF  = 'R_1.4'                     # scatter radius: 'R_Mmax' | 'R_1.4' | 'R_max'
# -----------------------------------------------------------------------------
# Draw low σ_crit first so high σ_crit ends up ON TOP. In 'curves' mode the topmost
# line is the only one you see, so without a sort that choice was the arbitrary
# order the grid cells happened to come back in. Reverse for the opposite emphasis.
_f8b_curves = sorted(_f8b_curves, key=lambda t: t[1])


def _f8b_profiles(curves, xk, yk, xg, stable=False):
    """y(xg) for every curve, on the common grid xg; NaN outside a curve's own span.

    `stable=True` clips each curve to its stable branch (up to M_max) first — that
    is what makes R(M) single-valued, and so interpolable, on the M-R plane.
    """
    out = []
    for c, _ in curves:
        x, y = np.asarray(c[xk], float), np.asarray(c[yk], float)
        if stable:
            k = int(np.argmax(np.asarray(c['M']))) + 1
            x, y = x[:k], y[:k]
        o = np.argsort(x)                        # np.interp needs increasing xp
        out.append(np.interp(xg, x[o], y[o], left=np.nan, right=np.nan))
    return np.asarray(out)


def _f8b_band(ax, prof, xg, colour, label, swap=False, min_n=3):
    """Median + p16..p84 envelope of a bundle of profiles.

    Grid columns backed by fewer than `min_n` curves are dropped: at the high-mass
    end only a few members of a bin still exist, and a percentile over one or two
    curves is noise drawn as if it were a band. `swap` puts the independent
    variable on the y-axis (the M-R panel interpolates R at fixed M, but plots R
    horizontally).
    """
    ok = np.isfinite(prof).sum(axis=0) >= min_n
    if not ok.any():
        return
    lo, md, hi = (np.nanpercentile(prof[:, ok], p, axis=0) for p in (16, 50, 84))
    g = xg[ok]
    if swap:
        ax.fill_betweenx(g, lo, hi, color=colour, alpha=0.30, lw=0, zorder=2)
        ax.plot(md, g, color=colour, lw=1.7, zorder=3, label=label)
    else:
        ax.fill_between(g, lo, hi, color=colour, alpha=0.30, lw=0, zorder=2)
        ax.plot(g, md, color=colour, lw=1.7, zorder=3, label=label)


def _f8b_bins(curves, nbin):
    """Equal-COUNT (quantile) σ_crit bins, as (lo, hi, members, bin colour).

    Equal-count rather than equal-width: σ_crit is far from uniform over the grid,
    so equal-width bins would leave some holding a handful of cells and others
    most of them. Colour is taken at the bin's median σ_crit on the SAME norm as
    the heatmap, so a band's colour still means the same σ_crit as everywhere else.
    """
    s = np.array([sc for _, sc in curves], float)
    e = np.quantile(s, np.linspace(0, 1, nbin + 1))
    e[-1] = np.nextafter(e[-1], np.inf)          # top edge inclusive
    out = []
    for k in range(nbin):
        m = (s >= e[k]) & (s < e[k + 1])
        if m.any():
            out.append((e[k], e[k + 1], [curves[i] for i in np.flatnonzero(m)],
                        _cmap8(_norm8(np.median(s[m])))))
    return out


def _f8b_summary(c):
    """(M_max, R) summary point of one replayed TOV curve, R chosen by F8C_RDEF:
         'R_Mmax' radius of the maximum-mass star (the compactness endpoint)
         'R_1.4'  radius of the 1.4 M_sun star (what NICER-type measurements pin)
         'R_max'  largest radius anywhere on the curve
    R is NaN when the branch never reaches 1.4 M_sun ('R_1.4' only)."""
    M, R = np.asarray(c['M'], float), np.asarray(c['R'], float)
    k = int(np.argmax(M))
    if F8C_RDEF == 'R_Mmax':
        r = R[k]
    elif F8C_RDEF == 'R_max':
        r = np.nanmax(R)
    else:
        r = np.interp(1.4, M[:k + 1], R[:k + 1], left=np.nan, right=np.nan)
    return float(M[k]), float(r)


# 2×2: (a) M-R, (b) P_CFL(mu_B), (c) the (M_max, R) scatter, (d) unused.
fig, _axes = paper_grid('2x2', mode='double', placeholder=False, square=False,
                        **(PAPER_STYLE | {'aspect': 1.0}))
axMR, axP, axS, _axCB = _axes.flat
# The 4th quadrant holds the colorbar instead of being hidden: attaching the bar to
# the three data panels would stretch it over the whole figure height. Frame off, but
# the axes still occupies its slot so constrained_layout keeps the grid square.
_axCB.set_frame_on(False); _axCB.set_xticks([]); _axCB.set_yticks([])
_bins = _f8b_bins(_f8b_curves, F8B_NBIN) if F8B_MODE == 'bands' else []

# (a) M-R: hadronic reference (black) + 2 M_sun guide; CFL bundle coloured by σ_crit
_hadr, = axMR.plot(tov_cold[:, 3], tov_cold[:, 4], 'k-', lw=2,
                   label='Hadronic (T=0)', zorder=5)
axMR.axhline(2.0, color='0.5', ls=':', lw=1.0, zorder=1)
if F8B_MODE == 'bands':
    _Mg = np.linspace(0.4, max(np.nanmax(c['M']) for c, _ in _f8b_curves), 160)
    for _lo, _hi, _mem, _col in _bins:               # R(M) on the stable branch
        _f8b_band(axMR, _f8b_profiles(_mem, 'M', 'R', _Mg, stable=True), _Mg,
                  _col, f'{_lo:.0f}–{_hi:.0f}', swap=True)
else:
    for _c, _sc in _f8b_curves:
        axMR.plot(_c['R'], _c['M'], color=_cmap8(_norm8(_sc)), lw=0.6, alpha=0.8)
axMR.set_xlim(7, 16); axMR.set_ylim(0, 3.0)
axMR.set_xlabel(r'$R$ [km]'); axMR.set_ylabel(r'$M$ [$M_\odot$]')
# TWO legends, not one: the hadronic curve is not a σ_crit bin, so folding it into
# the σ-titled box would put "Hadronic" under a "σ_crit [MeV/fm²]" heading. Corners
# are explicit (not loc='best') because 'best' ignores Text and would sometimes land
# on the (a) panel tag — move these two if the real curves fill those corners.
_lg_h = axMR.legend(handles=[_hadr], loc='upper right', fontsize=8)
if F8B_MODE == 'bands':
    axMR.add_artist(_lg_h)                       # keep it when the second is added
    axMR.legend(loc='lower left', fontsize=8,
                title=r'$\sigma_{\rm crit}$ [MeV/fm$^2$]',
                handles=[Line2D([], [], color=c, lw=1.7, label=f'{lo:.0f}–{hi:.0f}')
                         for lo, hi, _, c in _bins])
panel_label(axMR, '(a)')

# (b) P_CFL(mu_B): hadronic P_H (black) + the same bundle
axP.plot(mu_B_H_sorted, P_H_sorted, 'k-', lw=2, label=r'$P_H\ (T\approx0)$', zorder=5)
if F8B_MODE == 'bands':
    _mug = np.linspace(min(np.nanmin(c['mu']) for c, _ in _f8b_curves),
                       max(np.nanmax(c['mu']) for c, _ in _f8b_curves), 200)
    for _lo, _hi, _mem, _col in _bins:
        _f8b_band(axP, _f8b_profiles(_mem, 'mu', 'P', _mug), _mug, _col, None)
else:
    for _c, _sc in _f8b_curves:
        axP.plot(_c['mu'], _c['P'], color=_cmap8(_norm8(_sc)), lw=0.6, alpha=0.8)
axP.set_xlabel(r'$\mu_B$ [MeV]'); axP.set_ylabel(r'$P$ [MeV/fm$^3$]')
axP.legend(loc='lower right')            # 'upper left' is where the (b) tag lives
panel_label(axP, '(b)')

# (c) one POINT per parametrization instead of one curve — no overplotting, so the
#     σ_crit colour is directly readable. Same axes orientation as (a) (R on x,
#     M on y) so the two panels can be compared without flipping your head.
_MS = np.array([_f8b_summary(c) for c, _ in _f8b_curves])          # (M_max, R)
_SS = np.array([sc for _, sc in _f8b_curves])
_fin = np.isfinite(_MS[:, 1])
axS.axhline(2.0, color='0.5', ls=':', lw=1.0, zorder=1)
_sca = axS.scatter(_MS[_fin, 1], _MS[_fin, 0], c=_SS[_fin], cmap=_cmap8, norm=_norm8,
                   s=9, lw=0, alpha=0.85, zorder=3)
_RLAB = {'R_Mmax': r'$R(M_{\max})$', 'R_1.4': r'$R_{1.4}$', 'R_max': r'$R_{\max}$'}
axS.set_xlabel(_RLAB[F8C_RDEF] + ' [km]'); axS.set_ylabel(r'$M_{\max}$ [$M_\odot$]')
panel_label(axS, '(c)')
if (~_fin).any():                                  # only possible for R_1.4
    print(f"(c) {(~_fin).sum()}/{len(_fin)} curves have no {F8C_RDEF} "
          f"(branch never reaches 1.4 M_sun) and are not plotted")

# colorbar in the free 4th quadrant, sized as a slim bar inside it
fig.colorbar(_sca, cax=_axCB.inset_axes([0.04, 0.03, 0.07, 0.94]),
             label=r'$\sigma_{\rm crit}$ [MeV/fm$^2$]')
fig.savefig(f'../output/figures/paper_fig5a_MR_PmuB_sigmacrit_{xsd_tag}.pdf',
            bbox_inches='tight')
plt.show()


# %% [markdown]
# ### Paper Appendix A — electric-charge prescriptions: (a) $W(R)$, (b) $T_{\rm nuc}(n_B^H)$
#
# The two panels of App. A of the paper, merged into ONE figure.
# **(a)** work of formation $W(R)$ of an unpaired droplet at the centre of the
# reference PNS at $t_{T_\mathrm{max}}$ (the trapped star whose baryonic mass is
# that of a cold $1.4\,M_\odot$ NS — the same star $\sigma_{\rm crit}$ is
# evaluated on), under the five charge prescriptions: **LCN** (local neutrality,
# no Coulomb), **GCN** (global neutrality, no Coulomb), **GCN+Coulomb**
# (unscreened $\propto R^5$ — no critical droplet), **minimization**
# (`coulomb_minimize`, the scheme used in the paper) and **screening** (Debye).
# Circles mark each critical point.
# **(b)** the same prescriptions seen through the observable that matters: the
# nucleation curve $T_{\rm nuc}(n_B^H)$ (where $\tau=\tau_{\rm target}$) for the
# unpCFL droplet at two surface tensions (colour = prescription, line style =
# $\sigma$). The spread between prescriptions is much smaller than in (a),
# because $W_*\sim\sigma^{2\text{–}3}$ compresses the barrier uncertainty — and
# it stays small at both $\sigma$.

# %%
# ============================================================================
#  Appendix A.  Electric-charge prescription: barrier vs observable.
#    (a) W(R) at the reference PNS centre, five prescriptions (unpaired droplet)
#    (b) T_nuc(n_B^H) for LCN / GCN / minimization (unpCFL, two sigmas)
#  Physics: the prescription fixes how much net charge the droplet may carry and
#  what it costs. LCN (Q=0, no Coulomb) and GCN (charged, cost ignored) bracket
#  the truth; the unscreened GCN+Coulomb R^5 term overshoots and kills the
#  barrier peak; minimization and Debye screening sit inside the bracket and on
#  top of each other -- that agreement is the point of the appendix.
# ============================================================================
APP_SET = quark_param_sets[0]                    # Set A
APP_TAG = q_tag_of(APP_SET)
APP_PAR = get_alphabag_custom(alpha=APP_SET['alpha'], B4=APP_SET['B4'],
                              m_s=APP_SET['m_s'])
APP_SIG = 100.0                                  # sigma of both panels [MeV/fm^2]
APP_MT0 = 1.4                                    # reference PNS: Mb = Mb(cold 1.4 Msun)
APP_YL  = PNS_TMAX['YLH']                        # t_Tmax snapshot (hottest)
APP_PH  = 'unpaired'                               # phase of panel (b)

# Reference PNS centre (n_B, T) — taken from the star match, NOT hardcoded, so
# it can never drift from the star used in Figs. 3/5. Paper quotes ~3.5 n_sat, ~33 MeV.
APP_NB, APP_T, APP_HPT = nuc_an.central_state(APP_MT0, _star)
print(f"reference PNS centre: n_B = {APP_NB/n_sat:.2f} n_sat, T = {APP_T:.2f} MeV")

# Point at which panel (a) is evaluated. Set to (None, None) to fall back on the
# reference PNS centre above; a round (n_B, T) near it reads better in a caption
# and the conclusions of the appendix do not depend on the exact point.
APPA_NB, APPA_T = 3.5 * n_sat, 30.0
if APPA_NB is None or APPA_T is None:
    APPA_NB, APPA_T = APP_NB, APP_T
print(f"panel (a) evaluated at: n_B = {APPA_NB/n_sat:.2f} n_sat, T = {APPA_T:.2f} MeV")

# (charge mode, label, colour, linestyle). lcn/gcn/coulomb_minimize also have
# saved nucleation tables -> panel (b). NOTE screening draws on top of (and hides)
# minimization: the two agree to <1%, which is the point -- give screening a dashed
# style here if you want both visible.
_APP_MODES = [
    ('lcn',              'LCN',          OKAB_CAT[0], '-'),
    ('gcn',              'GCN',          OKAB_CAT[1], '-'),
    ('gcn_coulomb',      'GCN + Coul.',  OKAB_CAT[3], '-'),
    ('coulomb_minimize', 'minimization', OKAB_CAT[2], '-'),
    ('screening',        'screening',    OKAB_CAT[4], '-'),
]
_APP_TABLE_MODES = ['lcn', 'gcn', 'coulomb_minimize']    # panel (b): tables exist

fig, ((axA, axB),) = paper_grid('1x2', mode='double', placeholder=False, square=False,
                             **(PAPER_STYLE | {'aspect': 1.15}))

# ── (a) W(R) ────────────────────────────────────────────────────────────────
APP_Rg = np.linspace(0.02, 12.0, 500)            # droplet radius grid [fm]
_dA, _wmax, _lamD = [], 0.0, np.nan
for _ch, _lbl, _col, _ls in _APP_MODES:
    _eb = compute_energy_barrier(
        H['trapped'], APPA_NB, APPA_T, APP_SIG, electric_charge_mode=_ch,
        params=APP_PAR, flavor_mode='saddlepoint', quark_phase='unpaired',
        Y_L_H=APP_YL, R_values=APP_Rg)
    axA.plot(APP_Rg, _eb.W, color=_col, ls=_ls, lw=1.7, label=_lbl)
    if _ch == 'screening':
        _lamD = float(_eb.lambda_D)
    if np.isfinite(_eb.W).any():
        # Critical droplet = FIRST local maximum of W(R), not the global one: for
        # GCN+Coulomb the global max sits at the right edge of the R grid (the R^5
        # runaway), which is not a critical point.
        _dW = np.diff(_eb.W)
        _k = int(np.argmax(_dW < 0)) if (_dW < 0).any() else int(np.nanargmax(_eb.W))
        axA.plot(APP_Rg[_k], _eb.W[_k], 'o', ms=5, color=_col, mfc='white', zorder=5)
        print(f"  {_lbl:14s} R_* = {APP_Rg[_k]:.2f} fm, "
              f"W_*/T = {_eb.W[_k]/APPA_T:.1f}")
        _wmax = max(_wmax, float(_eb.W[_k]))     # kept for reference (see APPA_YLIM)
    _dA += [(_lbl, float(r), float(w)) for r, w in zip(APP_Rg, _eb.W)]
axA.axhline(0, color='0.6', lw=0.7, zorder=0)
# Fixed frame: only the barrier region is shown. The GCN+Coulomb curve dips below
# zero past its peak and then runs off the top as R^5 (~+1e5 MeV by R=12 fm) --
# both excursions are outside the frame on purpose; use (-0.35, 1.35)*_wmax to see
# the dip again.
APPA_YLIM = (0, 10000)
axA.set_xlim(0, 8.5); axA.set_ylim(*APPA_YLIM)
axA.set_xlabel(r'$R$ [fm]'); axA.set_ylabel(r'$W$ [MeV]')
axA.legend(loc='upper left')
panel_label(axA, '(a)', 'upper right')

# ── (b) T_nuc(n_B^H) from the saved nucleation tables ───────────────────────
# TWO surface tensions, so the reader sees that the prescription spread is small
# NOT just at one sigma: colour = prescription, line style = sigma. Any sigma in
# sigma_list works (the tables must exist); APP_SIG is the one used in panel (a).
APPA_SIGMAS = [APP_SIG, 150]
_SIG_LS = {sg: ls for sg, ls in zip(APPA_SIGMAS, ((0, (5, 2)), '-'))}

_dB, _sig_anchor = [], {}
for _sg in APPA_SIGMAS:
    for _ch, _lbl, _col, _ls in _APP_MODES:
        if _ch not in _APP_TABLE_MODES:
            continue
        stem = f"Htrapped_saddlepoint_{_ch}_{APP_PH}_{APP_TAG}_s{int(_sg)}"
        if stem not in nuc_sets:
            print(f"  [skip] missing table {stem}")
            continue
        _grids = nuc_sets[stem].hadronic_grids
        _iYL   = int(np.argmin(np.abs(_grids['Y_L_H'] - APP_YL)))
        _res   = compute_nucleation_density(nuc_sets[stem], tau_target=tau_target, scan='n_B')
        _nB, _Tn = nucleation_curve(_res, _iYL)
        _m = np.isfinite(_nB) & np.isfinite(_Tn)
        axB.plot(_nB[_m] / n_sat, _Tn[_m], color=_col, ls=_SIG_LS[_sg], lw=1.7)
        _sig_anchor.setdefault(_sg, []).append((_nB[_m] / n_sat, _Tn[_m]))
        _dB += [(_lbl, _sg, float(n), float(t))
                for n, t in zip(_nB[_m] / n_sat, _Tn[_m])]

# sigma written NEXT TO its group of curves instead of in the legend: the three
# prescriptions of one sigma sit close together, so one label per group is enough.
_xa = 3.0                                        # n_B/n_sat where the labels sit
for _sg, _curves in _sig_anchor.items():
    _ys = [float(np.interp(_xa, _x, _y)) for _x, _y in _curves]   # group spread at _xa
    _above = _sg > min(_sig_anchor)              # lowest sigma labelled BELOW its group
    axB.text(_xa, (max(_ys) + 2.5) if _above else (min(_ys) - 3.0),
             rf'$\sigma={int(_sg)}$ MeV fm$^{{-2}}$', fontsize=9,
             va='bottom' if _above else 'top', rotation=-10, rotation_mode='anchor')

# Stellar track: T(n_B^H) along the t_Tmax isentrope (Y_L^H, S), drawn out to the
# central density of the HEAVIEST PNS on that trapped sequence — beyond that no
# star exists, so the curve stops there. Where the track lies ABOVE a T_nuc curve
# the star nucleates.
_tov_tr  = tov_trapped[min(tov_trapped,                       # nearest (Y_L, S) sequence
                           key=lambda k: abs(k[0] - APP_YL) + abs(k[1] - PNS_TMAX['S']))]
_nBc_max = float(_tov_tr[int(np.argmax(_tov_tr[:, 4])), 2])   # cols: 2=n_Bc, 4=M
_nBiso   = np.linspace(1.0 * n_sat, _nBc_max, 200)
print(f"isentrope (App. B) drawn to n_B(M_PNS^max) = {_nBc_max/n_sat:.2f} n_sat "
      f"(M_PNS^max = {_tov_tr[:, 4].max():.2f} Msun)")

# top at 95 MeV so the sigma=150 curves are not cropped at low density
axB.set_xlim(1, max(10, np.ceil(_nBc_max / n_sat))); axB.set_ylim(1, 95)
axB.set_xlabel(r'$n_B^H / n_{\rm sat}$'); axB.set_ylabel(r'$T_{\rm nuc}$ [MeV]')
# Legend carries the prescriptions only; sigma is annotated on the curves above.
axB.legend(handles=[Line2D([], [], color=_c, label=_l)
                    for _ch, _l, _c, _ in _APP_MODES if _ch in _APP_TABLE_MODES],
           loc='upper right')
panel_label(axB, '(b)')

pd.DataFrame(_dA, columns=['prescription', 'R_fm', 'W_MeV']).to_csv(
    f'../output/figure_data/appA_WR_charge_{xsd_tag}.csv', index=False)
pd.DataFrame(_dB, columns=['prescription', 'sigma', 'nBH_over_n0', 'T_nuc_MeV']).to_csv(
    f'../output/figure_data/appA_Tnuc_charge_{xsd_tag}.csv', index=False)
fig.savefig(f'../output/figures/paper_appA_charge_prescriptions_{xsd_tag}.pdf',
            bbox_inches='tight')
plt.show()
print(f"lambda_D = {_lamD:.1f} fm at the reference centre")


# %% [markdown]
# ### Paper Appendix B — frozen-flavour vs saddle-point composition
#
# **(a)** $T_{\rm nuc}(n_B^H)$ (the $\tau=\tau_{\rm target}$ condition) at one
# surface tension for Set A: **frozen flavour** ($Y_C^{Q}=Y_C^{H}$,
# $Y_S^{Q}=Y_S^{H}$ — the droplet inherits the hadronic composition; unpaired
# only) against **saddle point** (composition minimizes $W$, i.e. strong
# equilibrium across the interface — the scheme used in the paper) for the three
# droplet phases. The frozen curve sits far above the others: it needs $T\sim70$
# MeV where the saddle-point ones need $\sim35$.
# **(b)** why: the composition of the critical droplet itself, $Y_i^{Q*}$ for
# $i=C,S,e,\nu$, at $T=30$ MeV. The saddle-point droplet goes straight to
# $Y_S\simeq0.9$ (nearly one strange quark per baryon, what makes it bulk-favoured),
# while the frozen droplet is pinned to the small hadronic $Y_S^{H}$. Under LCN
# $Y_e^{Q*}=Y_C^{Q*}$ by construction and $Y_\nu^{Q*}=0$ (no trapped neutrinos
# inside the droplet), so those two curves are degenerate with $Y_C$ and with zero.

# %%
# ============================================================================
#  Appendix B.  Frozen-flavour vs saddle-point composition (LCN).
#    (a) T_nuc(n_B^H), Set A at APPB_SIG: frozen/unpaired vs saddle-point
#        unpaired / CFL / unpCFL.
#    (b) critical-droplet composition Y_i^{Q*}(n_B^H) at APPB_T, both prescriptions.
#  Physics: a deconfined droplet is bulk-favored only if it is strange enough. In
#  the frozen limit it can only inherit the strangeness the BULK hadronic phase
#  already has, so it cannot nucleate until hyperons are abundant (high n_B^H);
#  the saddle-point droplet makes its own strangeness and nucleates throughout.
# ============================================================================
from eos.alphabag.thermodynamics_quarks import T_critical    # same Tc as the CFL EoS gap

APPB_SIG   = 100.0                         # sigma of panel (a) [MeV/fm^2]
APPB_SET   = quark_param_sets[0]           # Set A
APPB_CH    = 'lcn'                         # frozen exists for LCN only -> compare at LCN
APPB_T     = 30.0                          # temperature of panel (b) [MeV]
APPB_NB2   = np.linspace(1.0, 10.0, 70) * n_sat       # density grid of panel (b)
# (flavor, phase, label, colour, linestyle). Colour = composition prescription,
# line style = droplet phase (same convention as Fig. 1/4).
APPB_CURVES = [
    ('saddlepoint', 'unpaired', 'saddle point, unpaired', OKAB_CAT[0], ':'),
    ('saddlepoint', 'cfl',      'saddle point, CFL',      OKAB_CAT[0], '--'),
    ('saddlepoint', 'unpCFL',   'saddle point, unpCFL',   OKAB_CAT[0], '-'),
    ('frozen',      'unpaired', 'frozen, unpaired',       OKAB_CAT[3], ':'),
]
# species of panel (b): (attribute, label, colour). Y_e and Y_nu are not drawn:
# under LCN Y_e = Y_C identically and Y_nu = 0 (no trapped neutrinos in the droplet).
APPB_SPECIES = [('Y_C', r'$Y_C$', OKAB_CAT[3]),
                ('Y_S', r'$Y_S$', OKAB_CAT[0])]

fig, ((axA, axB),) = paper_grid('1x2', mode='double', placeholder=False, square=False,
                             **(PAPER_STYLE | {'aspect': 1.15}))

# ── (a) T_nuc(n_B^H), Set A ─────────────────────────────────────────────────
_tag   = q_tag_of(APPB_SET)
_T_CFL = float(T_critical(APPB_SET['Delta0']))       # CFL undefined above Tc
_dB2 = []
for _fl, _ph, _lbl, _col, _ls in APPB_CURVES:
    stem = f"Htrapped_{_fl}_{APPB_CH}_{_ph}_{_tag}_s{int(APPB_SIG)}"
    if stem not in nuc_sets:
        print(f"  [skip] missing table {stem}")
        continue
    _grids = nuc_sets[stem].hadronic_grids
    _iYL   = int(np.argmin(np.abs(_grids['Y_L_H'] - APP_YL)))
    _res   = compute_nucleation_density(nuc_sets[stem], tau_target=tau_target, scan='n_B')
    _nB, _Tn = nucleation_curve(_res, _iYL)
    _m = np.isfinite(_nB) & np.isfinite(_Tn)
    if _ph == 'cfl':
        _m &= (_Tn <= _T_CFL)                        # gap vanishes above Tc
    # PHASE_ALPHA fades the non-emphasised phases WITHIN the saddle-point family;
    # the frozen curve is the comparison the panel is about, so it stays opaque.
    axA.plot(_nB[_m] / n_sat, _Tn[_m], color=_col, ls=_ls, lw=PHASE_LW[_ph],
             alpha=(1.0 if _fl == 'frozen' else PHASE_ALPHA[_ph]), label=_lbl)
    print(f"  {_lbl:24s}: {int(_m.sum())} points"
          + (f", n_B >= {_nB[_m].min()/n_sat:.2f} n_sat" if _m.any() else " (no nucleation)"))
    _dB2 += [(_fl, _ph, float(n), float(t)) for n, t in zip(_nB[_m] / n_sat, _Tn[_m])]
axA.set_xlim(1, 10); axA.set_ylim(5, 80)
axA.set_xlabel(r'$n_B^H / n_{\rm sat}$'); axA.set_ylabel(r'$T_{\rm nuc}$ [MeV]')
# The legend lives in the band between the frozen curve (top of the panel) and
# the saddle-point ones (middle); anchored in axes coords so it stays there.
axA.legend(loc='upper left', bbox_to_anchor=(0.02, 0.90), fontsize=8)
panel_label(axA, '(a)')

# ── (b) critical-droplet composition at APPB_T ──────────────────────────────
# Composition is sigma-independent under LCN, so the sigma passed here only fixes
# R_c (unused): the Y_i are the ones entering W(R) at every sigma. Q* is returned
# even where no critical droplet exists (Delta_f >= 0), which is exactly the frozen
# case over most of this range -- that is what we want to show.
_comp = {}
for _fl in ('saddlepoint', 'frozen'):
    _rows = []
    for _nb in APPB_NB2:
        _hp = nuc_an.hadronic_point(H['trapped'], _nb, APP_YL, APPB_T)
        _Qs = nuc_an.critical_droplet_pt(APPB_SIG, _hp, APPB_T, _fl, APPB_CH,
                                         'unpaired', {}, APP_PAR,
                                         APPB_SET['Delta0'], _nuc_cfg)[2]
        _rows.append([np.nan if _Qs is None else float(getattr(_Qs, _y))
                      for _y, *_ in APPB_SPECIES])
    _comp[_fl] = np.array(_rows)                     # (n_density, n_species)

for _j, (_y, _ylbl, _col) in enumerate(APPB_SPECIES):
    axB.plot(APPB_NB2 / n_sat, _comp['saddlepoint'][:, _j], color=_col, ls='-', lw=1.8)
    axB.plot(APPB_NB2 / n_sat, _comp['frozen'][:, _j], color=_col, ls='--', lw=1.4)
axB.set_xlim(1, 10); axB.set_ylim(-0.03, 1.0)
axB.set_xlabel(r'$n_B^H / n_{\rm sat}$'); axB.set_ylabel(r'$Y_i^{Q*}$')
axB.legend(handles=[Line2D([], [], color=_c, ls='-', lw=1.8, label=_lb)
                    for _y, _lb, _c in APPB_SPECIES]
                   + [Line2D([], [], color='k', ls='-',  lw=1.8, label='saddle point'),
                      Line2D([], [], color='k', ls='--', lw=1.4, label='frozen')],
           loc='center left', ncol=2)
panel_label(axB, '(b)')

pd.DataFrame(_dB2, columns=['flavor_mode', 'phase', 'nBH_over_n0', 'T_nuc_MeV']
             ).to_csv(f'../output/figure_data/appB_Tnuc_{xsd_tag}.csv', index=False)
pd.DataFrame({'nBH_over_n0': APPB_NB2 / n_sat,
              **{f'{_y}_{_fl}': _comp[_fl][:, _j]
                 for _fl in _comp for _j, (_y, *_r) in enumerate(APPB_SPECIES)}}
             ).to_csv(f'../output/figure_data/appB_composition_{xsd_tag}.csv', index=False)
fig.savefig(f'../output/figures/paper_appB_frozen_vs_saddlepoint_{xsd_tag}.pdf',
            bbox_inches='tight')
plt.show()


# %% [markdown]
# ### $\Delta\sigma_{\rm crit}$ maps — sensitivity to $M_{T0}$ and to the droplet phase
#
# How much does $\sigma_{\rm crit}(B^{1/4},\Delta_0)$ actually move when you change
# an *input* rather than the quark parameters? Two comparisons, same machinery and
# same $(B^{1/4},\Delta_0)$ plane as Fig 5, but showing a **signed difference**:
#
# * **(A)** $\sigma_{\rm crit}(M_{T0}) - \sigma_{\rm crit}(1.4)$ for each
#   $M_{T0}\in$ `DSIG_MT0S` — the threshold mass sets which star must not have
#   nucleated, so a heavier $M_{T0}$ probes a denser centre and generally needs a
#   *larger* $\sigma$ to stay quiet.
# * **(B)** $\sigma_{\rm crit}(\mathrm{unpCFL}) - \sigma_{\rm crit}(\mathrm{CFL})$
#   at $M_{T0}=1.4$ — the droplet-phase choice. These coincide *exactly* wherever
#   the critical droplet is larger than the pairing-coherence radius ($R_*>R_\Delta$,
#   so the droplet is fully CFL); they can only differ in the small-droplet corner.
#
# Colour is **diverging with a neutral midpoint and symmetric limits**, so white
# means "no change" and the two hues are the sign — deliberately *not* Fig 5's
# sequential viridis, which encodes magnitude.
#
# **(A) reads one saved `.npz` per $M_{T0}$ and does not compute them.** To have
# them ready, set `MT0_grid_thr = [1.17, 1.3, 1.4, 1.5, 1.6]` in the IV.3 grid-scan
# cell and re-run it (`_run_scan` already loops over the list and saves one file
# each). Any missing $M_{T0}$ is reported and skipped.

# %%
# ============================================================================
#  Delta-sigma_crit maps + summary statistics.  Two comparisons, one machinery:
#    (A) sigma_crit(MT0) - sigma_crit(DSIG_REF_MT0)  over DSIG_MT0S   [phase fixed]
#    (B) sigma_crit(unpCFL) - sigma_crit(cfl)        at DSIG_REF_MT0  [MT0 fixed]
#  Everything is read from the saved IV.3 grids -- nothing is re-scanned here (a
#  grid scan is hours). Cells not viable in BOTH grids are left white: a difference
#  is only meaningful where both sides exist.
#
#  Why the stats are reported twice (all cells / differing cells): for (B) the two
#  phases are bit-identical wherever the critical droplet is bigger than the
#  coherence radius, which is most of the plane. An all-cells median is then 0 and
#  tells you nothing about the size of the effect where there IS one. So each row
#  reports the fraction of cells that actually move, and the spread among those.
# ============================================================================
set_paper_style()
# ---- knobs -----------------------------------------------------------------
DSIG_REF_MT0 = 1.40                     # baseline M_T0 [M_sun] everything is differenced against
DSIG_MT0S    = [1.17, 1.6]    # M_T0 values to compare against the baseline
DSIG_PHASE   = 'unpCFL'                 # droplet phase used for comparison (A)
DSIG_PHASES  = ('unpCFL', 'cfl')        # the pair compared in (B), as (minuend, subtrahend)
DSIG_ALPHA   = 0                        # alpha_s slice index mapped in (A) (index into alpha_slices)
DSIG_CMAP    = 'RdBu_r'                 # diverging, neutral midpoint: red = +, blue = -, white = 0
DSIG_VLIM    = None                     # colour limit +/-; None = robust value from the data
DSIG_PCT     = 99                       # percentile of |Delta| setting that limit when VLIM is None
DSIG_TOL     = 1e-3                     # |Delta| above this counts as "actually differs" [MeV/fm^2]
DSIG_FLAVOR, DSIG_CHARGE = 'saddlepoint', 'coulomb_minimize'
# ----------------------------------------------------------------------------
_DSIG_AXES = ('alpha_slices', 'B4_grid', 'Delta0_grid')


def _dsig_path(MT0, phase):
    return (f'../output/mc_cfl/sigma_crit_grid_{xsd_tag}_MT0{MT0:.2f}_'
            f'{DSIG_FLAVOR}-{DSIG_CHARGE}-{phase}.npz')


def _dsig_load(MT0, phase):
    """Saved sigma_crit grid for (MT0, phase), or None if it was never scanned."""
    p = _dsig_path(MT0, phase)
    return np.load(p, allow_pickle=False) if os.path.exists(p) else None


def _dsig_diff(gA, gB):
    """Signed sigma_crit difference gA - gB on the cells finite in BOTH, NaN elsewhere.

    Refuses to difference grids built on different axes: the arrays would still
    subtract element-wise and hand back a silently meaningless map.
    """
    for k in _DSIG_AXES:
        if not np.array_equal(gA[k], gB[k]):
            raise ValueError(f"grids differ in '{k}' -- cannot be differenced "
                             f"cell-by-cell (re-scan both on the same grid)")
    A, B = gA['sig_crit'], gB['sig_crit']
    both = np.isfinite(A) & np.isfinite(B)
    return np.where(both, A - B, np.nan), B          # (difference, reference for relative %)


def _dsig_stats(D, ref, tol=None):
    """Summary of a Delta-sigma_crit field, over all compared cells AND over the
    subset that actually moves (|Delta| > tol). Relative values are Delta/ref."""
    tol = DSIG_TOL if tol is None else tol
    m = np.isfinite(D)
    d, r = D[m], ref[m]
    if d.size == 0:
        return None
    mv = np.abs(d) > tol                              # cells that actually move
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
    """One formatted stats line. 'median' and the p16..p84 band are taken over the
    MOVING cells (see the cell header); min..max is over all compared cells."""
    if s is None:
        return f"{label:<34}{'-- no overlapping viable cells --':>65}"
    if not s.get('n_move'):
        return (f"{label:<34}{s['n']:>6}{'0 (0.0%)':>13}"
                f"{'identical everywhere':>56}")
    return (f"{label:<34}{s['n']:>6}{s['n_move']:>6} ({s['frac']:5.1f}%)"
            f"{s['med_m']:>+9.2f}{s['p16_m']:>+10.2f}..{s['p84_m']:<+10.2f}"
            f"{s['lo']:>+9.2f}..{s['hi']:<+8.2f}{s['rel_m']:>+7.1f}%")


def _dsig_vlim(fields):
    """Symmetric colour limit shared by a set of difference maps: the DSIG_PCT-th
    percentile of |Delta|, so a handful of outlier cells cannot flatten the rest.
    Symmetric is not cosmetic -- it is what puts 0 on the colormap's neutral
    midpoint, so white reliably means 'no change'."""
    if DSIG_VLIM is not None:
        return float(DSIG_VLIM)
    v = np.concatenate([np.abs(f[np.isfinite(f)]).ravel() for f in fields]) \
        if fields else np.array([])
    v = v[v > 0]
    return float(np.percentile(v, DSIG_PCT)) if v.size else 1.0


def _dsig_map(ax, D, B4, D0, vlim, title):
    """One difference panel in the Fig-5 plane. Returns the mesh for the colorbar."""
    # Cells missing from either grid are drawn GREY, not white: on a diverging map
    # white is the zero of the data ("no change"), so leaving no-data white as Fig 5
    # does would make "not viable" and "identical" look the same. 0.85 grey is the
    # same not-viable grey the unpaired heatmap further down already uses.
    _cm = plt.get_cmap(DSIG_CMAP).copy()
    _cm.set_bad('0.85')
    pcm = ax.pcolormesh(B4, D0, np.ma.masked_invalid(D), cmap=_cm,
                        vmin=-vlim, vmax=+vlim, shading='nearest')   # symmetric -> 0 = neutral
    ax.set_xlabel(r'$B^{1/4}$ [MeV]'); ax.set_ylabel(r'$\Delta_0$ [MeV]')
    ax.set_xlim(B4.min(), B4.max()); ax.set_ylim(0.1, D0.max())
    ax.set_title(title)
    return pcm


# ---- (A) M_T0 sensitivity ---------------------------------------------------
_ref = _dsig_load(DSIG_REF_MT0, DSIG_PHASE)
if _ref is None:
    print(f"(A) skipped: baseline grid missing -> {_dsig_path(DSIG_REF_MT0, DSIG_PHASE)}")
    _dA = []
else:
    _B4d, _D0d, _ald = _ref['B4_grid'], _ref['Delta0_grid'], _ref['alpha_slices']
    _ia = min(DSIG_ALPHA, len(_ald) - 1)
    _dA, _missA = [], []
    for _mt in DSIG_MT0S:
        _g = _dsig_load(_mt, DSIG_PHASE)
        if _g is None:
            _missA.append(_mt)
            continue
        _D, _R = _dsig_diff(_g, _ref)
        _dA.append((_mt, _D, _R))
    if _missA:
        print(f"(A) missing grids for M_T0 = {_missA} -> set "
              f"MT0_grid_thr = {sorted(set(DSIG_MT0S) | {DSIG_REF_MT0})} in the IV.3 "
              f"cell and re-run it, then re-run this cell.")

if _dA:
    _vA = _dsig_vlim([_D[_ia] for _, _D, _ in _dA])
    fig, _axA = paper_grid('2x2', mode='double', placeholder=False, square=False,
                           **(PAPER_STYLE | {'aspect': 1.0}))
    _axAf = list(_axA.flat)
    _pcm = None
    for _k, (_mt, _D, _) in enumerate(_dA[:len(_axAf)]):
        _pcm = _dsig_map(_axAf[_k], _D[_ia], _B4d, _D0d, _vA,
                         rf"$M_{{T0}}={_mt:g}\,M_\odot$")
        panel_label(_axAf[_k], f"({chr(97 + _k)})")
    for _ax in _axAf[len(_dA):]:                      # unused panels of the 2x2
        _ax.set_visible(False)
    fig.colorbar(_pcm, ax=_axA, label=(rf'$\sigma_{{\rm crit}}(M_{{T0}}) - '
                                       rf'\sigma_{{\rm crit}}({DSIG_REF_MT0:g})$'
                                       r' [MeV/fm$^2$]'),
                 fraction=0.046, pad=0.02)
    fig.suptitle(rf"$\alpha_s=\pi/2\times{_ald[_ia]/(np.pi/2):.1f}$, {DSIG_PHASE}",
                 fontsize=10)
    fig.savefig(f'../output/figures/dsigma_MT0_{xsd_tag}_{DSIG_PHASE}.pdf',
                bbox_inches='tight')
    plt.show()

# ---- (B) droplet-phase sensitivity at the baseline M_T0 ---------------------
_pA, _pB = DSIG_PHASES
_gA, _gB = _dsig_load(DSIG_REF_MT0, _pA), _dsig_load(DSIG_REF_MT0, _pB)
_dB = None
if _gA is None or _gB is None:
    print(f"(B) skipped: need both phase grids at M_T0={DSIG_REF_MT0:g} "
          f"({_pA}: {'ok' if _gA is not None else 'MISSING'}, "
          f"{_pB}: {'ok' if _gB is not None else 'MISSING'})")
else:
    _DB, _RB = _dsig_diff(_gA, _gB)
    _dB = (_DB, _RB)
    _B4b, _D0b, _alb = _gA['B4_grid'], _gA['Delta0_grid'], _gA['alpha_slices']
    _shw = [i for i in F8_SHOW if i < len(_alb)]      # same alpha panels as Fig 5
    _vB  = _dsig_vlim([_DB[i] for i in _shw])
    fig, _axB = paper_grid('1x2', mode='double', placeholder=False, square=False,
                           **(PAPER_STYLE | {'aspect': 1.0}))
    for _c, _i in enumerate(_shw[:2]):
        _pcmB = _dsig_map(_axB[0, _c], _DB[_i], _B4b, _D0b, _vB,
                          rf"$\alpha_s=\pi/2\times{_alb[_i]/(np.pi/2):.1f}$")
        panel_label(_axB[0, _c], f"({chr(97 + _c)})")
    fig.colorbar(_pcmB, ax=_axB,
                 label=rf'$\sigma_{{\rm crit}}$({_pA}) $-\ \sigma_{{\rm crit}}$({_pB})'
                       r' [MeV/fm$^2$]', fraction=0.046, pad=0.02)
    fig.savefig(f'../output/figures/dsigma_phase_{xsd_tag}_MT0{DSIG_REF_MT0:.2f}.pdf',
                bbox_inches='tight')
    plt.show()

# ---- typical differences, printed -------------------------------------------
# 'move' = cells with |Delta| > DSIG_TOL; median and the p16..p84 band are over
# THOSE cells; min..max spans all compared cells; 'rel' = median Delta/sigma_ref.
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
# ### $W_*/T$ over the accessible plane — is the barrier-to-temperature ratio an invariant?
#
# Classical nucleation theory fixes the *rate*, not the barrier:
# $\tau = 1/(V\Gamma)$ with $\Gamma=(\kappa\Omega_0/2\pi)\,e^{-W_*/T}$, so
# demanding $\tau=\tau_{\rm target}$ pins
#
# $$W_*/T \;=\; \ln\!\big(V\,\kappa\,\Omega_0\,\tau_{\rm target}/2\pi\big),$$
#
# which depends on the parametrization **only logarithmically, through the
# prefactor** $\kappa\Omega_0$. So at each cell's *own* $\sigma_{\rm crit}$ the map
# below should be nearly flat — that flatness is the claim, and the printed spread
# is the number to quote. Set `WT_SIGMA` to a fixed $\sigma$ instead and the same
# map becomes a genuine barrier landscape (there $W_*/T$ is free to vary, and
# large $W_*/T$ = hard to nucleate).
#
# $W_*$ is read at the **deciding shell** — the one with the shortest $\tau$ —
# because $\sigma_{\rm crit}$ is itself star-wide (min $\tau$ over `WT_SHELLS`
# shells), and that shell is not always the centre.

# %%
# ============================================================================
#  W*/T colour map over the (B^1/4, Delta_0) plane, for every accessible
#  (sigma_crit-viable) cell of the Fig 5 grid. Same plane, same alpha slices,
#  same masking convention as Fig 5; the quantity is W*/T instead of sigma_crit.
#  Needs the Fig 5 cell (_SIG8/_al8/_B48/_D08/_F8) and the IV.0 engine (_star,
#  _nuc_cfg, N_SHELLS, NB_SHELL_MIN).
# ============================================================================
set_paper_style()
WT_SHOW   = [0, 2]            # alpha_s slice indices to DRAW (2 panels, as Fig 5)
# Slices to SUMMARIZE. Deliberately decoupled from WT_SHOW: Fig 5 has room for two
# panels, but a claim about "the whole accepted island" must be computed over every
# alpha slice the sigma_crit range is quoted over — otherwise the printed statistic
# silently describes 2/3 of the island. Set to WT_SHOW to summarize only what is drawn.
WT_STATS  = None              # None -> every alpha slice in the grid; or a list of indices
WT_SIGMA  = None              # None -> each cell's OWN sigma_crit; a float -> that fixed sigma
WT_SHELLS = N_SHELLS          # 1 = centre only (fast); N_SHELLS = star-wide, as sigma_crit
WT_CMAP   = 'magma'           # sequential, one hue — deliberately NOT Fig 5's viridis
WT_CLIP   = (2, 98)           # robust colour limits [percentiles], so outliers don't flatten it
WT_NJOBS  = scan_n_jobs
# grid metadata: the method must match the grid the sigma_crit values came from,
# otherwise W* would be evaluated for a different droplet than the one that set sigma_crit.
WT_FLAVOR, WT_CHARGE, WT_PHASE = (str(_F8['flavor']), str(_F8['charge']), str(_F8['phase']))
# Shells are HADRONIC (star match + isentrope) -> quark-independent -> built ONCE.
_wt_shells = nuc_an.star_shell_states(float(_F8['MT0']), _star, WT_SHELLS,
                                      nB_min=NB_SHELL_MIN)
print(f"W*/T map: {WT_PHASE}/{WT_FLAVOR}/{WT_CHARGE}, MT0={float(_F8['MT0']):.2f}, "
      f"{len(_wt_shells)} shell(s) n_B={_wt_shells[0][0].n_B:.3f}..{_wt_shells[-1][0].n_B:.3f} fm⁻³, "
      f"σ={'σ_crit(cell)' if WT_SIGMA is None else f'{WT_SIGMA:g} MeV/fm²'}")


def _wt_cell(al, b4, dl, sc):
    """(W*/T, T, shell index) at the shell with the SHORTEST tau, for one cell.

    The deciding shell — not the centre — is the one whose barrier defines the
    star-wide threshold, so that is where W* is read. tau_pt returns +inf when the
    hadronic phase is stable (W*=inf, never nucleates) and NaN on solver failure;
    both are excluded from the argmin, and a cell with no usable shell is NaN.
    The per-shell cache is reused for the second call, so re-solving the winning
    shell for W* costs only the barrier maximization, not the composition solve.
    """
    sig = sc if WT_SIGMA is None else WT_SIGMA
    if not (np.isfinite(sc) and np.isfinite(sig)):
        return np.nan, np.nan, -1              # not accessible / no sigma to evaluate at
    with np.errstate(divide='ignore', invalid='ignore'):   # crossover_radius @ Delta0=0
        p = get_alphabag_custom(alpha=al, B4=b4, m_s=m_s_fixed)
        caches = [{} for _ in _wt_shells]
        t = np.array([nuc_an.tau_pt(sig, hp, T, WT_FLAVOR, WT_CHARGE, WT_PHASE,
                                    caches[k], p, dl, _nuc_cfg)
                      for k, (hp, T) in enumerate(_wt_shells)], float)
        ok = np.isfinite(t) & (t > 0)
        if not ok.any():
            return np.nan, np.nan, -1
        k = int(np.flatnonzero(ok)[np.argmin(t[ok])])
        hp, T = _wt_shells[k]
        W = nuc_an.critical_droplet_pt(sig, hp, T, WT_FLAVOR, WT_CHARGE, WT_PHASE,
                                       caches[k], p, dl, _nuc_cfg)[1]
    return (float(W) / T if np.isfinite(W) else np.nan), T, k


from joblib import Parallel, delayed
_WT8 = np.full(_SIG8.shape, np.nan)          # W*/T
_WTT = np.full(_SIG8.shape, np.nan)          # T at the deciding shell [MeV]
_WTK = np.full(_SIG8.shape, -1, dtype=int)   # which shell decided
_ND8, _NB8 = len(_D08), len(_B48)
# Compute the union: every slice that is drawn, plus every slice that is summarized.
_wt_stats = list(range(len(_al8))) if WT_STATS is None else list(WT_STATS)
for _ia in sorted(set(WT_SHOW) | set(_wt_stats)):
    _res = Parallel(n_jobs=WT_NJOBS)(
        delayed(_wt_cell)(_al8[_ia], _B48[j], _D08[i], _SIG8[_ia, i, j])
        for i in range(_ND8) for j in range(_NB8))
    _WT8[_ia], _WTT[_ia], _WTK[_ia] = (np.array([r[c] for r in _res]).reshape(_ND8, _NB8)
                                       for c in (0, 1, 2))
    print(f"  α_s slice {_ia}: {np.isfinite(_WT8[_ia]).sum()}/"
          f"{np.isfinite(_SIG8[_ia]).sum()} accessible cells returned a W*/T")

# ---- map --------------------------------------------------------------------
_wtv = _WT8[np.isfinite(_WT8)]
_wlo, _whi = np.percentile(_wtv, WT_CLIP) if _wtv.size else (0.0, 1.0)
_wtcm = plt.get_cmap(WT_CMAP).copy()
_wtcm.set_bad('0.85')                # not accessible = the repo's grey, never a colour
# Grid shape follows len(WT_SHOW), panels via axes.flat — same rule as Fig 5, so
# widening WT_SHOW to more alpha_s slices needs no edit here.
fig, axes = paper_grid('1x2' if len(WT_SHOW) <= 2 else '2x2', mode='double',
                       placeholder=False,
                       fontsize=11, labelsize=11, legendsize=9, aspect=1.0)
for _ax in axes.flat[len(WT_SHOW):]:
    _ax.set_visible(False)
_wpcm = None
for _c, _ia in enumerate(WT_SHOW):
    ax = axes.flat[_c]
    _wpcm = ax.pcolormesh(_B48, _D08, np.ma.masked_invalid(_WT8[_ia]), cmap=_wtcm,
                          vmin=_wlo, vmax=_whi, shading='nearest', zorder=2)
    # iso-W*/T lines: with the colour range this tight the eye cannot read gradients
    # off the fill alone, so put numbers on it. Levels at INTERIOR quartiles — a level
    # sitting on _wlo/_whi would just trace the clip boundary, not a real iso-line.
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
fig.savefig(f'../output/figures/WoverT_map_{xsd_tag}_MT0{float(_F8["MT0"]):.2f}.pdf',
            bbox_inches='tight')
plt.show()

# ---- how constant is it? the printed numbers are the paper claim -------------
# WT_WIN is the window the paper quotes a containment fraction for ("X% between 166
# and 170"), so it is printed as a column rather than recomputed by hand off the
# percentiles — a 16-84 band is 68% containment by construction and is NOT that number.
WT_WIN = (166.0, 170.0)
_WHDR = (f"{'slice':<26}{'n':>6}{'median':>9}{'p16':>8}{'p84':>8}{'±band':>8}"
         f"{'in %g-%g' % WT_WIN:>12}{'min':>8}{'max':>8}")
print('\n' + _WHDR); print('-' * len(_WHDR))
def _wt_row(lab, v):
    v = v[np.isfinite(v)]
    if v.size < 5:
        return f'{lab:<26}{v.size:>6d}   too few'
    p16, p50, p84 = np.percentile(v, [16, 50, 84])
    win = 100 * np.mean((v >= WT_WIN[0]) & (v <= WT_WIN[1]))
    # ±band = half the 16-84 width as a % of the median: "how constant is it"
    return (f'{lab:<26}{v.size:>6d}{p50:>9.1f}{p16:>8.1f}{p84:>8.1f}'
            f'{100 * 0.5 * (p84 - p16) / p50:>7.1f}%{win:>11.1f}%'
            f'{v.min():>8.1f}{v.max():>8.1f}')
print(_wt_row(f'ALL {len(_wt_stats)} slices (island)', _WT8[_wt_stats]))
for _ia in _wt_stats:
    print(_wt_row(f'  α_s = π/2 × {_al8[_ia]/(np.pi/2):.1f}'
                  + ('' if _ia in WT_SHOW else '  [not drawn]'), _WT8[_ia]))
if sorted(set(WT_SHOW)) != sorted(set(_wt_stats)):
    print(_wt_row('drawn slices only', _WT8[sorted(set(WT_SHOW))]))
if WT_SIGMA is None:
    # The analytic expectation, evaluated with the SAME V and tau_target the grid used.
    # Any residual spread above this is the prefactor kappa*Omega_0 moving, nothing else.
    print(f"\nCNT pins W*/T = ln(V κ Ω₀ τ/2π) with V={_nuc_cfg.V:.3g} fm³, "
          f"τ={_nuc_cfg.tau_target:g} s -> the map is flat to the extent that κΩ₀ is.")
_wk = _WTK[(_WTK >= 0)]
print(f"deciding shell: " + ', '.join(
    f"n_B={_wt_shells[k][0].n_B:.2f} → {100 * (_wk == k).mean():.0f}%"
    for k in range(len(_wt_shells)) if (_wk == k).any()))
if WT_SIGMA is None:
    # Cells >1% off the median are NOT physics: there tau(sigma_crit) != tau_target
    # because sigma_target_pt refined the crossing on a sub-bracket where a different
    # shell controlled tau_star (its scan is over min-over-shells, which is only
    # piecewise smooth). They show up as speckle; this is how many there are.
    _woff = np.abs(_WT8 - np.nanmedian(_WT8)) > 0.01 * np.nanmedian(_WT8)
    print(f"off-invariant cells (|ΔW*/T| > 1%): {_woff.sum()}/"
          f"{np.isfinite(_WT8).sum()} = {100 * _woff.sum() / max(np.isfinite(_WT8).sum(), 1):.1f}%"
          f" — σ_crit's crossing refinement, not a different barrier")

# ---- flat CSV, so the quoted number never has to be re-derived from a print ----
_wt_ia, _wt_i, _wt_j = np.nonzero(np.isfinite(_WT8))
pd.DataFrame({
    'alpha_s':     _al8[_wt_ia],
    'B4_MeV':      _B48[_wt_j],
    'Delta0_MeV':  _D08[_wt_i],
    'sigma_crit':  _SIG8[_wt_ia, _wt_i, _wt_j],
    'W_over_T':    _WT8[_wt_ia, _wt_i, _wt_j],
    'T_shell_MeV': _WTT[_wt_ia, _wt_i, _wt_j],
    'shell_nB':    [_wt_shells[k][0].n_B for k in _WTK[_wt_ia, _wt_i, _wt_j]],
    'drawn':       np.isin(_wt_ia, WT_SHOW),
}).to_csv(f'../output/figure_data/WoverT_map_{xsd_tag}_MT0{float(_F8["MT0"]):.2f}.csv',
          index=False)
print(f'wrote WoverT_map_{xsd_tag}_MT0{float(_F8["MT0"]):.2f}.csv '
      f'({len(_wt_ia)} cells) to ../output/figure_data/')


# %% [markdown]
# ### $\sigma_{\rm crit}(\alpha_s, B^{1/4})$ — unpaired matter, fixed $m_s$
#
# Companion to IV.2/IV.3 for the **unpaired** nucleation channel. An unpaired
# droplet has no CFL gap, so **everything here is $\Delta_0$-independent** — both
# $\sigma_{\rm crit}$ and the viability filter. Fixing $m_s$, this maps
# $\sigma_{\rm crit}$ over the $(B^{1/4}, \alpha_s)$ plane (the $\Delta_0$ axis of
# IV.3 is replaced by $\alpha_s$). The viability filter is built on the
# **unpaired** quark EoS, not the CFL one:
#
# 1. **No re-hadronization** — $P_{\rm unp}(\mu_B) > P_H(\mu_B)$ on the overlap.
# 2. **$M_{\max}$** — TOV integrated with the **unpaired** quark EoS, in
#    `M_max_window`.
#
# (No Witten / 2-flavour / $\Delta_0$: those belong to the CFL analysis.)

# %%
# =============================================================================
#  UNPAIRED sigma_crit heatmap over (B^1/4, alpha_s) at fixed m_s.
#  Reuses the engine: scan_unpaired_filters (unpaired-EoS viability) +
#  compute_sigma_crit. No Delta_0 anywhere.
# =============================================================================
import dataclasses
from matplotlib.colors import ListedColormap

# ---- user-configurable inputs ----------------------------------------------
U_MS           = m_s_fixed            # strange-quark mass [MeV] (the fixed one)
U_MT0          = 1.4                  # nucleation-threshold grav. mass [M_sun]
U_CHARGE       = 'coulomb_minimize'   # 'lcn' | 'gcn' | 'coulomb_minimize'
U_ALPHA_GRID   = np.linspace(np.pi/2*0.0, np.pi/2*0.3, 31)       # alpha_s   (y-axis)
U_B4_GRID      = np.linspace(130.0, 160.0, 51)   # B^1/4 [MeV] (x-axis)
U_APPLY_FILTER = True                 # grey out non-viable cells (unpaired filter)
U_NJOBS        = -1
# -----------------------------------------------------------------------------
# Unpaired-EoS viability (no-rehadronization + unpaired-TOV M_max) at this m_s.
_cfgU = dataclasses.replace(_filt_cfg, m_s=U_MS)
if U_APPLY_FILTER:
    _cflU, _mmU, _rsU = nuc_an.scan_unpaired_filters(
        U_ALPHA_GRID, U_B4_GRID, _cfgU, n_jobs=U_NJOBS)
else:
    _cflU = np.ones((len(U_ALPHA_GRID), 1, len(U_B4_GRID)), dtype=bool)

# unpaired sigma_crit at the PNS centre, on the viable cells. Delta_0 is a dummy
# here (the unpaired droplet ignores it); pass a harmless non-zero value.
_sigU = nuc_an.compute_sigma_crit(
    _cflU, U_MT0, 'saddlepoint', U_CHARGE, 'unpaired',
    U_ALPHA_GRID, U_B4_GRID, [100.0], _star, _nuc_cfg,
    m_s=U_MS, n_jobs=U_NJOBS)

sigU = _sigU[:, 0, :]; cflU = _cflU[:, 0, :]     # drop the singleton Delta_0 axis

# ---- plot: single heatmap, x = B^1/4, y = alpha_s ---------------------------
_fin = sigU[np.isfinite(sigU)]
_vmin, _vmax = (float(_fin.min()), float(_fin.max())) if _fin.size else (0.0, 1.0)
fig, ax = plt.subplots(figsize=(6.6, 5.2))
# categorical background for non-finite sigma_crit: grey=not viable,
# orange=viable but never nucleates, teal=nucleates for all tested sigma.
_cat = np.where(np.isfinite(sigU), np.nan,
                np.where(~cflU, 0.0, np.where(np.isposinf(sigU), 2.0, 1.0)))
ax.pcolormesh(U_B4_GRID, U_ALPHA_GRID, np.ma.masked_invalid(_cat),
              cmap=ListedColormap(['0.85', '#f4a261', '#2a9d8f']),
              vmin=0, vmax=2, shading='nearest')
pcm = ax.pcolormesh(U_B4_GRID, U_ALPHA_GRID, np.ma.masked_invalid(sigU),
                    cmap=plt.cm.viridis, vmin=_vmin, vmax=_vmax, shading='nearest')
if U_APPLY_FILTER and cflU.any() and (~cflU).any():       # viability boundary
    ax.contour(U_B4_GRID, U_ALPHA_GRID, cflU.astype(float), levels=[0.5],
               colors='k', linewidths=1.1, linestyles='--')
for p in quark_param_sets:                                # mark sets at this m_s
    if abs(p['m_s'] - U_MS) < 1.0:
        ax.scatter([p['B4']], [p['alpha']], s=120, marker='*',
                   c='yellow', edgecolors='k', zorder=5)
ax.set_xlabel(r'$B^{1/4}$ [MeV]'); ax.set_ylabel(r'$\alpha_s$')
ax.set_title(rf'unpaired $\sigma_{{\rm crit}}$ — $m_s={U_MS:.0f}$ MeV, '
             rf'$M_{{T0}}={U_MT0:g}\,M_\odot$, $Y_L={YLH}$, $S={S}$, '
             rf'$\tau={tau_target*1e3:g}$ ms — {xsd_tag}')
fig.colorbar(pcm, ax=ax, label=r'$\sigma_{\rm crit}$ [MeV/fm$^2$]')
fig.tight_layout(); plt.show()

# %% [markdown]
# ### UNPAIRED $\sigma_{\rm crit}(B^{1/4},\alpha_s)$ — three $m_s$ panels (full Fig. 5 layers)
#
# Same unpaired heatmap as above, one panel per strange-quark mass, carrying the
# same toggleable layers as Paper Fig. 5 — but every quantity is the **unpaired
# $\beta$-eq quark EoS**, never CFL (no $\Delta_0$, no pairing):
# - **heatmap + iso-$\sigma_{\rm crit}$** — $\sigma_{\rm crit}$ fill with black
#   iso-lines (50–250 MeV/fm²);
# - **iso-$M_{\max}$** — white dashed lines (every 0.2 $M_\odot$ above 2), from the
#   **unpaired**-TOV $M_{\max}$ on the mass-relevant cells;
# - **reject outlines** — coloured boundaries of the non-viable zones the unpaired
#   filter emits: vermillion $M_{\max}<2M_\odot$, blue re-hadr., purple quasi-re-hadr.,
#   grey no-solve (the unpaired gate has NO Witten/2-flavour column, by design);
# - **stability lines** — the two window bounds drawn as curves: green = **unpaired
#   3-flavour SQM absolute-stability** ($e/n_B|_{P=0}=930$, Witten), grey = **2-flavour
#   stability** ($e/n_B|_{P=0}=930$ of $ud$ matter). Both from the bulk EoS at $P=0$
#   (root-found per $\alpha_s$ on a coarse grid), NOT gates — they only annotate.
# - yellow stars mark the tabulated sets at that $m_s$.

# %%
# ============================================================================
#  UNPAIRED sigma_crit(B^1/4, alpha_s), one panel per m_s, full Fig-8 layer set.
#  scan_unpaired_filters gives (viable, M_max, reason) FREE -> iso-M_max + reject
#  outlines. The Witten (3-flavour unpaired SQM) and 2-flavour stability lines are
#  the only extra physics: e/n_B at P=0 vs 930 MeV, root-found in B^1/4 per alpha.
# ============================================================================
import dataclasses
from scipy.optimize import brentq
from matplotlib.ticker import MultipleLocator, FuncFormatter
# ---- user-configurable inputs ----------------------------------------------
U3_MS_LIST     = [80.0, 100.0, 150.0]            # three strange-quark masses [MeV]
U3_MT0         = U_MT0                            # nucleation-threshold grav. mass [M_sun]
U3_CHARGE      = U_CHARGE                         # charge prescription (as single-m_s cell)
U3_ALPHA_GRID  = U_ALPHA_GRID                     # alpha_s   (y-axis) — reuse cell above
U3_B4_GRID     = U_B4_GRID                        # B^1/4 [MeV] (x-axis)
U3_NJOBS       = -1
# ---- layer toggles: set any False to drop that layer -----------------------
U3_ISO_SIGMA   = True     # black iso-sigma_crit contour lines
U3_ISO_MMAX    = True     # white dashed iso-M_max lines (unpaired TOV)
U3_REJECT      = True      # coloured reject-zone boundary outlines + labels
U3_WITTEN      = True      # green line: unpaired 3-flavour SQM abs.-stability (e/n_B=930)
U3_TWOFLAV     = True      # grey line: 2-flavour stability (ud e/n_B=930)
U3_WINDOW_ONLY = True      # colour sigma_crit ONLY inside the 2-flav..Witten window
U3_FILL_QUASI  = True      # solve sigma_crit in quasi-re-hadr. cells too, so they fill
                           # when they meet the window (quasi status never gates the mask)
U3_STAB_NALPHA = 13        # alpha_s samples for the two stability curves (coarse=cheap)
# -----------------------------------------------------------------------------
# excluded-region reason -> (colour, label). Only the reasons the UNPAIRED filter
# can emit (no witten/twoflavor gate here); colours echo the Fig-8 Okabe-Ito set.
_RC3 = nuc_an.REASON_CODE
_C3  = dict(mmax='#D55E00', rehadr='#0072B2', rehad_quasi='#CC79A7', solve='#9a9a9a',
            witten='#009E73', twoflav='#8a8a8a')
_RSPEC3 = [(_RC3['mmax'],        _C3['mmax'],        r'$M_{\rm QS}^{\rm max}<2M_\odot$'),
           (_RC3['rehadr'],      _C3['rehadr'],      're-hadr.'),
           (_RC3['rehad_quasi'], _C3['rehad_quasi'], 'quasi-\nre-hadr.'),
           (_RC3['solve'],       _C3['solve'],       'no solve')]

def _ea_unp_P0(alpha, B4, ms):
    """e/n_B [MeV] at P=0 of UNPAIRED 3-flavour beta-eq SQM (= Witten energy per
    baryon). nan if the EoS has no P=0 crossing on the grid."""
    cfg = dataclasses.replace(_filt_cfg, m_s=ms)
    P, e, _mu, ok = nuc_an.unpaired_eos_at_params(alpha, B4, cfg)
    n, P, e = cfg.n_B_grid[ok], P[ok], e[ok]
    if n.size < 5:
        return np.nan
    cr = nuc_an.zero_crossing(n, P)
    if cr is None:
        return np.nan
    j, fr = cr
    nP0 = n[j] + fr * (n[j + 1] - n[j]); eP0 = e[j] + fr * (e[j + 1] - e[j])
    return float(eP0 / nP0) if nP0 > 0 else np.nan

def _B_at_930(scalar, alpha, lo, hi):
    """B^1/4 where scalar(alpha, B)=930 MeV, or nan if no single crossing brackets
    in [lo,hi] (e/n_B rises with B, so one clean root)."""
    f = lambda b: scalar(alpha, b) - 930.0
    flo, fhi = f(lo), f(hi)
    if not (np.isfinite(flo) and np.isfinite(fhi)) or flo * fhi > 0:
        return np.nan
    return brentq(f, lo, hi, xtol=0.05)

def _interp_curve(curve, src_al, target_al, fallback):
    """A stability curve B(alpha) sampled on the coarse src_al, interpolated onto
    the full target_al grid. All-NaN (no crossing in range) -> constant fallback."""
    if curve is None:
        return np.full_like(target_al, fallback, dtype=float)
    m = np.isfinite(curve)
    if m.sum() < 2:
        return np.full_like(target_al, fallback, dtype=float)
    return np.interp(target_al, src_al[m], curve[m])

# per-m_s: unpaired viability + M_max + reason (FREE) and sigma_crit at the PNS
# centre. Same two engine calls as the single-panel cell; drop the singleton D0 axis.
_u3 = []
for _ms in U3_MS_LIST:
    _cfg = dataclasses.replace(_filt_cfg, m_s=_ms)
    _ok, _mm, _rs = nuc_an.scan_unpaired_filters(U3_ALPHA_GRID, U3_B4_GRID, _cfg, n_jobs=U3_NJOBS)
    # quasi-re-hadr. is a soft rejection -> optionally solve sigma_crit there too so
    # the fill can extend into it (the zone stays outlined purple).
    _ok_sig = (_ok | (_rs == _RC3['rehad_quasi'])) if U3_FILL_QUASI else _ok
    _sig = nuc_an.compute_sigma_crit(
        _ok_sig, U3_MT0, 'saddlepoint', U3_CHARGE, 'unpaired',
        U3_ALPHA_GRID, U3_B4_GRID, [100.0], _star, _nuc_cfg, m_s=_ms, n_jobs=U3_NJOBS)
    _u3.append((_sig[:, 0, :], _ok[:, 0, :], _mm[:, 0, :], _rs[:, 0, :]))

# shared colour scale across the three panels (finite sigma_crit only)
_allfin = np.concatenate([s[np.isfinite(s)] for s, _, _, _ in _u3])
_v3min, _v3max = (float(_allfin.min()), float(_allfin.max())) if _allfin.size else (0.0, 1.0)

# stability curves: coarse alpha grid, root-found in B^1/4. 2-flavour is m_s-
# INDEPENDENT (no strange quarks) -> compute once; Witten is per-m_s.
# ponytail: serial root-find over U3_STAB_NALPHA alphas; bump it if a line looks jagged.
_stab_al = np.linspace(U3_ALPHA_GRID.min(), U3_ALPHA_GRID.max(), U3_STAB_NALPHA)
_lo3, _hi3 = float(U3_B4_GRID.min()), float(U3_B4_GRID.max())
_B2flav = (np.array([_B_at_930(lambda a, b: nuc_an.ud_eps_per_nB(a, b, _filt_cfg), al,
                               _lo3, _hi3) for al in _stab_al])
           if (U3_TWOFLAV or U3_WINDOW_ONLY) else None)

set_paper_style()
fig, axes = plt.subplots(1, len(U3_MS_LIST), figsize=(5.25 * len(U3_MS_LIST), 4.8),
                         squeeze=False, constrained_layout=True)
pcm = None
for _c, (_ms, (_sig, _ok, _mm, _rs)) in enumerate(zip(U3_MS_LIST, _u3)):
    ax = axes[0, _c]; ax.set_box_aspect(1)               # square panels, like Fig 5
    # Witten line B(alpha), per m_s (needed for both the window mask and its drawing).
    _Bw = (np.array([_B_at_930(lambda a, b: _ea_unp_P0(a, b, _ms), al, _lo3, _hi3)
                     for al in _stab_al]) if (U3_WITTEN or U3_WINDOW_ONLY) else None)
    # stability-window mask: keep sigma_crit only where 2-flav is unbound (right of
    # the grey line) AND 3-flav SQM is absolutely stable (left of the green line).
    if U3_WINDOW_ONLY:
        _b2f = _interp_curve(_B2flav, _stab_al, U3_ALPHA_GRID, U3_B4_GRID.min())
        _bwf = _interp_curve(_Bw, _stab_al, U3_ALPHA_GRID, U3_B4_GRID.max())
        # fill exactly where all four criteria hold: 2-flav unbound (right of grey)
        # AND 3-flav bound (left of green) = the window; Mmax>2 and no-rehadr are
        # automatic -- sigma_crit is NaN on the mmax/rehadr-rejected cells so they
        # never fill. quasi-rehad does NOT enter the mask (only outlined below).
        _win = ((U3_B4_GRID[None, :] >= _b2f[:, None]) &
                (U3_B4_GRID[None, :] <= _bwf[:, None]))
        _sigp = np.where(_win, _sig, np.nan)
    else:
        _sigp = _sig
    pcm = ax.pcolormesh(U3_B4_GRID, U3_ALPHA_GRID, np.ma.masked_invalid(_sigp),
                        cmap='viridis', vmin=_v3min, vmax=_v3max, shading='nearest', zorder=2)
    # (1) iso-sigma_crit contour lines (black), only levels inside this panel's range
    if U3_ISO_SIGMA:
        _lv = [l for l in (50, 100, 150, 200, 250)
               if np.isfinite(_sigp).any() and np.nanmin(_sigp) <= l <= np.nanmax(_sigp)]
        if _lv:
            _cs = ax.contour(U3_B4_GRID, U3_ALPHA_GRID, _sigp, levels=_lv,
                             colors='k', linewidths=0.7, alpha=0.6, zorder=3)
            ax.clabel(_cs, fmt='%.0f', fontsize=11, inline=True)
    # (2) iso-M_max (white dashed) on the mass-relevant cells (viable + the M<2 band)
    if U3_ISO_MMAX:
        _mmv = np.where(_ok | (_rs == _RC3['mmax']), _mm, np.nan)
        _lvm = [round(x, 1) for x in np.arange(2.2, 4.001, 0.2)
                if np.isfinite(_mmv).any() and np.nanmin(_mmv) <= round(x, 1) <= np.nanmax(_mmv)]
        if _lvm:
            _cm = ax.contour(U3_B4_GRID, U3_ALPHA_GRID, _mmv, levels=_lvm, colors='white',
                             linewidths=0.9, linestyles='--', alpha=0.9, zorder=3.5)
            ax.clabel(_cm, fmt='%.1f', fontsize=11, inline=True)
    # (3) reject-zone boundary outlines + labels (only reasons the unpaired gate emits)
    if U3_REJECT:
        for _code, _col, _lab in _RSPEC3:
            _m = _rs == _code
            if not _m.any():
                continue
            ax.contour(U3_B4_GRID, U3_ALPHA_GRID, _m.astype(float), levels=[0.5],
                       colors=[_col], linewidths=2.2, zorder=4)
            _r, _cc = np.where(_m)
            ax.text(np.median(U3_B4_GRID[_cc]), np.median(U3_ALPHA_GRID[_r]), _lab,
                    color=_col, fontsize=8.5, fontweight='bold', ha='center',
                    va='center', zorder=7)
    # (4) 2-flavour stability line (grey) — ud e/n_B|_{P=0}=930, m_s-independent
    if U3_TWOFLAV and _B2flav is not None and np.isfinite(_B2flav).any():
        ax.plot(_B2flav, _stab_al, color=_C3['twoflav'], lw=2.4, zorder=4.5)
        _k = np.isfinite(_B2flav)
        ax.text(_B2flav[_k][-1], _stab_al[_k][-1], '2 flav.\nstable ', color=_C3['twoflav'],
                fontsize=8.5, fontweight='bold', ha='right', va='center', zorder=7)
    # (5) Witten line (green) — unpaired 3-flavour SQM e/n_B|_{P=0}=930 (abs. stable left of it)
    if U3_WITTEN and _Bw is not None:
        if np.isfinite(_Bw).any():
            ax.plot(_Bw, _stab_al, color=_C3['witten'], lw=2.4, zorder=4.5)
            _k = np.isfinite(_Bw)
            ax.text(_Bw[_k][0], _stab_al[_k][0], ' SQM not\n abs. stable', color=_C3['witten'],
                    fontsize=8.5, fontweight='bold', ha='left', va='center', zorder=7)
    for p in quark_param_sets:                            # mark tabulated sets at this m_s
        if abs(p['m_s'] - _ms) < 1.0:
            ax.scatter([p['B4']], [p['alpha']], s=120, marker='*',
                       c='yellow', edgecolors='k', zorder=6)
    ax.set_title(rf"$m_s={_ms:.0f}$ MeV")
    ax.set_xlabel(r'$B^{1/4}$ [MeV]'); ax.set_ylabel(r'$\alpha_s$')
    # y-axis in units of pi/2: alpha_s = pi/2 * f (the param-set / Fig-8 convention).
    # ticks land on nice f=0.1 multiples; labels read "pi/2 . f".
    ax.yaxis.set_major_locator(MultipleLocator(np.pi / 2 * 0.1))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: r'$0$' if abs(v) < 1e-9 else rf'$\frac{{\pi}}{{2}}\,{v/(np.pi/2):.1f}$'))
    ax.set_xlim(U3_B4_GRID.min(), U3_B4_GRID.max())
    ax.set_ylim(U3_ALPHA_GRID.min(), U3_ALPHA_GRID.max())
    panel_label(ax, f"({chr(97 + _c)})")
fig.colorbar(pcm, ax=axes.ravel().tolist(), label=r'$\sigma_{\rm crit}$ [MeV/fm$^2$]',
             shrink=0.9)
fig.suptitle(rf"unpaired $\sigma_{{\rm crit}}$ — $M_{{T0}}={U3_MT0:g}\,M_\odot$, "
             rf"$Y_L={YLH}$, $S={S}$, $\tau={tau_target*1e3:g}$ ms — {xsd_tag}", y=1.02)
fig.savefig(f'../output/figures/sigcrit_unpaired_ms_panels_{xsd_tag}.pdf', bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Electromagnetic screening — reproducing the thesis "screen" figure
#
# $W(R)$ at one hadronic point across the electric-charge prescriptions of
# Sec. *Electric charge neutrality*: **LCN** (local neutrality, no Coulomb),
# **GCN** (global neutrality, no Coulomb), **GCN+Coulomb** (unscreened $\propto
# R^5$), **minimization** (`coulomb_minimize`, Coulomb inside the saddle), and
# **screening** (Debye-screened Coulomb, $\lambda_D$ from $\chi_e=\partial
# n_e/\partial\mu_e$). The dashed grey curve is `screening` with
# $\lambda_D=10^6$ fm — it must fall on top of **GCN+Coulomb** (unscreened
# limit), the end-to-end check that the new screened Coulomb term is wired
# correctly. Dots mark each critical point $R_*$.

# %% [markdown]
# ## Study — does $W(R)$ develop a minimum at large $R$?
#
# The thin-wall barrier $W(R)$ rises to the critical radius $R_*$ then, for a
# genuine first-order transition, falls as the bulk term ($\propto -R^3\Delta f$)
# takes over. With the Coulomb / Debye-screening charge terms a SECOND rise at
# large $R$ could in principle create a local **minimum** (a metastable finite
# droplet / frustrated phase). Here we push $R$ out to `WR_RMAX` fm and check: is
# there a turnover, or is $W$ monotone past $R_*$? `gcn_coulomb` is excluded (its
# unscreened $\propto R^5$ term blows up and is unphysical at large $R$).

# %%
# ============================================================================
#  W(R) to very large R for coulomb_minimize / screening; flag any local minimum
#  beyond the barrier peak. Reuses compute_energy_barrier (nucleation.critical).
# ============================================================================
set_paper_style()
WR_T      = 20.0                 # temperature [MeV]
WR_NBH    = 5.0                  # n_B^H / n_sat
WR_SIG    = 30.0                 # surface tension [MeV/fm^2]
WR_YL     = PNS_TMAX['YLH']      # lepton fraction (Y_L^H at t_Tmax = 0.25)
WR_SET    = quark_param_sets[0]  # quark parametrization
WR_PHASE  = 'unpaired'           # 'unpaired' | 'cfl' | 'unpCFL'
WR_RMAX   = 100.0                # fm — push R far out to expose any turnover
WR_N      = 1200
WR_MODES  = [('coulomb_minimize', STANDARD_COLORS['Blue']),
             ('screening',        STANDARD_COLORS['Orange'])]
_wr_par = get_alphabag_custom(alpha=WR_SET['alpha'], B4=WR_SET['B4'], m_s=WR_SET['m_s'])
_wr_Rg  = np.linspace(0.02, WR_RMAX, WR_N)

fig, (axF, axZ) = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
for _ch, _col in WR_MODES:
    _eb = compute_energy_barrier(
        H['trapped'], WR_NBH * n_sat, WR_T, WR_SIG, electric_charge_mode=_ch,
        params=_wr_par, flavor_mode='saddlepoint', quark_phase=WR_PHASE,
        Delta0=WR_SET['Delta0'], Y_L_H=WR_YL, R_values=_wr_Rg)
    _W = np.asarray(_eb.W, float)
    axF.plot(_wr_Rg, _W, color=_col, lw=1.6, label=_ch)
    axZ.plot(_wr_Rg, _W, color=_col, lw=1.6, label=_ch)
    if np.isfinite(_W).any():                       # barrier peak + any later local min
        _pk = int(np.nanargmax(_W))
        _mn = _pk + int(np.nanargmin(_W[_pk:]))
        _has_min = (_pk < _mn < len(_W) - 1 and _W[_mn] < _W[_pk]
                    and _W[-1] > _W[_mn] + 1e-6)     # dips then rises again => true minimum
        axF.plot(_wr_Rg[_pk], _W[_pk], 'o', ms=5, color=_col, mfc='white', zorder=5)
        if _has_min:
            axF.plot(_wr_Rg[_mn], _W[_mn], 's', ms=6, color=_col, zorder=6)
        print(f"{_ch:>18}: R_*={_wr_Rg[_pk]:.2f} fm, W_*={_W[_pk]:.1f} MeV;  "
              f"large-R minimum: {('YES @ R=%.1f fm' % _wr_Rg[_mn]) if _has_min else 'no (monotone past R_*)'}")
axF.axhline(0, color='0.6', lw=0.7, zorder=0)
axF.set_xlabel(r'$R$ [fm]'); axF.set_ylabel(r'$W$ [MeV]')
axF.set_title(rf'full range (to {WR_RMAX:g} fm)'); axF.legend(fontsize=8)
axZ.axhline(0, color='0.6', lw=0.7, zorder=0)
axZ.set_xlim(0, 12); axZ.set_xlabel(r'$R$ [fm]'); axZ.set_ylabel(r'$W$ [MeV]')
axZ.set_title('zoom near the barrier')
fig.suptitle(rf"$W(R)$ large-$R$ check — {WR_PHASE}, $T={WR_T:.0f}$ MeV, "
             rf"$n_B^H/n_0={WR_NBH:g}$, $\sigma={WR_SIG:.0f}$ — {q_tag_of(WR_SET)}", y=1.04)
plt.show()


# %% [markdown]
# ## Study — critical-droplet observables at nucleation (τ = 1 s)
#
# Typical values of $R_*$, $W_*/T$, $N_B^{Q*}=\tfrac43\pi R_*^3\,n_B^{Q*}$ (quark
# interior baryon number of the critical droplet) and $Y_S^H$, evaluated where the
# central nucleation time is $\tau=1$ s. **Plot 1** — along the $\tau=1$ s curve
# $T_{\rm nuc}(n_B^H)$ at fixed $(Y_L^H,\sigma)$ for one quark set: observables vs
# $n_B^H$, colour = $\sigma$. Goal: read off the generic magnitude/scaling of the
# droplet at the moment it nucleates.

# %%
# ============================================================================
#  Study helper + Plot 1. _droplet_obs: critical-droplet observables at one
#  hadronic point (reused by Plots 2a/2b). Plot 1: vs n_B^H along the tau=1 s
#  curve T_nuc(n_B^H) (compute_nucleation_density) at fixed (Y_L^H, sigma).
# ============================================================================
from dataclasses import replace as _dc_replace
set_paper_style()


def _droplet_obs(sigma, H_pt, T, flavor, charge, phase, params, Delta0, nuc, cache=None):
    """Critical-droplet observables at one hadronic point:
      R_* [fm], W_*/T, N_B^{Q*}=(4/3)π R_*³·n_B^{Q*} (QUARK interior density), Y_S^H.
    NaN where the droplet solve fails or the hadronic phase is stable (W_*=inf)."""
    R_c, W_c, Qs = nuc_an.critical_droplet_pt(
        sigma, H_pt, T, flavor, charge, phase,
        {} if cache is None else cache, params, Delta0, nuc)
    _YS = float(getattr(H_pt, 'Y_S', np.nan))
    if Qs is None or not (np.isfinite(R_c) and R_c > 0 and np.isfinite(W_c) and W_c > 0):
        return dict(R=np.nan, WoverT=np.nan, NB=np.nan, YS=_YS)
    return dict(R=float(R_c), WoverT=float(W_c) / float(T),
                NB=4.0 / 3.0 * np.pi * R_c**3 * float(Qs.n_B), YS=_YS)


SP1_SET     = quark_param_sets[1]                    # one quark parametrization
SP1_YLH     = PNS_TMAX['YLH']                        # Y_L^H (0.25 = t_Tmax)
SP1_PHASE   = 'cfl'                               # 'unpaired' | 'cfl' | 'unpCFL'
SP1_FLAVOR, SP1_CHARGE = 'saddlepoint', 'coulomb_minimize'
SP1_SIGMAS  = [80.0, 100.0, 150, 200.0]                   # colour = sigma
_nuc1s = _dc_replace(_nuc_cfg, tau_target=0.001)       # tau = 1 s
_tag1  = q_tag_of(SP1_SET)
_par1  = get_alphabag_custom(alpha=SP1_SET['alpha'], B4=SP1_SET['B4'], m_s=SP1_SET['m_s'])
_D01   = SP1_SET['Delta0']
_col1  = {sg: mpl.cm.viridis(t) for sg, t in
          zip(SP1_SIGMAS, np.linspace(0.15, 0.85, len(SP1_SIGMAS)))}

fig, ((axR, axW), (axN, axY)) = paper_grid('2x2', mode='double', placeholder=False, **PAPER_STYLE)
_P1 = [(axR, 'R', r'$R_*$ [fm]', 'linear'), (axW, 'WoverT', r'$W_*/T$', 'linear'),
       (axN, 'NB', r'$N_B^{Q*}$', 'linear'), (axY, 'YS', r'$Y_S^H$', 'linear')]
for sg in SP1_SIGMAS:
    stem = f"Htrapped_{SP1_FLAVOR}_{SP1_CHARGE}_{SP1_PHASE}_{_tag1}_s{int(sg)}"
    if stem not in nuc_sets:
        print(f"Plot 1 skip σ={sg:g}: {stem} not in nuc_sets"); continue
    _obs = nuc_sets[stem]
    _iYL = int(np.argmin(np.abs(_obs.hadronic_grids['Y_L_H'] - SP1_YLH)))
    _YLg = float(_obs.hadronic_grids['Y_L_H'][_iYL])
    res  = compute_nucleation_density(_obs, tau_target=0.001, scan='n_B')   # tau=1 s locus
    nBc_, Tc_ = nucleation_curve(res, _iYL)
    _cache, _rows = {}, []
    for nb, T in zip(nBc_, Tc_):
        if not (np.isfinite(nb) and np.isfinite(T)):
            _rows.append(dict(R=np.nan, WoverT=np.nan, NB=np.nan, YS=np.nan)); continue
        H_pt = nuc_an.hadronic_point(H['trapped'], nb, _YLg, T)
        _rows.append(_droplet_obs(sg, H_pt, T, SP1_FLAVOR, SP1_CHARGE, SP1_PHASE,
                                  _par1, _D01, _nuc1s, _cache))
    _x = np.asarray(nBc_, float) / n_sat
    for ax, key, _, _sc in _P1:
        ax.plot(_x, [r[key] for r in _rows], color=_col1[sg], lw=1.6,
                label=rf'$\sigma={int(sg)}$')
for ax, key, ylab, sc in _P1:
    ax.set_ylabel(ylab); ax.set_yscale(sc); ax.set_xlabel(r'$n_B^H/n_0$')
axR.legend(fontsize=8, title=r'$\sigma$ [MeV/fm$^2$]')
for _lab, ax in zip('abcd', (axR, axW, axN, axY)):
    panel_label(ax, f'({_lab})')
fig.suptitle(rf"Droplet observables along $\tau=1$ s — {SP1_PHASE}, "
             rf"$Y_L^H={SP1_YLH:g}$ — {_tag1}", y=1.02)
fig.savefig(f'../output/figures/study_droplet_vs_nBH_tau1s_{xsd_tag}_{_tag1}.pdf',
            bbox_inches='tight')
plt.show()


# %% [markdown]
# **Plot 2a** — scatter over ALL $\sigma_{\rm crit}$-viable Fig-5 cells
# $(B^{1/4},\Delta_0,\alpha_s)$ at $M_{T0}=1.4$: each observable vs the cell's
# $\sigma_{\rm crit}$, colour = $\alpha_s$. The central state is quark-independent
# (same $H_{\rm pt}$ for every cell); only the droplet solve uses the cell's
# $(\alpha_s,B^{1/4},\Delta_0)$. Reveals global trends across the accepted space.

# %%
# ============================================================================
#  Study Plot 2a: droplet observables vs sigma_crit, one point per viable Fig-5
#  cell. Needs the Fig 5 cell to have run (_SIG8, _al8, _B48, _D08).
# ============================================================================
set_paper_style()
SP2_MT0    = 1.4
SP2_PHASE  = 'unpCFL'
SP2_FLAVOR, SP2_CHARGE = 'saddlepoint', 'coulomb_minimize'
SP2_MS     = m_s_fixed
SP2_MAX    = 400                        # cap #cells (even subsample); None = all
_nBHc2, _Tc2, _Hpt2 = nuc_an.central_state(SP2_MT0, _star)   # quark-independent centre
_cells = [(_al8[ia], _B48[j], _D08[i], float(_SIG8[ia, i, j]))
          for ia in range(len(_al8)) for i in range(len(_D08)) for j in range(len(_B48))
          if np.isfinite(_SIG8[ia, i, j])]
if SP2_MAX and len(_cells) > SP2_MAX:
    _cells = _cells[::int(np.ceil(len(_cells) / SP2_MAX))]
_al_arr = np.array([c[0] for c in _cells]) / (np.pi / 2)
_sc_arr = np.array([c[3] for c in _cells])
_rows2 = [_droplet_obs(sc, _Hpt2, _Tc2, SP2_FLAVOR, SP2_CHARGE, SP2_PHASE,
                       get_alphabag_custom(alpha=al, B4=B4, m_s=SP2_MS), D0, _nuc_cfg)
          for al, B4, D0, sc in _cells]
print(f"Plot 2a: {len(_cells)} viable cells")

fig, ((axR, axW), (axN, axY)) = paper_grid('2x2', mode='double', placeholder=False, **PAPER_STYLE)
_P2 = [(axR, 'R', r'$R_*$ [fm]', 'linear'), (axW, 'WoverT', r'$W_*/T$', 'log'),
       (axN, 'NB', r'$N_B^{Q*}$', 'log'), (axY, 'YS', r'$Y_S^H$', 'linear')]
_sm2 = None
for ax, key, ylab, sc in _P2:
    _sm2 = ax.scatter(_sc_arr, np.array([r[key] for r in _rows2]), c=_al_arr,
                      s=14, cmap='plasma', alpha=0.8,
                      vmin=_al_arr.min(), vmax=_al_arr.max())
    ax.set_ylabel(ylab); ax.set_yscale(sc); ax.set_xlabel(r'$\sigma_{\rm crit}$ [MeV/fm$^2$]')
for _lab, ax in zip('abcd', (axR, axW, axN, axY)):
    panel_label(ax, f'({_lab})')
fig.colorbar(_sm2, ax=[axR, axW, axN, axY], label=r'$\alpha_s\,/\,(\pi/2)$', shrink=0.6)
fig.suptitle(rf"Droplet vs $\sigma_{{\rm crit}}$ over viable cells — "
             rf"$M_{{T0}}={SP2_MT0:g}\,M_\odot$, {SP2_PHASE}", y=1.02)
fig.savefig(f'../output/figures/study_droplet_vs_sigmacrit_scatter_{xsd_tag}.pdf',
            bbox_inches='tight')
plt.show()


# %% [markdown]
# ## Table — PNS evolution outcomes (NS / QS, conversion energy)
#
# For each (quark parametrization, σ) and an initial PNS gravitational mass at
# $t_0$ $(Y_L^H=0.35,\,S_H=1.5)$, follow the fixed-baryon-mass evolution:
# **t_0** columns $M^{\rm PNS}, M_B, n_{B,c}^H, Y_{S,c}^H$; **t_Tmax**
# $(Y_L^H=0.25,\,S_H=2.0)$ columns $M^{\rm PNS}, n_{B,c}^H, Y_{S,c}^H$; **remnant**
# type / mass / conversion energy. Decision: at each snapshot compute
# $\sigma_{\rm crit}$ (τ=`TAB_TAU`) at the centre; **σ<σ_crit ⇒ nucleates**. Converts
# at $t_0$ if σ<σ_crit(t_0) (then the $t_{T\max}$ columns read '--' — that snapshot
# is never reached), else at $t_Tmax$ if σ<σ_crit(t_Tmax), else stays NS. A **BH**
# forms when the fixed-$M_B$ star can't be held: promptly if $M^{t_0}>M_{\max}^{t_0}$,
# or after deleptonization when $M_B>M_{B,\max}^{t_{T\max}}$ (so the maximal PNS at
# $t_0$ collapses unless small σ converts it first).
# Conversion energy $E_{\rm conv}=(M^{\rm PNS}_{\rm convert}-M_{\rm QS})c^2$ at fixed
# $M_B$ (QS remnant from the row's own CFL EoS). Generalises the old fixed-σ table.

# %%
# ============================================================================
#  Outcomes table. New physics vs the rest of the notebook: (i) the nucleate-or-not
#  decision branch, (ii) M_QS(M_B) baryon-conserving remnant, (iii) E_conv, (iv) BH
#  collapse when M_B exceeds the supportable max (prompt at t_0 or on deleptonization).
#  All assumptions are flagged inline for validation against the old table.
# ============================================================================
M_SUN_C2_ERG = 1.7827e54                 # M_sun c^2 [erg]  (new constant)
TAB_SETS     = quark_param_sets          # rows: quark parametrizations
# σ [MeV/fm²]: span the σ_crit band at both snapshots so the remnant type varies
# down the column — low σ nucleates at t_0 (QS), mid only by t_Tmax (QS), high stays
# NS. σ_crit at the PNS centre sits ~50–200 here (cf. Fig 3d), so bracket it.
TAB_SIGMAS   = [50.0, 100.0, 150.0, 200.0, 250.0]
TAB_MPNS_T0  = [1.4, 1.6]                # initial PNS grav mass at t_0 [M_sun]
TAB_TAU      = 1e-3                       # s — nucleation-criterion tau for the decision
TAB_FLAVOR, TAB_CHARGE, TAB_PHASE = 'saddlepoint', 'coulomb_minimize', 'unpCFL'
_tab_nuc = _dc_replace(_nuc_cfg, tau_target=TAB_TAU)
_t0key = min(tov_trapped, key=lambda k: abs(k[0] - 0.35) + abs(k[1] - 1.5))
_tTkey = min(tov_trapped, key=lambda k: abs(k[0] - 0.25) + abs(k[1] - 2.0))
_tov0, _tovT = tov_trapped[_t0key], tov_trapped[_tTkey]
# BH bookkeeping: max PNS grav mass at t_0, and max baryonic mass the (deleptonized)
# t_Tmax branch can still support. A fixed-M_B star that exceeds these collapses.
_Mmax_t0   = float(_tov0[:int(np.argmax(_tov0[:, 4])) + 1, 4].max())
_Mb_max_tT = float(_tovT[:int(np.argmax(_tovT[:, 4])) + 1, 5].max())
# Include the maximal-mass PNS at t_0: it collapses to a BH after deleptonization
# unless small σ lets it nucleate to a QS first.
_TAB_MPNS  = list(TAB_MPNS_T0) + [round(_Mmax_t0, 3)]

def _mb_to_M_interp(arr):
    """M(M_B) on the stable branch — robust to the duplicate/non-monotone M_B the
    CFL QS sequence can have (_binterp's cubic rejects duplicate x). Sort + dedupe."""
    k = np.argmax(arr[:, 4]) + 1             # stable branch up to M_max (grav mass, col 4)
    x, y = arr[:k, 5], arr[:k, 4]            # M_B (col 5) -> M (col 4)
    o = np.argsort(x); x, y = x[o], y[o]
    u = np.concatenate(([True], np.diff(x) > 1e-9))   # drop duplicate M_B
    x, y = x[u], y[u]
    if len(x) < 4:
        return lambda q: np.nan
    return interp1d(x, y, kind='cubic', bounds_error=False, fill_value=np.nan)

_mqs_cache = {}
def _M_QS_of_Mb(pset):                   # baryon-conserving QS remnant mass M(M_B)
    k = (pset['alpha'], pset['B4'], pset['Delta0'], pset['m_s'])
    if k not in _mqs_cache:
        _mqs_cache[k] = _mb_to_M_interp(_cold_cfl_stable(pset)[0])
    return _mqs_cache[k]

def _nucleates(sc, sigma):               # sigma<sigma_crit => nucleates; +inf => always
    return bool(np.isposinf(sc) or (np.isfinite(sc) and sigma < sc))

def _sigma_crit_at(nBc, YL, S, Tc, params, D0):
    if not (np.isfinite(nBc) and np.isfinite(Tc)):
        return np.nan
    Hpt = nuc_an.hadronic_point(H['trapped'], nBc, YL, Tc)
    return nuc_an.sigma_target_pt(Hpt, Tc, TAB_FLAVOR, TAB_CHARGE, TAB_PHASE,
                                  params, D0, _tab_nuc)

def _outcome_row(pset, params, D0, sigma, MpnsT0):
    """One outcomes-table row: follow ONE fixed-baryon-mass PNS from t_0 to t_Tmax
    and decide NS / QS / BH for surface tension `sigma`.

    Returns (row dict, σ_crit(t_0), σ_crit(t_Tmax)) — the two σ_crit values are
    the deciding quantities, handed back so a caller can plot the decision
    itself, not just its outcome. σ_crit is NaN for a snapshot never reached
    (prompt BH, or conversion already happened at t_0).
    """
    sc0 = scT = np.nan
    Mb = float(_binterp(_tov0, 4, 5)(MpnsT0))       # M -> Mb (conserved after)
    # --- prompt BH: PNS above the t_0 maximum mass -> no stable star ---
    if (MpnsT0 > _Mmax_t0 + 1e-6) or not np.isfinite(Mb):
        return dict(
            params=q_tag_of(pset), sigma=sigma,
            M_t0=MpnsT0, Mb_t0=Mb, nBc_t0=np.nan, YSc_t0=np.nan,
            M_tT=np.nan, nBc_tT=np.nan, YSc_tT=np.nan,
            type='BH', M_rem=np.nan, E_conv_e53=np.nan), sc0, scT
    # --- t_0 central state + nucleation decision ---
    nBc0 = float(_binterp(_tov0, 4, 2)(MpnsT0))     # M -> n_Bc
    Tc0  = float(H['iso_trapped']['T'](nBc0, 0.35, 1.5))
    YS0  = float(H['trapped']['Y_S'](nBc0, 0.35, Tc0))
    sc0  = _sigma_crit_at(nBc0, 0.35, 1.5, Tc0, params, D0)
    if _nucleates(sc0, sigma):
        # nucleates already at t_0 -> QS now; t_Tmax snapshot never reached ('--')
        typ, Mconv = 'QS', MpnsT0
        MpnsT = nBcT = YST = np.nan
    elif Mb > _Mb_max_tT + 1e-6:
        # survives t_0 but deleptonization drops M_max below its M_B -> BH at t_Tmax
        typ, Mconv = 'BH', np.nan
        MpnsT = nBcT = YST = np.nan
    else:
        # --- t_Tmax at fixed M_B: decide again ---
        MpnsT = float(_binterp(_tovT, 5, 4)(Mb))    # Mb -> M
        nBcT  = float(_binterp(_tovT, 5, 2)(Mb))    # Mb -> n_Bc
        TcT   = float(H['iso_trapped']['T'](nBcT, 0.25, 2.0)) if np.isfinite(nBcT) else np.nan
        YST   = float(H['trapped']['Y_S'](nBcT, 0.25, TcT)) if np.isfinite(nBcT) else np.nan
        scT   = _sigma_crit_at(nBcT, 0.25, 2.0, TcT, params, D0)
        if _nucleates(scT, sigma):
            typ, Mconv = 'QS', MpnsT
        else:
            typ, Mconv = 'NS', np.nan
    # --- remnant mass + conversion energy ---
    if typ == 'QS':
        Mqs   = float(_M_QS_of_Mb(pset)(Mb))
        Mrem  = Mqs
        Econv = (Mconv - Mqs) * M_SUN_C2_ERG / 1e53 if np.isfinite(Mqs) else np.nan
    elif typ == 'NS':
        Mrem  = float(_binterp(tov_cold, 5, 4)(Mb))  # cold NS remnant at fixed M_B
        Econv = np.nan
    else:                                            # BH
        Mrem  = np.nan
        Econv = np.nan
    return dict(
        params=q_tag_of(pset), sigma=sigma,
        M_t0=MpnsT0, Mb_t0=Mb, nBc_t0=nBc0 / n_sat, YSc_t0=YS0,
        M_tT=MpnsT, nBc_tT=(nBcT / n_sat if np.isfinite(nBcT) else np.nan), YSc_tT=YST,
        type=typ, M_rem=Mrem, E_conv_e53=Econv), sc0, scT


_rows = []
for pset in TAB_SETS:
    params = get_alphabag_custom(alpha=pset['alpha'], B4=pset['B4'], m_s=pset['m_s'])
    D0 = pset['Delta0']
    for sigma in TAB_SIGMAS:
        for MpnsT0 in _TAB_MPNS:
            _rows.append(_outcome_row(pset, params, D0, sigma, MpnsT0)[0])

tab = pd.DataFrame(_rows)
_fmt = dict(M_t0='{:.2f}', Mb_t0='{:.2f}', nBc_t0='{:.2f}', YSc_t0='{:.2f}',
            M_tT='{:.2f}', nBc_tT='{:.2f}', YSc_tT='{:.2f}',
            M_rem='{:.2f}', E_conv_e53='{:.2f}')
_disp = tab.copy()
for _c, _f in _fmt.items():
    _disp[_c] = _disp[_c].map(lambda v, _f=_f: '--' if not np.isfinite(v) else _f.format(v))
print(_disp.to_string(index=False))
import os as _os
_os.makedirs('../output/figure_data', exist_ok=True)
tab.to_csv(f'../output/figure_data/table_outcomes_{xsd_tag}.csv', index=False)
# Minimal LaTeX tabular by hand (pandas.to_latex needs jinja2, not installed).
_cols = list(_disp.columns)
_tex = ['\\begin{tabular}{l' + 'c' * (len(_cols) - 1) + '}', '\\hline\\hline',
        ' & '.join(_cols) + ' \\\\', '\\hline']
_tex += [' & '.join(str(_disp.iloc[_i][_c]) for _c in _cols) + ' \\\\'
         for _i in range(len(_disp))]
_tex += ['\\hline\\hline', '\\end{tabular}']
with open(f'../output/figure_data/table_outcomes_{xsd_tag}.tex', 'w') as _f:
    _f.write('\n'.join(_tex))
print('\nwrote table_outcomes CSV + tex to ../output/figure_data/')


# %% [markdown]
# ### Observation-anchored parametrizations — $M$–$R$, outcomes table, remnant masses
#
# Pick two $(B^{1/4},\Delta_0)$ cells out of the Fig-5 $\sigma_{\rm crit}$ grid, each
# the **lowest-$\sigma_{\rm crit}$** cell whose cold CFL $M_{\max}$ matches a mass
# measurement: $2.18\,M_\odot$ (PSR J0952−0607 at $-1\sigma$: $2.35-0.17$) and
# $2.50\,M_\odot$ (GW190814 secondary). Lowest $\sigma_{\rm crit}$ = the *most
# conservative* choice — the parametrization that still supports the observed mass
# while demanding the least surface tension to survive as a hadronic star.
#
# Then, with each cell's own $\sigma_{\rm crit}$ used as the surface tension $\sigma$,
# re-run the PNS-evolution decision over $M^{\rm PNS}(t_0)=1.2\ldots1.78\,M_\odot$
# and show **(a)** $M$–$R$ vs the observational constraints, **(b)** the decision
# itself ($\sigma$ vs $\sigma_{\rm crit}$ at both snapshots), **(c)** the remnant mass
# produced, **(d)** the conversion energy.

# %%
# ============================================================================
#  Two observation-anchored quark parametrizations + their PNS-evolution outcomes.
#  Needs: the Fig 5 cell (_SIG8/_MM8/_al8/_B48/_D08/_F8) and the outcomes-table
#  cell above (_outcome_row and its helpers). Reuses _cold_cfl_stable for the QS
#  branch and add_observational_constraints for the M-R overlays.
# ============================================================================
set_paper_style()
# target cold-CFL M_max [M_sun] -> label. Both are 1D mass measurements, so the
# match is on M_max alone; TBL2_MTOL is how close the grid cell must land.
TBL2_TARGETS = {'PSR J0952$-$0607 ($-1\\sigma$)': 2.35 - 0.17,
                'GW190814 (secondary)':           2.50}
TBL2_MTOL    = 0.02                    # |M_max - target| window [M_sun]
# M_PNS(t_0) sweep. arange stops below 1.78, so append it explicitly (the user's
# upper end). 16 points x 2 sets x <=2 snapshots ~ 60 sigma_crit root-finds.
TBL2_MPNS    = np.append(np.arange(1.20, 1.78, 0.04), 1.78)
TBL2_COL     = [OKAB['blue'], OKAB['vermillion']]      # one colour per selected set


def _pick_cell(target, tol=TBL2_MTOL):
    """Lowest-σ_crit viable grid cell with |M_max - target| <= tol.

    Grid axes are (α_s slice, Δ_0 row, B^1/4 col) — the same order Fig 5 contours
    against (_B48, _D08), so the unravel below must not be transposed.
    """
    m = np.isfinite(_SIG8) & (np.abs(_MM8 - target) <= tol)
    if not m.any():
        raise ValueError(f'no viable cell within {tol} of M_max={target}; widen TBL2_MTOL')
    ia, ir, ic = np.unravel_index(np.argmin(np.where(m, _SIG8, np.inf)), _SIG8.shape)
    return (dict(alpha=float(_al8[ia]), B4=float(_B48[ic]),
                 Delta0=float(_D08[ir]), m_s=float(_F8['m_s'])),
            float(_SIG8[ia, ir, ic]), float(_MM8[ia, ir, ic]))


_sel = []                                  # (label, pset, sigma, M_max_grid)
for _lab, _tgt in TBL2_TARGETS.items():
    _p, _sc, _mm = _pick_cell(_tgt)
    _sel.append((_lab, _p, _sc, _mm))
    print(f"{_lab:<32} target M_max={_tgt:.2f} -> {q_tag_of(_p)}  "
          f"α_s={_p['alpha']:.3f}  M_max={_mm:.3f}  σ_crit={_sc:.1f} MeV/fm²")
# NOTE the grid σ_crit is the STAR-WIDE value (max over shells, MT0=1.4, t_Tmax
# snapshot), while _outcome_row's _sigma_crit_at is CENTRE-ONLY and so systematically
# smaller. Using the grid value as σ therefore does NOT put the 1.4 M_sun star exactly
# at threshold — panel (b) shows where the real crossing sits. Override with e.g.
# _sel = [(l, p, 0.8*s, m) for l, p, s, m in _sel] to slide σ down onto it.

# ---- outcomes over the M_PNS sweep -----------------------------------------
_r2 = []
for _lab, _p, _sig, _mm in _sel:
    _pr = get_alphabag_custom(alpha=_p['alpha'], B4=_p['B4'], m_s=_p['m_s'])
    for _M in TBL2_MPNS:
        _row, _sc0, _scT = _outcome_row(_p, _pr, _p['Delta0'], _sig, float(_M))
        _r2.append(_row | dict(label=_lab, sc_t0=_sc0, sc_tT=_scT, Mmax_QS=_mm))
tab2 = pd.DataFrame(_r2)
_disp2 = tab2.copy()
for _c, _f in (_fmt | {'sc_t0': '{:.1f}', 'sc_tT': '{:.1f}', 'Mmax_QS': '{:.2f}'}).items():
    _disp2[_c] = _disp2[_c].map(lambda v, _f=_f: '--' if not np.isfinite(v) else _f.format(v))
print()
print(_disp2.drop(columns=['Mmax_QS']).to_string(index=False))
tab2.to_csv(f'../output/figure_data/table_outcomes_obsanchored_{xsd_tag}.csv', index=False)

# ---- figure -----------------------------------------------------------------
fig, _ax2 = paper_grid('2x2', mode='double', placeholder=False, square=False,
                       **(PAPER_STYLE | {'aspect': 1.0}))
axMR2, axSC, axRem, axE = _ax2.flat

# (a) M-R: constraints behind (zorder 0-1), hadronic black, one QS branch per set.
add_observational_constraints(axMR2, CONTOUR_DIR, show_mass_bands=True,
                              inline_labels=True)
axMR2.plot(tov_cold[:, 3], tov_cold[:, 4], 'k-', lw=2, label='Hadronic (T=0)', zorder=5)
for (_lab, _p, _sig, _mm), _c in zip(_sel, TBL2_COL):
    _st, _mx = _cold_cfl_stable(_p)          # cached inside? no - one solve per set
    axMR2.plot(_st[:, 3], _st[:, 4], color=_c, lw=2, zorder=5,
               label=f"QS: $M_{{\\max}}$={_mx:.2f}, $\\sigma_{{\\rm crit}}$={_sig:.0f}")
axMR2.set_xlim(8, 16); axMR2.set_ylim(0, 3.0)
axMR2.set_xlabel(r'$R$ [km]'); axMR2.set_ylabel(r'$M$ [$M_\odot$]')
axMR2.legend(loc='lower left', fontsize=7)
panel_label(axMR2, '(a)')

# (b) the DECISION: σ (horizontal) vs σ_crit(M_PNS) at t_0 (solid) and t_Tmax
#     (dashed). σ below the σ_crit curve => nucleates. σ_crit(t_Tmax) is NaN wherever
#     the star already converted at t_0 (that snapshot is never reached) — the gap is
#     physical, not missing data.
for (_lab, _p, _sig, _mm), _c in zip(_sel, TBL2_COL):
    _g = tab2[tab2.label == _lab]
    axSC.plot(_g.M_t0, _g.sc_t0, color=_c, ls='-',  lw=1.8)
    axSC.plot(_g.M_t0, _g.sc_tT, color=_c, ls='--', lw=1.8)
    axSC.axhline(_sig, color=_c, ls=':', lw=1.2)
axSC.set_xlabel(r'$M^{\rm PNS}(t_0)$ [$M_\odot$]')
axSC.set_ylabel(r'$\sigma_{\rm crit}$ [MeV/fm$^2$]')
axSC.legend(handles=[Line2D([], [], color='0.3', ls=ls, label=lb) for ls, lb in
                     (('-', r'$\sigma_{\rm crit}(t_0)$'),
                      ('--', r'$\sigma_{\rm crit}(t_{T\max})$'),
                      (':', r'$\sigma$ (this row)'))],
            loc='best', fontsize=7)
panel_label(axSC, '(b)')

# (c) what actually comes out: remnant mass vs the initial PNS mass. Marker = fate,
#     colour = parametrization. Dotted diagonal = M_rem == M^PNS(t_0), so the vertical
#     drop off it IS the mass lost to the conversion (binding-energy release).
_MK = {'QS': ('o', 'full'), 'NS': ('s', 'none')}
# BH rows carry M_rem = NaN, so a marker legend entry for them would be a label with
# nothing on the plot. Shade their M_PNS range instead — the collapse threshold is
# set-INDEPENDENT (it is _Mmax_t0 / _Mb_max_tT, both hadronic), so one band covers both.
_bhM = tab2.M_t0[tab2.type == 'BH']
for _a in (axRem, axE):
    if len(_bhM):
        _a.axvspan(_bhM.min() - 0.02, TBL2_MPNS[-1] + 0.02, color='0.85', lw=0, zorder=0)
        _a.text(0.985, 0.04, 'BH', transform=_a.transAxes, ha='right', va='bottom',
                fontsize=8, color='0.35', fontweight='bold')
axRem.plot(TBL2_MPNS, TBL2_MPNS, color='0.6', ls=':', lw=1.0, zorder=1)
for (_lab, _p, _sig, _mm), _c in zip(_sel, TBL2_COL):
    _g = tab2[tab2.label == _lab]
    for _t, (_m, _fs) in _MK.items():
        _gg = _g[_g.type == _t]
        if len(_gg):
            axRem.plot(_gg.M_t0, _gg.M_rem, ls='none', marker=_m, ms=5, fillstyle=_fs,
                       mfc=(_c if _fs == 'full' else 'none'), mec=_c, mew=1.2, zorder=3)
axRem.set_xlabel(r'$M^{\rm PNS}(t_0)$ [$M_\odot$]')
axRem.set_ylabel(r'$M_{\rm remnant}$ [$M_\odot$]')
axRem.legend(handles=[Line2D([], [], color='0.3', ls='none', marker=_m, ms=5,
                             fillstyle=_fs, mfc=('0.3' if _fs == 'full' else 'none'),
                             label=_t) for _t, (_m, _fs) in _MK.items()]
                    + [Line2D([], [], color=_c, lw=2, label=_l)
                       for (_l, *_), _c in zip(_sel, TBL2_COL)],
             loc='lower right', fontsize=7)   # 'best' drifts onto the (c) panel tag
panel_label(axRem, '(c)')

# (d) conversion energy (QS rows only; NS/BH have none by construction)
for (_lab, _p, _sig, _mm), _c in zip(_sel, TBL2_COL):
    _g = tab2[(tab2.label == _lab) & np.isfinite(tab2.E_conv_e53)]
    axE.plot(_g.M_t0, _g.E_conv_e53, color=_c, marker='o', ms=4, lw=1.5)
axE.set_xlabel(r'$M^{\rm PNS}(t_0)$ [$M_\odot$]')
axE.set_ylabel(r'$E_{\rm conv}$ [$10^{53}$ erg]')
panel_label(axE, '(d)')

fig.savefig(f'../output/figures/obsanchored_outcomes_{xsd_tag}.pdf', bbox_inches='tight')
plt.show()
print(f'\nwrote table_outcomes_obsanchored_{xsd_tag}.csv + '
      f'obsanchored_outcomes_{xsd_tag}.pdf')


# %% [markdown]
# ## Export — figure data + guide for Claude Science
#
# Writes `../output/figure_data/`: (i) a flat CSV of the Fig-5 σ_crit grid, and
# (ii) `figures_guide.md` describing every paper/study figure (axes, encodings,
# what it shows, what to look for) so Claude Science can analyse and comment
# without the notebook. The outcomes table CSV/tex is already written above; the
# figure PDFs live in `../output/figures/`; TOV/nucleation `.dat` tables in
# `../output/tables_*`; the raw σ_crit grid `.npz` in `../output/mc_cfl/`.

# %%
# ============================================================================
#  Export cell: Fig-5 grid -> flat CSV, and a written figures_guide.md.
# ============================================================================
import os as _os
_os.makedirs('../output/figure_data', exist_ok=True)

# (1) Fig-5 sigma_crit grid -> long CSV (one row per (alpha,B4,Delta0) cell).
try:
    _g = np.load(f'../output/mc_cfl/sigma_crit_grid_{xsd_tag}_MT0{MT0_grid_thr[0]:.2f}_'
                 f'saddlepoint-coulomb_minimize-unpCFL.npz')
    _al, _B4, _D0 = _g['alpha_slices'], _g['B4_grid'], _g['Delta0_grid']
    _S, _M, _Rn = _g['sig_crit'], _g['M_max'], _g['reason']
    _rows_csv = ['alpha_s,B4_MeV,Delta0_MeV,sigma_crit_MeVfm2,M_max_Msun,reason_code']
    for ia in range(len(_al)):
        for i in range(len(_D0)):
            for j in range(len(_B4)):
                _rows_csv.append(f"{_al[ia]:.6g},{_B4[j]:.6g},{_D0[i]:.6g},"
                                 f"{_S[ia,i,j]:.6g},{_M[ia,i,j]:.6g},{int(_Rn[ia,i,j])}")
    open('../output/figure_data/fig5_sigmacrit_grid.csv', 'w').write('\n'.join(_rows_csv))
    print(f'wrote fig5_sigmacrit_grid.csv ({len(_rows_csv)-1} cells)')
except FileNotFoundError:
    print('Fig 5 npz not found — run the IV.3 sigma_crit scan first')

# (2) figures_guide.md
_GUIDE = r"""# Figure & table guide — 2-family quark nucleation paper

Reason codes (Fig 5 grid `reason_code`): 0=solve-fail, 1=Witten (SQM not abs. stable),
2=re-hadronization, 3=quasi-re-hadr., 4=M_max<2 Msun, 5=OK (accepted), 6=2-flavour bound.
`sigma_crit` NaN = non-viable cell; +inf = nucleates for all sigma tested.

## Fig 1 — nucleation barrier & critical quantities (fixed Y_L, sigma; one set)
(a) barrier W(R) at T=30 MeV for n_B^H/n_0 = 1,3,5 (colour) and the three phases
(unpaired ':' , CFL '--', unpCFL '-'), dot = critical point R_*. (b,c,d) R_*, W_*/T,
log10 tau vs n_B^H/n_0, coloured by T. Look for: barrier height/critical radius trends.

## Fig 1b — same panels, T fixed, colour = sigma (sigma-sweep variant of Fig 1).

## Fig 2 — stellar sequences, EoS & central density (M-R, P(mu_B), n_Bc vs M).
Hadronic vs quark branches; observational M-R constraints overlaid.

## Fig 3 — R_*, W_*/T, log10 tau, sigma_crit at the PNS centre vs M_PNS
2x2, colour = sigma, line style = phase. (a) R_*, (b) W_*/T, (c) log10 tau (dashed
line = tau target), (d) sigma_crit(tau=target) vs M_PNS (colour = quark set OR tau).
Red vertical guides = fixed M_PNS / baryon-matched cold star. Look for: where the
central droplet is largest/barrier lowest along the star sequence.

## Fig 4 — T_nuc(n_B^H) nucleation-condition curves (one panel per PNS thermal state)
The (n_B^H, T) locus where central tau = tau_target, per phase. Above/right of a
curve the star nucleates. Look for: which densities/temperatures permit nucleation.

## Fig 5 — sigma_crit(B^{1/4}, Delta_0) map, two alpha_s panels  [DATA: fig5_sigmacrit_grid.csv]
Viridis heatmap of the critical surface tension at the PNS centre (M_T0=1.4, tau=1 ms).
White solid = iso-sigma_crit contours; magenta dashed = iso-M_max; coloured outlines =
excluded zones (vermillion M_max<2, green Witten, grey 2-flavour, blue re-hadr.). Look
for: the accepted-parameter island and how sigma_crit varies with B^{1/4} and Delta_0.

## Fig 5a — M-R and P_CFL(mu_B) of the viable cells, coloured by sigma_crit
Every sigma_crit-viable cell replayed (cold CFL EoS + TOV); curves coloured by
sigma_crit on the Fig-5 scale; hadronic reference black. Look for: spread of QS
radii/max masses across the accepted region.

## (unnumbered) unpaired sigma_crit(B^{1/4}, alpha_s) — single and three-m_s panels.

## Study — droplet observables at nucleation (tau = 1 s)
Observables: R_* [fm]; W_*/T; N_B^{Q*}=(4/3)pi R_*^3 n_B^{Q*} (quark interior baryon
number of the critical droplet); Y_S^H (hadronic strangeness fraction).
- Plot 1: the four vs n_B^H/n_0 along the tau=1 s curve at fixed (Y_L^H, sigma), one
  quark set (colour = sigma).
- Plot 2a: scatter, the four vs sigma_crit, one point per viable Fig-5 cell (colour =
  alpha_s) — GLOBAL trends across parameter space.
- Plot 2b: the four vs M_PNS (colour = sigma), one quark set.
Goal: identify generic magnitudes/scalings of the critical droplet at nucleation
(is N_B^{Q*} ~ O(1-10)? does R_* cluster? does Y_S^H track density?).

## Study — W(R) large-R check
W(R) pushed to ~100 fm for coulomb_minimize / screening charge modes. Question: does
W(R) turn over into a local MINIMUM at large R (metastable finite droplet / frustrated
phase) or is it monotone past the barrier R_*? Printed verdict per mode.

## Table — PNS evolution outcomes  [DATA: table_outcomes_<tag>.csv/.tex]
Rows = (quark set, sigma, initial M^PNS at t_0). Columns: t_0 {M^PNS, M_B, n_Bc^H/n_0,
Y_Sc^H}; t_Tmax {M^PNS, n_Bc^H/n_0, Y_Sc^H}; remnant {type NS/QS, M, E_conv [1e53 erg]}.
Decision: sigma < sigma_crit(snapshot) => nucleates; converts at t_0 else t_Tmax else
stays NS. E_conv = (M^PNS_convert - M_QS)c^2 at fixed M_B. '--' = state/branch absent.
Look for: which (set, sigma) convert early (t_0), late (t_Tmax), or cool to NS, and the
energy released.
"""
open('../output/figure_data/figures_guide.md', 'w').write(_GUIDE)
print('wrote figures_guide.md')
print('Claude-Science bundle in ../output/figure_data/: figures_guide.md, '
      'fig1_bcd_*, fig2_sequences_*, fig3abc_*/fig3d_*, fig4_Tnuc_*, '
      'fig5_sigmacrit_grid.csv, table_outcomes_*.csv/.tex (+ PDFs in ../output/figures/). '
      'NOTE: fig1-4 CSVs are written when you run those figure cells.')
