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
# NOTE: force-reinstall pulls origin/main — PUSH local commits first or this
# reinstalls stale GitHub code (repo is metastability-nucleation, not nucleation).
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
          '../output/figures'):
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
    dict(alpha=0.08*np.pi/2, B4=158.0, Delta0=157.0, m_s=100.0),
    #dict(alpha=0.3*np.pi/2, B4=145.0, Delta0=80.0, m_s=100.0),
    dict(alpha=0.1*np.pi/2, B4=145.0, Delta0=80.0, m_s=100.0),
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
alpha_slices     = [0.1 * np.pi / 2, 0.2 * np.pi / 2, 0.3 * np.pi / 2]   # α_s panels
B4_grid_scan     = np.linspace(130.0, 180.0, 51)         # B^1/4 [MeV]
Delta0_grid_scan = np.linspace(0.0, 200.0, 51)           # Δ_0  [MeV]
scan_n_jobs      = -1                                     # joblib workers (-1 = all cores)
# ---- methods to run: (flavor, charge, phase). The FIRST is the "primary" one,
#      it drives results_scan + the reload / M-R cells below. Trim to taste.
scan_cases = [
    ('saddlepoint', 'coulomb_minimize', 'unpCFL'),
    ('saddlepoint', 'coulomb_minimize', 'cfl'),
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
alpha_slices     = [0.1 * np.pi / 2, 0.2 * np.pi / 2, 0.3 * np.pi / 2]   # α_s panels
B4_grid_scan     = np.linspace(130.0, 180.0, 51)         # B^1/4 [MeV]
Delta0_grid_scan = np.linspace(0.0, 200.0, 51)           # Δ_0  [MeV]
scan_n_jobs      = -1                                     # joblib workers (-1 = all cores)
# ---- methods to run: (flavor, charge, phase). The FIRST is the "primary" one,
#      it drives results_scan + the reload / M-R cells below. Trim to taste.
scan_cases = [
    ('saddlepoint', 'coulomb_minimize', 'unpCFL'),
    ('saddlepoint', 'coulomb_minimize', 'cfl'),
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
for F1_SET in quark_param_sets[::-1]:
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
    fig, ((axA, axB), (axC, axD)) = paper_grid('2x2', mode='double', placeholder=False)

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
                             mfc=(_cA[_ci] if _PHASE_FILL[_ph] else 'white'), zorder=5)
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
F1_SET     = quark_param_sets[1]       # <-- SELECT quark parametrization (also sets Delta0)
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
fig, ((axA, axB), (axC, axD)) = paper_grid('2x2', mode='double', placeholder=False)

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
F2_SET   = quark_param_sets[1]                   # quark parametrization (ref plot + Delta0)
_QS_GREENS = ['#74c476', '#31a354', '#006d2c']   # QS star colours, per quark set (light->dark)

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


# QS = cold CFL star for the single reference quark parametrization (F2_SET).
_qs_tov = []
for _p in [F2_SET]:
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
        arr=_st, c='#238b45', lbl='QS',
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
fig, ((axA, axB), (axC, axD)) = paper_grid('2x2', mode='double', placeholder=False)

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
FN2_SET             = quark_param_sets[1]                 # <-- quark parametrization (panels a–c)
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
fig, ((axA, axB), (axC, axD)) = paper_grid('2x2', mode='double', placeholder=False)
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

# common cosmetics: red vertical mass guides, xlim, per-panel x-label, panel tag (no grid)
for _lab, ax in zip(('(a)', '(b)', '(c)', '(d)'), (axA, axB, axC, axD)):
    ax.set_xlim(0.7, M_max)
    ax.set_xlabel(r'$M_{\rm PNS}\ [M_\odot]$')         # every panel labelled (no sharex)
    for mv in FN2_MVERT:                       # dashed: fixed M_PNS
        ax.axvline(mv, color=FN2_VGUIDE_COL, ls='--', lw=1.0, zorder=1)
    for md in M_dot:                           # dotted: Mb-matched to cold
        if np.isfinite(md):
            ax.axvline(md, color=FN2_VGUIDE_COL, ls=':', lw=1.0, zorder=1)
    panel_label(ax, _lab, corner='upper left')

# legends: phase (line style) on (a); σ colour on (b); panel (d) carries its own.
_ph = [Line2D([], [], color='k', ls=ls, label=lbl)
       for lbl, ls in [('unpaired', ':'), ('CFL', '--'), ('unpCFL', '-')]]
_sg = [Line2D([], [], color=_sig_col[sg], ls='-', label=rf'$\sigma={int(sg)}$')
       for sg in FN2_SIGMAS]
axA.legend(handles=_ph, loc='best')
axB.legend(handles=_sg, loc='best', title=r'$\sigma$ [MeV/fm$^2$]')

fig.savefig(f'../output/figures/paper_fig3_Rstar_Wc_tau_sigmacrit_{xsd_tag}_{_tag}.pdf',
            bbox_inches='tight')
plt.show()


# %% [markdown]
# ### Paper Figure 4 — $T_{\rm nuc}(n_B^H)$ nucleation conditions
#
# One figure per quark parametrization. $\tau=\tau_{\rm target}$ curves in
# $(n_B^H, T)$, one line per $\sigma$ (colour) and phase (line style). The
# stellar isentrope is drawn in the snapshot colour (orange $t_0$, red
# $t_{T_\mathrm{max}}$); markers on it mark the PNS central density (filled +
# labelled: fixed $M=1,1.2,1.4,1.6$, star = $M_{\max}$; open: $M_b$-matched to a
# cold star). **(a)** $t_0$: $Y_L^H=0.35,\,S_H=1.5$   **(b)** $t_{T_\mathrm{max}}$: $Y_L^H=0.25,\,S_H=2$.

# %%
# ============================================================================
#  Figure 4.  T_nuc(n_B^H) nucleation-condition curves — ONE figure per quark set.
#    Two panels: (a) t_0 (Y_L^H=0.35, S_H=1.5),  (b) t_Tmax (Y_L^H=0.25, S_H=2.0).
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
F6_SIGMAS_BY_SET[q_tag_of(quark_param_sets[0])] = [150, 200, 250]
F6_SIGMAS_BY_SET[q_tag_of(quark_param_sets[1])] = [50, 100, 150]
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

# one figure per quark parametrization
for FN_SET in quark_param_sets:
    _tag    = q_tag_of(FN_SET)
    F6_SIGMAS = F6_SIGMAS_BY_SET.get(_tag, F6_SIGMAS_DEFAULT)    # per-set σ selection
    _sig_col  = {sg: mpl.cm.viridis(t) for sg, t in             # colour = σ, per set
                 zip(F6_SIGMAS, np.linspace(0.15, 0.85, len(F6_SIGMAS)))}
    _T_CFL  = float(T_critical(FN_SET['Delta0']))   # CFL Tc = 0.57*2^(1/3)*Delta0 [MeV]
    _panels = [dict(**PNS_T0,   lab='(a)'),         # left  = t_0
               dict(**PNS_TMAX, lab='(b)')]         # right = t_Tmax

    # PRD size: mode='double' (4.75" wide, half height → panels match a 2×2 top row).
    # Change to 'single'/'double' to resize.
    fig, axes = paper_grid('1x2', mode='double', placeholder=False)
    axes = axes[0]      # paper_grid axes are 2-D (1×2); flatten to a 1-D row of 2 panels
    # NOTE: no sharey — both panels carry their own T_nuc axis (same ylim set below)
    for ax, pan in zip(axes, _panels):
        YL, S, col = pan['YLH'], pan['S'], pan['color']
        YL_used = None
        for sg in F6_SIGMAS:
            for ph, ls in _PHASE_LS.items():          # unpCFL drawn last -> on top
                stem = f"Htrapped_{FN_FLAVOR}_{FN_CHARGE}_{ph}_{_tag}_s{int(sg)}"
                if stem not in nuc_sets:
                    continue
                grids   = nuc_sets[stem].hadronic_grids
                iYL     = int(np.argmin(np.abs(grids['Y_L_H'] - YL)))
                YL_used = grids['Y_L_H'][iYL]
                res = compute_nucleation_density(nuc_sets[stem],
                                                 tau_target=FN_TAU, scan='n_B')
                nB, T = nucleation_curve(res, iYL)
                m = np.isfinite(nB) & np.isfinite(T)
                if FN_CUT_CFL_ABOVE_TC and ph == 'cfl':
                    m &= (T <= _T_CFL)                # CFL undefined above Tc
                if m.any():
                    ax.plot(nB[m] / n_sat, T[m], color=_sig_col[sg], ls=ls,
                            lw=_PHASE_LW[ph], alpha=_PHASE_ALPHA[ph])

        ax.set_xlim(0.5, 12); ax.set_ylim(1, 80)      # T [MeV], n_B^H/n_sat

        # isentrope T(n_B^H) at (Y_L, S), solid in the snapshot colour + PNS markers.
        # Drawn from 0.5 n_sat up to the PNS M_max central density (branch tip).
        YL_iso = YL if YL_used is None else float(YL_used)
        T_iso  = lambda nBc: H['iso_trapped']['T'](nBc, YL_iso, S)
        tov_tr = _tov_trapped_seq(YL_iso, S)
        nBc_mx = tov_tr[int(np.argmax(tov_tr[:, _M])), _NBC]     # M_max central density
        nB_iso = np.linspace(0.5 * n_sat, nBc_mx, 200)
        ax.plot(nB_iso / n_sat, T_iso(nB_iso), color=col, ls='-', lw=1.8, zorder=3)
        _iso_markers(ax, tov_tr, T_iso, col)

        ax.set_xlabel(r'$n_B^H / n_{\rm sat}$')
        ax.set_ylabel(r'$T_{\rm nuc}$ [MeV]')         # on BOTH panels (incl. right)
        ax.set_title(rf"{pan['lbl']}:  $Y_L^H={YL_iso:.2f}$,  $S_H={S:g}$")
        ax.set_box_aspect(1)                          # square panels
        panel_label(ax, pan['lab'])

    # Legends: phase (line style, Fig-1 lw/alpha) top-right of (a); sigma right of
    # (b) — number-only labels under a small units title.
    _ph_handles = [Line2D([], [], color='k', ls=_PHASE_LS[p], lw=_PHASE_LW[p],
                          alpha=_PHASE_ALPHA[p], label=_PHASE_LBL[p]) for p in _PHASE_LS]
    _sig_handles = [Line2D([], [], color=_sig_col[sg], ls='-', label=f'{int(sg)}')
                    for sg in F6_SIGMAS]
    axes[0].legend(handles=_ph_handles, loc='upper right')
    _lg = axes[1].legend(handles=_sig_handles, loc='upper right',
                         title=r'$\sigma\;[\mathrm{MeV\,fm^{-2}}]$')
    _lg.get_title()#.set_fontsize(12)                   # smaller units title

    fig.savefig(f'../output/figures/paper_fig4_Tnuc_{xsd_tag}_{_tag}.pdf',
                bbox_inches='tight')
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
F8_SHOW          = [0, 2]     # which alpha_s panel indices to draw (pi/2 x 0.1, 0.3)
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

# PRD size: mode='double' (4.75" wide, 1×2). Change to 'single'/'double' to resize.
# paper_grid('1x2') is 2 columns → assumes len(F8_SHOW) == 2 (F8_SHOW = [0, 1]).
fig, axes = paper_grid('1x2', mode='double', placeholder=False)   # panels already square
pcm = None
for _c, _ia in enumerate(F8_SHOW):
    ax = axes[0, _c]
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
F8B_ALPHAS     = [0, 1, 2]              # α_s slice indices to include (into _al8)
F8B_MAX_CURVES = 300                    # even subsample of viable cells (None = all)
# viable (finite-σ_crit) cells across the chosen α slices — same grid as Fig 5
_al_idx  = [i for i in F8B_ALPHAS if i < len(_al8)]
_f8b_curves = nuc_an.replay_accepted(_SIG8[_al_idx], _al8[_al_idx], _B48, _D08,
                                     _filt_cfg, max_curves=F8B_MAX_CURVES,
                                     n_jobs=scan_n_jobs)
_norm8 = plt.Normalize(vmin=_vmin8, vmax=_vmax8)     # == Fig 5 heatmap colour scale
_cmap8 = plt.cm.viridis

fig, _axes = paper_grid('1x2', mode='double', placeholder=False)
axMR, axP = _axes[0]                     # paper_grid axes are 2-D (1×2) → row 0
# (a) M-R: hadronic reference (black) + 2 M_sun guide; CFL curves coloured by σ_crit
axMR.plot(tov_cold[:, 3], tov_cold[:, 4], 'k-', lw=2, label='Hadronic (T=0)', zorder=5)
axMR.axhline(2.0, color='0.5', ls=':', lw=1.0, zorder=1)
for _c, _sc in _f8b_curves:
    axMR.plot(_c['R'], _c['M'], color=_cmap8(_norm8(_sc)), lw=0.6, alpha=0.8)
axMR.set_xlim(7, 16); axMR.set_ylim(0, 3.0)
axMR.set_xlabel(r'$R$ [km]'); axMR.set_ylabel(r'$M$ [$M_\odot$]')
axMR.legend(loc='lower right'); panel_label(axMR, '(a)')
# (b) P_CFL(mu_B): hadronic P_H (black) + CFL curves coloured by σ_crit
axP.plot(mu_B_H_sorted, P_H_sorted, 'k-', lw=2, label=r'$P_H\ (T\approx0)$', zorder=5)
for _c, _sc in _f8b_curves:
    axP.plot(_c['mu'], _c['P'], color=_cmap8(_norm8(_sc)), lw=0.6, alpha=0.8)
axP.set_xlabel(r'$\mu_B$ [MeV]'); axP.set_ylabel(r'$P$ [MeV/fm$^3$]')
axP.legend(loc='upper left'); panel_label(axP, '(b)')
# colorbar as tall as the square right panel (same inset trick as Fig 5)
_sm = plt.cm.ScalarMappable(norm=_norm8, cmap=_cmap8); _sm.set_array([])
_cax8b = axP.inset_axes([1.05, 0.0, 0.05, 1.0])
fig.colorbar(_sm, cax=_cax8b, label=r'$\sigma_{\rm crit}$ [MeV/fm$^2$]')

fig.savefig(f'../output/figures/paper_fig5a_MR_PmuB_sigmacrit_{xsd_tag}.pdf',
            bbox_inches='tight')
plt.show()


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

# %%
# ============================================================================
#  Screening validation: W(R) for the five charge prescriptions at one point.
# ============================================================================
SC_T    = 20.0                      # temperature [MeV]
SC_NBH  = 5.0                       # n_B^H / n_sat
SC_SIG  = 30.0                      # surface tension [MeV/fm^2]
SC_PAR  = get_alphabag_custom(alpha=m5_set['alpha'], B4=m5_set['B4'], m_s=m5_set['m_s'])
SC_Rg   = np.linspace(0.02, 12.0, 500)

# (charge_mode, label, colour, extra kwargs) — colours echo thesis Fig. screen
_SC_MODES = [
    ('lcn',              'LCN',            'tab:red',    {}),
    ('gcn',              'GCN (no Coul.)', 'tab:purple', {}),
    ('gcn_coulomb',      'GCN + Coul.',    'tab:blue',   {}),
    ('coulomb_minimize', 'Minimization',   'tab:green',  {}),
    ('screening',        'Screening',      'tab:orange', {}),
]

fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
# Scale the y-axis to the PHYSICAL barriers (LCN/GCN/minimization/screening).
# The unscreened GCN+Coulomb term grows as R^5 and can run far off-scale — that
# blow-up is exactly the pathology screening/minimization cure (thesis Fig. screen),
# so it is allowed to leave the top of the frame rather than compress everything.
_wmax = 0.0
for _ch, _lbl, _col, _kw in _SC_MODES:
    _eb = compute_energy_barrier(
        H['trapped'], SC_NBH * n_sat, SC_T, SC_SIG,
        electric_charge_mode=_ch, params=SC_PAR, flavor_mode='saddlepoint',
        quark_phase='unpaired', Y_L_H=m5_YL_used, R_values=SC_Rg, **_kw)
    ax.plot(SC_Rg, _eb.W, color=_col, lw=1.9, label=_lbl)
    if np.isfinite(_eb.W).any():
        _k = int(np.nanargmax(_eb.W))
        ax.plot(SC_Rg[_k], _eb.W[_k], 'o', ms=6, color=_col, mfc='white', zorder=5)
        if _ch != 'gcn_coulomb':               # exclude the R^5 blow-up from the y-scale
            _wmax = max(_wmax, float(_eb.W[_k]))

# Unscreened limit: screening with a huge lambda_D must overlie GCN+Coulomb.
_eb_big = compute_energy_barrier(
    H['trapped'], SC_NBH * n_sat, SC_T, SC_SIG, electric_charge_mode='screening',
    params=SC_PAR, flavor_mode='saddlepoint', quark_phase='unpaired',
    Y_L_H=m5_YL_used, R_values=SC_Rg, lambda_D=1e8)
ax.plot(SC_Rg, _eb_big.W, color='0.4', lw=1.4, ls='--',
        label=r'Screening $\lambda_D{\to}\infty$')

# annotate the physical Debye length used by the screening curve
_eb_scr = compute_energy_barrier(
    H['trapped'], SC_NBH * n_sat, SC_T, SC_SIG, electric_charge_mode='screening',
    params=SC_PAR, flavor_mode='saddlepoint', quark_phase='unpaired',
    Y_L_H=m5_YL_used, R_values=SC_Rg)
ax.axhline(0, color='0.6', lw=0.7, zorder=0)
ax.set_xlim(0, 8)
if _wmax > 0:
    ax.set_ylim(-0.2 * _wmax, 1.3 * _wmax)
ax.set_xlabel(r'$R$ [fm]'); ax.set_ylabel(r'$W$ [MeV]')
ax.set_title(rf"Charge prescriptions, $T={SC_T:.0f}$ MeV, "
             rf"$n_B^H/n_\mathrm{{sat}}={SC_NBH:g}$, $\sigma={SC_SIG:.0f}$ "
             rf"MeV/fm$^2$, $\lambda_D={_eb_scr.lambda_D:.1f}$ fm — {m5_tag}")
ax.legend(loc='upper right', fontsize=8)
plt.show()


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


SP1_SET     = quark_param_sets[0]                    # one quark parametrization
SP1_YLH     = PNS_TMAX['YLH']                        # Y_L^H (0.25 = t_Tmax)
SP1_PHASE   = 'unpCFL'                               # 'unpaired' | 'cfl' | 'unpCFL'
SP1_FLAVOR, SP1_CHARGE = 'saddlepoint', 'coulomb_minimize'
SP1_SIGMAS  = [50.0, 100.0, 200.0]                   # colour = sigma
_nuc1s = _dc_replace(_nuc_cfg, tau_target=1.0)       # tau = 1 s
_tag1  = q_tag_of(SP1_SET)
_par1  = get_alphabag_custom(alpha=SP1_SET['alpha'], B4=SP1_SET['B4'], m_s=SP1_SET['m_s'])
_D01   = SP1_SET['Delta0']
_col1  = {sg: mpl.cm.viridis(t) for sg, t in
          zip(SP1_SIGMAS, np.linspace(0.15, 0.85, len(SP1_SIGMAS)))}

fig, ((axR, axW), (axN, axY)) = paper_grid('2x2', mode='double', placeholder=False)
_P1 = [(axR, 'R', r'$R_*$ [fm]', 'linear'), (axW, 'WoverT', r'$W_*/T$', 'log'),
       (axN, 'NB', r'$N_B^{Q*}$', 'log'), (axY, 'YS', r'$Y_S^H$', 'linear')]
for sg in SP1_SIGMAS:
    stem = f"Htrapped_{SP1_FLAVOR}_{SP1_CHARGE}_{SP1_PHASE}_{_tag1}_s{int(sg)}"
    if stem not in nuc_sets:
        print(f"Plot 1 skip σ={sg:g}: {stem} not in nuc_sets"); continue
    _obs = nuc_sets[stem]
    _iYL = int(np.argmin(np.abs(_obs.hadronic_grids['Y_L_H'] - SP1_YLH)))
    _YLg = float(_obs.hadronic_grids['Y_L_H'][_iYL])
    res  = compute_nucleation_density(_obs, tau_target=1.0, scan='n_B')   # tau=1 s locus
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

fig, ((axR, axW), (axN, axY)) = paper_grid('2x2', mode='double', placeholder=False)
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
# **Plot 2b** — vs $M_{\rm PNS}$ (colour = $\sigma$) for one quark set: the four
# observables along the $t_{T\rm max}$ central sequence (extends Fig 3 to
# $N_B^{Q*}$ and $Y_S^H$). Fixed $\sigma$ values, so $\tau$ varies along each curve.

# %%
# ============================================================================
#  Study Plot 2b: droplet observables vs M_PNS, colour = sigma, one quark set.
# ============================================================================
set_paper_style()
SP2B_SET   = quark_param_sets[0]
SP2B_PHASE = 'unpCFL'
SP2B_FLAVOR, SP2B_CHARGE = 'saddlepoint', 'coulomb_minimize'
SP2B_SIGMAS = [50.0, 100.0, 200.0]
_yl2, _s2 = PNS_TMAX['YLH'], PNS_TMAX['S']
_tov2 = tov_trapped[min(tov_trapped, key=lambda k: abs(k[0] - _yl2) + abs(k[1] - _s2))]
_i2 = np.argmax(_tov2[:, 4]) + 1
_m2 = _tov2[:_i2, 4] >= 0.6
_Ms, _nBs = _tov2[:_i2, 4][_m2], _tov2[:_i2, 2][_m2]
_Ts = np.asarray(H['iso_trapped']['T'](_nBs, _yl2, _s2))
_par2b = get_alphabag_custom(alpha=SP2B_SET['alpha'], B4=SP2B_SET['B4'], m_s=SP2B_SET['m_s'])
_D02b = SP2B_SET['Delta0']
_col2b = {sg: mpl.cm.viridis(t) for sg, t in
          zip(SP2B_SIGMAS, np.linspace(0.15, 0.85, len(SP2B_SIGMAS)))}

fig, ((axR, axW), (axN, axY)) = paper_grid('2x2', mode='double', placeholder=False)
_P2b = [(axR, 'R', r'$R_*$ [fm]', 'linear'), (axW, 'WoverT', r'$W_*/T$', 'log'),
        (axN, 'NB', r'$N_B^{Q*}$', 'log'), (axY, 'YS', r'$Y_S^H$', 'linear')]
for sg in SP2B_SIGMAS:
    _cache, _rows = {}, []
    for nb, T in zip(_nBs, _Ts):
        H_pt = nuc_an.hadronic_point(H['trapped'], nb, _yl2, T)
        _rows.append(_droplet_obs(sg, H_pt, T, SP2B_FLAVOR, SP2B_CHARGE, SP2B_PHASE,
                                  _par2b, _D02b, _nuc_cfg, _cache))
    for ax, key, _, _sc in _P2b:
        ax.plot(_Ms, [r[key] for r in _rows], color=_col2b[sg], lw=1.6,
                label=rf'$\sigma={int(sg)}$')
for ax, key, ylab, sc in _P2b:
    ax.set_ylabel(ylab); ax.set_yscale(sc); ax.set_xlabel(r'$M_{\rm PNS}\ [M_\odot]$')
axR.legend(fontsize=8, title=r'$\sigma$ [MeV/fm$^2$]')
for _lab, ax in zip('abcd', (axR, axW, axN, axY)):
    panel_label(ax, f'({_lab})')
fig.suptitle(rf"Droplet vs $M_{{\rm PNS}}$ — {SP2B_PHASE}, {q_tag_of(SP2B_SET)}", y=1.02)
fig.savefig(f'../output/figures/study_droplet_vs_MPNS_{xsd_tag}_{q_tag_of(SP2B_SET)}.pdf',
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
# at $t_0$ if σ<σ_crit(t_0), else at $t_Tmax$ if σ<σ_crit(t_Tmax), else stays NS.
# Conversion energy $E_{\rm conv}=(M^{\rm PNS}_{\rm convert}-M_{\rm QS})c^2$ at fixed
# $M_B$ (QS remnant from the row's own CFL EoS). Generalises the old fixed-σ table.

# %%
# ============================================================================
#  Outcomes table. New physics vs the rest of the notebook: (i) the nucleate-or-not
#  decision branch, (ii) M_QS(M_B) baryon-conserving remnant, (iii) E_conv. All
#  assumptions are flagged inline for validation against the old table.
# ============================================================================
M_SUN_C2_ERG = 1.7827e54                 # M_sun c^2 [erg]  (new constant)
TAB_SETS     = quark_param_sets          # rows: quark parametrizations
TAB_SIGMAS   = [30.0]                    # and sigmas [MeV/fm^2]
TAB_MPNS_T0  = [1.4, 1.6]                # initial PNS grav mass at t_0 [M_sun]
TAB_TAU      = 1e-3                       # s — nucleation-criterion tau for the decision
TAB_FLAVOR, TAB_CHARGE, TAB_PHASE = 'saddlepoint', 'coulomb_minimize', 'unpCFL'
_tab_nuc = _dc_replace(_nuc_cfg, tau_target=TAB_TAU)
_t0key = min(tov_trapped, key=lambda k: abs(k[0] - 0.35) + abs(k[1] - 1.5))
_tTkey = min(tov_trapped, key=lambda k: abs(k[0] - 0.25) + abs(k[1] - 2.0))
_tov0, _tovT = tov_trapped[_t0key], tov_trapped[_tTkey]

_mqs_cache = {}
def _M_QS_of_Mb(pset):                   # baryon-conserving QS remnant mass M(M_B)
    k = (pset['alpha'], pset['B4'], pset['Delta0'], pset['m_s'])
    if k not in _mqs_cache:
        _mqs_cache[k] = _binterp(_cold_cfl_stable(pset)[0], 5, 4)   # cols 5=Mb -> 4=M
    return _mqs_cache[k]

def _nucleates(sc, sigma):               # sigma<sigma_crit => nucleates; +inf => always
    return bool(np.isposinf(sc) or (np.isfinite(sc) and sigma < sc))

def _sigma_crit_at(nBc, YL, S, Tc, params, D0):
    if not (np.isfinite(nBc) and np.isfinite(Tc)):
        return np.nan
    Hpt = nuc_an.hadronic_point(H['trapped'], nBc, YL, Tc)
    return nuc_an.sigma_target_pt(Hpt, Tc, TAB_FLAVOR, TAB_CHARGE, TAB_PHASE,
                                  params, D0, _tab_nuc)

_rows = []
for pset in TAB_SETS:
    params = get_alphabag_custom(alpha=pset['alpha'], B4=pset['B4'], m_s=pset['m_s'])
    D0 = pset['Delta0']
    for sigma in TAB_SIGMAS:
        for MpnsT0 in TAB_MPNS_T0:
            # --- t_0 (fixed M_B thereafter) ---
            Mb   = float(_binterp(_tov0, 4, 5)(MpnsT0))     # M -> Mb
            nBc0 = float(_binterp(_tov0, 4, 2)(MpnsT0))     # M -> n_Bc
            Tc0  = float(H['iso_trapped']['T'](nBc0, 0.35, 1.5))
            YS0  = float(H['trapped']['Y_S'](nBc0, 0.35, Tc0))
            # --- t_Tmax at fixed M_B (NaN if beyond that branch's max) ---
            MpnsT = float(_binterp(_tovT, 5, 4)(Mb))        # Mb -> M
            nBcT  = float(_binterp(_tovT, 5, 2)(Mb))        # Mb -> n_Bc
            TcT   = float(H['iso_trapped']['T'](nBcT, 0.25, 2.0)) if np.isfinite(nBcT) else np.nan
            YST   = float(H['trapped']['Y_S'](nBcT, 0.25, TcT)) if np.isfinite(nBcT) else np.nan
            # --- decision: sigma_crit at each snapshot, pick outcome ---
            sc0 = _sigma_crit_at(nBc0, 0.35, 1.5, Tc0, params, D0)
            scT = _sigma_crit_at(nBcT, 0.25, 2.0, TcT, params, D0)
            if _nucleates(sc0, sigma):
                typ, Mconv = 'QS', MpnsT0
            elif _nucleates(scT, sigma):
                typ, Mconv = 'QS', MpnsT
            else:
                typ, Mconv = 'NS', np.nan
            if typ == 'QS':
                Mqs   = float(_M_QS_of_Mb(pset)(Mb))
                Mrem  = Mqs
                Econv = (Mconv - Mqs) * M_SUN_C2_ERG / 1e53 if np.isfinite(Mqs) else np.nan
            else:
                Mrem  = float(_binterp(tov_cold, 5, 4)(Mb))  # cold NS remnant at fixed M_B
                Econv = np.nan
            _rows.append(dict(
                params=q_tag_of(pset), sigma=sigma,
                M_t0=MpnsT0, Mb_t0=Mb, nBc_t0=nBc0 / n_sat, YSc_t0=YS0,
                M_tT=MpnsT, nBc_tT=nBcT / n_sat, YSc_tT=YST,
                type=typ, M_rem=Mrem, E_conv_e53=Econv))

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
with open(f'../output/figure_data/table_outcomes_{xsd_tag}.tex', 'w') as _f:
    _f.write(_disp.to_latex(index=False, escape=False))
print('\nwrote table_outcomes CSV + tex to ../output/figure_data/')


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
      'table_outcomes_*.csv/.tex, fig5_sigmacrit_grid.csv (+ PDFs in ../output/figures/).')
