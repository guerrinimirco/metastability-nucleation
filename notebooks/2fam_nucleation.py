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
# matter (proto-neutron star / merger remnant), for the 2-family / strange-quark
# hypothesis.
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
#   surface tension at the PNS centre (IV.1–IV.3), the paper figures (IV.4) and
#   the cross-method comparisons (IV.5).
#
# **Parts I, III and IV run without Part II** (they read the saved tables). Typical
# session after the tables exist: run **I → III → IV**. "Run All" executes
# top-to-bottom without errors (Part II then regenerates every table first).
#
# Heavy numerics live in the installed packages: `nucleation.analysis`
# (σ_crit engine, CFL filters, replay) and `nucleation.energy_barrier` /
# `nucleation.general_nucleation` (Q\* solvers, barriers, rates); plot styling in
# `nucleation.analysis.figure`. The notebook holds parameters, generation loops
# and figures only.

# %% [markdown]
# # Part I — Setup & parameters
#
# *Always run.* Imports and all tunable knobs; no heavy computation here.

# %% [markdown]
# ## I.1 — Imports & installs
#
# Single setup cell: re-running it from a fresh kernel restores every symbol used
# downstream. Keep the GitHub-install lines commented while developing the `eos` /
# `nucleation` packages locally (editable install) — repo edits then take effect
# after a kernel restart only.
#
# > ⚠️ Running the GitHub `pip install` line **replaces** the editable install
# > with a frozen copy, silently shadowing local repo edits. To restore:
# > `pip install -e ../../nucleation --no-deps` and check that
# > `import nucleation; nucleation.__file__` points into the repo.

# %%
import sys, os

# ─── Install the two custom packages from GitHub (latest commit) ─────────────
# Comment these out and uncomment the local-path block when developing
# the packages locally — edits in ../../eos or ../../nucleation then take
# effect immediately without a re-install.
# !{sys.executable} -m pip install --no-deps --force-reinstall git+https://github.com/guerrinimirco/eos.git --quiet
print("eos package loaded successfully!")
# !{sys.executable} -m pip install --no-deps --force-reinstall git+https://github.com/guerrinimirco/metastability-nucleation.git --quiet
print("nucleation package loaded successfully!")

# Local-dev alternative:
# # !{sys.executable} -m pip install -e ../../eos --quiet
# # !{sys.executable} -m pip install -e ../../nucleation --quiet


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
from nucleation.energy_barrier.small_droplet import (
    compute_Qstar_table, load_Qstar_table, build_Qstar_interpolators,
    compute_thermal_nucleation_observables,
    build_thermal_nucleation_interpolators,
    load_thermal_nucleation_table,
    compute_nucleation_density, nucleation_curve,
    export_table, QstarTableData,
)
from nucleation.energy_barrier.small_droplet.solvers import get_solver_Qs
from nucleation.energy_barrier.small_droplet.observables import compute_energy_barrier
import nucleation.analysis as nuc_an               # sigma_crit scan engine

# ─── Figure styling + observational-constraint overlays (shared, in-package) ─
from nucleation.analysis.figure import (
    set_paper_style, panel_label, STANDARD_COLORS,
    add_observational_constraints,
)

# Precomputed NICER/HESS contours live in the sibling `eos` project; the path
# is relative to the notebooks/ cwd. Regenerate offline if the samples change.
CONTOUR_DIR = "../../eos/plot/data/contours"


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
# n_B (baryon number density). 300 points, 0.1 → 12 × n_sat — covers from the
#   crust edge up to ≳10 n_sat (well above any NS central density).
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
    name=f"2fam_phi_{xsd_tag}",       # ← derived from xsd_tag, not hardcoded
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
    #dict(alpha=0.1*np.pi/2, B4=165.0, Delta0=180.0, m_s=100.0),
    #dict(alpha=0.1*np.pi/2, B4=170.0, Delta0=175.0, m_s=100.0),
    dict(alpha=0.18*np.pi/2, B4=150.0, Delta0=126.0, m_s=100.0),  # marginalized posterior
    dict(alpha=0.08*np.pi/2, B4=158.0, Delta0=157.0, m_s=100.0)   # maximum posterior
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
sigma_list = [50.0, 100.0, 150.0, 200.0]

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
    verbose=True,
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
            verbose=False,
        )

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
# Critical surface tension at the PNS centre (IV.1–IV.3) and the paper figures
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
#  sigma_crit engine binding.  (cheap — run once; required by IV.1–IV.3)
# =============================================================================

# ---- user-configurable inputs ----------------------------------------------
# Star match (Y_L, S) and CFL-filter acceptance constants.
YLH, S       = 0.25, 2.0        # trapped/isentropic PNS the σ_crit is evaluated at
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
_filt_cfg = nuc_an.FilterConfig(
    P_H_of_muB=P_H_of_muB, mu_B_H_sorted=mu_B_H_sorted, P_H_sorted=P_H_sorted,
    m_s=m_s_fixed, n_B_grid=n_B_grid_cfl, e_c_vec_tov=e_c_vec_tov,
    M_max_window=M_max_window, e_over_nB_max=e_over_nB_max)
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

def _run_scan(MT0s, fl, ch, ph, do_plot=True):
    """CFL filter (cached) + σ_crit per MT0; save each grid to .npz; optional plot."""
    res = nuc_an.run_sigma_crit_scan(
        MT0s, fl, ch, ph, alpha_slices, B4_grid_scan, Delta0_grid_scan,
        _filt_cfg, _nuc_cfg, _star, n_jobs=scan_n_jobs, reuse_filter=True,
        xsd_tag=xsd_tag, extra_save=dict(YLH=YLH, S=S),
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
               'rehadr': 're-hadronizes', 'mmax': 'M_max outside window',
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
# ## IV.1 — $\sigma_{\rm crit}$ at a star's central conditions
#
# $\sigma$ is the key unknown of the nucleation problem: larger $\sigma$ ⇒ taller
# barrier ⇒ longer $\tau$ (monotone). The **critical** surface tension
# $\sigma_{\rm crit}$ is the value at which $\tau=\tau_{\rm target}$ at the stellar
# centre — any $\sigma<\sigma_{\rm crit}$ nucleates in time, any larger does not.
#
# The centre is that of a trapped/isentropic star whose **baryon number** equals a
# cold ($T=0$) β-eq star of gravitational mass $M_{T0}$:
# $M_{T0}\!\to\!M_b$ (cold TOV) $\to n_B^{Hc}$ (trapped TOV) $\to T_c$ (isentrope)
# $\to$ `brentq` on $\log_{10}\tau-\log_{10}\tau_{\rm target}$.

# %%
# =============================================================================
#  sigma_crit at the central (nBHc, YLH, T_c) of a trapped/isentropic star whose
#  baryonic mass matches a cold (T=0) beta-eq star of gravitational mass MT0.
#  Uses nuc_an.central_state / sigma_target_pt directly (engine cell bound _star).
# =============================================================================
# ---- knobs ------------------------------------------------------------------
MT0                  = 1.4          # cold beta-eq gravitational mass [M_sun]
use_set_index        = 2           # index into quark_param_sets (see I.3)
quark_phase          = 'unpCFL'     # 'unpaired' | 'cfl' | 'unpCFL'
flavor_mode          = 'saddlepoint'  # 'frozen' | 'saddlepoint'
electric_charge_mode = 'coulomb_minimize'  # 'lcn' | 'gcn' | 'coulomb_minimize'
# -----------------------------------------------------------------------------

_qp = quark_param_sets[use_set_index]
params_q = get_alphabag_custom(alpha=_qp['alpha'], B4=_qp['B4'], m_s=_qp['m_s'])
Delta0_q = _qp['Delta0']
print(f"quark set {q_tag_of(_qp)}  ->  {flavor_mode}/{electric_charge_mode}/{quark_phase}")

# Central state of the matched star (nB, T, hadronic background H_pt).
nBHc, T_c, H_pt = nuc_an.central_state(MT0, _star)
print(f"MT0={MT0:.3f}:  nBHc={nBHc:.4f} fm^-3,  T_c={T_c:.3f} MeV")

# tau(sigma) at the centre, and the sigma where tau = tau_target.
def tau_at(sigma):
    """Central tau at one sigma (handles unpaired / cfl / unpCFL)."""
    return nuc_an.tau_pt(sigma, H_pt, T_c, flavor_mode, electric_charge_mode,
                         quark_phase, {}, params_q, Delta0_q, _nuc_cfg)

print(f"tau(sigma={sig_lo:6.2f}) = {tau_at(sig_lo):.3e} s")
print(f"tau(sigma={sig_hi:6.2f}) = {tau_at(sig_hi):.3e} s")

sigma_target = nuc_an.sigma_target_pt(H_pt, T_c, flavor_mode, electric_charge_mode,
                                      quark_phase, params_q, Delta0_q, _nuc_cfg)
if np.isfinite(sigma_target):
    print(f"\n=> sigma_target = {sigma_target:.4f} MeV/fm^2   (tau = {tau_at(sigma_target):.3e} s)")
elif np.isposinf(sigma_target):
    print(f"\n=> nucleates for ALL sigma up to {sig_hi:g}: sigma_target > sigma_hi")
else:
    print(f"\n=> no nucleation within [{sig_lo:g}, {sig_hi:g}] MeV/fm^2 (sigma_target = NaN)")


# %% [markdown]
# ### Paper figure — $\tau(\sigma)$ at the PNS centre with $\sigma_{\rm crit}$
#
# The physics of IV.1 as one publication panel: the central nucleation time
# $\tau(\sigma)$ for the three droplet phases, the $\tau=\tau_{\rm target}$
# horizon, and a star at each phase's $\sigma_{\rm crit}$. Saved as PDF to
# `../output/figures/`.

# %%
# ============================================================================
#  Paper figure: tau(sigma) at the star centre, one curve per droplet phase.
# ============================================================================
# ---- user-configurable inputs ----------------------------------------------
F0_MT0     = MT0                     # cold-star mass whose centre is probed
F0_PHASES  = [('unpaired', '#e08214', ':',  'unpaired'),   # (phase, colour, ls, label)
              ('cfl',      '#2c7fb8', '--', 'CFL'),
              ('unpCFL',   '#d7301f', '-',  'unpCFL')]
F0_SIGMAS  = np.linspace(max(sig_lo, 1.0), sig_hi, 60)   # sigma grid [MeV/fm^2]
F0_SAVE    = f'../output/figures/tau_sigma_centre_{xsd_tag}_{q_tag_of(_qp)}.pdf'
# ---- computation ------------------------------------------------------------
# tau(sigma) per phase; a fresh cache per phase lets lcn/gcn reuse the solve.
_f0_tau = {}
for _ph, _, _, _ in F0_PHASES:
    _cache = {}
    _f0_tau[_ph] = np.array([
        nuc_an.tau_pt(s, H_pt, T_c, flavor_mode, electric_charge_mode,
                      _ph, _cache, params_q, Delta0_q, _nuc_cfg)
        for s in F0_SIGMAS])
# sigma_crit per phase via the canonical root-find (matches the IV.3 heatmaps).
_f0_sc = {_ph: nuc_an.sigma_target_pt(H_pt, T_c, flavor_mode, electric_charge_mode,
                                      _ph, params_q, Delta0_q, _nuc_cfg)
          for _ph, _, _, _ in F0_PHASES}
# ---- layout ------------------------------------------------------------------
set_paper_style()
fig, ax = plt.subplots(figsize=(5.4, 4.4), constrained_layout=True)
for _ph, _col, _ls, _lbl in F0_PHASES:
    _t = _f0_tau[_ph]
    with np.errstate(divide='ignore', invalid='ignore'):
        _y = np.log10(_t)
    _m = np.isfinite(_y)
    ax.plot(F0_SIGMAS[_m], _y[_m], color=_col, ls=_ls, lw=2.0, label=_lbl)
    _sc = _f0_sc[_ph]
    if np.isfinite(_sc):                             # sigma_crit marker on the horizon
        ax.plot(_sc, np.log10(tau_target), '*', ms=15, color=_col,
                mec='k', mew=0.5, zorder=6)
ax.axhline(np.log10(tau_target), color='0.25', ls=(0, (1, 1)), lw=1.0)
ax.text(0.985, np.log10(tau_target), rf'$\tau={tau_target*1e3:g}$ ms  ',
        transform=ax.get_yaxis_transform(), ha='right', va='bottom', fontsize=11,
        color='0.25')
ax.set_xlim(F0_SIGMAS[0], F0_SIGMAS[-1])
ax.set_xlabel(r'$\sigma$ [MeV fm$^{-2}$]')
ax.set_ylabel(r'$\log_{10}\,\tau$ [s]')
ax.legend(loc='lower right', title=(rf'$M_{{T0}}={F0_MT0:g}\,M_\odot$,'
                                    rf'  $Y_L={YLH:g}$, $S={S:g}$'))
fig.savefig(F0_SAVE)
print(f"saved -> {F0_SAVE}")
plt.show()


# %% [markdown]
# ## IV.2 — $\sigma_{\rm crit}$ table: $M_{T0}$ × method, per quark set
#
# $\sigma_{\rm crit}$ for a few $M_{T0}$ across the methods in `_sig_methods`
# (including the two-layer unpCFL droplet), for every set in `quark_param_sets`.
# Reuses the IV.1 pipeline; the central states are $M_{T0}$-only (hadronic EoS is
# fixed) so they are computed once.

# %%
# =============================================================================
# Critical surface tension table: sigma_target over MT0 x method, for each quark
# parametrization, at the trapped/isentropic star centre (YLH, S, tau from above).
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
#  Grid knobs + one baseline scan.  (heavy: coulomb_minimize re-solves per σ)
# =============================================================================
# ---- knobs ------------------------------------------------------------------
scan_flavor, scan_charge, scan_phase = 'saddlepoint', 'coulomb_minimize', 'unpCFL'
MT0_grid_thr = [1.4]                                  # nucleation threshold(s)
alpha_slices     = [0.1 * np.pi / 2, 0.2 * np.pi / 2, 0.3 * np.pi / 2]   # α_s panels
B4_grid_scan     = np.linspace(130.0, 180.0, 51)         # MeV (B^1/4; 18.0 was a typo)
Delta0_grid_scan = np.linspace(0.0, 200.0, 51)           # MeV
scan_n_jobs      = -1                                    # joblib workers (-1 = all)
# -----------------------------------------------------------------------------

results_scan = _run_scan(MT0_grid_thr, scan_flavor, scan_charge, scan_phase,
                         do_plot=True)  # CFL filter scanned once, then reused


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
# ### Run + save all coulomb_minimize phases (unpaired / CFL / unpCFL)
#
# Loop the three saddlepoint coulomb_minimize phases. The CFL filter is scanned
# once (cached on the grid) and reused across phases; each is saved to its own
# `.npz`. Slow — unpCFL re-solves the droplet twice per σ.

# %%
# =============================================================================
#  Run + save all coulomb_minimize phases at MT0_grid_thr (grid from the cell above).
# =============================================================================
all_cases = [
    ('saddlepoint', 'coulomb_minimize', 'unpCFL'),
    ('saddlepoint', 'coulomb_minimize', 'unpaired'),
    ('saddlepoint', 'coulomb_minimize', 'cfl'),
]
results_all = {}
for fl, ch, ph in all_cases:
    print(f"\n=== {fl}/{ch}/{ph}  @ MT0={MT0_grid_thr} ===", flush=True)
    results_all[(fl, ch, ph)] = _run_scan(MT0_grid_thr, fl, ch, ph, do_plot=True)
print("\nAll cases done. Saved sigma_crit_grid_*_coulomb_minimize-*.npz in ../output/mc_cfl/.")


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
F1_SET    = quark_param_sets[0]                 # which (alpha, B4, Delta0) set
F1_FLAVOR = 'saddlepoint'                        # 'frozen' | 'saddlepoint'
F1_CHARGE = 'coulomb_minimize'                   # 'lcn' | 'gcn' | 'coulomb_minimize'
F1_YLH    = 0.25
F1_SIGMA  = 100.0
F1_TW     = 30.0                                 # panel (a) temperature [MeV]
F1_DENS   = [1.0, 3, 7]                      # panel (a) n_B^H / n_0
F1_TEMPS  = [10.0, 20.0, 30.0, 40.0, 50.0]       # panels (b,c,d) temperatures [MeV]

_PHASE_LS    = {'unpCFL': '-', 'cfl': '--', 'unpaired': ':'}   # line style per phase
_PHASE_FILL  = {'unpCFL': True, 'cfl': False, 'unpaired': False}
_PHASE_LBL   = {'unpCFL': 'unpCFL', 'cfl': 'CFL', 'unpaired': 'unpaired'}
_PHASE_LW    = {'unpCFL': 2.6, 'cfl': 1.7, 'unpaired': 1.7}    # solid (unpCFL) drawn thicker
_PHASE_ALPHA = {'unpCFL': 1.0, 'cfl': 0.6, 'unpaired': 0.55}   # fade dashed/dotted phases

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


fig, ((axA, axB), (axC, axD)) = plt.subplots(2, 2, figsize=(9.5, 9.0),
                                             constrained_layout=True)

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
axA.set_xlim(0, 8); axA.set_ylim(-0.2 * _Wmax, 1.18 * _Wmax)
axA.set_xlabel(r'$R$ [fm]'); axA.set_ylabel(r'$W$ [MeV]')
axA.set_title(rf'$Y_L^H={F1_YLH}$, $T={F1_TW:.0f}$ MeV, '
              rf'$\sigma={F1_SIGMA:.0f}$ MeV/fm$^2$')
panel_label(axA, '(a)', corner='lower')
_lgA = axA.legend([Line2D([], [], color=_cA[i], lw=2) for i in range(len(F1_DENS))],
                  [rf'$n_B^H/n_\mathrm{{sat}}={int(x)}$' for x in F1_DENS], loc='upper left')
axA.add_artist(_lgA)
# phase legend in the opposite (also empty) top corner -> no overlap with density legend or peak
axA.legend([Line2D([], [], color='0.3', ls=_PHASE_LS[p]) for p in _PHASE_LS],
           [_PHASE_LBL[p] for p in _PHASE_LS], loc='upper right')

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
axB.set_ylim(1, 8); axB.set_title(rf'$Y_L^H={F1_YLH}$, $\sigma={F1_SIGMA:.0f}$ MeV/fm$^2$')
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

plt.show()

# ---- Alternative panel (d): log10 tau vs T (density = colour, phase = style).
#      Both x-axis choices are produced; pick whichever reads better for the paper.
figD, axDt = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
for _ci, _x in enumerate(F1_DENS):
    _i = int(np.argmin(np.abs(_nBg / n_sat - _x)))
    for _ph, _ls in _PHASE_LS.items():
        _o = _f1_get(_ph)
        if _o is None:
            continue
        with np.errstate(divide='ignore', invalid='ignore'):
            axDt.plot(_Tg, np.log10(_o.tau[_i, _iYL, :]), color=_cA[_ci], ls=_ls,
                      lw=_PHASE_LW[_ph], alpha=_PHASE_ALPHA[_ph])
axDt.axhline(np.log10(1e-3), color='k', ls=(0, (1, 1)), lw=0.9)
axDt.set_xlabel(r'$T$ [MeV]'); axDt.set_ylabel(r'$\log_{10}\,\tau$ [s]')
axDt.set_xlim(0, 80); axDt.set_ylim(-60, 60)
axDt.set_title(rf'$Y_L^H={F1_YLH}$, $\sigma={F1_SIGMA:.0f}$ MeV/fm$^2$')
axDt.legend([Line2D([], [], color=_cA[i], lw=2) for i in range(len(F1_DENS))]
            + [Line2D([], [], color='0.3', ls=_PHASE_LS[p]) for p in _PHASE_LS],
            [rf'$n_B^H/n_\mathrm{{sat}}={int(x)}$' for x in F1_DENS] + [_PHASE_LBL[p] for p in _PHASE_LS],
            loc='upper right', ncol=2)
plt.show()


# %% [markdown]
# ### Paper Figure 1 (σ-sweep) — same panels but T fixed, coloured by σ
#
# Mirror of Fig 1 with the roles of $T$ and $\sigma$ swapped: fix $T$, sweep
# $\sigma$ (colour). **(a)** $W(R)$ at one $n_B^H$; **(b,c,d)** $R_*$, $W_*/T$,
# $\log_{10}\tau$ vs $n_B^H/n_\mathrm{sat}$.  Reuses the Fig-1 knobs/helpers
# (`_f1_get`, `_f1_Rx`, `_iT`, `_nBg`, `_Tg`, `_iYL`, `_params`, `_PHASE_*`), so
# run the Figure-1 cell first.  Tables must exist for each σ in `F1S_SIGMAS`.

# %%
# ============================================================================
#  Figure 1 variant: fix T, sweep σ (colour = σ).  Panels mirror Fig 1(a-d).
# ============================================================================
F1S_T      = 30.0                      # fixed temperature [MeV]
F1S_SIGMAS = sigma_list                # σ values to sweep (colour); tables must exist
F1S_DENS_A = 3.0                       # panel (a): single n_B^H/n_sat for the W(R) sweep

_cS = plt.cm.viridis(np.linspace(0.12, 0.85, len(F1S_SIGMAS)))

fig, ((axA, axB), (axC, axD)) = plt.subplots(2, 2, figsize=(9.5, 9.0),
                                             constrained_layout=True)

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
F2C_YL, F2C_T   = 0.25, 30.0                     # panel (c)/ref: fixed lepton frac, T [MeV]
F2C_SIGMA       = 150.0                          # panel (c)/ref: sigma for coulomb_minimize
_QS_GREENS = ['#74c476', '#31a354', '#006d2c']   # QS star colours, per quark set (light->dark)

# larger in-plot fonts (persists for later figures too; move to the import cell for all).
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 15, 'axes.titlesize': 14,
                     'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 10})

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
        add_crust_table='No', compute_baryonic_mass=True, compute_tidal=False, verbose=False)
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
         cdot=(0, -9, 'center', 'top'),   cstar=(6, -3, 'left', 'top'),   # below line (green sits above)
         ddot=(-8, 2, 'right', 'bottom'),  dstar=(-6, -2, 'right', 'top')),
]
for _i, (_p, _st) in enumerate(_qs_tov):
    _F2SEQ.append(dict(
        arr=_st, c='#238b45', lbl='QS',
        a=(0.85, 0.15, 0.08, 'left', 'bottom'),
        b=(0.9, 0.05, -0.04, 'left', 'top'),
        d=(0.92, 0.03, 0.2, 'left', 'bottom'),
        cdot=(0, 8, 'center', 'bottom'),  cstar=(7, 3, 'left', 'bottom'),
        ddot=(0, -9, 'center', 'top'),   dstar=(7, 0, 'left', 'center')))
_F2SEQ += [
    dict(arr=tov_trapped[(0.35, 1.5)], c='#fd8d3c', lbl=r'PNS ($t_0$)', yls=(0.35, 1.5),
         a=(0.92,  0.35, -0.02, 'left',  'top'),
         b=(0.92,  0.05, -0.10, 'left',  'top'),
         pc=(0.6,  0.02, -1.8,  'left',  'top'),
         d=(0.5,   0.03, -0.5,  'left',  'top'),
         cdot=(0, -9, 'center', 'top'),   cstar=(8, -3, 'left', 'top'),
         ddot=(8, -1, 'left', 'top'),     dstar=(7, -2, 'left', 'top')),
    dict(arr=tov_trapped[(0.25, 2.0)], c='#e31a1c', lbl=r'PNS ($t_{T_\mathrm{max}}$)', yls=(0.25, 2.0),
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
    ax.text(xs[j] + dx, ys[j] + dy, text, color=color, fontsize=12.5, fontweight='bold',
            ha=ha, va=va, zorder=6, clip_on=True)


_MDOTS = (1.0, 1.2, 1.4, 1.6)          # gravitational masses [M_sun] to mark (dots)


def _mass_marks(ax, arr, yvals, color, dot_off=(0, 7, 'center', 'bottom'),
                star_off=(6, 0, 'left', 'center'), label=True):
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
                            color=color, zorder=8)
    ax.plot(Mb[-1], yv[-1], '*', ms=13, color=color, mec='white', mew=0.6, zorder=7)  # M_max
    if label:
        _sx, _sy, _sha, _sva = star_off
        ax.annotate(r"$M_{\max}$", (Mb[-1], yv[-1]), textcoords='offset points',
                    xytext=(_sx, _sy), ha=_sha, va=_sva, fontsize=8.5, color=color, zorder=8)


def _grid(ax):
    """Gridlines disabled (uncomment the body to restore light gray gridlines)."""
    # ax.grid(True, color='gray', alpha=0.25, lw=0.6, zorder=0)
    pass


def _qs_P(nB_arr, flavor, charge, phase, params, Delta0,
          YL=F2C_YL, T=F2C_T, sigma=F2C_SIGMA):
    """Q* droplet total pressure P_total(n_B^H) for one (flavor, charge, phase, set)."""
    kw = dict(quark_phase=phase, Delta0=Delta0,
              include_photons=True, include_gluons=True, include_thermal_neutrinos=True)
    if charge == 'coulomb_minimize':
        kw['sigma'] = sigma                       # required by the coulomb_minimize solver
    solver = get_solver_Qs(flavor, charge, params, **kw)
    P = np.full(nB_arr.shape, np.nan)
    for _i, _nB in enumerate(nB_arr):
        _out = solver(nuc_an.hadronic_point(H['trapped'], _nB, YL, T))
        _Qs = _out[0] if isinstance(_out, tuple) else _out
        if _Qs is not None:
            P[_i] = getattr(_Qs, 'P_total', np.nan)
    return P


fig, ((axA, axB), (axC, axD)) = plt.subplots(2, 2, figsize=(9.5, 9.0),
                                             constrained_layout=True)

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
_grid(axA); panel_label(axA, '(a)')

# (b) M vs M_baryonic.  Carries the shared curve legend (NS / QS / PNS) for all panels.
for _s in _F2SEQ:
    _a = _stable(_s['arr'])
    axB.plot(_a[:, 5], _a[:, 4], color=_s['c'], lw=2.4, label=_s['lbl'])
axB.set_xlabel(r'$M_B$ [$M_\odot$]'); axB.set_ylabel(r'$M$ [$M_\odot$]')
axB.set_xlim(0.6, 3.4); axB.set_ylim(0.6, 2.6)
axB.legend(loc='lower right', fontsize=10, frameon=False)
_grid(axB); panel_label(axB, '(b)')

# (c) central temperature T_c vs baryonic mass M_B.  NS/QS are cold: drawn as a
#     coloured T_c=0 horizontal line over their M_B span; the two PNS sequences
#     carry T_c(n_B^c) from their isentrope.  Every curve gets M-dot + M_max-star.
for _s in _F2SEQ:
    _a = _stable(_s['arr'])
    if _s.get('yls') is None:                       # cold NS/QS -> T_c=0 line over its M_B span
        _mb = _a[:, 5]
        axC.plot([_mb.min(), _mb.max()], [0.0, 0.0], color=_s['c'], lw=2.4)
        _mass_marks(axC, _a, np.zeros(len(_a)), _s['c'], _s['cdot'], _s['cstar'])
        continue
    _yl, _S = _s['yls']
    _Tc = np.asarray(H['iso_trapped']['T'](_a[:, 2], _yl, _S))    # central T on the isentrope
    axC.plot(_a[:, 5], _Tc, color=_s['c'], lw=2.4)
    _mass_marks(axC, _a, _Tc, _s['c'], _s['cdot'], _s['cstar'])
axC.set_xlabel(r'$M_B$ [$M_\odot$]'); axC.set_ylabel(r'$T_c$ [MeV]')
axC.set_xlim(0.6, 3.4); axC.set_ylim(bottom=-1)
_grid(axC); panel_label(axC, '(c)')

# (d) central density vs baryonic mass; every curve gets M-dot + M_max-star labels.
for _s in _F2SEQ:
    _a = _stable(_s['arr'])
    _y = _a[:, 2] / n_sat
    axD.plot(_a[:, 5], _y, color=_s['c'], lw=2.4)
    _mass_marks(axD, _a, _y, _s['c'], _s['ddot'], _s['dstar'])
axD.set_xlabel(r'$M_B$ [$M_\odot$]'); axD.set_ylabel(r'$n_B^c/n_\mathrm{sat}$')
axD.set_xlim(0.6, 3.4); _grid(axD); panel_label(axD, '(d)')

plt.show()

# %%
# ---- Reference only: Q* pressure vs n_B^H across charge/flavor prescriptions ----
#      saddlepoint {lcn, gcn, coulomb_minimize} + frozen lcn, for unpaired & CFL.
#      (frozen has no CFL phase -> that combo is skipped.)
_nBr = np.linspace(0.16, 1.2, 30)
_REF = [('saddlepoint', 'lcn',              '#1f77b4', 'saddle / LCN'),
        ('saddlepoint', 'gcn',              '#2ca02c', 'saddle / GCN'),
        ('saddlepoint', 'coulomb_minimize', '#d62728', 'saddle / Coul-min'),
        ('frozen',      'lcn',              '#9467bd', 'frozen / LCN')]
_REF_PH = {'unpaired': '-', 'cfl': '--'}

figR, axR = plt.subplots(figsize=(6.4, 5.0), constrained_layout=True)
axR.plot(_nBr / n_sat, [float(H['trapped']['P'](_nB, F2C_YL, F2C_T)) for _nB in _nBr],
         color='0.15', lw=2.6)
for _flav, _chg, _col, _lbl in _REF:
    for _ph, _ls in _REF_PH.items():
        if _flav == 'frozen' and _ph == 'cfl':
            continue
        axR.plot(_nBr / n_sat, _qs_P(_nBr, _flav, _chg, _ph, _f2params, F2_SET['Delta0']),
                 color=_col, ls=_ls, lw=1.8, alpha=(1.0 if _ls == '-' else 0.65))
axR.set_xlabel(r'$n_B^H/n_\mathrm{sat}$'); axR.set_ylabel(r'$P$ [MeV/fm$^3$]')
axR.set_title(rf'$Q^*$ pressure — $Y_L^H={F2C_YL}$, $T={F2C_T:.0f}$ MeV, '
              rf'$\sigma={F2C_SIGMA:.0f}$ MeV/fm$^2$')
axR.legend([Line2D([], [], color='0.15', lw=2.6)]
           + [Line2D([], [], color=_c, lw=2) for _, _, _c, _ in _REF]
           + [Line2D([], [], color='0.3', ls=_ls) for _ls in _REF_PH.values()],
           ['H (trapped)'] + [_l for _, _, _, _l in _REF] + ['unpaired', 'CFL'],
           fontsize=8, ncol=2, loc='upper left')
plt.show()


# %% [markdown]
# ### Paper Figure 6 — $T_{\rm nuc}(n_B^H)$ nucleation conditions
#
# $\tau=\tau_{\rm target}$ curves in $(n_B^H, T)$, one line per $\sigma$ (colour)
# and phase (line style). Black = stellar isentrope; dots on it mark the PNS
# central density (filled: fixed $M$; open: $M_b$-matched to a cold star).
# **(a)** $S=1, Y_L=0.35$   **(b)** $S=2, Y_L=0.25$.

# %%
# ============================================================================
#  Figure 6.  T_nuc(n_B) nucleation-condition curves — two (Y_L, S) panels.
#    colour = sigma;  line style: unpaired ':', CFL '--', unpCFL '-'.
#    red dash-dot = stellar isentrope T(n_B) at (Y_L, S).
#    Dots ON the isentrope mark the PNS central density n_Bc:
#      filled = PNS(Y_L,S) at gravitational mass M = 1,1.3,1.4,1.5, M_max;
#      open   = PNS(Y_L,S) whose baryonic mass = Mb of a COLD (T=0) star with
#               M(T0) = 1,1.3,1.4,1.5  (the same star, hot proto-NS vs cold).
# ============================================================================
set_paper_style()

# ---- knobs ----
FN_SET    = quark_param_sets[0]          # quark parametrization
FN_FLAVOR = 'saddlepoint'
FN_CHARGE = 'gcn'
FN_TAU    = 1e-3                         # s  (tau_target)
FN_CUT_CFL_ABOVE_TC = True               # mask the CFL curve where T_nuc > T_CFL (gap vanishes above Tc)
M_GRAV    = [1.0,  1.4]         # filled dots (+ M_max) — grav. mass
M_COLD    = [1.0, 1.4]         # open dots — cold-star grav. mass, Mb-matched
_tag = q_tag_of(FN_SET)
from eos.alphabag.thermodynamics_quarks import T_critical   # same Tc as the CFL EoS gap
_T_CFL = float(T_critical(FN_SET['Delta0']))   # CFL Tc = 0.57*2^(1/3)*Delta0 [MeV] (matches gap_cfl)

_panels   = [dict(YL=0.35, S=1.0, lab='(a)'),    # left
             dict(YL=0.25, S=2.0, lab='(b)')]    # right
_phase_ls = [('unpaired', ':'), ('cfl', '--'), ('unpCFL', '-')]

# sigma -> colour (viridis over the available sigma_list)
_sig_col = {sg: mpl.cm.viridis(t) for sg, t in
            zip(sigma_list, np.linspace(0.15, 0.85, len(sigma_list)))}

# TOV columns: 0=e_c 1=P_c 2=n_Bc 3=R 4=M 5=Mb.
_NBC, _M, _MB = 2, 4, 5

def _branch_interp(arr, xcol, ycol):
    """interp ycol(xcol) along a TOV sequence's stable branch (up to M_max)."""
    i = np.argmax(arr[:, _M]) + 1
    return interp1d(arr[:i, xcol], arr[:i, ycol], kind='cubic',
                    bounds_error=False, fill_value=np.nan)

def _tov_trapped_seq(YL, S):
    """Nearest trapped/isentropic TOV sequence to (YL, S)."""
    key = min(tov_trapped, key=lambda k: abs(k[0] - YL) + abs(k[1] - S))
    return tov_trapped[key]

fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), sharey=True)
for ax, pan in zip(axes, _panels):
    YL, S = pan['YL'], pan['S']
    YL_used = None
    for sg in sigma_list:
        for ph, ls in _phase_ls:
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
                m &= (T <= _T_CFL)                    # CFL undefined above Tc -> drop those points
            if m.any():
                ax.plot(nB[m] / n_sat, T[m], color=_sig_col[sg], ls=ls, lw=1.6)

    ax.set_xlim(0.5, 12); ax.set_ylim(0, 100)     # T [MeV], n_B^H/n_sat

    # ---- isentrope + PNS central-density markers (on the isentrope) ----
    YL_iso = YL if YL_used is None else float(YL_used)
    T_iso  = lambda nBc: H['iso_trapped']['T'](nBc, YL_iso, S)   # T at a density

    nB_iso = np.linspace(1, 10, 200) * n_sat
    ax.plot(nB_iso / n_sat, T_iso(nB_iso), color='red', ls='-.', lw=1.3,
            label=rf'isentrope $S={S:g}$', zorder=3)

    tov_tr = _tov_trapped_seq(YL_iso, S)

    # filled: PNS at gravitational mass M (+ M_max at the sequence's tip)
    nBc_M   = _branch_interp(tov_tr, _M, _NBC)
    nBc_fil = [float(nBc_M(M)) for M in M_GRAV]
    i_mx    = int(np.argmax(tov_tr[:, _M]))
    nBc_fil.append(float(tov_tr[i_mx, _NBC]))              # M_max
    nBc_fil = np.array(nBc_fil)
    ax.scatter(nBc_fil / n_sat, T_iso(nBc_fil), s=34, c='k', zorder=5)

    # open: PNS whose baryonic mass = Mb of the cold (T=0) star at M(T0)
    Mb_of_M   = _branch_interp(tov_cold, _M, _MB)          # cold: M -> Mb
    nBc_of_Mb = _branch_interp(tov_tr,  _MB, _NBC)         # trapped: Mb -> n_Bc
    nBc_open  = np.array([float(nBc_of_Mb(Mb_of_M(M0))) for M0 in M_COLD])
    ax.scatter(nBc_open / n_sat, T_iso(nBc_open), s=34,
               facecolors='none', edgecolors='k', linewidths=1.2, zorder=5)

    ax.set_xlabel(r'$n_B^H / n_{\rm sat}$')
    ax.set_title(rf"$Y_L\approx{YL_iso:.2f}$, $S={S:g}$")
    ax.set_box_aspect(1)                                   # square panels
    panel_label(ax, pan['lab'])
axes[0].set_ylabel(r'$T_{\rm nuc}$ [MeV]')

# Legends: phase (line style) + markers on the left; sigma (colour) on the right.
_ph_handles = [Line2D([], [], color='k', ls=ls, label=lbl)
               for lbl, ls in [('unpaired', ':'), ('CFL', '--'), ('unpCFL', '-')]]
_mk_handles = [
    Line2D([], [], marker='o', color='k', ls='none',
           label=r'PNS $M$=1,1.3,1.4,1.5,max'),
    Line2D([], [], marker='o', mfc='none', mec='k', color='k', ls='none',
           label=r'PNS $M_b\!=\!M_b(M_{T0})$'),
]
_sig_handles = [Line2D([], [], color=_sig_col[sg], ls='-',
                       label=rf'$\sigma={int(sg)}$') for sg in sigma_list]
leg1 = axes[0].legend(handles=_ph_handles, loc='upper left', title='phase')
axes[0].add_artist(leg1)
axes[0].legend(handles=_mk_handles, loc='lower right', fontsize=10)
axes[1].legend(handles=_sig_handles, loc='upper right',
               title=r'$\sigma$ [MeV/fm$^2$]')

fig.suptitle(rf"$\tau={FN_TAU*1e3:g}$ ms — {FN_FLAVOR}/{FN_CHARGE} — {_tag}", y=1.02)
fig.tight_layout(); plt.show()


# %% [markdown]
# ### Paper Figure 7 — $R^*$, $W^*/T$, $\log_{10}\tau$ at the PNS centre vs $M_{\rm PNS}$
#
# Critical droplet quantities along the trapped/isentropic sequence at fixed
# $(Y_L, S)$. Rows = observable, columns = quark set, colour = $\sigma$, line style
# = phase. Vertical guides: fixed $M_{\rm PNS}$ (dashed) and $M_b$-matched to a cold
# star (dotted).

# %%
# ============================================================================
#  Figure 7.  R*, W*/T and log10(tau) at the PNS centre vs M_PNS.
#    rows = observable;  columns = quark parametrization.
#    colour = sigma;  line style: unpaired ':', CFL '--', unpCFL '-'.
#  Vertical guides: dashed = M_PNS = 1, 1.4;  dotted = the M_PNS whose baryonic
#  mass equals Mb of a COLD (T=0) star with M(T0) = 1, 1.4 (same star, cooled).
#  The (M_PNS -> n_Bc -> T_c) map is hadronic-only: built ONCE, reused.
# ============================================================================
set_paper_style()

# ---- knobs ----
FN2_YL, FN2_S       = 0.25, 2.0          # trapped-PNS (Y_L, S)
FN2_FLAVOR          = 'saddlepoint'
FN2_CHARGE          = 'coulomb_minimize'
FN2_TAU             = 1e-3               # s (target line on the log10 tau row)
FN2_MVERT           = [1.0, 1.4]         # masses for the vertical guides
_phase_ls = [('unpaired', ':'), ('cfl', '--'), ('unpCFL', '-')]
_sig_col  = {sg: mpl.cm.viridis(t) for sg, t in
             zip(sigma_list, np.linspace(0.15, 0.85, len(sigma_list)))}

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

_tags = [q_tag_of(p) for p in quark_param_sets]

# rows: (label, y-array from a nuc interpolator dict, yscale)
_rows = [
    (r'$R^*$ [fm]',
     lambda itp: np.array([itp['R_c'](nb, _YL, T) for nb, T in zip(nBc_seq, T_seq)]),
     'linear', (0, 8)),
    (r'$W^*/T$',
     lambda itp: np.array([itp['W_c'](nb, _YL, T) for nb, T in zip(nBc_seq, T_seq)]) / T_seq,
     'linear', (0, 1000)),
    (r'$\log_{10}\,\tau$ [s]',
     lambda itp: np.array([itp['log10_tau'](nb, _YL, T) for nb, T in zip(nBc_seq, T_seq)]),
     'linear', None),
]

nrow, ncol = len(_rows), len(_tags)
fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.3 * nrow),
                         sharex='col', sharey='row', squeeze=False)
for j, tag in enumerate(_tags):
    _a = next((nuc_sets[k] for k in nuc_sets if k.endswith(f"_{tag}_s{int(sigma_list[0])}")), None)
    _YL = (float(_a.hadronic_grids['Y_L_H'][
               np.argmin(np.abs(_a.hadronic_grids['Y_L_H'] - FN2_YL))])
           if _a is not None else FN2_YL)
    for sg in sigma_list:
        for ph, ls in _phase_ls:
            stem = f"Htrapped_{FN2_FLAVOR}_{FN2_CHARGE}_{ph}_{tag}_s{int(sg)}"
            if stem not in nuc_sets:
                continue
            itp = build_thermal_nucleation_interpolators(nuc_sets[stem])
            for i, (_, getter, _sc, _yl) in enumerate(_rows):
                axes[i, j].plot(M_seq, getter(itp),
                                color=_sig_col[sg], ls=ls, lw=1.6)
    axes[0, j].set_title(tag, fontsize=11)
    axes[-1, j].set_xlabel(r'$M_{\rm PNS}\ [M_\odot]$')

for i, (ylab, _, sc, ylim) in enumerate(_rows):
    axes[i, 0].set_ylabel(ylab)
    if ylim is not None:
        axes[i, 0].set_ylim(*ylim)          # shared across the row
    for j in range(ncol):
        ax = axes[i, j]
        ax.set_yscale(sc); ax.grid(alpha=0.3); ax.set_xlim(0.6, M_max)
        for mv in FN2_MVERT:                       # dashed: fixed M_PNS
            ax.axvline(mv, color='0.35', ls='--', lw=1.0, zorder=1)
        for md in M_dot:                           # dotted: Mb-matched to cold
            if np.isfinite(md):
                ax.axvline(md, color='0.35', ls=':', lw=1.0, zorder=1)
        if ylab.startswith(r'$\log_{10}'):
            ax.axhline(np.log10(FN2_TAU), color='0.5', ls='--', lw=1.0)

# Legends: phase (line style), sigma (colour), and the vertical guides.
_ph = [Line2D([], [], color='k', ls=ls, label=lbl)
       for lbl, ls in [('unpaired', ':'), ('CFL', '--'), ('unpCFL', '-')]]
_sg = [Line2D([], [], color=_sig_col[sg], ls='-', label=rf'$\sigma={int(sg)}$')
       for sg in sigma_list]
_vg = [Line2D([], [], color='0.35', ls='--', lw=1.0,
              label=r'$M_{\rm PNS}=1,1.4$'),
       Line2D([], [], color='0.35', ls=':', lw=1.0,
              label=r'$M_b\!=\!M_b(M_{T0}{=}1,1.4)$')]
leg1 = axes[0, 0].legend(handles=_ph, loc='best', title='phase', fontsize=9)
axes[0, 0].add_artist(leg1)
axes[0, -1].legend(handles=_sg, loc='best', title=r'$\sigma$ [MeV/fm$^2$]', fontsize=9)
axes[-1, 0].legend(handles=_vg, loc='best', fontsize=8)

fig.suptitle(rf"PNS centre — $Y_L={FN2_YL:g}$, $S={FN2_S:g}$, "
             rf"{FN2_FLAVOR}/{FN2_CHARGE}", y=1.005)
fig.tight_layout(); plt.show()


# %% [markdown]
# ## IV.5 — Method comparison at fixed conditions
#
# Cross-method plots (saddle LCN / GCN / Coulomb-min + frozen LCN) built from
# the loaded `nuc_sets` tables and the engine's per-point droplet solver.
# Self-contained after Parts I + III + IV.0.

# %%
# =============================================================================
#  Shared selectors for IV.5: parametrization, Y_L slice, table lookup.
# =============================================================================
# ---- user-configurable inputs ----------------------------------------------
m5_set    = quark_param_sets[0]      # which (alpha, B4, Delta0, m_s) set
m5_YL     = 0.25                     # lepton fraction (nearest grid value used)
# -----------------------------------------------------------------------------
m5_tag = q_tag_of(m5_set)
_m5_ref = next(o for k, o in nuc_sets.items()
               if k.endswith(f"_{m5_tag}_s{int(sigma_list[0])}"))
m5_nBg = _m5_ref.hadronic_grids['n_B_H']
m5_Tg  = _m5_ref.hadronic_grids['T']
m5_iYL = int(np.argmin(np.abs(_m5_ref.hadronic_grids['Y_L_H'] - m5_YL)))
m5_YL_used = _m5_ref.hadronic_grids['Y_L_H'][m5_iYL]

def m5_get(flavor, charge, phase, sg):
    """nuc_sets table for one (flavor, charge, phase, sigma) of the IV.5 set."""
    return nuc_sets.get(f"Htrapped_{flavor}_{charge}_{phase}_{m5_tag}_s{int(sg)}")

# (flavor, charge, colour, label) — the four methods, from the I.4 table
# (single source; its per-method line style is unused here, σ sets the style).
_M4 = [(fl, ch, c, lbl) for fl, ch, c, _ls, lbl in _methods]
_M4_LS = ['-', '--']                       # line style per sigma
print(f"IV.5 selectors: set {m5_tag}, Y_L≈{m5_YL_used:.2f}")


# %% [markdown]
# ### $T_{\rm nuc}(n_B^H)$ across charge/flavor methods, 2 σ, one phase
#
# $T_{\rm nuc}$ at τ=`tau_target`, one droplet phase (`PT_PHASE`). **Colour =
# method, line style = σ**. frozen has only the unpaired phase, so it is
# skipped unless `PT_PHASE='unpaired'`.

# %%
# ============================================================================
#  T_nuc vs n_B^H/n_sat — colour = charge/flavor method, line style = σ.
# ============================================================================
# ---- user-configurable inputs ----------------------------------------------
PT_PHASE  = 'unpaired'                        # 'unpaired' | 'cfl' | 'unpCFL'
PT_SIGMAS = [sigma_list[0], sigma_list[2]]    # two σ (line style)
# ---- layout ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
for _fl, _ch, _col, _lbl in _M4:
    if _fl == 'frozen' and PT_PHASE != 'unpaired':
        continue                              # frozen supports only the unpaired phase
    for _si, _sig in enumerate(PT_SIGMAS):
        _o = m5_get(_fl, _ch, PT_PHASE, _sig)
        if _o is None:
            continue
        _res = compute_nucleation_density(_o, tau_target=tau_target, scan='n_B')
        _nB, _T = nucleation_curve(_res, m5_iYL)
        _m = np.isfinite(_nB) & np.isfinite(_T)
        if _m.any():
            ax.plot(_nB[_m] / n_sat, _T[_m], color=_col, ls=_M4_LS[_si % len(_M4_LS)],
                    lw=1.7, marker='.', ms=4)
ax.set_xlabel(r'$n_B^H/n_\mathrm{sat}$'); ax.set_ylabel(r'$T_{\rm nuc}$ [MeV]')
ax.set_title(rf"τ={tau_target*1e3:g} ms, $Y_L\approx{m5_YL_used:.2f}$, {PT_PHASE} — {m5_tag}")
ax.grid(alpha=0.3)
_lg1 = ax.legend([Line2D([], [], color=_c, lw=2) for _, _, _c, _ in _M4],
                 [_l for _, _, _, _l in _M4], loc='upper right', fontsize=8, title='method')
ax.add_artist(_lg1)
ax.legend([Line2D([], [], color='0.3', ls=_M4_LS[i]) for i in range(len(PT_SIGMAS))],
          [rf'$\sigma={int(s)}$' for s in PT_SIGMAS], loc='lower left', fontsize=8, title=r'$\sigma$')
plt.show()


# %% [markdown]
# ### $W(R)$ across charge/flavor methods, 2 σ, one phase
#
# Barrier $W(R)$ at fixed $(T, n_B^H)$, one phase (`PW_PHASE`). **Colour =
# method, line style = σ**; dot = critical point. frozen drawn only for the
# unpaired phase. $W(R)$ recomputed with `compute_energy_barrier`.

# %%
# ============================================================================
#  W(R) at fixed (T, n_B^H) — colour = charge/flavor method, line style = σ.
# ============================================================================
# ---- user-configurable inputs ----------------------------------------------
PW_PHASE  = 'unpaired'                         # 'unpaired' | 'cfl' | 'unpCFL'
PW_SIGMAS = [sigma_list[0], sigma_list[2]]     # two σ (line style)
PW_T      = 30.0                               # fixed temperature [MeV]
PW_NBH    = 3.0                                # fixed n_B^H / n_sat
# ---- computation ------------------------------------------------------------
_PW_par = get_alphabag_custom(alpha=m5_set['alpha'], B4=m5_set['B4'], m_s=m5_set['m_s'])
_PW_D0  = m5_set['Delta0']
_PW_Rx  = float(nuc_an.crossover_radius(PW_T, _PW_D0))   # unpCFL crossover radius
_PW_Rg  = np.linspace(0.01, 14.0, 400)
# ---- layout ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
_wmax = 0.0
for _fl, _ch, _col, _lbl in _M4:
    if _fl == 'frozen' and PW_PHASE != 'unpaired':
        continue
    for _si, _sig in enumerate(PW_SIGMAS):
        try:
            _eb = compute_energy_barrier(
                H['trapped'], PW_NBH * n_sat, PW_T, _sig,
                electric_charge_mode=_ch, params=_PW_par, flavor_mode=_fl,
                quark_phase=PW_PHASE, Delta0=_PW_D0, Y_L_H=m5_YL_used, R_values=_PW_Rg,
                switching_mode='step', Rx=(_PW_Rx if PW_PHASE == 'unpCFL' else None))
        except Exception:
            continue                           # e.g. a (flavor,phase) combo not supported
        ax.plot(_PW_Rg, _eb.W, color=_col, ls=_M4_LS[_si % len(_M4_LS)], lw=1.8)
        if np.isfinite(_eb.W).any():
            _k = int(np.nanargmax(_eb.W))
            ax.plot(_PW_Rg[_k], _eb.W[_k], 'o', ms=6, color=_col, mfc='white', zorder=5)
            _wmax = max(_wmax, float(_eb.W[_k]))
if np.isfinite(_PW_Rx):
    ax.axvspan(0, _PW_Rx, color='tab:blue', alpha=0.07, zorder=0)   # R <= R_x(T)
ax.axhline(0, color='0.6', lw=0.7, zorder=0)
ax.set_xlim(0, 8)
if _wmax > 0:
    ax.set_ylim(-0.15 * _wmax, 1.2 * _wmax)
ax.set_xlabel(r'$R$ [fm]'); ax.set_ylabel(r'$W$ [MeV]')
ax.set_title(rf"{PW_PHASE}, $T={PW_T:.0f}$ MeV, $n_B^H/n_\mathrm{{sat}}={PW_NBH:g}$, "
             rf"$Y_L\approx{m5_YL_used:.2f}$ — {m5_tag}")
_lg1 = ax.legend([Line2D([], [], color=_c, lw=2) for _, _, _c, _ in _M4],
                 [_l for _, _, _, _l in _M4], loc='upper right', fontsize=8, title='method')
ax.add_artist(_lg1)
ax.legend([Line2D([], [], color='0.3', ls=_M4_LS[i]) for i in range(len(PW_SIGMAS))],
          [rf'$\sigma={int(s)}$' for s in PW_SIGMAS], loc='center right', fontsize=8, title=r'$\sigma$')
plt.show()


# %% [markdown]
# ### Critical-droplet observables vs σ at the τ=τ_target PNS core
#
# Fix the proto-NS isentrope $(Y_L, S)$. For each (σ, phase) the τ=τ_target
# locus $T_{\rm nuc}(n_B^H)$ crosses that isentrope at one density — the PNS
# core that *just* nucleates in τ_target. There the engine's
# `critical_droplet_pt` gives the droplet, plotted **vs σ** (one line per
# phase): $R^*$, $W^*/T$, $M_{\rm PNS}$, interior $n_B^{c*}$, baryon count
# $N_B=n_B^{c*}\frac43\pi R^{*3}$, and $Y_S^H$ (hadronic strangeness fraction).

# %%
# =============================================================================
#  Critical-droplet observables vs σ for {unpaired, CFL, unpCFL}; the barrier
#  physics comes from nuc_an.critical_droplet_pt (same code as tau_pt).
# =============================================================================
# ---- user-configurable inputs ----------------------------------------------
_D7_flavor, _D7_charge = 'saddlepoint', 'coulomb_minimize'
_D7_methods = [('unpaired', 'tab:orange', '--', 'unpaired'),   # (phase, colour, ls, label)
               ('cfl',      'tab:green',  '-.', 'CFL'),
               ('unpCFL',   'tab:blue',   '-',  'unpCFL')]
_D7_YL = m5_YL_used                    # PNS lepton fraction (grid value)
_D7_S  = 2.0                           # PNS isentrope
# ---- computation ------------------------------------------------------------
_D7_par = get_alphabag_custom(alpha=m5_set['alpha'], B4=m5_set['B4'], m_s=m5_set['m_s'])
_D7_D0  = m5_set['Delta0']

# S-isentrope T(n_B) and the central-density -> M_PNS map (stable branch).
_D7_iso = lambda nB: np.asarray(H['iso_trapped']['T'](nB, _D7_YL, _D7_S))
_pns_full = tov_trapped[min(tov_trapped, key=lambda k:           # nearest (Y_L, S)
                            abs(k[0] - _D7_YL) + abs(k[1] - _D7_S))]  # cols: 2=n_Bc, 4=M
_pns = _pns_full[:int(np.argmax(_pns_full[:, 4])) + 1]           # stable branch
_M_of_nBc = lambda nB: float(np.interp(nB, _pns[:, 2], _pns[:, 4]))


def _s2_crossing(nB, Tn):
    """n_B where the τ-locus T_nuc(n_B) meets the isentrope (first sign change);
    that density is 'on the isentrope, τ=tau_target' — the core that just nucleates."""
    m = np.isfinite(nB) & np.isfinite(Tn)
    if m.sum() < 2:
        return np.nan
    x, d = nB[m], Tn[m] - _D7_iso(nB[m])
    s = np.where(np.diff(np.sign(d)) != 0)[0]
    if s.size == 0:
        return np.nan
    i = s[0]
    return float(x[i] - d[i] * (x[i + 1] - x[i]) / (d[i + 1] - d[i]))   # linear root


def _crossing_observables(sigma, phase):
    """One point per (σ, phase): on the (Y_L, S) isentrope, the density where
    τ=tau_target; return its critical-droplet observables (or None)."""
    obs = m5_get(_D7_flavor, _D7_charge, phase, sigma)
    if obs is None:
        return None
    res = compute_nucleation_density(obs, tau_target=tau_target, scan='n_B')
    nBx = _s2_crossing(*nucleation_curve(res, m5_iYL))    # τ-locus ∩ isentrope
    if not np.isfinite(nBx):
        return None
    Tx = float(_D7_iso(np.array([nBx]))[0])   # isentrope T there (= T_nuc at crossing)
    H_pt = nuc_an.hadronic_point(H['trapped'], nBx, _D7_YL, Tx)
    R_c, W_c, Qs = nuc_an.critical_droplet_pt(
        sigma, H_pt, Tx, _D7_flavor, _D7_charge, phase, {}, _D7_par, _D7_D0, _nuc_cfg)
    if Qs is None or not (np.isfinite(R_c) and R_c > 0 and np.isfinite(W_c)):
        return None
    nBc = float(Qs.n_B)
    NB = nBc * (4.0 / 3.0) * np.pi * R_c ** 3   # baryons in a sharp droplet of radius R*
    return dict(Rstar=R_c, WoverT=W_c / Tx, Mpns=_M_of_nBc(nBx),
                nBc=nBc, NB=NB, YSH=float(H_pt.Y_S))


# One crossing point per (σ, phase).
_D7 = {ph: {k: np.full(len(sigma_list), np.nan) for k in
            ('Rstar', 'WoverT', 'Mpns', 'nBc', 'NB', 'YSH')}
       for ph, _, _, _ in _D7_methods}
for _ph, _, _, _ in _D7_methods:
    for _i, _sig in enumerate(sigma_list):
        _o = _crossing_observables(_sig, _ph)
        if _o is not None:
            for _k in _D7[_ph]:
                _D7[_ph][_k][_i] = _o[_k]
    print(f"  {_ph:<8}: {int(np.sum(np.isfinite(_D7[_ph]['Rstar'])))}/{len(sigma_list)} σ-points",
          flush=True)

# ---- layout: 6 panels vs σ, one line per phase --------------------------------
_D7_panels = [('Rstar', r'$R^*$ [fm]'), ('WoverT', r'$W^*/T$'),
              ('Mpns', r'$M_{\rm PNS}$ [$M_\odot$]'), ('nBc', r'$n_B^{c*}$ [fm$^{-3}$]'),
              ('NB', r'$N_B$'), ('YSH', r'$Y_S^H$')]
_sig_arr = np.asarray(sigma_list, float)
fig, axs = plt.subplots(2, 3, figsize=(15, 9), sharex=True, constrained_layout=True)
for _ph, _c, _ls, _lbl in _D7_methods:
    for _ax, (_key, _ylab) in zip(axs.flat, _D7_panels):
        _ax.plot(_sig_arr, _D7[_ph][_key], color=_c, ls=_ls, lw=1.6,
                 marker='o', ms=5, label=_lbl)
for _ax, (_key, _ylab) in zip(axs.flat, _D7_panels):
    _ax.set_ylabel(_ylab); _ax.grid(alpha=0.3)
for _ax in axs[-1]:
    _ax.set_xlabel(r'$\sigma$ [MeV/fm$^2$]')
axs.flat[0].legend(fontsize=9)
fig.suptitle(rf"critical droplet at τ={tau_target*1e3:g} ms on the $S={_D7_S:g}$, "
             rf"$Y_L={_D7_YL:.2f}$ PNS isentrope — {_D7_flavor}/{_D7_charge}, {m5_tag}", y=1.04)
plt.show()
