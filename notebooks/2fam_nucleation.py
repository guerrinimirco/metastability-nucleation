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
# # 2-family nucleation
#
# Build the hadronic-phase EoS (SFHo / `2fam_phi` parametrization, with hyperons + Δ resonances),
# compute TOV configurations (mass-radius sequence), then run a Monte-Carlo search for CFL
# quark-matter parameters compatible with:
#
# 1. **Witten's absolute-stability bound** — $(\varepsilon/n_B)|_{P=0,T=0} < 930$ MeV;
# 2. **No re-hadronization** — $P_{\rm CFL}(\mu_B) > P_H(\mu_B)$ on the $\mu_B$ overlap;
# 3. **$M_{\max} \geq 2\,M_\odot$** — consistent with the heaviest observed pulsars.
#

# %% [markdown]
# ## Imports & installs
#
# A single setup cell. Re-running it from a fresh kernel restores every symbol used downstream.
# Comment out the `--force-reinstall` lines and uncomment the local-dev block while iterating
# on the `eos` / `nucleation` packages on disk.
#

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


# ─── Standard scientific Python ──────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
from functools import partial                                  # currying helper
from scipy.interpolate import RegularGridInterpolator, interp1d


# ─── Plot styling helpers (shared look & feel across figures) ────────────────
from eos.general.plotting_info import (
    set_global_style, setup_scientific_figure, apply_style,
    add_panel_labels, STANDARD_COLORS, STYLE, LABELS, FONTS,
)


# ─── SFHo equation of state (hadronic phase) ─────────────────────────────────
from eos.sfho.compute_tables import (
    TableSettings, compute_table,
    load_eos_table as load_eos_table_sfho,
    build_interpolators as build_interpolators_sfho,
)
from eos.sfho.parameters import create_custom_parametrization


# ─── Alpha-bag equation of state (quark phase) ───────────────────────────────
from eos.alphabag.compute_tables import (
    AlphaBagTableSettings, compute_alphabag_table,
    load_eos_tables_multi as load_eos_tables_multi_alphabag,
    build_interpolators as build_interpolators_alphabag,
)
from eos.alphabag.parameters import get_alphabag_custom
from eos.alphabag.eos import solve_cfl


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
    export_thermal_nucleation_table, load_thermal_nucleation_table,
    compute_nucleation_temperature, compute_nucleation_density,
)


# ─── Output directories (created once, idempotent) ───────────────────────────
for d in ('../output/tables_Hphase', '../output/tables_tov', '../output/mc_cfl'):
    os.makedirs(d, exist_ok=True)


# %% [markdown]
# # Hadronic phase EoS

# %% [markdown]
# ## Info & parameters
#
# We use the SFHo relativistic mean-field (RMF) family, in the `2fam_phi` variant
# (φ-meson coupled to strangeness ⇒ allows hyperons + Δ resonances). The
# "custom parametrization" knobs below are the parameters that are **not pinned**
# by saturation properties of symmetric nuclear matter: the hyperon potential
# depths ($U_{HN}$) and the Δ–meson coupling ratios ($x_{y\Delta}$).
#

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
# ## Compute the four hadronic EoS tables
#
# We need four equilibrium scenarios — they cover the typical thermal/leptonic
# states of hot dense matter in a proto-NS or post-merger remnant:
#
# | Tag             | Equilibrium                              | Grid axes            | Physical regime                                       |
# |-----------------|------------------------------------------|----------------------|-------------------------------------------------------|
# | `betaeq`        | β-equilibrium, **free-streaming** ν      | $(n_B, T)$           | After ν-sphere: $\mu_\nu = 0$, ν leave the matter   |
# | `iso_betaeq`    | β-equilibrium, **isentropic**            | $(n_B, S)$           | Same physics, parametrized by $S$ instead of $T$      |
# | `trapped`       | Fixed $Y_L$, **trapped** ν               | $(n_B, Y_L, T)$      | Inside ν-sphere: $\mu_\nu \neq 0$, $Y_L$ conserved |
# | `iso_trapped`   | Fixed $Y_L$, trapped ν, **isentropic**   | $(n_B, Y_L, S)$      | Same physics, parametrized by $S$                     |
#
# Each call writes a `.dat` table to `../output/tables_Hphase/`. Re-running is
# idempotent (overwrites the file with the latest params) — safe to re-execute
# after a parameter change.
#

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


# %% [markdown]
# ## Load tables and build interpolators
#
# `build_interpolators_sfho` wraps each table column in a
# `scipy.interpolate.RegularGridInterpolator`. Access is uniform across all four cases:
#
# ```python
# H['betaeq']['P'](n_B, T)              # pressure, β-equilibrium
# H['trapped']['mu_e'](n_B, Y_L, T)     # electron chem. pot., trapped ν
# H['iso_betaeq']['T'](n_B, S)          # iso tables: T is an *output*, S the input
# ```
#
# Standard keys exposed per case: `P`, `eps`, `s`, `f`, `T` (iso tables only),
# `mu_B`, `mu_C`, `mu_S`, `mu_e`, `mu_nu` (trapped only), `mu_L` (trapped only),
# `Y_C`, `Y_S`.
#

# %%
# =============================================================================
#  Load the four .dat tables and build vectorized interpolators.
# =============================================================================

# (case label) → (filename stem, optional suffix, equilibrium tag for loader)
_cases = {
    'betaeq':      ('eos_hadronic_betaeq_sfho_2famphi',  '',            'beta_eq'),
    'trapped':     ('eos_hadronic_trapped_sfho_2famphi', '',            'trapped_neutrinos'),
    'iso_betaeq':  ('eos_hadronic_betaeq_sfho_2famphi',  '_isentropic', 'isentropic_beta_eq'),
    'iso_trapped': ('eos_hadronic_trapped_sfho_2famphi', '_isentropic', 'isentropic_trapped'),
}

H, H_table = {}, {}
for key, (stem, suf, eq) in _cases.items():
    path = f'../output/tables_Hphase/{stem}_{xsd_tag}{suf}.dat'
    H_table[key] = load_eos_table_sfho(path, eq)
    H[key]       = build_interpolators_sfho(H_table[key])

# Show the actual grid extents for sanity-checking.
for key, t in H_table.items():
    rng = ', '.join(f"{ax} ∈ [{t.grids[ax][0]:.4g}, {t.grids[ax][-1]:.4g}]" for ax in t.grids)
    print(f"  {key:12s}  {rng}")


# ── Per-baryon convenience callables (s/n_B, f/n_B, Y_nu = Y_L − Y_C). ──────
# Everything else is reached directly via H[case][quantity](args).
S_betaeq         = lambda nB, T:       H['betaeq']['s'](nB, T) / nB
F_betaeq         = lambda nB, T:       H['betaeq']['f'](nB, T) / nB
S_trapped        = lambda nB, YL, T:   H['trapped']['s'](nB, YL, T) / nB
F_trapped        = lambda nB, YL, T:   H['trapped']['f'](nB, YL, T) / nB
F_iso_betaeq     = lambda nB, S:       H['iso_betaeq']['f'](nB, S) / nB
F_iso_trapped    = lambda nB, YL, S:   H['iso_trapped']['f'](nB, YL, S) / nB
Y_nu_trapped     = lambda nB, YL, T:   YL - H['trapped']['Y_C'](nB, YL, T)
Y_nu_iso_trapped = lambda nB, YL, S:   YL - H['iso_trapped']['Y_C'](nB, YL, S)


# %% [markdown]
# # TOV computation

# %% [markdown]
# ## Cold β-equilibrium star (T = 0)
#
# Take the cold-limit slice of the β-equilibrium table, splice an SFHo crust below
# $n_{\rm transition}$ via a tanh blend, then integrate the TOV equation on a
# log-spaced grid of central energy densities. The result is the M(R) sequence;
# `truncate_to_stable_branch` cuts it at the maximum mass.
#

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


# %% [markdown]
# ## Trapped, isentropic profiles — sweep over (Y_L, S)
#
# For each $(Y_L, S)$ pair we slice the iso-trapped table to get $P, \varepsilon$
# on the $n_B$ grid, splice an SFHo trapped-ν crust at the same $(Y_L, S)$ for
# consistency, and integrate TOV. Results stored in `tov_results_trapped[(Y_L, S)]`.
#

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


# %% [markdown]
# ## Plots: $P(n_B)$ and $M$–$R$
#
# Two-panel diagnostic. Left: pressure vs density (linear/linear so the
# high-density branch dominates the eye). Right: $M$–$R$ sequence (stable branch only).

# %%
# =============================================================================
#  Two-panel diagnostic plot:
#    left  → P(n_B) for cold β-eq and a few trapped-isentropic cases
#    right → M–R sequence for the same cases (stable branch only)
#  TOV result columns (after truncate_to_stable_branch):
#    0=e_c   1=P_c   2=n_Bc   3=R [km]   4=M [M_⊙]   5=M_b
# =============================================================================

cases = [(0.20, 1.5), (0.30, 2.0), (0.40, 2.5)]   # (Y_L, S) curves to overlay

fig, (ax_P, ax_MR) = plt.subplots(1, 2, figsize=(11, 4.5))

# ── P(n_B): linear/linear so the high-density branch dominates the eye ──────
ax_P.plot(n_B_values / n_sat, H['betaeq']['P'](n_B_values, 0.01),
          'k-', lw=2, label='T = 0 (β-eq)')
for YL, S in cases:
    ax_P.plot(n_B_values / n_sat,
              H['iso_trapped']['P'](n_B_values, YL, S),
              label=f'Y_L={YL}, S={S}')
ax_P.set_xlabel(r'$n_B / n_{\rm sat}$')
ax_P.set_ylabel(r'$P$ [MeV/fm$^3$]')
ax_P.legend(fontsize=8); ax_P.grid(alpha=0.3)

# ── M–R: stable branch only (truncate_to_stable_branch cuts past M_max) ─────
ax_MR.plot(results_tov[:, 3], results_tov[:, 4],
           'k-', lw=2, label='T = 0 (β-eq)')
for YL, S in cases:
    r = tov_results_trapped[(YL, S)]
    ax_MR.plot(r[:, 3], r[:, 4], label=f'Y_L={YL}, S={S}')
ax_MR.set_xlim(8, 16)
ax_MR.set_ylim(0.2, 2.5)
ax_MR.set_xlabel(r'$R$ [km]')
ax_MR.set_ylabel(r'$M$ [$M_\odot$]')
ax_MR.legend(fontsize=8); ax_MR.grid(alpha=0.3)

fig.tight_layout(); plt.show()


# %% [markdown]
# # Quark phase — Monte-Carlo CFL parameter search

# %% [markdown]
# ## Filters
#
# We sample $(\alpha_s, \Delta_0, B^{1/4})$ uniformly in physically motivated boxes
# and keep only those parameter sets passing **three filters in sequence**:
#
# **(1) Witten's conjecture** — for absolutely-stable quark matter we need
# $$\left.\frac{\varepsilon}{n_B}\right|_{P=0,\,T=0} \;<\; 930~\text{MeV}$$
# (the energy/baryon of $^{56}$Fe).
#
# **(2) No re-hadronization** — if the deconfined CFL phase has higher pressure
# than the hadronic phase at every shared $\mu_B$, the system stays deconfined
# once formed:
# $$P_{\rm CFL}(\mu_B) \;>\; P_H(\mu_B) \quad \text{on the }\mu_B\text{ overlap.}$$
#
# **(3) $2\,M_\odot$ constraint** — the quark-star branch (TOV without a crust)
# must support $M_{\max} \geq M_{\rm target}\,(=2\,M_\odot)$, consistent with
# observed heavy pulsars.
#
# The hadronic comparison curve $P_H(\mu_B)$ is pre-built once at $T\sim 0$ from
# the β-equilibrium table loaded above. Accepted candidates are appended to a CSV
# in `../output/mc_cfl/` so multiple MC runs accumulate.
#

# %%
# =============================================================================
#  Monte-Carlo search for CFL parameter sets (α_s, Δ_0, B^(1/4)).
#  Three filters applied sequentially; ordered cheap → expensive.
# =============================================================================

# ---- knobs ──────────────────────────────────────────────────────────────────
N_samples       = 1000          # number of random parameter triples to try
rng_seed        = 0
M_max_window   = (2.0, 2.5)   # M_⊙ — accept the candidate iff M_max ∈ this window
e_over_nB_max   = 930.0         # MeV — Witten bound (energy/baryon of ^56 Fe)

# Sampling boxes (uniform priors); physical ranges from the literature.
alpha_range  = (0.0,  0.6)
B4_range     = (130.0, 200.0)   # MeV (B^(1/4))
Delta0_range = (10.0, 250.0)    # MeV (CFL gap at μ → ∞)
m_s_fixed    = 100.0            # MeV (strange-quark mass; fixed for now)

# CFL EOS evaluation grid (T=0).
n_B_grid_cfl = np.linspace(0.05, 2.5, 250)   # fm^-3
T_eos        = 0.0

# Hadronic comparison evaluated at T ≈ 0 on the existing β-eq table grid.
T_hadr_eval  = H_table['betaeq'].grids['T'][0]

# Central-density grid for the quark-star TOV (no crust).
e_c_vec_tov  = generate_ec_logspace(e_min=100, e_max=2000, n_points=80)

# Progress reporting cadence (one short summary line every K samples).
PROGRESS_EVERY = 25


# ---- Pre-build P_H(μ_B) at T~0 from the hadronic interpolators ──────────────
# P(μ_B) is invertible via interp1d because μ_B(n_B) is monotone at fixed T.
n_B_H_grid = H_table['betaeq'].grids['n_B']
mu_B_H_arr = H['betaeq']['mu_B'](n_B_H_grid, T_hadr_eval)
P_H_arr    = H['betaeq']['P'](   n_B_H_grid, T_hadr_eval)

ok_H = np.isfinite(mu_B_H_arr) & np.isfinite(P_H_arr)
order = np.argsort(mu_B_H_arr[ok_H])
mu_B_H_sorted = mu_B_H_arr[ok_H][order]
P_H_sorted    = P_H_arr[ok_H][order]

P_H_of_muB = interp1d(mu_B_H_sorted, P_H_sorted,
                      kind='linear', bounds_error=False, fill_value=np.nan)


def _cfl_eos_at_params(alpha, B4, Delta0):
    """Solve CFL β-eq EOS at T=0 over n_B_grid_cfl. Returns P, e, mu, ok mask."""
    p = get_alphabag_custom(alpha=alpha, B4=B4, m_s=m_s_fixed)
    P  = np.full_like(n_B_grid_cfl, np.nan)
    e  = np.full_like(n_B_grid_cfl, np.nan)
    mu = np.full_like(n_B_grid_cfl, np.nan)
    ok = np.zeros_like(n_B_grid_cfl, dtype=bool)
    guess = None
    for i, nB in enumerate(n_B_grid_cfl):
        try:
            r = solve_cfl(nB, T_eos, Delta0, p,
                          include_photons=False, include_gluons=True,
                          initial_guess=guess)
        except Exception:
            r = None
        if r is None or not r.converged or r.error > 1e-6:
            continue
        P[i], e[i], mu[i] = r.P_total, r.e_total, r.mu_B
        guess = np.array([r.mu_u, r.mu_d, r.mu_s])   # warm-start next solve
        ok[i] = True
    return P, e, mu, ok


def _zero_crossing(x, y):
    """First sign change in y(x); returns (j, frac) so x_cross = x[j] + frac*(x[j+1]-x[j])."""
    s = np.sign(y)
    idx = np.where(np.diff(s) != 0)[0]
    if idx.size == 0:
        return None
    j = idx[0]
    frac = -y[j] / (y[j+1] - y[j])
    return j, frac


# ---- MC loop ───────────────────────────────────────────────────────────────
# Status conventions used in the per-sample line:
#   OK         → passed all three filters, appended to `accepted`
#   rej:solve  → CFL solver did not converge on enough n_B points
#   rej:witten → e/n_B at P=0 exceeded 930 MeV (or no P=0 crossing)
#   rej:rehadr → P_CFL ≤ P_H somewhere on the μ_B overlap
#   rej:mmax   → TOV failed, or M_max outside M_max_window
import time, sys
rng = np.random.default_rng(rng_seed)
accepted = []
rej = dict(solve=0, witten=0, rehadr=0, mmax=0)
t0   = time.time()

print(f"Starting MC: {N_samples} samples  (progress summary every {PROGRESS_EVERY})")
print("-" * 90)

for k in range(N_samples):
    alpha  = rng.uniform(*alpha_range)
    B4     = rng.uniform(*B4_range)
    Delta0 = rng.uniform(*Delta0_range)

    # Header that every per-sample line shares.
    head = f"[{k+1:4d}/{N_samples}] α={alpha:.3f}  B4={B4:6.2f}  Δ0={Delta0:6.2f}"

    # Solve the CFL EoS for this parameter triple over the n_B grid.
    P_cfl, e_cfl, mu_cfl, ok = _cfl_eos_at_params(alpha, B4, Delta0)
    if ok.sum() < 20:                           # not enough converged points
        rej['solve'] += 1
        print(f"{head}  → rej:solve   (only {ok.sum()} converged)")
        sys.stdout.flush(); continue
    n_ok, P_ok, e_ok, mu_ok = (n_B_grid_cfl[ok], P_cfl[ok],
                                e_cfl[ok], mu_cfl[ok])

    # (1) Witten — locate P=0 along the n_B axis, evaluate e/n_B there.
    cross = _zero_crossing(n_ok, P_ok)
    if cross is None or P_ok[-1] <= 0:
        rej['witten'] += 1
        print(f"{head}  → rej:witten  (no P=0 crossing)")
        sys.stdout.flush(); continue
    j, frac = cross
    n_P0 = n_ok[j] + frac * (n_ok[j+1] - n_ok[j])
    e_P0 = e_ok[j] + frac * (e_ok[j+1] - e_ok[j])
    e_per_nB = e_P0 / n_P0
    if e_per_nB >= e_over_nB_max:
        rej['witten'] += 1
        print(f"{head}  → rej:witten  (e/nB|P=0 = {e_per_nB:6.1f} MeV)")
        sys.stdout.flush(); continue

    # (2) No re-hadronization — on the shared μ_B window require P_CFL > P_H.
    mu_lo = max(mu_ok.min(), mu_B_H_sorted.min())
    mu_hi = min(mu_ok.max(), mu_B_H_sorted.max())
    if mu_hi <= mu_lo:
        rej['rehadr'] += 1
        print(f"{head}  → rej:rehadr  (no μ_B overlap)")
        sys.stdout.flush(); continue
    mu_chk = np.linspace(mu_lo, mu_hi, 500)
    P_cfl_of_mu = interp1d(mu_ok, P_ok, kind='linear',
                           bounds_error=False, fill_value=np.nan)
    Pc, Ph = P_cfl_of_mu(mu_chk), P_H_of_muB(mu_chk)
    m = np.isfinite(Pc) & np.isfinite(Ph)
    if not np.all(Pc[m] > Ph[m]):
        rej['rehadr'] += 1
        # Quantify how badly it failed: largest amount by which P_H exceeded P_CFL.
        gap = np.nanmax((Ph - Pc)[m]) if m.any() else float('nan')
        print(f"{head}  → rej:rehadr  (P_H − P_CFL up to {gap:6.1f} MeV/fm³)")
        sys.stdout.flush(); continue

    # (3) TOV M_max on quark-only star (no crust).
    pos = P_ok > 0
    if pos.sum() < 10:
        rej['mmax'] += 1
        print(f"{head}  → rej:mmax    (only {pos.sum()} positive-P points)")
        sys.stdout.flush(); continue
    try:
        tov_full = compute_tov_sequence(
            EOSTable_for_TOV(P=P_ok[pos], epsilon=e_ok[pos], nB=n_ok[pos]),
            e_c_vec=e_c_vec_tov, add_crust_table='No',
            compute_baryonic_mass=True, compute_tidal=False, verbose=False,
        )
        _, M_max, e_c_at_Mmax = truncate_to_stable_branch(tov_full, verbose=False)
    except Exception as ex:
        rej['mmax'] += 1
        print(f"{head}  → rej:mmax    (TOV error: {type(ex).__name__})")
        sys.stdout.flush(); continue
    if not np.isfinite(M_max) or not (M_max_window[0] <= M_max <= M_max_window[1]):
        rej['mmax'] += 1
        why = ("too low"  if M_max < M_max_window[0]
               else "too high" if M_max > M_max_window[1]
               else "non-finite")
        print(f"{head}  → rej:mmax    (M_max = {M_max:.3f} M_⊙ — {why} "
              f"vs window {M_max_window})")
        sys.stdout.flush(); continue

    # Passed all filters.
    accepted.append(dict(
        alpha=alpha, B4=B4, Delta0=Delta0, m_s=m_s_fixed,
        e_over_nB_P0=float(e_per_nB), n_B_P0=float(n_P0),
        M_max=float(M_max), e_c_Mmax=float(e_c_at_Mmax),
    ))
    print(f"{head}  → OK          e/nB|P=0={e_per_nB:6.2f} MeV   "
          f"M_max={M_max:.3f} M_⊙  (in {M_max_window})")
    sys.stdout.flush()

    # Periodic running summary so the tail of a long log is easy to read.
    if (k + 1) % PROGRESS_EVERY == 0:
        dt = time.time() - t0
        rate = (k + 1) / dt if dt > 0 else 0.0
        eta_s = (N_samples - (k + 1)) / rate if rate > 0 else 0.0
        print(f"  ── progress: {k+1}/{N_samples}   accepted={len(accepted)}   "
              f"rej(solve/wit/reh/mm)={rej['solve']}/{rej['witten']}/{rej['rehadr']}/{rej['mmax']}   "
              f"rate={rate:5.2f}/s   ETA={eta_s/60:5.1f} min")
        sys.stdout.flush()

# ---- Final summary ─────────────────────────────────────────────────────────
print("-" * 90)
print(f"Accepted: {len(accepted)} / {N_samples}")
print(f"Rejections: solve={rej['solve']}  witten={rej['witten']}  "
      f"rehadr={rej['rehadr']}  mmax={rej['mmax']}")
print(f"Total runtime: {(time.time() - t0)/60:.1f} min")


# ---- Persist accepted candidates (cumulative across MC runs) ────────────────
import csv, datetime
out_csv = f'../output/mc_cfl/cfl_accepted_{xsd_tag}.csv'
write_header = not os.path.exists(out_csv)
with open(out_csv, 'a', newline='') as f:
    w = csv.writer(f)
    if write_header:
        w.writerow(['alpha', 'B4', 'Delta0', 'm_s',
                    'e_over_nB_P0', 'n_B_P0', 'M_max', 'e_c_Mmax',
                    'x_sigma_delta', 'timestamp'])
    ts = datetime.datetime.now().isoformat(timespec='seconds')
    for d in accepted:
        w.writerow([d['alpha'], d['B4'], d['Delta0'], d['m_s'],
                    d['e_over_nB_P0'], d['n_B_P0'], d['M_max'], d['e_c_Mmax'],
                    x_sigma_delta, ts])
print(f"Appended {len(accepted)} accepted candidates → {out_csv}")

# Numpy-friendly view for downstream analysis / plotting.
if accepted:
    accepted_arr = {k: np.array([d[k] for d in accepted]) for k in accepted[0]}
    print("\nRanges of accepted sets:")
    for k in ('alpha', 'B4', 'Delta0', 'e_over_nB_P0', 'M_max'):
        a = accepted_arr[k]
        print(f"  {k:14s}: [{a.min():.3f}, {a.max():.3f}]")

# %% [markdown]
# ## Visualize accepted MC samples — parameter space
#
# Two views of the accepted set:
#
# - **Pairplot (corner)** — pairwise scatter + marginal histograms for the three
#   sampled CFL parameters $(\alpha_s, B^{1/4}, \Delta_0)$ plus the two main
#   derived quantities ($\varepsilon/n_B|_{P=0}$ from Witten, and $M_{\max}$).
#   Color = $M_{\max}$, so you can see which candidates barely pass vs. easily pass.
# - **3D scatter** — same parameter triple in 3D, again colored by $M_{\max}$.
#   Useful for spotting the *shape* of the allowed region in parameter space.

# %%
# =============================================================================
#  Pairplot + 3D scatter of accepted CFL candidates.
#  Loads the cumulative CSV (so multiple MC runs at the same x_sigma_delta
#  pile up here).
# =============================================================================
import pandas as pd
import seaborn as sns
# matplotlib ≥ 3.2 auto-registers the '3d' projection; no Axes3D import needed.

mc_csv = f'../output/mc_cfl/cfl_accepted_{xsd_tag}.csv'
mc = pd.read_csv(mc_csv)
print(f"Loaded {len(mc)} accepted samples from {mc_csv}.")

# ── 1) Pairplot (corner): pairwise scatter + marginal histograms ────────────
# Columns: α_s + B^1/4 + Δ_0 + M_max.
# e/n_B|P=0 dropped — it's a sanity check, not a free parameter, and bunches
# near 930 MeV (the Witten bound) cluttering the corner.
# `hue` is intentionally OMITTED: with a continuous variable like M_max,
# seaborn splits each diagonal histogram bin per unique M_max value, which
# gives 1-sample-tall stacks instead of a real marginal.  M_max colour info
# lives on the 3D scatter below.
cols = ['alpha', 'B4', 'Delta0', 'M_max']
g = sns.pairplot(mc, vars=cols,
                 diag_kind='hist',
                 plot_kws=dict(s=14, alpha=0.7, color='steelblue'),
                 corner=True)
g.fig.suptitle(f"Accepted CFL parameter sets ({xsd_tag})", y=1.02)
# Pretty axis labels (TeX) so the column names read as physics symbols.
_nice = {'alpha': r'$\alpha_s$', 'B4': r'$B^{1/4}$ [MeV]',
         'Delta0': r'$\Delta_0$ [MeV]', 'M_max': r'$M_{\max}$ [$M_\odot$]'}
for ax in g.axes.flatten():
    if ax is None:                            # corner=True leaves empty cells
        continue
    if ax.get_xlabel() in _nice: ax.set_xlabel(_nice[ax.get_xlabel()])
    if ax.get_ylabel() in _nice: ax.set_ylabel(_nice[ax.get_ylabel()])
plt.show()

# ── 2) 3D scatter of (α_s, B^1/4, Δ_0), color = M_max ───────────────────────
fig = plt.figure(figsize=(7, 5))
ax  = fig.add_subplot(111, projection='3d')
sc  = ax.scatter(mc['alpha'], mc['B4'], mc['Delta0'],
                 c=mc['M_max'], cmap='viridis', s=18)
fig.colorbar(sc, ax=ax, label=r'$M_{\max}\;[M_\odot]$', pad=0.1)
ax.set_xlabel(r'$\alpha_s$')
ax.set_ylabel(r'$B^{1/4}$ [MeV]')
ax.set_zlabel(r'$\Delta_0$ [MeV]')
ax.set_title(f"Accepted CFL parameter sets ({xsd_tag})")
plt.tight_layout(); plt.show()


# %% [markdown]
# ## All accepted EoS + M–R curves
#
# To see *physically* how the accepted CFL EoSs look, we **re-solve** the CFL
# EoS for every accepted $(\alpha_s, B^{1/4}, \Delta_0)$, redo the TOV
# integration (no crust), and overlay:
#
# - **Left**: $P(\mu_B)$ for every accepted candidate, with the cold-hadronic
#   curve $P_H(\mu_B)$ drawn in black for reference.  All accepted curves must
#   lie *above* the hadronic one on their μ_B overlap (filter #2).
# - **Right**: $M$–$R$ sequence (stable branch) for every accepted candidate,
#   with the cold-hadronic $M$–$R$ in black and the $2\,M_\odot$ threshold
#   marked.
#
# Color encodes $M_{\max}$, so the same color in both panels refers to the same
# candidate.  Re-solving is fast because the count of accepted candidates is
# small (≲ a few % of `N_samples`).

# %%
# =============================================================================
#  Replay each accepted (α, B^1/4, Δ_0): solve CFL EoS, run TOV, collect curves.
#  Reuses _cfl_eos_at_params defined in the MC cell above — make sure that
#  cell has been executed first.
# =============================================================================
import matplotlib.cm as cm

print(f"Re-solving CFL EoS + TOV for {len(mc)} accepted samples...")
curves = []
for i, row in mc.iterrows():
    P_cfl, e_cfl, mu_cfl, ok = _cfl_eos_at_params(row.alpha, row.B4, row.Delta0)
    if ok.sum() < 20:
        continue
    n_ok  = n_B_grid_cfl[ok]
    P_ok  = P_cfl[ok]
    e_ok  = e_cfl[ok]
    mu_ok = mu_cfl[ok]
    pos = P_ok > 0
    if pos.sum() < 10:
        continue
    try:
        tov_full = compute_tov_sequence(
            EOSTable_for_TOV(P=P_ok[pos], epsilon=e_ok[pos], nB=n_ok[pos]),
            e_c_vec=e_c_vec_tov, add_crust_table='No',
            compute_baryonic_mass=True, compute_tidal=False, verbose=False,
        )
        tov_stable, _, _ = truncate_to_stable_branch(tov_full, verbose=False)
    except Exception:
        continue
    curves.append(dict(
        mu=mu_ok, P=P_ok,
        R=tov_stable[:, 3], M=tov_stable[:, 4],
        M_max=float(row.M_max),
    ))
print(f"Reconstructed {len(curves)} / {len(mc)} accepted curves.")

# ── Build a M_max → color map shared by both panels ─────────────────────────
M_max_vals = np.array([c['M_max'] for c in curves])
norm = plt.Normalize(vmin=M_max_vals.min(), vmax=M_max_vals.max())
cmap = cm.viridis

fig, (axP, axMR) = plt.subplots(1, 2, figsize=(12, 4.8))

# ── (a) P(μ_B) — hadronic reference + every accepted CFL curve ─────────────
axP.plot(mu_B_H_sorted, P_H_sorted, 'k-', lw=2, label='Hadronic (T≈0)')
for c in curves:
    axP.plot(c['mu'], c['P'], color=cmap(norm(c['M_max'])), lw=0.7, alpha=0.8)
axP.set_xlabel(r'$\mu_B$ [MeV]')
axP.set_ylabel(r'$P$ [MeV/fm$^3$]')
axP.legend(fontsize=8); axP.grid(alpha=0.3)
axP.set_title(r'$P(\mu_B)$  —  accepted CFL vs hadronic')

# ── (b) M–R — hadronic reference + 2 M_sun line + every accepted M(R) ──────
axMR.plot(results_tov[:, 3], results_tov[:, 4], 'k-', lw=2, label='Hadronic (T=0)')
axMR.axhline(2.0, color='red', linestyle=':', lw=1, label=r'$2\,M_\odot$')
for c in curves:
    axMR.plot(c['R'], c['M'], color=cmap(norm(c['M_max'])), lw=0.7, alpha=0.8)
axMR.set_xlim(7, 16); axMR.set_ylim(0, 3.0)
axMR.set_xlabel(r'$R$ [km]')
axMR.set_ylabel(r'$M$ [$M_\odot$]')
axMR.legend(fontsize=8); axMR.grid(alpha=0.3)
axMR.set_title('Mass–radius sequences (stable branch)')

# Shared colorbar on the right.
sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
cb = fig.colorbar(sm, ax=[axP, axMR], shrink=0.85, pad=0.02)
cb.set_label(r'$M_{\max}\;[M_\odot]$')

fig.suptitle(f"Accepted CFL candidates ({xsd_tag})", y=1.02)
plt.show()

# %% [markdown]
# # Quark phase — chosen parametrization → tables
#
# The Monte-Carlo above *searches* the $(\alpha_s, B^{1/4}, \Delta_0)$ box. Here we
# instead fix **one** quark parametrization by hand (given as input below) and turn
# it into the final EoS tables used downstream by the nucleation code.
#
# Flow:
# 1. **Validate at $T=0$** — for the chosen set, check Witten's bound, no
#    re-hadronization vs the cold hadronic curve, and that the quark-star branch
#    gives $M_{\max}\in[2,2.5]\,M_\odot$. Plot the $M$–$R$ sequence.
# 2. **Tabulate** — compute the **unpaired** and **CFL** αBag tables over the full
#    $(n_B, T)$ grid in β-equilibrium, save them, and build interpolators.
#
# The *same* $(\alpha_s, B^{1/4}, m_s)$ is used for both phases; $\Delta_0$ is the
# CFL pairing gap and is simply ignored by the unpaired phase.

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
    dict(alpha=0.1*np.pi/2, B4=165.0, Delta0=180.0, m_s=100.0),
]

sigma_list = [50.0, 100.0, 150.0]      # surface tensions to scan [MeV/fm^2]

def q_tag_of(p):
    """Unique filename/key tag for one parameter set."""
    return (f"B4{int(round(p['B4']))}_D{int(round(p['Delta0']))}"
            f"_a{p['alpha']:.2f}_ms{int(round(p['m_s']))}")

print(f"{len(quark_param_sets)} quark set(s):")
for p in quark_param_sets:
    print(f"  {q_tag_of(p)}  (α_s={p['alpha']:.3f}, B¼={p['B4']}, "
          f"Δ_0={p['Delta0']}, m_s={p['m_s']})")
print(f"σ scan: {sigma_list} MeV/fm²")


# %% [markdown]
# ## Validate the chosen CFL set at $T=0$
#
# Three filters (same physics as the MC), then the $M$–$R$ plot:
#
# 1. **Witten** — $(\varepsilon/n_B)|_{P=0} < 930$ MeV.
# 2. **No re-hadronization** — $P_{\rm CFL}(\mu_B) > P_H(\mu_B)$ on the $\mu_B$ overlap.
# 3. **Mass** — quark-star (no crust) $M_{\max}\in[2,2.5]\,M_\odot$.

# %%
# =============================================================================
#  Validate EVERY CFL parametrization in quark_param_sets at T = 0.
#  Self-contained: depends only on the hadronic interpolators H + the αBag solver.
# =============================================================================
M_window_q   = (2.0, 2.5)   # M_⊙ — target M_max window
witten_max_q = 930.0        # MeV — Witten bound (e/baryon of ^56Fe)
n_B_val      = np.linspace(0.05, 2.5, 250)   # fm^-3 — reaches down to P=0

# P_H(μ_B) at T≈0 — set-independent, built once.
T_cold_q   = H_table['betaeq'].grids['T'][0]
n_B_H      = H_table['betaeq'].grids['n_B']
mu_H, P_H_ = H['betaeq']['mu_B'](n_B_H, T_cold_q), H['betaeq']['P'](n_B_H, T_cold_q)
m_         = np.isfinite(mu_H) & np.isfinite(P_H_)
o          = np.argsort(mu_H[m_]); mu_H_s, P_H_s = mu_H[m_][o], P_H_[m_][o]

mr_curves = []   # (tag, tov_q, all_pass) for the overlay plot
for p in quark_param_sets:
    stag = q_tag_of(p)
    pq   = get_alphabag_custom(alpha=p['alpha'], B4=p['B4'], m_s=p['m_s'])

    # ── CFL EoS at T=0 ──────────────────────────────────────────────────────
    P_q = np.full_like(n_B_val, np.nan); e_q = np.full_like(n_B_val, np.nan)
    mu_q = np.full_like(n_B_val, np.nan); ok_q = np.zeros_like(n_B_val, dtype=bool)
    guess = None
    for i, nB in enumerate(n_B_val):
        try:
            r = solve_cfl(nB, 0.0, p['Delta0'], pq,
                          include_photons=False, include_gluons=True, initial_guess=guess)
        except Exception:
            r = None
        if r is None or not r.converged or r.error > 1e-6:
            continue
        P_q[i], e_q[i], mu_q[i] = r.P_total, r.e_total, r.mu_B
        guess = np.array([r.mu_u, r.mu_d, r.mu_s])
        ok_q[i] = True
    n_ok, P_ok, e_ok, mu_ok = n_B_val[ok_q], P_q[ok_q], e_q[ok_q], mu_q[ok_q]
    if ok_q.sum() < 5 or not (P_ok[0] < 0 < P_ok[-1]):
        print(f"── {stag}: CFL solve insufficient / no P=0 crossing — skipped"); continue

    # (1) Witten — interpolate (n_B, e) at P=0 (P monotone in n_B).
    n_P0 = np.interp(0.0, P_ok, n_ok); e_P0 = np.interp(0.0, P_ok, e_ok)
    e_per_nB = e_P0 / n_P0
    pass_witten = e_per_nB < witten_max_q

    # (2) No re-hadronization — P_CFL > P_H on the μ_B overlap.
    mu_lo, mu_hi = max(mu_ok.min(), mu_H_s.min()), min(mu_ok.max(), mu_H_s.max())
    mu_chk = np.linspace(mu_lo, mu_hi, 500)
    Pc = interp1d(mu_ok, P_ok, bounds_error=False, fill_value=np.nan)(mu_chk)
    Ph = interp1d(mu_H_s, P_H_s, bounds_error=False, fill_value=np.nan)(mu_chk)
    mm = np.isfinite(Pc) & np.isfinite(Ph)
    pass_rehadr = (mu_hi > mu_lo) and bool(np.all(Pc[mm] > Ph[mm]))
    worst = np.nanmax((Ph - Pc)[mm]) if mm.any() else np.nan

    # (3) Mass — TOV on the quark-only star (no crust).
    pos = P_ok > 0
    tov_q_full = compute_tov_sequence(
        EOSTable_for_TOV(P=P_ok[pos], epsilon=e_ok[pos], nB=n_ok[pos]),
        e_c_vec=generate_ec_logspace(e_min=100, e_max=2000, n_points=100),
        add_crust_table='No', compute_baryonic_mass=True, compute_tidal=False, verbose=False)
    tov_q, M_max_q, _ = truncate_to_stable_branch(tov_q_full, verbose=False)
    pass_mass = bool(np.isfinite(M_max_q) and M_window_q[0] <= M_max_q <= M_window_q[1])

    allp = pass_witten and pass_rehadr and pass_mass
    _yn = lambda b: 'PASS' if b else 'FAIL'
    print(f"── {stag}:  Witten e/nB|P0={e_per_nB:6.1f} [{_yn(pass_witten)}] | "
          f"rehadr max(P_H−P_CFL)={worst:6.1f} [{_yn(pass_rehadr)}] | "
          f"M_max={M_max_q:.3f} [{_yn(pass_mass)}]  →  {'ALL PASS' if allp else 'FAIL'}")
    mr_curves.append((stag, tov_q, allp))

# ── M–R plot: hadronic reference + every set + the [2, 2.5] M_⊙ window ──────
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.plot(results_tov[:, 3], results_tov[:, 4], 'k-', lw=2, label='Hadronic (T=0)')
for (stag, tov_q, allp), col in zip(mr_curves, plt.cm.viridis(np.linspace(0, 1, max(len(mr_curves), 1)))):
    ax.plot(tov_q[:, 3], tov_q[:, 4], color=col, lw=2,
            label=f"{stag} {'✓' if allp else '✗'}")
ax.axhspan(*M_window_q, color='red', alpha=0.08)
ax.axhline(2.0, color='red', ls=':', lw=1, label=r'$2\,M_\odot$')
ax.set_xlim(7, 16); ax.set_ylim(0, 3.0)
ax.set_xlabel(r'$R$ [km]'); ax.set_ylabel(r'$M$ [$M_\odot$]')
ax.set_title('CFL quark stars at T=0 (✓ = all filters pass)')
ax.legend(fontsize=7); ax.grid(alpha=0.3)
fig.tight_layout(); plt.show()


# %% [markdown]
# ## Check: is CFL the stable *bulk* phase vs unpaired?
#
# The unpCFL droplet model assigns the bulk to **CFL for $R>R_x$** and unpaired
# for $R\le R_x$ — purely by the coherence radius $R_x=\hbar c/\Delta(T)$, **not**
# by a free-energy comparison (`switching_step`: $S=1$ ⇒ CFL above $R_x$). That is
# only physical if CFL is genuinely the more stable bulk phase. At fixed $\mu_B$
# the stable phase has the **higher pressure**, so we verify
# $P_{\rm CFL}(\mu_B) > P_{\rm unp}(\mu_B)$ at $T=0$, β-eq, for every set.

# %%
# =============================================================================
#  Bulk stability: P_CFL(μ_B) vs P_unp(μ_B) at T=0, β-eq, per parameter set.
#  Stable phase = higher P at given μ_B. CFL stable ⇒ R>Rx→CFL is justified.
# =============================================================================
from eos.alphabag.eos import solve_alphabag_beta_eq

nB_chk = np.linspace(0.1, 2.5, 200)   # fm^-3
fig, ax = plt.subplots(figsize=(6.5, 5))
for p, col in zip(quark_param_sets,
                  plt.cm.viridis(np.linspace(0, 1, max(len(quark_param_sets), 1)))):
    stag = q_tag_of(p)
    pq   = get_alphabag_custom(alpha=p['alpha'], B4=p['B4'], m_s=p['m_s'])
    mu_u, P_u, mu_c, P_c = [], [], [], []
    gu = gc = None
    for nB in nB_chk:
        try:
            r = solve_alphabag_beta_eq(nB, 0.0, pq, include_photons=False,
                                       include_gluons=True, initial_guess=gu)
            if r.converged:
                mu_u.append(r.mu_B); P_u.append(r.P_total)
                gu = np.array([r.mu_u, r.mu_d, r.mu_s, r.mu_e])
        except Exception:
            pass
        try:
            rc = solve_cfl(nB, 0.0, p['Delta0'], pq, include_photons=False,
                           include_gluons=True, initial_guess=gc)
            if rc.converged:
                mu_c.append(rc.mu_B); P_c.append(rc.P_total)
                gc = np.array([rc.mu_u, rc.mu_d, rc.mu_s])
        except Exception:
            pass
    mu_u, P_u, mu_c, P_c = map(np.array, (mu_u, P_u, mu_c, P_c))

    # Compare on the common μ_B overlap.
    lo, hi = max(mu_u.min(), mu_c.min()), min(mu_u.max(), mu_c.max())
    mug = np.linspace(lo, hi, 400)
    Pu = interp1d(mu_u, P_u, bounds_error=False)(mug)
    Pc = interp1d(mu_c, P_c, bounds_error=False)(mug)
    mm = np.isfinite(Pu) & np.isfinite(Pc)
    cfl_stable = bool(np.all(Pc[mm] >= Pu[mm]))
    worst = np.nanmax((Pu - Pc)[mm]) if mm.any() else np.nan   # >0 ⇒ unpaired wins somewhere
    print(f"{stag}: CFL {'STABLE' if cfl_stable else 'NOT stable'} vs unpaired  "
          f"(max(P_unp − P_CFL) = {worst:+.2f} MeV/fm³)")

    ax.plot(mu_u, P_u, color=col, ls='--', lw=1.3, label=f'{stag} unp')
    ax.plot(mu_c, P_c, color=col, ls='-',  lw=1.8, label=f'{stag} CFL')

ax.set_xlabel(r'$\mu_B$ [MeV]'); ax.set_ylabel(r'$P$ [MeV/fm$^3$]')
ax.set_title('Bulk stability: CFL (solid) vs unpaired (dashed), T=0 β-eq')
ax.legend(fontsize=7); ax.grid(alpha=0.3)
fig.tight_layout(); plt.show()


# %% [markdown]
# ## Compute the unpaired & CFL tables
#
# One αBag table per phase, β-equilibrium, over the full $(n_B, T)$ grid.
# `compute_alphabag_table` dispatches on `phase`; for the unpaired table the
# `Delta0_values` entry is present but unused. Each call writes a `.dat` to
# `../output/tables_Qphase/` (idempotent — re-running overwrites).

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
# # Q* tables over the parameter set(s)
#
# For every quark parameter set in `quark_param_sets`, compute the Q* (critical
# droplet) tables in the **saddlepoint** flavor mode for three charge treatments —
# **LCN** (maxwell, local neutrality), **GCN** (gibbs, global), and
# **coulomb_minimize** (minimize surface+Coulomb) — and both quark phases
# (**unpaired**, **CFL**), against both hadronic backgrounds (β-eq and trapped-ν),
# scanning σ ∈ `sigma_list`.
#
# LCN/GCN solves are σ-independent (only $R_c=-2\sigma/\Delta f$ scales), so we
# solve once per (set, background, charge, phase) and rescale $R_c$ per σ.
# `coulomb_minimize` makes $R_c$ a solver unknown ⇒ genuinely σ-dependent, so it
# is re-solved at every σ.

# %%
# =============================================================================
#  Compute Q* over quark_param_sets × {β-eq, trapped} × {lcn, gcn,
#  coulomb_minimize} × {unpaired, cfl} × σ.  Stored in Qstar_sets by stem key.
# =============================================================================
from nucleation.energy_barrier.small_droplet import export_table, QstarTableData

os.makedirs('../output/tables_Qstar', exist_ok=True)

_bg_tab  = {'Hbetaeq': H_table['betaeq'], 'Htrapped': H_table['trapped']}
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
# ## Load the Q* tables
#
# Reload everything from disk (so this rebuilds `Qstar_sets` without re-solving).

# %%
# =============================================================================
#  Load all Q* tables → Qstar_sets, keyed by filename stem.
# =============================================================================
import glob

Qstar_sets = {}
for path in sorted(glob.glob('../output/tables_Qstar/Qstar_*.dat')):
    stem = os.path.splitext(os.path.basename(path))[0].removeprefix('Qstar_')
    t = load_Qstar_table(path)
    Qstar_sets[stem] = dict(table=t, interp=build_Qstar_interpolators(t))

print(f"Loaded {len(Qstar_sets)} Q* tables.")


# %% [markdown]
# ## Preliminary plot: $P_H$ vs $P_{Q^*}$ vs $n_B^H$
#
# Compare the hadronic pressure with the Q* droplet pressure for the saddlepoint
# LCN/GCN treatments (unpaired + CFL). Row 0: β-eq at two temperatures. Rows 1–2:
# trapped-ν over $(T, Y_L)$. The hadron is metastable where a $P_{Q^*}$ curve
# rises above $P_H$.

# %%
# =============================================================================
#  P_H vs P_Q* vs n_B^H for one selected set + σ. Edit the selectors below.
# =============================================================================
set_sel   = quark_param_sets[1]
stag_sel  = q_tag_of(set_sel)
sigma_sel = sigma_list[0]      # P_total is σ-independent for lcn/gcn
T_list_be = [10, 50]           # β-eq temperatures (row 0)
T_list_tr = [10, 50]           # trapped temperatures (rows 1-2)
YL_list   = [0.2, 0.4]         # trapped lepton fractions (columns)

nBg = H_table['betaeq'].grids['n_B']
x   = nBg / n_sat

def _over_nB(P_interp, *fixed):
    """Scalar-arg interpolator → array over n_B at fixed (T) or (Y_L, T)."""
    return np.array([float(P_interp(n, *fixed)) for n in nBg])

# (charge, phase, color, linestyle, label)
_curves = [
    ('lcn', 'unpaired', 'tab:orange', '--', 'LCN unp'),
    ('gcn', 'unpaired', 'tab:red',    '--', 'GCN unp'),
    ('lcn', 'cfl',      'tab:green',  '-',  'LCN CFL'),
    ('gcn', 'cfl',      'tab:blue',   '-',  'GCN CFL'),
]

fig, axes = plt.subplots(3, 2, figsize=(12, 13), sharex=True)

# Row 0: β-equilibrium.
for j, Tv in enumerate(T_list_be):
    ax = axes[0, j]
    ax.plot(x, _over_nB(H['betaeq']['P'], Tv), 'k-', lw=2, label=r'$P_H$')
    for charge, ph, c, ls, lbl in _curves:
        stem = qs_stem(stag_sel, 'Hbetaeq', charge, ph, sigma_sel)
        if stem in Qstar_sets:
            ax.plot(x, _over_nB(Qstar_sets[stem]['interp']['P'], Tv),
                    color=c, ls=ls, lw=1.3, label=lbl)
    ax.set_title(rf'$\beta$-eq,  $T={Tv}$ MeV'); ax.grid(alpha=0.3)
    if j == 0:
        ax.set_ylabel(r'$P$ [MeV/fm$^3$]'); ax.legend(fontsize=7, ncol=2)

# Rows 1-2: trapped neutrinos over (T, Y_L).
for i, Tv in enumerate(T_list_tr):
    for j, YLv in enumerate(YL_list):
        ax = axes[i + 1, j]
        ax.plot(x, _over_nB(H['trapped']['P'], YLv, Tv), 'k-', lw=2, label=r'$P_H$')
        for charge, ph, c, ls, lbl in _curves:
            stem = qs_stem(stag_sel, 'Htrapped', charge, ph, sigma_sel)
            if stem in Qstar_sets:
                ax.plot(x, _over_nB(Qstar_sets[stem]['interp']['P'], YLv, Tv),
                        color=c, ls=ls, lw=1.3, label=lbl)
        ax.set_title(rf'trapped,  $T={Tv}$ MeV,  $Y_L={YLv}$'); ax.grid(alpha=0.3)
        if j == 0:
            ax.set_ylabel(r'$P$ [MeV/fm$^3$]')

for ax in axes[-1, :]:
    ax.set_xlabel(r'$n_B^H / n_{\rm sat}$')
for ax in axes.flat:
    ax.set_xlim(0, 10)
fig.suptitle(rf"$P_H$ vs $P_{{Q^*}}$  —  {stag_sel}, σ={int(sigma_sel)} MeV/fm²", y=1.005)
fig.tight_layout(); plt.show()


# %% [markdown]
# # Thermal nucleation observables — trapped-ν background
#
# Compute $R_c, W_c, \Gamma, \tau$ over $(n_B, Y_L, T)$ for each parameter set ×
# σ. Methods:
# **saddlepoint** LCN / GCN / coulomb_minimize for the phases **unpaired**, **CFL**,
# and **unpCFL** (CFL core + unpaired mantle, switching at $R_x(T)=\hbar c/\Delta(T)$),
# all reusing the Q* tables in `Qstar_sets`; plus **frozen** LCN (unpaired only;
# Q* recomputed internally since frozen tables aren't pre-stored).
#
# *Only the trapped background for now.* Results go to `nuc_sets`, keyed by stem
# `Htrapped_{flavor}_{charge}_{phase}_{tag}_s{σ}`.

# %%
# =============================================================================
#  Compute trapped nucleation observables. Reuses Q* tables where available.
# =============================================================================
os.makedirs('../output/tables_nucleation', exist_ok=True)
V_nuc = 4.18879e51   # fm^3 — system volume for τ (sphere of radius 100 m)

# unpCFL crossover radius R_x(T) = ħc/Δ(T), BCS gap Δ(T)=Δ0·√(1−(T/T_c)²),
# T_c = 0.57·Δ0; ∞ above T_c (no pairing → purely unpaired droplet).
from eos.general.physics_constants import hc

def crossover_radius(T, Delta0, T_c_factor=0.57):
    T_c = T_c_factor * Delta0
    ratio = np.asarray(T / T_c)
    gap = np.where(ratio < 1, Delta0 * np.sqrt(np.maximum(0, 1 - ratio**2)), 0.0)
    return np.where(gap > 0, hc / gap, np.inf)

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
    Rx   = crossover_radius(Rx_T, p['Delta0'])
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
# ## Load the trapped nucleation tables
#
# Reload from disk (rebuild `nuc_sets` without recomputing).

# %%
# =============================================================================
#  Load trapped nucleation observables → nuc_sets, keyed by filename stem.
# =============================================================================
import glob

nuc_sets = {}
for path in sorted(glob.glob('../output/tables_nucleation/nucleation_Htrapped_*.dat')):
    stem = os.path.splitext(os.path.basename(path))[0].removeprefix('nucleation_')
    nuc_sets[stem] = load_thermal_nucleation_table(path)

print(f"Loaded {len(nuc_sets)} trapped nucleation tables.")


# %% [markdown]
# ## Plot: $R_c$, $W_c/T$, $\log_{10}\Gamma$, $\log_{10}\tau$ vs $n_B^H$
#
# Three 4-panel views at a fixed $Y_L$ slice (set by `set_sel`, `YL_sel`):
#
# 1. **fixed $T,\sigma$** — overlay the methods (frozen LCN, saddle LCN/GCN, Coulomb min);
# 2. **fixed $T$, one method+phase** — overlay $\sigma$;
# 3. **fixed $\sigma$, one method+phase** — overlay $T$.
#
# Phase ∈ {`unpaired`, `cfl`, `unpCFL`}.

# %%
# =============================================================================
#  Shared selectors + a 4-row plotting helper for the trapped nucleation views.
# =============================================================================
set_sel  = quark_param_sets[0]
stag_sel = q_tag_of(set_sel)
YL_sel   = 0.2                     # nearest grid value used

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


# %%
# =============================================================================
#  (1) Fixed T, Y_L, σ — overlay methods (phase = unpaired → all four).
# =============================================================================
T_sel, sigma_sel, phase_sel = 30.0, sigma_list[0], 'unpaired'

_methods = [
    ('frozen',      'lcn',              'tab:gray',   ':',  'frozen LCN'),
    ('saddlepoint', 'lcn',              'tab:orange', '--', 'saddle LCN'),
    ('saddlepoint', 'gcn',              'tab:red',    '-',  'saddle GCN'),
    ('saddlepoint', 'coulomb_minimize', 'tab:blue',   '-.', 'Coulomb min'),
]
curves = [(_get(fl, ch, phase_sel, sigma_sel), _iT(T_sel), c, ls, lbl)
          for fl, ch, c, ls, lbl in _methods]
nuc_panels(curves, rf"trapped {phase_sel} — $T\approx{Tg[_iT(T_sel)]:.0f}$ MeV, "
                   rf"$Y_L\approx{YL_used:.2f}$, σ={int(sigma_sel)} — {stag_sel}")


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
# ## Nucleation density $n_B(T)$ at $\tau = \tau_{\rm target}$ (trapped)
#
# For each method, root-find the curve $\tau = \tau_{\rm target}$ in the
# $(n_B, T)$ plane (`scan='n_B'`), at the selected $Y_L$ slice.

# %%
# =============================================================================
#  Trapped nucleation curve τ = τ_target, per method (selected set/σ/phase/Y_L).
# =============================================================================
tau_target = 1e-3   # s

def nuc_curve(res, iYL=0):
    """(n_B, T) of the τ=target curve; for trapped take the iYL Y_L slice."""
    if getattr(res, 'scan', 'T') == 'n_B':
        nB, T = np.asarray(res.n_B_arr), np.asarray(res.T_nuc)
    else:
        nB, T = np.asarray(res.n_B_nuc), np.asarray(res.T_arr)
    if T.ndim == 2:
        T  = T[:, iYL]
        nB = nB[:, iYL] if nB.ndim == 2 else nB
    return nB, T

# Chosen densities (in n_sat units) at which to read off T_nuc.
nB_vec = np.array([2.0, 3.0, 4.0, 5.0, 6.0]) * n_sat

fig, ax = plt.subplots(figsize=(7, 5))
for flavor, charge, c, ls, lbl in _methods:
    stem = f"Htrapped_{flavor}_{charge}_{phase_sel}_{stag_sel}_s{int(sigma_sel)}"
    if stem not in nuc_sets:
        continue
    res = compute_nucleation_density(nuc_sets[stem], tau_target=tau_target, scan='n_B')
    nB, T = nuc_curve(res, iYL)
    m = np.isfinite(nB) & np.isfinite(T)
    if m.any():
        ax.plot(nB[m] / n_sat, T[m], color=c, ls=ls, lw=1.5, marker='.', ms=4, label=lbl)
        # T at the requested densities: interp along n_B (the monotonic axis);
        # NaN outside the converged n_B range so we don't extrapolate.
        T_vec = np.interp(nB_vec, nB[m], T[m], left=np.nan, right=np.nan)
        ax.plot(nB_vec / n_sat, T_vec, color=c, ls='none', marker='o', ms=8,
                mfc='none', mew=1.5)
        print(f"  {lbl}: {int(np.sum(res.converged))}/{res.converged.size} converged")
        for x, t in zip(nB_vec / n_sat, T_vec):
            print(f"      n_B/n_sat={x:.1f} -> T_nuc={t:.2f} MeV" if np.isfinite(t)
                  else f"      n_B/n_sat={x:.1f} -> T_nuc=  (outside range)")

ax.set_xlabel(r'$n_B^H / n_{\rm sat}$'); ax.set_ylabel(r'$T_{\rm nuc}$ [MeV]')
ax.set_title(rf"τ={tau_target*1e3:g} ms — trapped, {phase_sel}, $Y_L\approx{YL_used:.2f}$, "
             rf"{stag_sel}, σ={int(sigma_sel)}")
ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout(); plt.show()


# %% [markdown]
# ## $T_{\rm nuc}(n_B^H)$ for unpaired / CFL / unpCFL — coulomb_minimize
#
# Nucleation curve τ = τ_target overlaying the three quark phases, for the
# coulomb_minimize saddlepoint method (selected set / σ / Y_L).

# %%
# =============================================================================
#  T_nuc vs n_B^H/n_sat at τ=τ_target, coulomb_minimize, phases unp/CFL/unpCFL.
# =============================================================================
flavor_d, charge_d, sigma_d = 'saddlepoint', 'coulomb_minimize', sigma_list[0]

_phase_styles = [
    ('unpaired', 'tab:orange', '--', 'unpaired'),
    ('cfl',      'tab:green',  '-.', 'CFL'),
    ('unpCFL',   'tab:blue',   '-',  'unpCFL'),
]

fig, ax = plt.subplots(figsize=(7, 5))
for ph, c, ls, lbl in _phase_styles:
    stem = f"Htrapped_{flavor_d}_{charge_d}_{ph}_{stag_sel}_s{int(sigma_d)}"
    if stem not in nuc_sets:
        continue
    res = compute_nucleation_density(nuc_sets[stem], tau_target=tau_target, scan='n_B')
    nB, T = nuc_curve(res, iYL)
    m = np.isfinite(nB) & np.isfinite(T)
    if m.any():
        ax.plot(nB[m] / n_sat, T[m], color=c, ls=ls, lw=1.5, marker='.', ms=4, label=lbl)
    print(f"  {lbl}: {int(np.sum(res.converged))}/{res.converged.size} converged")

ax.set_xlabel(r'$n_B^H / n_{\rm sat}$'); ax.set_ylabel(r'$T_{\rm nuc}$ [MeV]')
ax.set_title(rf"τ={tau_target*1e3:g} ms — trapped, coulomb_minimize, "
             rf"$Y_L\approx{YL_used:.2f}$, σ={int(sigma_d)} — {stag_sel}")
ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout(); plt.show()

