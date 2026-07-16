"""Critical-surface-tension parameter scans for 2-family quark nucleation.

Reusable engine extracted from the ``2fam_nucleation`` notebook. Three layers:

* **CFL filters** -- Witten bound, no-rehadronization, TOV M_max window
  (``cfl_eos_at_params``, ``passes_cfl_filters``).
* **Single-point nucleation** -- the surface tension ``sigma_crit`` at which the
  central nucleation time equals a target, for one (alpha, B4, Delta0) at one
  star (``central_state``, ``sigma_target_pt``); supports unpaired / cfl /
  unpCFL droplets and lcn / gcn / coulomb_minimize charge modes.
* **Scan + plot** -- grid scan over (alpha, B4, Delta0) with a filter cache and
  optional joblib parallelism (``scan_cfl_filters``, ``compute_sigma_crit``,
  ``run_sigma_crit_scan``), the sigma_crit heatmaps (``plot_sigma_crit_grid``)
  and the M-R / P(mu_B) replay (``replay_cfl``).

Notebook globals are replaced by three small config objects: ``FilterConfig``
(CFL-filter setup), ``NucConfig`` (nucleation/brentq setup) and ``StarMatch``
(the MT0 -> central-conditions map). Build the last via ``make_star_match`` and
the P_H(mu_B) comparator via ``build_PH_of_muB``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq

from eos.alphabag.parameters import get_alphabag_custom
from eos.alphabag.eos import (solve_cfl, solve_alphabag_beta_eq,
                              solve_alphabag_fixed_yc_ys)
from eos.alphabag.thermodynamics_quarks import T_critical   # CFL gap melting temp (single source)
from eos.general.physics_constants import hc
from eos.tov.solver import (EOSTable_for_TOV, generate_ec_logspace,
                            compute_tov_sequence, truncate_to_stable_branch)
from nucleation.critical import critical_droplet, _build_H_from_interp
from nucleation.composition import get_solver_Qs
from nucleation.rates import nucleation_rate, nucleation_time

try:
    import joblib  # noqa: F401  -- presence check; Parallel imported in-function
    _HAVE_JOBLIB = True
except Exception:
    _HAVE_JOBLIB = False

# Filter-reason -> integer code. The three filters short-circuit cheap->expensive,
# so codes are *nested*: contouring this field at 1.5 / 2.5 gives the Witten /
# no-rehadr pass edges, and 3.5 (== cfl_ok edge) the full-acceptance edge.
REASON_CODE = {'solve': 0, 'witten': 1, 'rehadr': 2, 'mmax': 3, 'OK': 4,
               'twoflavor': 5}  # 2-flavor (ud) matter is bound -> nuclei unstable


# =============================================================================
#  Config objects
# =============================================================================
@dataclass
class FilterConfig:
    """Everything the CFL filters need beyond (alpha, B4, Delta0).

    P_H_of_muB / mu_B_H_sorted / P_H_sorted come from ``build_PH_of_muB``."""
    P_H_of_muB: Callable
    mu_B_H_sorted: np.ndarray
    P_H_sorted: np.ndarray
    m_s: float = 100.0
    n_B_grid: np.ndarray = field(default_factory=lambda: np.linspace(0.05, 2.5, 250))
    e_c_vec_tov: np.ndarray = field(
        default_factory=lambda: generate_ec_logspace(e_min=100, e_max=2000, n_points=80))
    M_max_window: tuple = (2.0, np.inf)
    e_over_nB_max: float = 930.0          # 3-flavor SQM must be bound: e/nB|P=0 < this
    T_eos: float = 0.0
    # 2-flavor (ud+e, charge-neutral, beta-eq) matter must NOT be bound, else ordinary
    # nuclei would decay to it -> require e/nB|P=0 > e_over_nB_2flavor (Delta0-independent).
    check_2flavor: bool = True
    e_over_nB_2flavor: float = 930.0
    # TOV backend for the M_max filter and the M-R replay: 'scipy' (trusted
    # reference) or 'fast' (numba, ~100x). The scan/replay already parallelise
    # over grid cells with joblib, so the fast solver is called serially here
    # (compute_tov_sequence is single-threaded) to avoid numba-threads x joblib oversubscription.
    tov_backend: str = 'scipy'


@dataclass
class NucConfig:
    """Nucleation / root-find setup for sigma_crit."""
    sig_lo: float = 1.0
    sig_hi: float = 300.0
    n_sigma_scan: int = 40          # coarse sigma grid for the robust sigma_crit scan
    tau_target: float = 1e-3        # s
    V: float = 4.18879e51           # fm^3 (sphere R = 100 m)
    include_photons: bool = True
    include_gluons: bool = True
    include_thermal_neutrinos: bool = True


@dataclass
class StarMatch:
    """Maps a cold-beta-eq gravitational mass MT0 to trapped central conditions."""
    H_iso_trapped: dict          # interpolator dict with key 'T'
    H_trapped: dict              # interpolator dict: P, eps, mu_*, Y_C, Y_S
    YLH: float
    S: float
    Mb_of_MT0: Callable          # from cold beta-eq TOV
    nBHc_of_Mb: Callable         # from trapped/isentropic TOV at (YLH, S)


def build_PH_of_muB(H_betaeq: dict, n_B_grid: np.ndarray, T: float):
    """Cold P_H(mu_B) comparator for the no-rehadronization filter.

    Returns (P_H_of_muB, mu_B_H_sorted, P_H_sorted)."""
    mu = H_betaeq['mu_B'](n_B_grid, T)
    P = H_betaeq['P'](n_B_grid, T)
    ok = np.isfinite(mu) & np.isfinite(P)
    order = np.argsort(mu[ok])
    mu_s, P_s = mu[ok][order], P[ok][order]
    return interp1d(mu_s, P_s, kind='linear', bounds_error=False, fill_value=np.nan), mu_s, P_s


def make_star_match(H: dict, YLH: float, S: float,
                    tov_betaeq_path: str, tov_trapped_path: str) -> StarMatch:
    """Build a StarMatch from the H interpolators + the two TOV tables.

    Columns: cold beta-eq TOV 4=M,5=Mb; trapped TOV 2=n_Bc,4=M,5=Mb."""
    tov0 = np.loadtxt(tov_betaeq_path)
    M0, Mb0 = tov0[:, 4], tov0[:, 5]
    i0 = int(np.argmax(M0)) + 1
    Mb_of_MT0 = interp1d(M0[:i0], Mb0[:i0], kind='cubic', bounds_error=True)
    tovh = np.loadtxt(tov_trapped_path)
    nBc, Mh, Mbh = tovh[:, 2], tovh[:, 4], tovh[:, 5]
    ih = int(np.argmax(Mh)) + 1
    nBHc_of_Mb = interp1d(Mbh[:ih], nBc[:ih], kind='cubic', bounds_error=True)
    return StarMatch(H_iso_trapped=H['iso_trapped'], H_trapped=H['trapped'],
                     YLH=YLH, S=S, Mb_of_MT0=Mb_of_MT0, nBHc_of_Mb=nBHc_of_Mb)


# =============================================================================
#  CFL EoS + filters
# =============================================================================
def cfl_eos_at_params(alpha, B4, Delta0, cfg: FilterConfig):
    """Solve the CFL beta-eq EoS at T=cfg.T_eos over cfg.n_B_grid (warm-started).
    Returns (P, e, mu, ok-mask)."""
    p = get_alphabag_custom(alpha=alpha, B4=B4, m_s=cfg.m_s)
    n = cfg.n_B_grid
    P = np.full_like(n, np.nan); e = np.full_like(n, np.nan); mu = np.full_like(n, np.nan)
    ok = np.zeros_like(n, dtype=bool)
    guess = None
    for i, nB in enumerate(n):
        try:
            r = solve_cfl(nB, cfg.T_eos, Delta0, p, include_photons=False,
                          include_gluons=True, initial_guess=guess)
        except Exception:
            r = None
        if r is None or not r.converged or r.error > 1e-6:
            continue
        P[i], e[i], mu[i] = r.P_total, r.e_total, r.mu_B
        guess = np.array([r.mu_u, r.mu_d, r.mu_s])
        ok[i] = True
    return P, e, mu, ok


def unpaired_eos_at_params(alpha, B4, cfg: FilterConfig):
    """Solve the UNPAIRED beta-eq alpha-Bag EoS at T=cfg.T_eos over cfg.n_B_grid
    (warm-started). Returns (P, e, mu, ok-mask). No Delta_0 — an unpaired droplet
    has no CFL gap, so the unpaired-matter viability filters never reference it."""
    p = get_alphabag_custom(alpha=alpha, B4=B4, m_s=cfg.m_s)
    n = cfg.n_B_grid
    P = np.full_like(n, np.nan); e = np.full_like(n, np.nan); mu = np.full_like(n, np.nan)
    ok = np.zeros_like(n, dtype=bool)
    guess = None
    for i, nB in enumerate(n):
        try:
            r = solve_alphabag_beta_eq(nB, cfg.T_eos, p, include_photons=False,
                                       include_gluons=True, initial_guess=guess)
        except Exception:
            r = None
        if r is None or not r.converged or r.error > 1e-6:
            continue
        P[i], e[i], mu[i] = r.P_total, r.e_total, r.mu_B
        guess = np.array([r.mu_u, r.mu_d, r.mu_s, r.mu_e])   # beta-eq has 4 unknowns
        ok[i] = True
    return P, e, mu, ok


def zero_crossing(x, y):
    """First sign change in y(x); returns (j, frac) with x_cross = x[j]+frac*dx."""
    s = np.sign(y)
    idx = np.where(np.diff(s) != 0)[0]
    if idx.size == 0:
        return None
    j = idx[0]
    return j, -y[j] / (y[j + 1] - y[j])


def ud_eps_per_nB(alpha, B4, cfg: FilterConfig):
    """eps/n_B [MeV] at P=0 for cold, charge-neutral, beta-equilibrated 2-flavor
    (ud+e, Y_S=0) unpaired matter -- the 'is 2-flavor matter bound?' line. np.nan
    if no P=0 crossing. INDEPENDENT of Delta0 and m_s (no strange quarks). The SQM
    hypothesis needs this ABOVE ~930 MeV so ordinary (ud) nuclei do not decay to
    quark matter -- the companion to the 3-flavor Witten bound."""
    p = get_alphabag_custom(alpha=alpha, B4=B4, m_s=cfg.m_s)
    n = cfg.n_B_grid
    P = np.full_like(n, np.nan)
    e = np.full_like(n, np.nan)

    def solve(nB, yc, guess):
        return solve_alphabag_fixed_yc_ys(nB, yc, 0.0, cfg.T_eos, p,
                                          include_photons=False, include_gluons=True,
                                          include_electrons=True, initial_guess=guess)

    guess = None
    for i, nB in enumerate(n):
        # beta-eq quark charge fraction Y_C: mu_d - mu_u - mu_e = 0 (electrons neutralise).
        def g(yc, _nB=nB, _g=guess):
            r = solve(_nB, yc, _g)
            return r.mu_d - r.mu_u - r.mu_e
        try:
            if g(1e-4) * g(0.34) > 0:      # no beta-eq root bracketed at this density
                continue
            yc = brentq(g, 1e-4, 0.34, xtol=1e-5)
            r = solve(nB, yc, guess)
        except Exception:
            continue
        if not r.converged or r.error > 1e-6:
            continue
        P[i], e[i] = r.P_total, r.e_total
        guess = np.array([r.mu_u, r.mu_d, r.mu_s])   # warm-start the next density
    ok = np.isfinite(P)
    if ok.sum() < 5:
        return np.nan
    cr = zero_crossing(n[ok], P[ok])
    if cr is None:
        return np.nan
    j, fr = cr
    nn, ee = n[ok], e[ok]
    nP0 = nn[j] + fr * (nn[j + 1] - nn[j])
    eP0 = ee[j] + fr * (ee[j + 1] - ee[j])
    return float(eP0 / nP0) if nP0 > 0 else np.nan


def passes_cfl_filters(alpha, B4, Delta0, cfg: FilterConfig):
    """(ok, M_max, reason) for the three CFL filters, ordered cheap->expensive:
    (1) Witten e/n_B|_{P=0} < cfg.e_over_nB_max, (2) P_CFL > P_H on the mu_B
    overlap, (3) TOV M_max in cfg.M_max_window. M_max is finite once TOV runs."""
    P_cfl, e_cfl, mu_cfl, ok = cfl_eos_at_params(alpha, B4, Delta0, cfg)
    if ok.sum() < 20:
        return False, np.nan, 'solve'
    n_ok, P_ok, e_ok, mu_ok = cfg.n_B_grid[ok], P_cfl[ok], e_cfl[ok], mu_cfl[ok]

    cr = zero_crossing(n_ok, P_ok)
    if cr is None or P_ok[-1] <= 0:
        return False, np.nan, 'witten'
    j, fr = cr
    n_P0 = n_ok[j] + fr * (n_ok[j + 1] - n_ok[j])
    e_P0 = e_ok[j] + fr * (e_ok[j + 1] - e_ok[j])
    if e_P0 / n_P0 >= cfg.e_over_nB_max:
        return False, np.nan, 'witten'

    mu_lo = max(mu_ok.min(), cfg.mu_B_H_sorted.min())
    mu_hi = min(mu_ok.max(), cfg.mu_B_H_sorted.max())
    if mu_hi <= mu_lo:
        return False, np.nan, 'rehadr'
    mu_chk = np.linspace(mu_lo, mu_hi, 500)
    Pc = interp1d(mu_ok, P_ok, kind='linear', bounds_error=False,
                  fill_value=np.nan)(mu_chk)
    Ph = cfg.P_H_of_muB(mu_chk)
    m = np.isfinite(Pc) & np.isfinite(Ph)
    if not np.all(Pc[m] > Ph[m]):
        return False, np.nan, 'rehadr'

    pos = P_ok > 0
    if pos.sum() < 10:
        return False, np.nan, 'mmax'
    try:
        tov = compute_tov_sequence(
            EOSTable_for_TOV(P=P_ok[pos], epsilon=e_ok[pos], nB=n_ok[pos]),
            e_c_vec=cfg.e_c_vec_tov, add_crust_table='No',
            compute_baryonic_mass=True, compute_tidal=False, verbose=False,
            backend=cfg.tov_backend)
        _, M_max, _ = truncate_to_stable_branch(tov, verbose=False)
    except Exception:
        return False, np.nan, 'mmax'
    lo, hi = cfg.M_max_window
    if not np.isfinite(M_max) or not (lo <= M_max <= hi):
        return False, M_max, 'mmax'
    return True, M_max, 'OK'


def passes_unpaired_filters(alpha, B4, cfg: FilterConfig):
    """(ok, M_max, reason) for UNPAIRED strange-quark matter — the filter built
    on the UNPAIRED alpha-Bag EoS (no Delta_0):

      (1) no re-hadronization: P_unp(mu_B) > P_H(mu_B) on the mu_B overlap;
      (2) TOV M_max in cfg.M_max_window, structure integrated with the
          UNPAIRED quark EoS (not CFL).

    Unlike ``passes_cfl_filters`` there is no Witten / 2-flavour column here and
    no Delta_0 anywhere: the whole point is that the unpaired sigma_crit map is
    gap-independent, so its viability gate must be too."""
    P_q, e_q, mu_q, ok = unpaired_eos_at_params(alpha, B4, cfg)
    if ok.sum() < 20:
        return False, np.nan, 'solve'
    n_ok, P_ok, e_ok, mu_ok = cfg.n_B_grid[ok], P_q[ok], e_q[ok], mu_q[ok]

    # (1) no re-hadronization — P_unp must exceed P_H on the mu_B overlap.
    mu_lo = max(mu_ok.min(), cfg.mu_B_H_sorted.min())
    mu_hi = min(mu_ok.max(), cfg.mu_B_H_sorted.max())
    if mu_hi <= mu_lo:
        return False, np.nan, 'rehadr'
    mu_chk = np.linspace(mu_lo, mu_hi, 500)
    Pc = interp1d(mu_ok, P_ok, kind='linear', bounds_error=False,
                  fill_value=np.nan)(mu_chk)
    Ph = cfg.P_H_of_muB(mu_chk)
    m = np.isfinite(Pc) & np.isfinite(Ph)
    if not np.all(Pc[m] > Ph[m]):
        return False, np.nan, 'rehadr'

    # (2) M_max from the UNPAIRED quark EoS.
    pos = P_ok > 0
    if pos.sum() < 10:
        return False, np.nan, 'mmax'
    try:
        tov = compute_tov_sequence(
            EOSTable_for_TOV(P=P_ok[pos], epsilon=e_ok[pos], nB=n_ok[pos]),
            e_c_vec=cfg.e_c_vec_tov, add_crust_table='No',
            compute_baryonic_mass=True, compute_tidal=False, verbose=False,
            backend=cfg.tov_backend)
        _, M_max, _ = truncate_to_stable_branch(tov, verbose=False)
    except Exception:
        return False, np.nan, 'mmax'
    lo, hi = cfg.M_max_window
    if not np.isfinite(M_max) or not (lo <= M_max <= hi):
        return False, M_max, 'mmax'
    return True, M_max, 'OK'


# =============================================================================
#  Droplet-level re-hadronization test  (ΔP = P_Q* - P_H over n_BH)
# =============================================================================
# This is the *droplet* re-hadronization test, distinct from the bulk-EoS one in
# passes_*_filters. There Q* is the bulk quark EoS crossed against P_H in mu_B
# space. Here, for each hadronic density n_BH we solve the Q* droplet at the LOCAL
# hadronic conditions (n_BH, Y_LH, T) -- so mu_B^Q* = mu_B^H by the saddlepoint
# equations -- and compare its pressure P_Q* to P_H(n_BH). Physics: at matched
# mu_B a higher-pressure phase is favoured, so ΔP = P_Q* - P_H > 0 means the quark
# droplet is the stable phase and cannot re-hadronize at that density.

def rehad_pressure_profile(
        H_interp, params, n_B_H_grid, *, Y_L_H=None, T=None,
        flavor_mode='saddlepoint', electric_charge_mode='gcn',
        quark_phase='unpaired', Delta0=None,
        include_photons=True, include_gluons=True,
        include_thermal_neutrinos=True):
    """ΔP(n_BH) = P_Q*(n_BH conditions) - P_H(n_BH), swept over n_B_H_grid.

    Q* is solved at each hadronic point via the same composition machinery the
    barrier uses (get_solver_Qs); failed solves give NaN. n_B_H_grid must be
    ascending (the strong flag differentiates along it). Returns the grid (echoed)
    and the ΔP array [MeV/fm^3]."""
    eq_type = H_interp['eq_type']
    solver = get_solver_Qs(
        flavor_mode, electric_charge_mode, params, quark_phase=quark_phase,
        Delta0=Delta0, include_photons=include_photons,
        include_gluons=include_gluons,
        include_thermal_neutrinos=include_thermal_neutrinos)

    dP = np.full(len(n_B_H_grid), np.nan)
    guess = None
    for i, n_B_H in enumerate(n_B_H_grid):
        if eq_type == 'beta_eq':
            pt = (n_B_H, T)
        elif eq_type == 'trapped_neutrinos':
            pt = (n_B_H, Y_L_H, T)
        elif eq_type == 'fixed_yc':
            pt = (n_B_H, Y_L_H, T)   # Y_L_H reused as Y_C_H here
        else:
            raise ValueError(f"Unsupported eq_type: '{eq_type}'")
        H = _build_H_from_interp(H_interp, pt, eq_type)
        Qs = solver(H, initial_guess=guess)
        if isinstance(Qs, tuple):          # coulomb_minimize returns (Q*, R_c)
            Qs = Qs[0]
        if Qs is None:
            continue
        dP[i] = Qs.P_total - H.P_total
        guess = np.array([Qs.mu_u, Qs.mu_d, Qs.mu_s, Qs.mu_e])
    return np.asarray(n_B_H_grid), dP


def rehad_flags(n_B_H_grid, dP):
    """(no_rehad, no_rehad_strong, n_BH_cross) from a ΔP(n_BH) profile.

    * no_rehad         -- once ΔP first reaches >=0 (crossing n_BH_cross), it stays
                          strictly >0 for every larger n_BH (quark phase never
                          gives pressure back to hadronic matter).
    * no_rehad_strong  -- ΔP is strictly increasing in n_BH everywhere: the quark
                          droplet gets monotonically more favoured with density.

    NaNs (failed Q* solves) are dropped first. Flags are False and n_BH_cross NaN
    if there is no crossing or too few finite points."""
    n = np.asarray(n_B_H_grid, float)
    d = np.asarray(dP, float)
    fin = np.isfinite(d)
    n, d = n[fin], d[fin]
    if d.size < 2:
        return False, False, np.nan

    no_rehad_strong = bool(np.all(np.diff(d) > 0.0))

    cross_idx = np.argmax(d >= 0.0) if np.any(d >= 0.0) else -1
    if cross_idx < 0:
        return False, no_rehad_strong, np.nan
    n_BH_cross = float(n[cross_idx])
    no_rehad = bool(np.all(d[cross_idx + 1:] > 0.0))
    return no_rehad, no_rehad_strong, n_BH_cross


def replay_cfl(alpha, B4, Delta0, cfg: FilterConfig, e_c_vec=None):
    """Cold CFL EoS + TOV for one set -> dict(mu, P, R, M) or None on failure.

    Plot-oriented replay: the M-R curve only feeds a figure, so the TOV sweep
    uses a coarser central-density grid than the acceptance filter (40 points
    instead of cfg.e_c_vec_tov's 80), skips the baryonic-mass quadrature (M_b
    is never plotted) and slices the stable branch at argmax(M) directly
    (no find_mmax_precise refinement). ~2x faster per set than the filter path
    with a visually identical curve. Pass e_c_vec to override the grid.
    (Raw TOV cols without M_b: 0=e_c 1=n_c 2=P_c 3=R[km] 4=M[Msun].)"""
    P, e, mu, ok = cfl_eos_at_params(alpha, B4, Delta0, cfg)
    if ok.sum() < 20:
        return None
    n_ok, P_ok, e_ok, mu_ok = cfg.n_B_grid[ok], P[ok], e[ok], mu[ok]
    pos = P_ok > 0
    if pos.sum() < 10:
        return None
    if e_c_vec is None:
        e_c_vec = generate_ec_logspace(e_min=100, e_max=2000, n_points=40)
    try:
        tov = compute_tov_sequence(
            EOSTable_for_TOV(P=P_ok[pos], epsilon=e_ok[pos], nB=n_ok[pos]),
            e_c_vec=e_c_vec, add_crust_table='No',
            compute_baryonic_mass=False, compute_tidal=False, verbose=False,
            backend=cfg.tov_backend)
    except Exception:
        return None
    if tov.shape[0] < 3:
        return None
    i_max = int(np.argmax(tov[:, 4])) + 1          # stable branch: up to M_max
    return dict(mu=mu_ok, P=P_ok, R=tov[:i_max, 3], M=tov[:i_max, 4])


def replay_accepted(sig_crit, alpha_slices, B4_grid, Delta0_grid,
                    cfg: FilterConfig, max_curves=None, n_jobs=-1, verbose=True):
    """Replay the cold CFL EoS + TOV of every ACCEPTED cell of a sigma_crit grid.

    Accepted = finite sigma_crit (CFL-pass AND nucleating). Each replay is an
    independent (EoS solve + TOV) -> joblib-parallel over cells. ``max_curves``
    evenly subsamples the accepted list first: for an overlay plot a few
    hundred curves are visually identical to several thousand at a fraction of
    the cost (None = replay all).

    Returns
    -------
    list of (curve_dict, sigma_crit) with curve_dict from ``replay_cfl``
    (keys mu, P, R, M); failed replays are dropped.
    """
    alpha_slices = np.atleast_1d(alpha_slices)
    accept = [(alpha_slices[ia], B4_grid[jx], Delta0_grid[i],
               float(sig_crit[ia, i, jx]))
              for ia in range(len(alpha_slices))
              for i in range(len(Delta0_grid))
              for jx in range(len(B4_grid))
              if np.isfinite(sig_crit[ia, i, jx])]
    n_total = len(accept)
    if max_curves and n_total > max_curves:
        accept = accept[::int(np.ceil(n_total / max_curves))]
    if verbose:
        print(f"replay: {len(accept)}/{n_total} accepted cells "
              f"(max_curves={max_curves}), "
              f"{'parallel' if (n_jobs != 1 and _HAVE_JOBLIB) else 'serial'}...",
              flush=True)
    if n_jobs != 1 and _HAVE_JOBLIB:
        from joblib import Parallel, delayed
        out = Parallel(n_jobs=n_jobs, verbose=(5 if verbose else 0))(
            delayed(replay_cfl)(a, b, d, cfg) for a, b, d, _ in accept)
    else:
        out = [replay_cfl(a, b, d, cfg) for a, b, d, _ in accept]
    curves = [(c, sc) for c, (_, _, _, sc) in zip(out, accept) if c is not None]
    if verbose:
        print(f"replay: reconstructed {len(curves)}/{len(accept)} curves.", flush=True)
    return curves


# =============================================================================
#  Single-point nucleation
# =============================================================================
def crossover_radius(T, Delta0):
    """unpCFL coherence radius R_x(T) = hc/Delta(T), using the SAME CFL gap as the
    EoS: Delta(T)=Delta0*sqrt(1-(T/Tc)^2) with Tc=T_critical(Delta0) (=0.57*2^(1/3)
    *Delta0, incl. the CFL enhancement). inf at/above Tc (no pairing -> unpaired).
    Tc is taken from eos.T_critical so R_x can never drift from the gap in gap_cfl."""
    if Delta0 <= 0:
        return np.inf
    T_c = T_critical(Delta0)
    ratio = np.asarray(T / T_c)
    gap = np.where(ratio < 1, Delta0 * np.sqrt(np.maximum(0, 1 - ratio**2)), 0.0)
    return np.where(gap > 0, hc / gap, np.inf)


def hadronic_point(H_trapped: dict, nB, YL, T):
    """Trapped hadronic state (SimpleNamespace) at one (n_B, Y_L, T) point.

    Evaluates the trapped-neutrino interpolator dict at the point and packs
    every field the Q* solvers and ``driving_force`` need. Charge neutrality
    of the bulk hadronic phase fixes Y_e = Y_C; lepton-number conservation
    fixes Y_nu = Y_L - Y_C."""
    pt = (nB, YL, T)
    h = SimpleNamespace(
        n_B=float(nB), T=float(T),
        P_total=float(H_trapped['P'](*pt)),   e_total=float(H_trapped['eps'](*pt)),
        mu_B=float(H_trapped['mu_B'](*pt)),   mu_C=float(H_trapped['mu_C'](*pt)),
        mu_S=float(H_trapped['mu_S'](*pt)),   mu_e=float(H_trapped['mu_e'](*pt)),
        mu_nu=float(H_trapped['mu_nu'](*pt)),
        Y_C=float(H_trapped['Y_C'](*pt)),     Y_S=float(H_trapped['Y_S'](*pt)))
    h.Y_e = h.Y_C
    h.Y_nu = float(YL) - h.Y_C
    return h


def central_state(MT0, star: StarMatch):
    """(nBHc, T_c, H_pt) at the centre of the trapped star with Mb = Mb(MT0)."""
    Mb = float(star.Mb_of_MT0(MT0))
    nBHc = float(star.nBHc_of_Mb(Mb))
    T_c = float(star.H_iso_trapped['T'](nBHc, star.YLH, star.S))
    H_pt = hadronic_point(star.H_trapped, nBHc, star.YLH, T_c)
    return nBHc, T_c, H_pt


def critical_droplet_pt(sigma, H_pt, T_c, flavor, charge, phase, cache,
                        params, Delta0, nuc: NucConfig):
    """Critical droplet at one hadronic point: (R_c [fm], W_c [MeV], Qs).

    Thin wrapper over ``nucleation.critical.critical_droplet`` -- the single
    barrier engine shared with the tables and the notebook W(R) plots, so the
    physics can never diverge. ``cache`` (a dict) memoizes the sigma-independent
    composition solves across the sigma scan.

    Returns
    -------
    (R_c, W_c, Qs) with the conventions
      * (nan, nan, None)  -- solver failed / no critical droplet found;
      * (nan, inf,  Qs)   -- hadronic phase STABLE (Delta_f >= 0): infinite
                             barrier, nucleation never happens;
      * finite values     -- genuine critical droplet; Qs is the droplet state
                             (for unpCFL: the phase selected at the peak).
    """
    Rx = float(crossover_radius(T_c, Delta0)) if phase == 'unpCFL' else None
    cd = critical_droplet(
        H_pt, sigma=sigma, params=params, flavor_mode=flavor,
        electric_charge_mode=charge, quark_phase=phase, Delta0=Delta0, Rx=Rx,
        cache=cache, include_photons=nuc.include_photons,
        include_gluons=nuc.include_gluons,
        include_thermal_neutrinos=nuc.include_thermal_neutrinos)
    return cd.R_c, cd.W_c, cd.Qs


def tau_pt(sigma, H_pt, T_c, flavor, charge, phase, cache, params, Delta0, nuc: NucConfig):
    """Central nucleation time tau [s] at one sigma, for one droplet phase.

    Thin wrapper over ``critical_droplet_pt``: converts (R_c, W_c, Qs) to a
    Langer rate and tau = 1/(Gamma V). Returns NaN on solver failure, +inf
    when the hadronic phase is stable (W_c = inf -> never nucleates)."""
    R_c, W_c, Qs = critical_droplet_pt(sigma, H_pt, T_c, flavor, charge, phase,
                                       cache, params, Delta0, nuc)
    if np.isposinf(W_c):
        return np.inf
    if Qs is None or not (np.isfinite(R_c) and R_c > 0
                          and np.isfinite(W_c) and W_c > 0):
        return np.nan
    Gamma = float(nucleation_rate(W_c, R_c, sigma, T_c, H_pt, Qs))
    return float(nucleation_time(Gamma, nuc.V)) if Gamma > 0 else np.inf


def sigma_target_pt(H_pt, T_c, flavor, charge, phase, params, Delta0, nuc: NucConfig):
    """sigma where tau(sigma) = nuc.tau_target at the star centre.

    Robust to solver drop-outs. tau(sigma) can be non-monotonic and can be NaN
    over whole sigma sub-ranges where no critical droplet exists (e.g. the unpCFL
    forced-unpaired-below-Rx window). A naive brentq on [sig_lo, sig_hi] treats
    such a NaN edge as a tau=tau_target crossing and returns a spurious (tiny)
    sigma_crit. Instead we scan a coarse sigma grid and locate the crossing ONLY
    between adjacent CONVERGED (finite, positive tau) grid points, keeping the
    largest-sigma upward crossing (the ultimate high-sigma suppression edge), then
    brentq-refine on that sub-bracket.

    Returns
    -------
    float
      finite : sigma_crit (largest-sigma tau=tau_target crossing);
      +inf   : tau < tau_target for EVERY converged sigma up to sig_hi
               (nucleates for all tested sigma -> sigma_crit lies above sig_hi);
      NaN    : no converged nucleating point (never reaches tau_target from below,
               or no critical droplet anywhere in [sig_lo, sig_hi]).
    """
    cache = {}
    lo = np.log10(nuc.tau_target)

    def tau(s):
        return tau_pt(s, H_pt, T_c, flavor, charge, phase, cache, params, Delta0, nuc)

    sig = np.linspace(nuc.sig_lo, nuc.sig_hi, int(nuc.n_sigma_scan))
    t = np.array([tau(s) for s in sig])
    with np.errstate(divide='ignore', invalid='ignore'):
        g = np.where(np.isfinite(t) & (t > 0), np.log10(t) - lo, np.nan)

    # sigma_crit = largest sigma that still nucleates (tau <= tau_target), then look
    # at what happens just above it. Using the LARGEST such sigma automatically steps
    # over any lower-sigma drop-out gaps where no critical droplet exists (so those
    # NaN edges are never mistaken for the threshold).
    nuc_idx = np.where(np.isfinite(g) & (g < 0))[0]
    if nuc_idx.size == 0:
        return np.nan                                # never nucleates in [sig_lo, sig_hi]
    i = int(nuc_idx.max())
    if i == len(sig) - 1:
        return np.inf                                # still nucleating at sig_hi
    gj = g[i + 1]
    if np.isfinite(gj) and gj >= 0:                  # converged upward crossing -> refine
        try:
            return float(brentq(lambda s: np.log10(tau(s)) - lo,
                                 sig[i], sig[i + 1], xtol=1e-3, rtol=1e-5))
        except (ValueError, FloatingPointError):
            return float(sig[i] + (sig[i + 1] - sig[i]) * g[i] / (g[i] - gj))
    # successor is NaN (no critical droplet) or tau=inf: nucleation stops between
    # sig[i] and sig[i+1]. But this is only a *physical* sigma_crit if tau was
    # genuinely approaching tau_target at sig[i]. In the re-hadronization boundary
    # band the only "nucleating" points are a barrierless island (W_c~=0, tau ~
    # 1e-70 s, i.e. spinodal, not thermal nucleation) that then vanishes: there is
    # no tau=tau_target threshold at all. Flag those as NaN rather than emit a fake
    # low sigma_crit painted as "easy to nucleate".
    # ponytail: -6 = tau within 1e6 of target; raise/lower if boundary cells shift.
    if g[i] < -6.0:
        return np.nan
    # Refine the drop-out edge with a fine local sub-scan.
    last = sig[i]
    for s in np.linspace(sig[i], sig[i + 1], 12)[1:]:
        ts = tau(s)
        if np.isfinite(ts) and ts > 0 and (np.log10(ts) - lo) < 0:
            last = float(s)
        else:
            break
    return float(last)


# =============================================================================
#  Grid scan (filter cache + optional joblib)
# =============================================================================
_filter_cache = {}


def _grid_key(alpha_slices, B4_grid, Delta0_grid, cfg: FilterConfig):
    # Include the acceptance constants so the cache invalidates when they change.
    return (tuple(np.round(np.atleast_1d(alpha_slices), 6)),
            tuple(np.round(np.asarray(B4_grid), 6)),
            tuple(np.round(np.asarray(Delta0_grid), 6)),
            float(cfg.m_s), tuple(cfg.M_max_window), float(cfg.e_over_nB_max),
            bool(cfg.check_2flavor), float(cfg.e_over_nB_2flavor))


def scan_cfl_filters(alpha_slices, B4_grid, Delta0_grid, cfg: FilterConfig,
                     reuse=True, verbose=True, n_jobs=-1):
    """(cfl_ok, M_max, reason) stacks of shape (NA, ND, NB) over the grid.

    The expensive part (CFL EoS + TOV per cell); INDEPENDENT of MT0 and of the
    nucleation method, so cached on the grid + acceptance constants. Cells are
    independent -> joblib-parallel (n_jobs=-1 all cores, 1 serial; serial if no
    joblib)."""
    alpha_slices = np.atleast_1d(np.asarray(alpha_slices, float))
    key = _grid_key(alpha_slices, B4_grid, Delta0_grid, cfg)
    if reuse and key in _filter_cache:
        if verbose:
            print("CFL filter: reusing cached grid result.", flush=True)
        return _filter_cache[key]
    NA, ND, NB = len(alpha_slices), len(Delta0_grid), len(B4_grid)
    cfl = np.zeros((NA, ND, NB), dtype=bool)
    mm = np.full((NA, ND, NB), np.nan)
    rs = np.zeros((NA, ND, NB), dtype=np.int8)
    idx = [(ia, i, jx) for ia in range(NA) for i in range(ND) for jx in range(NB)]
    use_par = (n_jobs != 1) and _HAVE_JOBLIB
    if verbose:
        print(f"CFL filter scan: {NA}x{ND}x{NB}={len(idx)} cells, "
              f"{('parallel n_jobs=%d' % n_jobs) if use_par else 'serial'}...", flush=True)
    t0 = time.perf_counter()
    if use_par:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=n_jobs, verbose=(10 if verbose else 0))(
            delayed(passes_cfl_filters)(alpha_slices[ia], B4_grid[jx], Delta0_grid[i], cfg)
            for (ia, i, jx) in idx)
    else:
        res = [passes_cfl_filters(alpha_slices[ia], B4_grid[jx], Delta0_grid[i], cfg)
               for (ia, i, jx) in idx]
    for (ia, i, jx), (ok, M_max, reason) in zip(idx, res):
        cfl[ia, i, jx] = ok
        mm[ia, i, jx] = M_max
        rs[ia, i, jx] = REASON_CODE[reason]
    n_2f = 0
    if cfg.check_2flavor:
        # 2-flavor (ud) stability: reject (alpha,B4) where ud matter is bound
        # (e/nB|P=0 <= threshold) -> ordinary nuclei would decay. Delta0-independent,
        # so one solve per (alpha,B4) column, broadcast across Delta0.
        cols = [(ia, jx) for ia in range(NA) for jx in range(NB)]
        if use_par:
            from joblib import Parallel, delayed
            ud = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(ud_eps_per_nB)(alpha_slices[ia], B4_grid[jx], cfg)
                for (ia, jx) in cols)
        else:
            ud = [ud_eps_per_nB(alpha_slices[ia], B4_grid[jx], cfg) for (ia, jx) in cols]
        for (ia, jx), v in zip(cols, ud):
            if np.isfinite(v) and v <= cfg.e_over_nB_2flavor:
                bad = rs[ia, :, jx] == REASON_CODE['OK']   # only override passing cells
                cfl[ia, bad, jx] = False
                rs[ia, bad, jx] = REASON_CODE['twoflavor']
                n_2f += int(bad.sum())
    if verbose:
        npass = int(cfl.sum())
        c = {n: int((rs == k).sum()) for n, k in REASON_CODE.items()}  # reason histogram
        Mok = mm[cfl]
        mr = (f"M_max in [{Mok.min():.2f}, {Mok.max():.2f}] M_sun"
              if Mok.size else "no surviving cells")
        print(f"  CFL filter done in {time.perf_counter()-t0:.1f}s: "
              f"pass {npass}/{len(idx)} ({100*npass/max(len(idx),1):.0f}%), {mr}\n"
              f"    rejected -> no CFL solve: {c['solve']}, "
              f"Witten (unstable / e/nB>{cfg.e_over_nB_max:g}): {c['witten']}, "
              f"re-hadronizes (P_CFL<P_H): {c['rehadr']}, "
              f"M_max outside {cfg.M_max_window}: {c['mmax']}, "
              f"2-flavor bound (e/nB<={cfg.e_over_nB_2flavor:g}): {c['twoflavor']}",
              flush=True)
    _filter_cache[key] = (cfl, mm, rs)
    return cfl, mm, rs


def scan_unpaired_filters(alpha_slices, B4_grid, cfg: FilterConfig,
                          reuse=True, verbose=True, n_jobs=-1):
    """(ok, M_max, reason) stacks over the (alpha_s, B4) grid for UNPAIRED matter.

    Companion to ``scan_cfl_filters`` for the gap-free unpaired channel. Returns
    the same 3-D ``(NA, ND, NB)`` layout with a SINGLETON Delta_0 axis (ND=1), so
    the result drops straight into ``compute_sigma_crit`` / ``plot_sigma_crit_grid``
    (which expect an (alpha, Delta_0, B4) stack) without special-casing. Per cell
    it runs ``passes_unpaired_filters`` (no-rehadronization + unpaired-EoS M_max);
    joblib-parallel over cells, cached on the grid + acceptance constants."""
    alpha_slices = np.atleast_1d(np.asarray(alpha_slices, float))
    key = ('unpaired', tuple(np.round(alpha_slices, 6)),
           tuple(np.round(np.asarray(B4_grid), 6)),
           float(cfg.m_s), tuple(cfg.M_max_window))
    if reuse and key in _filter_cache:
        if verbose:
            print("Unpaired filter: reusing cached grid result.", flush=True)
        return _filter_cache[key]
    NA, NB = len(alpha_slices), len(B4_grid)
    idx = [(ia, jx) for ia in range(NA) for jx in range(NB)]
    use_par = (n_jobs != 1) and _HAVE_JOBLIB
    if verbose:
        print(f"Unpaired filter scan: {NA}x{NB}={len(idx)} cells, "
              f"{('parallel n_jobs=%d' % n_jobs) if use_par else 'serial'}...", flush=True)
    t0 = time.perf_counter()
    if use_par:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=n_jobs, verbose=(10 if verbose else 0))(
            delayed(passes_unpaired_filters)(alpha_slices[ia], B4_grid[jx], cfg)
            for (ia, jx) in idx)
    else:
        res = [passes_unpaired_filters(alpha_slices[ia], B4_grid[jx], cfg)
               for (ia, jx) in idx]
    cfl = np.zeros((NA, 1, NB), dtype=bool)
    mm = np.full((NA, 1, NB), np.nan)
    rs = np.zeros((NA, 1, NB), dtype=np.int8)
    for (ia, jx), (ok, M_max, reason) in zip(idx, res):
        cfl[ia, 0, jx] = ok
        mm[ia, 0, jx] = M_max
        rs[ia, 0, jx] = REASON_CODE[reason]
    if verbose:
        npass = int(cfl.sum())
        c = {n: int((rs == k).sum()) for n, k in REASON_CODE.items()}
        print(f"  Unpaired filter done in {time.perf_counter()-t0:.1f}s: "
              f"pass {npass}/{len(idx)} ({100*npass/max(len(idx),1):.0f}%); "
              f"rejected -> no solve: {c['solve']}, re-hadronizes: {c['rehadr']}, "
              f"M_max outside {cfg.M_max_window}: {c['mmax']}", flush=True)
    _filter_cache[key] = (cfl, mm, rs)
    return cfl, mm, rs


def compute_sigma_crit(cfl_ok, MT0, flavor, charge, phase, alpha_slices, B4_grid,
                       Delta0_grid, star: StarMatch, nuc: NucConfig, m_s=100.0,
                       n_jobs=-1, verbose=True):
    """sigma_crit stack at one MT0 / method, only on CFL-pass cells. H_pt (plain
    floats) is computed once and shared -> joblib-parallel over CFL-pass cells."""
    alpha_slices = np.atleast_1d(np.asarray(alpha_slices, float))
    NA, ND, NB = cfl_ok.shape
    nBHc, T_c, H_pt = central_state(MT0, star)
    sig = np.full((NA, ND, NB), np.nan)
    cells = [(ia, i, jx) for ia in range(NA) for i in range(ND) for jx in range(NB)
             if cfl_ok[ia, i, jx]]
    use_par = (n_jobs != 1) and _HAVE_JOBLIB
    if verbose:
        print(f"  sigma_crit MT0={MT0:.2f} {flavor}/{charge}/{phase}: {len(cells)} "
              f"CFL-pass cells (nBHc={nBHc:.4f}, T_c={T_c:.2f} MeV), "
              f"{'parallel' if use_par else 'serial'}", flush=True)
    if not cells:
        return sig
    pars = [get_alphabag_custom(alpha=alpha_slices[ia], B4=B4_grid[jx], m_s=m_s)
            for (ia, i, jx) in cells]
    D0s = [Delta0_grid[i] for (ia, i, jx) in cells]
    t0 = time.perf_counter()
    if use_par:
        from joblib import Parallel, delayed
        vals = Parallel(n_jobs=n_jobs, verbose=(10 if verbose else 0))(
            delayed(sigma_target_pt)(H_pt, T_c, flavor, charge, phase, pp, d0, nuc)
            for pp, d0 in zip(pars, D0s))
    else:
        vals = [sigma_target_pt(H_pt, T_c, flavor, charge, phase, pp, d0, nuc)
                for pp, d0 in zip(pars, D0s)]
    for (ia, i, jx), v in zip(cells, vals):
        sig[ia, i, jx] = v
    if verbose:
        fin = sig[np.isfinite(sig)]
        nnuc, nbrk = fin.size, len(cells) - fin.size
        rng = (f"sigma_crit in [{fin.min():.1f}, {fin.max():.1f}] MeV/fm^2"
               if nnuc else "none nucleate in bracket")
        print(f"    -> nucleating (finite sigma_crit): {nnuc}/{len(cells)} CFL-pass "
              f"cells in {time.perf_counter()-t0:.1f}s; {nbrk} never reach tau_target "
              f"in sigma=[{nuc.sig_lo:g}, {nuc.sig_hi:g}]. {rng}", flush=True)
    return sig


def run_sigma_crit_scan(MT0_grid_thr, flavor, charge, phase, alpha_slices,
                        B4_grid, Delta0_grid, cfg: FilterConfig, nuc: NucConfig,
                        star: StarMatch, n_jobs=-1, reuse_filter=True,
                        save_path_fmt=None, xsd_tag='', extra_save=None):
    """CFL filters (cached) + sigma_crit per MT0. Returns
    {MT0: dict(sig_crit, cfl_ok, M_max, reason)}. If save_path_fmt is given it is
    formatted with (xsd=xsd_tag, MT0=MT0, flavor, charge, phase) and an .npz saved."""
    alpha_slices = np.atleast_1d(np.asarray(alpha_slices, float))
    cfl, mm, rs = scan_cfl_filters(alpha_slices, B4_grid, Delta0_grid, cfg,
                                   reuse=reuse_filter, n_jobs=n_jobs)
    out = {}
    for MT0 in np.atleast_1d(MT0_grid_thr):
        MT0 = float(MT0)
        sig = compute_sigma_crit(cfl, MT0, flavor, charge, phase, alpha_slices,
                                 B4_grid, Delta0_grid, star, nuc, m_s=cfg.m_s,
                                 n_jobs=n_jobs)
        if save_path_fmt:
            npz = save_path_fmt.format(xsd=xsd_tag, MT0=MT0, flavor=flavor,
                                       charge=charge, phase=phase)
            payload = dict(B4_grid=B4_grid, Delta0_grid=Delta0_grid,
                           alpha_slices=alpha_slices, sig_crit=sig, cfl_ok=cfl,
                           M_max=mm, reason=rs, MT0=MT0, m_s=cfg.m_s,
                           mass_window=np.array(cfg.M_max_window, dtype=float),
                           flavor=flavor, charge=charge, phase=phase,
                           tau=nuc.tau_target)
            if extra_save:
                payload.update(extra_save)
            np.savez(npz, **payload)
            print(f"  saved -> {npz}", flush=True)
        out[MT0] = dict(sig_crit=sig, cfl_ok=cfl, M_max=mm, reason=rs)
    return out


# =============================================================================
#  Plotting
# =============================================================================
def plot_sigma_crit_grid(B4_grid, Delta0_grid, alpha_slices, sig_crit, cfl_ok,
                         M_max, reason, scan_label, mass_window=(2.0, np.inf),
                         title_extra='', show_split_grey=True, show_cfl_boundary=True,
                         show_filter_lines=True, show_mmax_lines=True,
                         show_sigcrit_lines=False, show_param_sets=True,
                         param_sets=None, mc_csv=None, alpha_tol=0.04):
    """sigma_crit heatmaps over (B4, Delta0), one panel per alpha, toggleable
    overlays. Pure function of the scan arrays -> use it to re-plot a reloaded npz."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    try:
        import pandas as pd
    except Exception:
        pd = None

    alpha_slices = np.atleast_1d(alpha_slices)
    NA = len(alpha_slices)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad((0, 0, 0, 0) if show_split_grey else 'lightgray')
    finite = sig_crit[np.isfinite(sig_crit)]
    vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)

    mc = None
    if show_param_sets and mc_csv and pd is not None:
        try:
            mc = pd.read_csv(mc_csv)
        except Exception:
            mc = None

    fig, axes = plt.subplots(1, NA, figsize=(4.6 * NA, 4.4), squeeze=False, sharey=True)
    for ia, alpha in enumerate(alpha_slices):
        ax = axes[0, ia]
        sa, ca, ma, ra = sig_crit[ia], cfl_ok[ia], M_max[ia], reason[ia].astype(float)
        if show_split_grey:
            # categorical background for the non-finite sigma_crit cells:
            #   0 grey  = not CFL-viable
            #   1 orange= CFL-viable but never nucleates in [sig_lo, sig_hi]  (NaN)
            #   2 teal  = CFL-viable and nucleates for ALL tested sigma        (+inf,
            #             i.e. sigma_crit lies above sig_hi -> most permissive)
            cat = np.where(np.isfinite(sa), np.nan,
                           np.where(~ca, 0.0, np.where(np.isposinf(sa), 2.0, 1.0)))
            ax.pcolormesh(B4_grid, Delta0_grid, np.ma.masked_invalid(cat),
                          cmap=ListedColormap(['0.85', '#f4a261', '#2a9d8f']),
                          vmin=0, vmax=2, shading='nearest')
        pcm = ax.pcolormesh(B4_grid, Delta0_grid, np.ma.masked_invalid(sa),
                            cmap=cmap, vmin=vmin, vmax=vmax, shading='nearest')
        if show_cfl_boundary and ca.any() and (~ca).any():
            ax.contour(B4_grid, Delta0_grid, ca.astype(float), levels=[0.5],
                       colors='k', linewidths=1.1, linestyles='--')
        if show_filter_lines:
            # Witten / no-rehadronization pass edges, labelled inline (no legend).
            for lev, name in [(1.5, 'Witten'), (2.5, 'rehadronization')]:
                if (ra > lev).any() and (ra < lev).any():
                    cf = ax.contour(B4_grid, Delta0_grid, ra, levels=[lev],
                                    colors='tab:blue', linewidths=1.1)
                    ax.clabel(cf, fmt={lev: name}, fontsize=8, inline=True)
        if show_mmax_lines and np.isfinite(ma).any():
            win = sorted(lv for lv in mass_window if np.isfinite(lv))
            wskip = {round(w, 1) for w in win}
            # white iso-M_max guides every 0.2 Msun, from 1.8 up to whatever appears
            # in the panel, skipping the red acceptance-window level(s).
            wl = [round(x, 1) for x in np.arange(1.8, 4.0001, 0.2)
                  if round(x, 1) not in wskip
                  and np.nanmin(ma) <= round(x, 1) <= np.nanmax(ma)]
            if wl:
                cs = ax.contour(B4_grid, Delta0_grid, ma, levels=wl,
                                colors='white', linewidths=0.9, alpha=0.9)
                ax.clabel(cs, fmt=r'%.1f $M_\odot$', fontsize=7, inline=True)
            if win:
                csw = ax.contour(B4_grid, Delta0_grid, ma, levels=win,
                                 colors='red', linewidths=1.6)
                ax.clabel(csw, fmt=r'%.2f $M_\odot$', fontsize=7, inline=True)
        if show_sigcrit_lines and np.isfinite(sa).any():
            cc = ax.contour(B4_grid, Delta0_grid, sa, levels=6,
                            colors='k', linewidths=0.5, alpha=0.5)
            ax.clabel(cc, fmt='%.0f', fontsize=6, inline=True)
        if show_param_sets:
            if mc is not None and {'alpha', 'B4', 'Delta0'} <= set(mc.columns):
                sel = np.abs(mc['alpha'].to_numpy() - alpha) < alpha_tol
                ax.scatter(mc['B4'].to_numpy()[sel], mc['Delta0'].to_numpy()[sel],
                           s=8, c='k', marker='.', alpha=0.5)
            for p in (param_sets or []):
                ja = int(np.argmin(np.abs(alpha_slices - p['alpha'])))
                if ja == ia:
                    ax.scatter([p['B4']], [p['Delta0']], s=110, marker='*',
                               c='yellow', edgecolors='k', zorder=5)
        # In-region text labels (replace the legend): centroid of each region's cells.
        for mask, txt, col in [(ra == REASON_CODE['mmax'], r'$M_{\max}<2$', '0.2'),
                               (ra == REASON_CODE['twoflavor'], '2-flavor\nbound', 'darkred'),
                               (ca & np.isnan(sa), 'no nucleation', 'saddlebrown'),
                               (ca & np.isposinf(sa), r'$\sigma_{\rm c}>\sigma_{\rm hi}$', 'teal')]:
            if mask.any():
                r_, c_ = np.where(mask)
                ax.text(float(B4_grid[c_].mean()), float(Delta0_grid[r_].mean()),
                        txt, color=col, fontsize=8, fontweight='bold',
                        ha='center', va='center')
        ax.set_title(rf"$\alpha_s$={alpha:.2f}")
        ax.set_xlabel(r'$B^{1/4}$ [MeV]')
        ax.set_xlim(np.min(B4_grid), np.max(B4_grid))
        ax.set_ylim(np.min(Delta0_grid), np.max(Delta0_grid))
        if ia == 0:
            ax.set_ylabel(r'$\Delta_0$ [MeV]')

    fig.colorbar(pcm, ax=axes.ravel().tolist(), label=r'$\sigma_{\rm crit}$ [MeV/fm$^2$]')
    fig.suptitle(rf"Acceptable region — {scan_label}{title_extra}", y=1.04)
    plt.show()
    return fig
