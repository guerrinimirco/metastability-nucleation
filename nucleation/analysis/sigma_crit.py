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
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq

from eos.alphabag.parameters import get_alphabag_custom
from eos.alphabag.eos import solve_cfl
from eos.general.physics_constants import hc
from eos.tov.solver import (EOSTable_for_TOV, generate_ec_logspace,
                            compute_tov_sequence, truncate_to_stable_branch)
from nucleation.energy_barrier.small_droplet.solvers import get_solver_Qs
from nucleation.energy_barrier.small_droplet.barrier import (
    driving_force, critical_radius_noCoulomb, critical_work_noCoulomb,
    work_of_formation, switching_step)
from nucleation.energy_barrier.small_droplet.observables import _find_Rc_Wc_step
from nucleation.general_nucleation.thermal import nucleation_rate, nucleation_time

try:
    import joblib  # noqa: F401  -- presence check; Parallel imported in-function
    _HAVE_JOBLIB = True
except Exception:
    _HAVE_JOBLIB = False

# Filter-reason -> integer code. The three filters short-circuit cheap->expensive,
# so codes are *nested*: contouring this field at 1.5 / 2.5 gives the Witten /
# no-rehadr pass edges, and 3.5 (== cfl_ok edge) the full-acceptance edge.
REASON_CODE = {'solve': 0, 'witten': 1, 'rehadr': 2, 'mmax': 3, 'OK': 4}


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
    e_over_nB_max: float = 930.0
    T_eos: float = 0.0


@dataclass
class NucConfig:
    """Nucleation / root-find setup for sigma_crit."""
    sig_lo: float = 1.0
    sig_hi: float = 300.0
    tau_target: float = 1e-3        # s
    V: float = 4.18879e51           # fm^3 (sphere R = 100 m)
    T_c_factor: float = 0.57        # CFL: T_c = T_c_factor * Delta0
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


def zero_crossing(x, y):
    """First sign change in y(x); returns (j, frac) with x_cross = x[j]+frac*dx."""
    s = np.sign(y)
    idx = np.where(np.diff(s) != 0)[0]
    if idx.size == 0:
        return None
    j = idx[0]
    return j, -y[j] / (y[j + 1] - y[j])


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
            compute_baryonic_mass=True, compute_tidal=False, verbose=False)
        _, M_max, _ = truncate_to_stable_branch(tov, verbose=False)
    except Exception:
        return False, np.nan, 'mmax'
    lo, hi = cfg.M_max_window
    if not np.isfinite(M_max) or not (lo <= M_max <= hi):
        return False, M_max, 'mmax'
    return True, M_max, 'OK'


def replay_cfl(alpha, B4, Delta0, cfg: FilterConfig):
    """Cold CFL EoS + TOV for one set -> dict(mu, P, R, M) or None on failure.
    (TOV cols 3=R[km], 4=M[Msun].)"""
    P, e, mu, ok = cfl_eos_at_params(alpha, B4, Delta0, cfg)
    if ok.sum() < 20:
        return None
    n_ok, P_ok, e_ok, mu_ok = cfg.n_B_grid[ok], P[ok], e[ok], mu[ok]
    pos = P_ok > 0
    if pos.sum() < 10:
        return None
    try:
        tov = compute_tov_sequence(
            EOSTable_for_TOV(P=P_ok[pos], epsilon=e_ok[pos], nB=n_ok[pos]),
            e_c_vec=cfg.e_c_vec_tov, add_crust_table='No',
            compute_baryonic_mass=True, compute_tidal=False, verbose=False)
        st, _, _ = truncate_to_stable_branch(tov, verbose=False)
    except Exception:
        return None
    return dict(mu=mu_ok, P=P_ok, R=st[:, 3], M=st[:, 4])


# =============================================================================
#  Single-point nucleation
# =============================================================================
def crossover_radius(T, Delta0, T_c_factor=0.57):
    """unpCFL coherence radius R_x(T) = hc/Delta(T), Delta(T)=Delta0*sqrt(1-(T/Tc)^2),
    Tc = T_c_factor*Delta0; inf above Tc (no pairing -> purely unpaired droplet)."""
    T_c = T_c_factor * Delta0
    ratio = np.asarray(T / T_c)
    gap = np.where(ratio < 1, Delta0 * np.sqrt(np.maximum(0, 1 - ratio**2)), 0.0)
    return np.where(gap > 0, hc / gap, np.inf)


def central_state(MT0, star: StarMatch):
    """(nBHc, T_c, H_pt) at the centre of the trapped star with Mb = Mb(MT0)."""
    Mb = float(star.Mb_of_MT0(MT0))
    nBHc = float(star.nBHc_of_Mb(Mb))
    T_c = float(star.H_iso_trapped['T'](nBHc, star.YLH, star.S))
    pt = (nBHc, star.YLH, T_c)
    Ht = star.H_trapped
    H_pt = SimpleNamespace(
        n_B=nBHc, T=T_c,
        P_total=float(Ht['P'](*pt)),   e_total=float(Ht['eps'](*pt)),
        mu_B=float(Ht['mu_B'](*pt)),   mu_C=float(Ht['mu_C'](*pt)),
        mu_S=float(Ht['mu_S'](*pt)),   mu_e=float(Ht['mu_e'](*pt)),
        mu_nu=float(Ht['mu_nu'](*pt)),
        Y_C=float(Ht['Y_C'](*pt)),     Y_S=float(Ht['Y_S'](*pt)))
    H_pt.Y_e, H_pt.Y_nu = H_pt.Y_C, star.YLH - H_pt.Y_C
    return nBHc, T_c, H_pt


def _solve_phase(sigma, H_pt, flavor, charge, phase, cache, params, Delta0, nuc):
    """(Qs, R_c_or_None). lcn/gcn are sigma-independent -> cache per phase."""
    sig_indep = charge in ('lcn', 'gcn', 'gcn_coulomb')
    if sig_indep and phase in cache:
        return cache[phase]
    kw = dict(quark_phase=phase, Delta0=Delta0,
              include_photons=nuc.include_photons, include_gluons=nuc.include_gluons,
              include_thermal_neutrinos=nuc.include_thermal_neutrinos)
    if not sig_indep:
        kw['sigma'] = sigma
    out = get_solver_Qs(flavor, charge, params, **kw)(H_pt)
    res = (out, None) if sig_indep else (out if out is not None else (None, None))
    if sig_indep:
        cache[phase] = res
    return res


def _Rc_single(Df, R_c_solver, sigma, charge):
    if charge in ('lcn', 'gcn'):
        return float(critical_radius_noCoulomb(Df, sigma)) if Df < 0 else np.nan
    if R_c_solver is None or not np.isfinite(R_c_solver) or R_c_solver <= 0:
        return np.nan
    return float(R_c_solver)


def tau_pt(sigma, H_pt, T_c, flavor, charge, phase, cache, params, Delta0, nuc: NucConfig):
    """Central nucleation time tau at one sigma, for one droplet phase.
    unpCFL combines unpaired + cfl via the step switching at R_x(T_c)."""
    if phase == 'unpCFL':
        Qu, Rcu = _solve_phase(sigma, H_pt, flavor, charge, 'unpaired', cache, params, Delta0, nuc)
        Qc, Rcc = _solve_phase(sigma, H_pt, flavor, charge, 'cfl', cache, params, Delta0, nuc)
        if Qu is None or Qc is None:
            return np.nan
        Dfu, Dfc = float(driving_force(Qu, H_pt)), float(driving_force(Qc, H_pt))
        Rc_u = _Rc_single(Dfu, Rcu, sigma, charge)
        Rc_c = _Rc_single(Dfc, Rcc, sigma, charge)
        Rx = float(crossover_radius(T_c, Delta0, nuc.T_c_factor))
        dnC_u = (float(Qu.Y_C) - float(Qu.Y_e)) * float(Qu.n_B)
        dnC_c = (float(Qc.Y_C) - float(Qc.Y_e)) * float(Qc.n_B)
        R_c, W_c, S = _find_Rc_Wc_step(
            np.asarray(Rc_u, float), np.asarray(Rc_c, float), Rx,
            lambda R: switching_step(R, Rx),
            np.asarray(Dfu, float), np.asarray(Dfc, float),
            np.asarray(dnC_u, float), np.asarray(dnC_c, float), sigma)
        R_c, W_c, S = float(R_c), float(W_c), float(S)
        if not (np.isfinite(R_c) and R_c > 0 and np.isfinite(W_c) and W_c > 0):
            return np.nan
        Qs_sel = Qc if S > 0.5 else Qu
        Gamma = float(nucleation_rate(W_c, R_c, sigma, T_c, H_pt, Qs_sel))
        return float(nucleation_time(Gamma, nuc.V)) if Gamma > 0 else np.inf

    Qs, R_c = _solve_phase(sigma, H_pt, flavor, charge, phase, cache, params, Delta0, nuc)
    if Qs is None:
        return np.nan
    Df = float(driving_force(Qs, H_pt))
    if Df >= 0:
        return np.inf
    if charge in ('lcn', 'gcn'):
        Rc_v, W_c = (float(critical_radius_noCoulomb(Df, sigma)),
                     float(critical_work_noCoulomb(Df, sigma)))
    else:
        if R_c is None or not np.isfinite(R_c) or R_c <= 0:
            return np.nan
        Rc_v = float(R_c)
        dnC = (float(Qs.Y_C) - float(Qs.Y_e)) * float(Qs.n_B)
        W_c = float(work_of_formation(Rc_v, Df, sigma, dnC))
    if not np.isfinite(W_c) or W_c <= 0:
        return np.nan
    Gamma = float(nucleation_rate(W_c, Rc_v, sigma, T_c, H_pt, Qs))
    return float(nucleation_time(Gamma, nuc.V)) if Gamma > 0 else np.inf


def sigma_target_pt(H_pt, T_c, flavor, charge, phase, params, Delta0, nuc: NucConfig):
    """sigma where tau(sigma) = nuc.tau_target at the centre, by brentq on
    [nuc.sig_lo, nuc.sig_hi]. NaN if the crossing is not bracketed."""
    cache = {}

    def g(s):
        t = tau_pt(s, H_pt, T_c, flavor, charge, phase, cache, params, Delta0, nuc)
        if not np.isfinite(t):
            return 100.0 if (np.isnan(t) or t > nuc.tau_target) else -100.0
        return np.log10(t) - np.log10(nuc.tau_target)

    if g(nuc.sig_lo) * g(nuc.sig_hi) > 0:
        return np.nan
    return float(brentq(g, nuc.sig_lo, nuc.sig_hi, xtol=1e-3, rtol=1e-5))


# =============================================================================
#  Grid scan (filter cache + optional joblib)
# =============================================================================
_filter_cache = {}


def _grid_key(alpha_slices, B4_grid, Delta0_grid, cfg: FilterConfig):
    # Include the acceptance constants so the cache invalidates when they change.
    return (tuple(np.round(np.atleast_1d(alpha_slices), 6)),
            tuple(np.round(np.asarray(B4_grid), 6)),
            tuple(np.round(np.asarray(Delta0_grid), 6)),
            float(cfg.m_s), tuple(cfg.M_max_window), float(cfg.e_over_nB_max))


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
    from matplotlib.lines import Line2D
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
            cat = np.where(np.isfinite(sa), np.nan, np.where(ca, 1.0, 0.0))
            ax.pcolormesh(B4_grid, Delta0_grid, np.ma.masked_invalid(cat),
                          cmap=ListedColormap(['0.85', '#f4a261']),
                          vmin=0, vmax=1, shading='nearest')
        pcm = ax.pcolormesh(B4_grid, Delta0_grid, np.ma.masked_invalid(sa),
                            cmap=cmap, vmin=vmin, vmax=vmax, shading='nearest')
        if show_cfl_boundary and ca.any() and (~ca).any():
            ax.contour(B4_grid, Delta0_grid, ca.astype(float), levels=[0.5],
                       colors='k', linewidths=1.1, linestyles='--')
        if show_filter_lines:
            for lev, col in [(1.5, 'tab:cyan'), (2.5, 'tab:pink')]:
                if (ra > lev).any() and (ra < lev).any():
                    ax.contour(B4_grid, Delta0_grid, ra, levels=[lev],
                               colors=col, linewidths=1.0, linestyles=':')
        if show_mmax_lines and np.isfinite(ma).any():
            cs = ax.contour(B4_grid, Delta0_grid, ma, levels=[1.8, 2.2, 2.4, 2.6],
                            colors='white', linewidths=0.9, alpha=0.9)
            ax.clabel(cs, fmt='%.1f', fontsize=7, inline=True)
            win = sorted(lv for lv in mass_window if np.isfinite(lv))
            if win:
                csw = ax.contour(B4_grid, Delta0_grid, ma, levels=win,
                                 colors='red', linewidths=1.6)
                ax.clabel(csw, fmt='%.2f', fontsize=7, inline=True)
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
        ax.set_title(rf"$\alpha_s$={alpha:.2f}")
        ax.set_xlabel(r'$B^{1/4}$ [MeV]')
        ax.set_xlim(np.min(B4_grid), np.max(B4_grid))
        ax.set_ylim(np.min(Delta0_grid), np.max(Delta0_grid))
        if ia == 0:
            ax.set_ylabel(r'$\Delta_0$ [MeV]')

    fig.colorbar(pcm, ax=axes.ravel().tolist(), label=r'$\sigma_{\rm crit}$ [MeV/fm$^2$]')
    handles = []
    if show_split_grey:
        handles += [Line2D([], [], marker='s', ls='', color='0.85', label='CFL rejected'),
                    Line2D([], [], marker='s', ls='', color='#f4a261', label='CFL ok, no nucl.')]
    if show_cfl_boundary:
        handles.append(Line2D([], [], color='k', ls='--', label='CFL-pass edge'))
    if show_filter_lines:
        handles += [Line2D([], [], color='tab:cyan', ls=':', label='Witten edge'),
                    Line2D([], [], color='tab:pink', ls=':', label='no-rehadr edge')]
    if show_param_sets:
        handles += [Line2D([], [], marker='*', ls='', color='yellow', mec='k', label='param sets'),
                    Line2D([], [], marker='.', ls='', color='k', label='MC accepted')]
    if handles:
        fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=7,
                   bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(rf"Acceptable region — {scan_label}{title_extra}", y=1.04)
    plt.show()
    return fig
