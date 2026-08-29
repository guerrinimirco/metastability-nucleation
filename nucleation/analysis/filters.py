"""
CFL / unpaired acceptance filters
=================================

Which (alpha_s, B^1/4, Delta_0) cells describe a physically admissible strange
star at all, before any nucleation question is asked. Applied cheap-to-expensive
and short-circuiting, so a rejected cell never pays for a TOV solve:

  1. Witten bound        -- strange quark matter must be absolutely stable
                            (e/n_B < 930 MeV at P=0), and 2-flavour (ud) matter
                            must NOT be, or ordinary nuclei would decay.
  2. No re-hadronization -- the quark EoS must stay above the hadronic one, in
                            bulk (Delta P(mu_B) at T=0) and at droplet level.
  3. TOV M_max window    -- the cold sequence must reach the observed maximum
                            mass.

The reason a cell fails is reported as an integer code (REASON_CODE) whose
ordering is nested, so contouring that field draws each filter's pass edge.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq

from nucleation.quark import custom_params
from eos.alphabag.solver import (solve_cfl, solve_beta_eq_neutrinoless,
                                 solve_fixed_yc_ys)
from eos.alphabag.species import SpeciesFlags
from eos.general.state import EOSTable_for_TOV
from eos.astro.tov.solver import (compute_tov_sequence,
                                  truncate_to_stable_branch)
from nucleation.composition import get_solver_Qs
from nucleation.critical import _build_H_from_interp
from nucleation.analysis.config import FilterConfig

# =============================================================================
#  CFL EoS + filters
# =============================================================================
def cfl_eos_at_params(alpha, B4, Delta0, cfg: FilterConfig):
    """Solve the CFL beta-eq EoS at T=cfg.T_eos over cfg.n_B_grid (warm-started).
    Returns (P, e, mu, ok-mask)."""
    p = custom_params(alpha=alpha, B4=B4, m_s=cfg.m_s)
    n = cfg.n_B_grid
    P = np.full_like(n, np.nan); e = np.full_like(n, np.nan); mu = np.full_like(n, np.nan)
    ok = np.zeros_like(n, dtype=bool)
    guess = None
    for i, nB in enumerate(n):
        try:
            # No gluons: in the paired phase all eight are Meissner-massive,
            # only the rotated photon stays massless, and eos refuses the flag
            # rather than dropping the sector silently. At T_eos = 0 the gas
            # contributed nothing anyway, so no number here moves.
            r = solve_cfl(p, nB, cfg.T_eos, Delta0,
                          SpeciesFlags(photons=False, gluons=False),
                          initial_guess=guess)
        except (RuntimeError, ValueError):
            r = None
        if r is None or not r.converged or r.error > 1e-6:
            continue
        P[i], e[i], mu[i] = r.P, r.eps, r.mu_B
        guess = np.array([r.mu_u, r.mu_d, r.mu_s])
        ok[i] = True
    return P, e, mu, ok


def unpaired_eos_at_params(alpha, B4, cfg: FilterConfig):
    """Solve the UNPAIRED beta-eq alpha-Bag EoS at T=cfg.T_eos over cfg.n_B_grid
    (warm-started). Returns (P, e, mu, ok-mask). No Delta_0 — an unpaired droplet
    has no CFL gap, so the unpaired-matter viability filters never reference it."""
    p = custom_params(alpha=alpha, B4=B4, m_s=cfg.m_s)
    n = cfg.n_B_grid
    P = np.full_like(n, np.nan); e = np.full_like(n, np.nan); mu = np.full_like(n, np.nan)
    ok = np.zeros_like(n, dtype=bool)
    guess = None
    for i, nB in enumerate(n):
        try:
            # Gluons on, as published: this is the unpaired phase, where the
            # eight of them are massless and the sector is a real choice. It
            # reads as an asymmetry against the CFL path above, which must say
            # False -- at cfg.T_eos = 0 both gases contribute exactly zero, so
            # the two branches are still being compared on equal terms.
            r = solve_beta_eq_neutrinoless(p, nB, cfg.T_eos,
                                       SpeciesFlags(photons=False, gluons=True),
                                       initial_guess=guess)
        except (RuntimeError, ValueError):
            r = None
        if r is None or not r.converged or r.error > 1e-6:
            continue
        P[i], e[i], mu[i] = r.P, r.eps, r.mu_B
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
    p = custom_params(alpha=alpha, B4=B4, m_s=cfg.m_s)
    n = cfg.n_B_grid
    P = np.full_like(n, np.nan)
    e = np.full_like(n, np.nan)

    def solve(nB, yc, guess):
        # Unpaired again, so gluons on for the same reason. The electrons are
        # NOT a species flag: on a fixed-fraction mode they are what `leptons=`
        # adds to enforce total neutrality, and they must be here -- the root
        # this brackets is mu_d - mu_u - mu_e.
        return solve_fixed_yc_ys(p, nB, yc, 0.0, cfg.T_eos,
                                          SpeciesFlags(photons=False, gluons=True),
                                          leptons=True, initial_guess=guess)

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
        except (RuntimeError, ValueError):
            continue
        if not r.converged or r.error > 1e-6:
            continue
        P[i], e[i] = r.P, r.eps
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
    (1) Witten e/n_B|_{P=0} < cfg.e_over_nB_max, (2) bulk no-rehadronization on
    ΔP(mu_B)=P_CFL-P_H at T=0 (both conditions, see _rehad_reason), (3) TOV M_max
    in cfg.M_max_window. M_max is finite once TOV runs."""
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

    rehad = _rehad_reason(mu_ok, P_ok, cfg)
    if rehad is not None:
        return False, np.nan, rehad

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

      (1) bulk no-rehadronization on ΔP(mu_B)=P_unp-P_H at T=0 (both conditions,
          see _rehad_reason), built on the UNPAIRED quark EoS;
      (2) Witten bound: unpaired 3-flavor SQM absolutely stable,
          e/n_B|_{P=0} < cfg.e_over_nB_max;
      (3) bulk no-rehadronization on Delta P(mu_B) = P_unp - P_H at T=0
          (both weak and strong conditions, see ``_rehad_reason``);
      (4) 2-flavor (ud) matter NOT bound (else ordinary nuclei would decay):
          e/n_B|_{P=0} > cfg.e_over_nB_2flavor (skipped if not cfg.check_2flavor);
      (5) TOV M_max in cfg.M_max_window, structure integrated with the
          UNPAIRED quark EoS.

    The FULL two-families acceptance, with every test built on the UNPAIRED
    EoS -- CFL/Delta_0 never enters (the unpaired sigma_crit map is
    gap-independent, so its viability gate must be too). Ordered
    cheap -> expensive; codes shared with ``passes_cfl_filters``."""
    P_q, e_q, mu_q, ok = unpaired_eos_at_params(alpha, B4, cfg)
    if ok.sum() < 20:
        return False, np.nan, 'solve'
    n_ok, P_ok, e_ok, mu_ok = cfg.n_B_grid[ok], P_q[ok], e_q[ok], mu_q[ok]

    # (2) Witten: unpaired SQM must be absolutely stable (e/nB at P=0 < bound).
    cr = zero_crossing(n_ok, P_ok)
    if cr is None or P_ok[-1] <= 0:
        return False, np.nan, 'witten'
    j, fr = cr
    n_P0 = n_ok[j] + fr * (n_ok[j + 1] - n_ok[j])
    e_P0 = e_ok[j] + fr * (e_ok[j + 1] - e_ok[j])
    if e_P0 / n_P0 >= cfg.e_over_nB_max:
        return False, np.nan, 'witten'

    # (3) bulk no-rehadronization (weak + strong).
    rehad = _rehad_reason(mu_ok, P_ok, cfg)
    if rehad is not None:
        return False, np.nan, rehad

    # (4) 2-flavor (ud) stability: bound ud matter would destabilise nuclei.
    #     Gap-free by construction (no strange quarks), so genuinely unpaired.
    if cfg.check_2flavor:
        v = ud_eps_per_nB(alpha, B4, cfg)
        if np.isfinite(v) and v <= cfg.e_over_nB_2flavor:
            return False, np.nan, 'twoflavor'

    # (5) M_max from the UNPAIRED quark EoS.
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


def _rehad_reason(mu_q, P_q, cfg: FilterConfig, n_mu=500):
    """Bulk no-rehadronization filter reason for one cell, from the T=0 pressure
    difference ΔP(mu_B) = P_quark(mu_B) - P_H(mu_B) on the mu_B overlap.

    None if the cell passes BOTH conditions, else the failing reason:
      'rehadr'      -- no_rehad fails: after the phases first cross, the quark
                       pressure dips back below P_H (matter could re-hadronize);
      'rehad_quasi' -- no_rehad passes but ΔP is not monotonically increasing in
                       mu_B (quark phase not becoming steadily more favoured).
    With cfg.merge_rehad_labels both map to 'rehadr'. mu_q/P_q are the CFL (or
    unpaired) EoS points already solved over cfg.n_B_grid -- no extra solves."""
    mu_lo = max(mu_q.min(), cfg.mu_B_H_sorted.min())
    mu_hi = min(mu_q.max(), cfg.mu_B_H_sorted.max())
    if mu_hi <= mu_lo:
        return 'rehadr'
    mu = np.linspace(mu_lo, mu_hi, n_mu)
    Pc = interp1d(mu_q, P_q, kind='linear', bounds_error=False,
                  fill_value=np.nan)(mu)
    dP = Pc - cfg.P_H_of_muB(mu)               # ΔP(mu_B) on the overlap
    no_re, strong, _ = rehad_flags(mu, dP)
    if not no_re:
        return 'rehadr'
    if not strong:
        return 'rehadr' if cfg.merge_rehad_labels else 'rehad_quasi'
    return None

