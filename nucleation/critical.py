"""
Critical droplet engine
=======================

THE single place that turns a hadronic point + physical prescriptions into a
critical droplet (R_c, W_c, Q*), and that builds the barrier profile W(R).
Every consumer -- sigma_crit scans, the grid tables, the notebook's W(R) plots
-- routes through here, so the barrier physics can never diverge between them.

Two entry points
----------------
* ``critical_droplet(H, ...)`` -> ``CriticalDroplet`` (R_c, W_c, Q* at one point).
  Analytic fast paths for lcn / gcn / gcn_coulomb / screening (composition is
  R-independent); the self-consistent 5-eq solver for coulomb_minimize (R and
  Q* co-solved); phase-switched piecewise barrier for unpCFL.
* ``compute_energy_barrier(...)`` / ``barrier_profile(...)`` -> W(R) arrays for
  plotting, plus ``compute_Qs_along_R`` (per-R Q* sweep at one point).

W_c conventions (used by sigma_crit)
------------------------------------
* finite R_c, W_c : genuine critical droplet.
* R_c = NaN, W_c = +inf : hadronic phase STABLE (Delta_f >= 0) -> never nucleates.
* R_c = NaN, W_c = NaN : solver failed / no critical droplet found.
(The grid tables in ``nucleation.tables`` keep W_c = NaN rather than +inf, to
match the on-disk .dat semantics -- that split is deliberate.)
"""

import numpy as np
from types import SimpleNamespace
from dataclasses import dataclass

from nucleation.barrier import (
    driving_force, bulk_W, surface_W, coulomb_W, coulomb_screened_W,
    work_of_formation,
    critical_radius_noCoulomb, critical_work_noCoulomb,
    critical_radius_coulomb, critical_radius_coulomb_screened,
    electron_susceptibility, debye_length,
    get_switching_function,
)
from nucleation.composition import (
    get_solver_Qs,
    solve_coulomb_minimize_at_R, solve_coulomb_minimize_cfl_at_R,
)


# =============================================================================
# Output dataclasses
# =============================================================================
@dataclass
class CriticalDroplet:
    """Critical droplet at one hadronic point.

    R_c : fm (NaN on failure or when H stable).
    W_c : MeV (+inf if H stable, NaN on failure).
    Qs  : droplet EOS state (for unpCFL, the phase selected at the peak); None on failure.
    Delta_f, delta_n_C : bulk driving force / net charge density at R_c.
    lambda_D : Debye length (NaN unless screening).
    S : unpCFL phase blend at the peak (NaN otherwise; 0 unpaired, 1 CFL).
    """
    R_c: float
    W_c: float
    Qs: object
    Delta_f: float = np.nan
    delta_n_C: float = np.nan
    lambda_D: float = np.nan
    S: float = np.nan


@dataclass
class EnergyBarrierResult:
    """W(R) profile at a single physical point (for plotting)."""
    R: np.ndarray
    W: np.ndarray
    W_bulk: np.ndarray
    W_surface: np.ndarray
    W_coulomb: np.ndarray
    Delta_f: np.ndarray
    delta_n_C: np.ndarray
    sigma: float
    lambda_D: float = np.nan


# =============================================================================
# Small shared helpers  (internal)
# =============================================================================
def _dnC(Qs):
    """Droplet net charge density delta_n_C = (Y_C - Y_e) n_B."""
    return float((Qs.Y_C - Qs.Y_e) * Qs.n_B)


def _lambda_D_of(H, lambda_D=None):
    """Debye length for the screening mode: explicit override or chi_e(H)."""
    if lambda_D is not None:
        return float(lambda_D)
    chi = electron_susceptibility(H.mu_e, H.T)
    return float(debye_length(chi))


# =============================================================================
# Per-point critical droplet -- single quark phase  (internal)
# =============================================================================
def _critical_single_phase(H, sigma, params, flavor_mode, charge, phase,
                           Delta0, cache, lambda_D, solver_kw):
    """Solve one quark phase (unpaired OR cfl) at H -> CriticalDroplet.

    lcn/gcn/gcn_coulomb/screening: composition is sigma-independent, so it is
    cached per phase in ``cache`` (a dict) to avoid re-solving across sigma.
    coulomb_minimize: the 5-eq self-consistent solver (composition + R together).
    """
    sig_indep = charge in ('lcn', 'gcn', 'gcn_coulomb', 'screening')

    # Composition solve (cached for sigma-independent charge modes).
    if sig_indep and cache is not None and (phase, charge) in cache:
        solved = cache[(phase, charge)]
    else:
        kw = dict(quark_phase=phase, Delta0=Delta0, **solver_kw)
        if charge == 'coulomb_minimize':
            kw['sigma'] = sigma
        solved = get_solver_Qs(flavor_mode, charge, params, **kw)(H)
        if sig_indep and cache is not None:
            cache[(phase, charge)] = solved

    if solved is None:
        return CriticalDroplet(np.nan, np.nan, None)

    # coulomb_minimize returns (Qs, R_c_from_solver); other modes return Qs.
    if charge == 'coulomb_minimize':
        Qs, R_c_solver = solved
    else:
        Qs, R_c_solver = solved, None

    Df = float(driving_force(Qs, H))
    if Df >= 0:                       # hadronic phase stable -> no barrier
        return CriticalDroplet(np.nan, np.inf, Qs, Delta_f=Df)

    if charge in ('lcn', 'gcn'):
        R_c = float(critical_radius_noCoulomb(Df, sigma))
        W_c = float(critical_work_noCoulomb(Df, sigma))
        return CriticalDroplet(R_c, W_c, Qs, Delta_f=Df, delta_n_C=0.0)

    dnC = _dnC(Qs)
    if charge == 'gcn_coulomb':
        R_c = float(critical_radius_coulomb(Df, sigma, dnC))
    elif charge == 'screening':
        R_c = float(critical_radius_coulomb_screened(Df, sigma, dnC, lambda_D))
    else:  # coulomb_minimize
        R_c = float(R_c_solver) if (R_c_solver is not None
                                    and np.isfinite(R_c_solver)
                                    and R_c_solver > 0) else np.nan
    if not np.isfinite(R_c):
        return CriticalDroplet(np.nan, np.nan, Qs, Delta_f=Df, delta_n_C=dnC)
    lam = lambda_D if charge == 'screening' else None
    W_c = float(work_of_formation(R_c, Df, sigma, dnC, lambda_D=lam))
    return CriticalDroplet(R_c, W_c, Qs, Delta_f=Df, delta_n_C=dnC,
                           lambda_D=(lambda_D if charge == 'screening' else np.nan))


# =============================================================================
# Per-point critical droplet -- public engine
# =============================================================================
def critical_droplet(H, *, sigma, params, flavor_mode='saddlepoint',
                     electric_charge_mode='gcn', quark_phase='unpaired',
                     Delta0=None, Rx=None, cache=None, lambda_D=None,
                     include_photons=True, include_gluons=True,
                     include_thermal_neutrinos=True):
    """Critical droplet (R_c, W_c, Q*) at hadronic point H.

    Parameters mirror the composition prescriptions. For ``quark_phase='unpCFL'``
    the unpaired core (R <= Rx) and CFL mantle (R > Rx) are combined into one
    piecewise barrier and the global peak is returned (Rx = coherence radius,
    supplied by the caller). ``cache`` (a dict) memoizes the sigma-independent
    composition solves across repeated sigma calls.

    Returns a ``CriticalDroplet``.
    """
    solver_kw = dict(include_photons=include_photons,
                     include_gluons=include_gluons,
                     include_thermal_neutrinos=include_thermal_neutrinos)
    lam = (_lambda_D_of(H, lambda_D)
           if electric_charge_mode == 'screening' else np.nan)

    if quark_phase != 'unpCFL':
        return _critical_single_phase(
            H, sigma, params, flavor_mode, electric_charge_mode, quark_phase,
            Delta0, cache, lam, solver_kw)

    # ---- unpCFL: combine unpaired core + CFL mantle at crossover Rx ----------
    if Rx is None:
        raise ValueError("Rx required for quark_phase='unpCFL'")
    cd_u = _critical_single_phase(
        H, sigma, params, flavor_mode, electric_charge_mode, 'unpaired',
        Delta0, cache, lam, solver_kw)
    cd_c = _critical_single_phase(
        H, sigma, params, flavor_mode, electric_charge_mode, 'cfl',
        Delta0, cache, lam, solver_kw)
    # The CFL core is required; the unpaired mantle may legitimately have no
    # critical droplet (falls back to the CFL upper branch).
    if cd_c.Qs is None:
        return CriticalDroplet(np.nan, np.nan, None)
    Dfc, Rc_c, dnC_c = cd_c.Delta_f, cd_c.R_c, cd_c.delta_n_C
    if cd_u.Qs is not None:
        Dfu, Rc_u, dnC_u = cd_u.Delta_f, cd_u.R_c, cd_u.delta_n_C
    else:
        Dfu = Rc_u = dnC_u = np.nan
    R_c, W_c, S = _find_Rc_Wc_step(
        np.asarray(Rc_u, float), np.asarray(Rc_c, float), float(Rx),
        np.asarray(Dfu, float), np.asarray(Dfc, float),
        np.asarray(dnC_u, float), np.asarray(dnC_c, float), sigma)
    R_c, W_c, S = float(R_c), float(W_c), float(S)
    if np.isposinf(W_c):                    # both phases non-favoured
        return CriticalDroplet(np.nan, np.inf, cd_c.Qs, S=S)
    Qs_sel = cd_c.Qs if S > 0.5 else cd_u.Qs
    if Qs_sel is None:                      # unpaired peak selected but absent
        return CriticalDroplet(np.nan, np.nan, None, S=S)
    return CriticalDroplet(R_c, W_c, Qs_sel, S=S)


###############################################################################
#                                                                             #
#  unpCFL piecewise-barrier peak logic (moved verbatim from observables.py)    #
#                                                                             #
###############################################################################
def _blend_phase(S, A_unp, B_cfl):
    """NaN-safe blend (1-S) A_unp + S B_cfl.

    Where S==0 the result is exactly A_unp (even if B_cfl is NaN), where S==1
    exactly B_cfl -- avoids 0*NaN poisoning when one phase failed but the
    switching function selects the other (the common step case).
    """
    S = np.asarray(S, dtype=float)
    mixed = (1.0 - S) * A_unp + S * B_cfl
    return np.where(S <= 0.0, A_unp, np.where(S >= 1.0, B_cfl, mixed))


def _find_Rc_Wc_step(R_c_unp, R_c_cfl, Rx,
                     Delta_f_unp, Delta_f_cfl,
                     delta_n_C_unp, delta_n_C_cfl, sigma,
                     Delta_f_unp_atRx=None, delta_n_C_unp_atRx=None):
    """Critical radius + barrier height of the step-switched unpCFL barrier.

    The barrier is the GLOBAL maximum of the piecewise W(R): the unpaired branch
    on [0, Rx] and the more-stable branch (smaller CNT radius) on (Rx, inf). We
    form both candidate peaks and compare their HEIGHTS (not just radii), so a
    case where both critical radii exceed Rx but the upper-branch peak is lower
    than the unpaired kink value correctly pins R_c = Rx (unpaired).

    Returns (R_c, W_c, S_at_Rc) with S in {0 (unpaired), 1 (CFL)}. W_c = +inf
    where neither phase is favoured (H stable against both).
    """
    unp_exists = np.isfinite(R_c_unp)
    cfl_exists = np.isfinite(R_c_cfl)

    # below-region candidate: unpaired barrier on [0, Rx] (peak at R*_unp if
    # inside, else pinned at the kink Rx).
    unp_interior = unp_exists & (R_c_unp <= Rx)
    R_below = np.where(unp_interior, R_c_unp, Rx)
    if Delta_f_unp_atRx is not None:                     # coulomb_minimize kink
        at_kink = ~unp_interior
        Df_u = np.where(at_kink, Delta_f_unp_atRx, Delta_f_unp)
        dnC_u = np.where(at_kink, delta_n_C_unp_atRx, delta_n_C_unp)
    else:
        Df_u, dnC_u = Delta_f_unp, delta_n_C_unp
    W_below = work_of_formation(R_below, Df_u, sigma, dnC_u)
    W_below = np.where(np.isfinite(Df_u), W_below, np.nan)

    # above-region candidate: the MORE STABLE phase (smaller R_c), only a genuine
    # peak when its critical radius exceeds Rx.
    R_c_stable = np.fmin(R_c_cfl, R_c_unp)
    cfl_is_stable = cfl_exists & (~unp_exists | (R_c_cfl <= R_c_unp))
    above_valid = np.isfinite(R_c_stable) & (R_c_stable > Rx)
    R_above = np.where(above_valid, R_c_stable, np.nan)
    S_stable = np.where(cfl_is_stable, 1.0, 0.0)
    Df_above = _blend_phase(S_stable, Delta_f_unp, Delta_f_cfl)
    dnC_above = _blend_phase(S_stable, delta_n_C_unp, delta_n_C_cfl)
    W_above = work_of_formation(R_above, Df_above, sigma, dnC_above)
    W_above = np.where(above_valid, W_above, np.nan)

    # global maximum of the piecewise barrier
    Wb = np.where(np.isfinite(W_below), W_below, -np.inf)
    Wa = np.where(np.isfinite(W_above), W_above, -np.inf)
    above_wins = Wa > Wb
    R_c = np.where(above_wins, R_above, R_below)
    W_c = np.where(above_wins, W_above, W_below)
    S_at_Rc = np.where(above_wins, S_stable, 0.0)
    valid = np.isfinite(W_below) | np.isfinite(W_above)
    R_c = np.where(valid, R_c, np.nan)
    W_c = np.where(valid, W_c, np.nan)

    # no-barrier guard: if Delta_f >= 0 for BOTH phases, W rises without bound
    # above Rx and there is NO critical droplet -> W_c = +inf (never nucleates).
    favoured_any = (Delta_f_unp < 0) | (Delta_f_cfl < 0)      # NaN < 0 is False
    finite_any = np.isfinite(Delta_f_unp) | np.isfinite(Delta_f_cfl)
    no_barrier = finite_any & ~favoured_any
    R_c = np.where(no_barrier, np.nan, R_c)
    W_c = np.where(no_barrier, np.inf, W_c)
    return R_c, W_c, S_at_Rc


def _find_Rc_Wc_tanh(R_c_unp, R_c_cfl, Rx, S_func,
                     Delta_f_unp, Delta_f_cfl,
                     delta_n_C_unp, delta_n_C_cfl, sigma,
                     R_c_guess=None):
    """Critical radius + barrier height for smooth (tanh) unpCFL switching.

    Numerically maximises W(R) per grid point via minimize(-W), seeded by the
    step-formula R_c.
    """
    from scipy.optimize import minimize
    shape = Delta_f_unp.shape
    if R_c_guess is None:
        R_c_guess = np.where(R_c_unp <= Rx, R_c_unp, np.maximum(Rx, R_c_cfl))
    R_c = np.full(shape, np.nan)
    W_c = np.full(shape, np.nan)
    fdu, fdc = Delta_f_unp.ravel(), Delta_f_cfl.ravel()
    fnu, fnc = delta_n_C_unp.ravel(), delta_n_C_cfl.ravel()
    fg = R_c_guess.ravel()
    fR, fW = R_c.ravel(), W_c.ravel()
    for k in range(fR.size):
        gk = fg[k]
        if not np.isfinite(gk) or gk <= 0:
            continue
        du, dc, nu, nc = fdu[k], fdc[k], fnu[k], fnc[k]

        def _neg_W(R, _du=du, _dc=dc, _nu=nu, _nc=nc):
            S = S_func(R)
            Df = (1 - S) * _du + S * _dc
            dnC = (1 - S) * _nu + S * _nc
            return -work_of_formation(R, Df, sigma, dnC)

        sol = minimize(lambda R: _neg_W(R[0]), x0=[gk])
        if sol.x[0] > 0:
            fR[k] = sol.x[0]
            fW[k] = -sol.fun
    return fR.reshape(shape), fW.reshape(shape)


###############################################################################
#                                                                             #
#  W(R) profile at a single point (compute_energy_barrier)                    #
#                                                                             #
###############################################################################
def _build_H_from_interp(H_interp, pt, eq_type):
    """Evaluate hadronic interpolators at one point -> SimpleNamespace."""
    T = pt[-1]
    mu_nu_val = float(H_interp['mu_nu'](*pt)) if 'mu_nu' in H_interp else 0.0
    H = SimpleNamespace(
        P_total=float(H_interp['P'](*pt)), mu_B=float(H_interp['mu_B'](*pt)),
        mu_C=float(H_interp['mu_C'](*pt)), mu_S=float(H_interp['mu_S'](*pt)),
        mu_e=float(H_interp['mu_e'](*pt)), mu_nu=mu_nu_val, T=T)
    if 'Y_C' in H_interp:
        H.Y_C = float(H_interp['Y_C'](*pt))
    elif eq_type == 'fixed_yc':
        H.Y_C = pt[1]
    if hasattr(H, 'Y_C'):
        H.Y_e = H.Y_C
    if 'Y_S' in H_interp:
        H.Y_S = float(H_interp['Y_S'](*pt))
    return H


def _Qs_from_interp(Q_interp, pt, eq_type, mu_nu_val, Y_L_H=None):
    """Evaluate Q* interpolators at one point -> SimpleNamespace."""
    q_keys = ['n_B', 'P_total', 'mu_B', 'mu_C', 'mu_S', 'mu_e',
              'Y_C', 'Y_S', 'Y_e']
    q_vals = {k: float(Q_interp[k](*pt)) for k in q_keys}
    q_vals['mu_nu'] = mu_nu_val
    if eq_type == 'trapped_neutrinos':
        q_vals['Y_nu'] = Y_L_H - q_vals['Y_C']
    else:
        q_vals['Y_nu'] = 0.0
    return SimpleNamespace(**q_vals)


def _Delta_f_and_dnC_from_Qs(Qs, H, electric_charge_mode):
    """Scalar (Delta_f, delta_n_C) from a Q* state (dnC=0 for lcn/gcn)."""
    df = float(driving_force(Qs, H))
    dnC = 0.0 if electric_charge_mode in ('lcn', 'gcn') else _dnC(Qs)
    return df, dnC


def compute_energy_barrier(
    H_interp, n_B_H, T, sigma, Q_interp=None, electric_charge_mode='gcn',
    params=None, flavor_mode='saddlepoint', quark_phase='unpaired',
    Delta0=None, Y_L_H=None, Y_C_H=None, R_values=None,
    include_photons=True, include_gluons=True, include_thermal_neutrinos=True,
    switching_mode='step', Rx=None, switching_width=None,
    Q_interp_unp=None, Q_interp_cfl=None, lambda_D=None,
):
    """Compute the work-of-formation profile W(R) at one physical point.

    Q* is obtained by one of three paths: (A) coulomb_minimize -> Q*(R) re-solved
    at each R; (B) ``Q_interp`` provided -> Q* interpolated (lcn/gcn/gcn_coulomb/
    screening); (C) fallback -> Q* solved once via ``get_solver_Qs``. For
    ``quark_phase='unpCFL'`` the unpaired and CFL barriers are blended by S(R).
    ``electric_charge_mode='screening'`` uses the GCN composition with a
    Debye-screened Coulomb term (lambda_D from chi_e(H), or the explicit override).

    Returns an ``EnergyBarrierResult``.
    """
    eq_type = H_interp['eq_type']
    if eq_type == 'beta_eq':
        pt = (n_B_H, T)
    elif eq_type == 'trapped_neutrinos':
        pt = (n_B_H, Y_L_H, T)
    elif eq_type == 'fixed_yc':
        pt = (n_B_H, Y_C_H, T)
    else:
        raise ValueError(f"Unsupported eq_type: '{eq_type}'")

    H = _build_H_from_interp(H_interp, pt, eq_type)
    if R_values is None:
        R_values = np.linspace(0, 20.0, 500)
    R_values = np.asarray(R_values, dtype=float)
    n_R = len(R_values)
    lam = (_lambda_D_of(H, lambda_D)
           if electric_charge_mode == 'screening' else None)

    Delta_f = np.full(n_R, np.nan)
    delta_n_C = np.full(n_R, np.nan)

    common_solver_kw = dict(include_photons=include_photons,
                            include_gluons=include_gluons,
                            include_thermal_neutrinos=include_thermal_neutrinos)

    if quark_phase == 'unpCFL':
        if Rx is None:
            raise ValueError("Rx required for quark_phase='unpCFL'")
        if Delta0 is None:
            raise ValueError("Delta0 required for quark_phase='unpCFL'")
        S_func = get_switching_function(switching_mode, Rx, switching_width)

        if electric_charge_mode == 'coulomb_minimize':
            if params is None:
                raise ValueError("params required for coulomb_minimize")
            prev_u = prev_c = None
            for i, R in enumerate(R_values):
                if R == 0:
                    Delta_f[i] = 0.0
                    delta_n_C[i] = 0.0
                    continue
                Qu = solve_coulomb_minimize_at_R(
                    R, H, params, sigma, **common_solver_kw, initial_guess=prev_u)
                Qc = solve_coulomb_minimize_cfl_at_R(
                    R, H, params, Delta0, sigma, **common_solver_kw,
                    initial_guess=prev_c)
                Sv = float(S_func(R))
                if Qu is not None and Qc is not None:
                    Delta_f[i] = ((1 - Sv) * float(driving_force(Qu, H))
                                  + Sv * float(driving_force(Qc, H)))
                    delta_n_C[i] = (1 - Sv) * _dnC(Qu) + Sv * _dnC(Qc)
                    prev_u = np.array([Qu.mu_u, Qu.mu_d, Qu.mu_s, Qu.mu_e])
                    prev_c = np.array([Qc.mu_u, Qc.mu_d, Qc.mu_s, Qc.mu_e])
                elif Qu is not None:
                    Delta_f[i] = (1 - Sv) * float(driving_force(Qu, H))
                    delta_n_C[i] = (1 - Sv) * _dnC(Qu)
                    prev_u = np.array([Qu.mu_u, Qu.mu_d, Qu.mu_s, Qu.mu_e])
                elif Qc is not None:
                    Delta_f[i] = Sv * float(driving_force(Qc, H))
                    delta_n_C[i] = Sv * _dnC(Qc)
                    prev_c = np.array([Qc.mu_u, Qc.mu_d, Qc.mu_s, Qc.mu_e])
        else:
            df_u = df_c = np.nan
            dnC_u = dnC_c = 0.0
            if Q_interp_unp is not None:
                Qu = _Qs_from_interp(Q_interp_unp, pt, eq_type, H.mu_nu, Y_L_H)
                df_u, dnC_u = _Delta_f_and_dnC_from_Qs(Qu, H, electric_charge_mode)
            elif params is not None:
                Qu = get_solver_Qs(flavor_mode, electric_charge_mode, params,
                                   quark_phase='unpaired', sigma=sigma,
                                   **common_solver_kw)(H)
                if Qu is not None:
                    df_u, dnC_u = _Delta_f_and_dnC_from_Qs(
                        Qu, H, electric_charge_mode)
            else:
                raise ValueError("Q_interp_unp or params required for unpCFL")
            if Q_interp_cfl is not None:
                Qc = _Qs_from_interp(Q_interp_cfl, pt, eq_type, H.mu_nu, Y_L_H)
                df_c, dnC_c = _Delta_f_and_dnC_from_Qs(Qc, H, electric_charge_mode)
            elif params is not None:
                Qc = get_solver_Qs(flavor_mode, electric_charge_mode, params,
                                   quark_phase='cfl', Delta0=Delta0, sigma=sigma,
                                   **common_solver_kw)(H)
                if Qc is not None:
                    df_c, dnC_c = _Delta_f_and_dnC_from_Qs(
                        Qc, H, electric_charge_mode)
            else:
                raise ValueError("Q_interp_cfl or params required for unpCFL")
            S_arr = S_func(R_values)
            Delta_f[:] = (1 - S_arr) * df_u + S_arr * df_c
            delta_n_C[:] = (1 - S_arr) * dnC_u + S_arr * dnC_c

    elif electric_charge_mode == 'coulomb_minimize':
        if params is None:
            raise ValueError("params required for coulomb_minimize")
        prev = None
        for i, R in enumerate(R_values):
            if R == 0:
                Delta_f[i] = 0.0
                delta_n_C[i] = 0.0
                continue
            if quark_phase == 'cfl':
                Qs_R = solve_coulomb_minimize_cfl_at_R(
                    R, H, params, Delta0, sigma, include_photons, include_gluons,
                    include_thermal_neutrinos, initial_guess=prev)
            else:
                Qs_R = solve_coulomb_minimize_at_R(
                    R, H, params, sigma, include_photons, include_gluons,
                    include_thermal_neutrinos, initial_guess=prev)
            if Qs_R is None:
                continue
            Delta_f[i] = float(driving_force(Qs_R, H))
            delta_n_C[i] = _dnC(Qs_R)
            prev = np.array([Qs_R.mu_u, Qs_R.mu_d, Qs_R.mu_s, Qs_R.mu_e])

    elif Q_interp is not None:
        Qs = _Qs_from_interp(Q_interp, pt, eq_type, H.mu_nu, Y_L_H)
        df, dnC = _Delta_f_and_dnC_from_Qs(Qs, H, electric_charge_mode)
        Delta_f[:] = df
        delta_n_C[:] = dnC

    else:
        if params is None:
            raise ValueError("Either Q_interp or params must be provided")
        Qs = get_solver_Qs(flavor_mode, electric_charge_mode, params,
                           quark_phase=quark_phase, Delta0=Delta0, sigma=sigma,
                           **common_solver_kw)(H)
        if Qs is not None:
            # coulomb_minimize path returns a tuple, handled above; here Qs is a state
            df, dnC = _Delta_f_and_dnC_from_Qs(Qs, H, electric_charge_mode)
            Delta_f[:] = df
            delta_n_C[:] = dnC

    W_b = bulk_W(R_values, Delta_f)
    W_s = surface_W(R_values, sigma)
    if lam is None:
        W_c = coulomb_W(R_values, delta_n_C)
    else:
        W_c = coulomb_screened_W(R_values, delta_n_C, lam)
    W = W_b + W_s + W_c

    return EnergyBarrierResult(
        R=R_values, W=W, W_bulk=W_b, W_surface=W_s, W_coulomb=W_c,
        Delta_f=Delta_f, delta_n_C=delta_n_C, sigma=sigma,
        lambda_D=(lam if lam is not None else np.nan))


# =============================================================================
# Q* swept over R at one hadronic point (coulomb_minimize W(R) plotting)
# =============================================================================
def compute_Qs_along_R(H_point, R_vec, params, sigma,
                       quark_phase='unpaired', Delta0=None,
                       include_photons=True, include_gluons=True,
                       include_thermal_neutrinos=True, initial_guess=None):
    """coulomb_minimize Q* as a function of R at one hadronic point.

    Fixes the hadronic point and sweeps R (warm-started), returning per-R Q*
    quantities plus the derived delta_n_C, Delta_f and barrier W(R). Used to plot
    the coulomb_minimize barrier shape.
    """
    from nucleation.tables import _BASE_DATA_KEYS
    solver_kw = dict(include_photons=include_photons,
                     include_gluons=include_gluons,
                     include_thermal_neutrinos=include_thermal_neutrinos)
    R_vec = np.atleast_1d(np.asarray(R_vec, dtype=float))
    n = R_vec.size
    out = {k: np.full(n, np.nan) for k in _BASE_DATA_KEYS}
    out['converged'] = np.zeros(n, dtype=bool)
    out['R'] = R_vec.copy()
    out['delta_n_C'] = np.full(n, np.nan)
    out['Delta_f'] = np.full(n, np.nan)
    out['W'] = np.full(n, np.nan)

    guess = initial_guess
    for k in range(n):
        R_val = float(R_vec[k])
        if not np.isfinite(R_val):
            continue
        if quark_phase == 'cfl':
            result = solve_coulomb_minimize_cfl_at_R(
                R_val, H_point, params, Delta0, sigma, **solver_kw,
                initial_guess=guess)
        else:
            result = solve_coulomb_minimize_at_R(
                R_val, H_point, params, sigma, **solver_kw, initial_guess=guess)
        if result is None:
            continue
        for key in _BASE_DATA_KEYS:
            out[key][k] = getattr(result, key)
        out['converged'][k] = True
        guess = np.array([result.mu_u, result.mu_d, result.mu_s, result.mu_e])
        Qs_pt = SimpleNamespace(
            P_total=result.P_total, n_B=result.n_B, mu_B=result.mu_B,
            mu_C=result.mu_C, mu_S=result.mu_S, mu_e=result.mu_e,
            mu_nu=H_point.mu_nu, Y_C=result.Y_C, Y_S=result.Y_S,
            Y_e=result.Y_e, Y_nu=H_point.Y_nu)
        dnC = _dnC(result)
        Df = float(driving_force(Qs_pt, H_point))
        out['delta_n_C'][k] = dnC
        out['Delta_f'][k] = Df
        out['W'][k] = work_of_formation(R_val, Df, sigma, dnC)
    return out
