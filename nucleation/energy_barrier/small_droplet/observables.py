"""
Nucleation Observables
======================

Energy barrier, thermal nucleation, and quantum nucleation observables
for the hadron-to-quark phase transition in the small-droplet (CNT) limit.

Sections
--------
1. **General**: Energy barrier W(R) at a single physical point, and
   nucleation temperature (root-finding along T for any log10_tau
   interpolator).

2. **Thermal**: Grid-level thermal nucleation observables (R_c, W_c,
   Gamma, tau) and interpolator builders.

3. **Quantum**: Grid-level quantum (WKB) tunneling observables
   (tau_qt, A, E_0, nu_0).

Usage
-----
Energy barrier with pre-built Q* interpolators (lcn/gcn/gcn_coulomb)::

    >>> from nucleation.energy_barrier.small_droplet import compute_energy_barrier
    >>> barrier = compute_energy_barrier(
    ...     H_interp, n_B_H=0.19, T=10.0, sigma=30.0,
    ...     Q_interp=Q_interp,
    ...     electric_charge_mode='gcn',
    ...     Y_L_H=0.4,
    ... )
    >>> print(barrier.W, barrier.Delta_f)

Energy barrier with coulomb_minimize (R-dependent Q*, solved on the fly)::

    >>> barrier = compute_energy_barrier(
    ...     H_interp, n_B_H=0.19, T=10.0, sigma=30.0,
    ...     electric_charge_mode='coulomb_minimize',
    ...     params=my_params,
    ... )

Thermal nucleation observables::

    >>> from nucleation.energy_barrier.small_droplet import (
    ...     compute_thermal_nucleation_observables,
    ...     build_thermal_nucleation_interpolators,
    ... )
    >>> obs = compute_thermal_nucleation_observables(
    ...     hadronic_table, params=my_params, sigma=30.0,
    ...     flavor_mode='saddlepoint', electric_charge_mode='gcn',
    ... )
    >>> interp = build_thermal_nucleation_interpolators(obs)
    >>> tau_val = interp['tau'](n_B_H, T)

Quantum nucleation observables::

    >>> from nucleation.energy_barrier.small_droplet import (
    ...     compute_quantum_nucleation_observables,
    ... )
    >>> qobs = compute_quantum_nucleation_observables(
    ...     hadronic_table, Qstar_table, sigma=30.0,
    ...     electric_charge_mode='gcn',
    ... )
    >>> print(qobs.tau_qt, qobs.A)

Nucleation temperature (general — works with any log10_tau interpolator)::

    >>> from nucleation.energy_barrier.small_droplet import (
    ...     compute_nucleation_temperature,
    ...     build_thermal_nucleation_interpolators,
    ... )
    >>> interp = build_thermal_nucleation_interpolators(thermal_obs)
    >>> T_result = compute_nucleation_temperature(
    ...     interp['log10_tau'],
    ...     thermal_obs.hadronic_grids,
    ...     thermal_obs.eq_type,
    ...     tau_target=1.0,
    ... )
    >>> print(T_result.T_nuc)
"""

import os
import numpy as np
from types import SimpleNamespace
from dataclasses import dataclass
from scipy.optimize import root_scalar, minimize
from scipy.interpolate import RegularGridInterpolator, interp1d

from nucleation.energy_barrier.small_droplet.barrier import (
    driving_force,
    bulk_W, surface_W, coulomb_W,
    work_of_formation,
    critical_work_noCoulomb,
    critical_radius_noCoulomb,
    get_switching_function,
)
from nucleation.energy_barrier.small_droplet.solvers import (
    get_solver_Qs,
    solve_saddlepoint_minimizecoulomb_at_R,
    solve_saddlepoint_minimizecoulomb_cfl_at_R,
)
from nucleation.general_nucleation.thermal import nucleation_rate, nucleation_time
from nucleation.energy_barrier.small_droplet.table import (
    compute_Qstar_table, GRID_AXES,
)


###############################################################################
#                                                                             #
#  1. GENERAL — Energy barrier and nucleation temperature                     #
#                                                                             #
###############################################################################

# =============================================================================
# Output dataclasses
# =============================================================================
@dataclass
class EnergyBarrierResult:
    """Energy barrier W(R) at a single physical point.

    Attributes
    ----------
    R : np.ndarray
        Radius array (fm).
    W : np.ndarray
        Total work of formation W(R) (MeV).
    W_bulk : np.ndarray
        Bulk (volume) contribution (MeV).
    W_surface : np.ndarray
        Surface contribution (MeV).
    W_coulomb : np.ndarray
        Coulomb contribution (MeV).
    Delta_f : np.ndarray
        Bulk driving force (MeV/fm^3). Array matching R; constant
        for lcn/gcn/gcn_coulomb, R-dependent for coulomb_minimize.
    delta_n_C : np.ndarray
        Net charge density (fm^-3). Array matching R; zero for
        lcn/gcn, constant for gcn_coulomb, R-dependent for
        coulomb_minimize.
    sigma : float
        Surface tension (MeV/fm^2).
    """
    R: np.ndarray
    W: np.ndarray
    W_bulk: np.ndarray
    W_surface: np.ndarray
    W_coulomb: np.ndarray
    Delta_f: np.ndarray
    delta_n_C: np.ndarray
    sigma: float


@dataclass
class NucleationTemperatureResult:
    """Result of nucleation temperature computation.

    Attributes
    ----------
    hadronic_grids : dict
        Input grids (n_B_H, and Y_L_H or Y_C_H for multi-dim cases).
    T_nuc : np.ndarray
        Nucleation temperature (MeV). NaN where no solution found.
    tau_target : float
        Target nucleation time (s).
    converged : np.ndarray
        Boolean mask: True where root finding succeeded.
    residual : np.ndarray
        |log10(tau(T_nuc)) - log10(tau_target)|. NaN where not converged.
    eq_type : str
        Equilibrium type.
    """
    hadronic_grids: dict
    T_nuc: np.ndarray
    tau_target: float
    converged: np.ndarray
    residual: np.ndarray
    eq_type: str


# =============================================================================
# Energy barrier W(R) at a single physical point
# =============================================================================
def _build_H_from_interp(H_interp, pt, eq_type):
    """Evaluate hadronic interpolators at a single point.

    Returns a SimpleNamespace with P_total, mu_B, mu_C, mu_S, mu_e,
    mu_nu, T, and optionally Y_C, Y_S, Y_e.

    T is taken from pt[-1]. For fixed_yc, Y_C is taken from pt[1].
    For other eq_types, Y_C is read from H_interp['Y_C'].
    """
    T = pt[-1]
    mu_nu_val = float(H_interp['mu_nu'](*pt)) if 'mu_nu' in H_interp else 0.0
    H = SimpleNamespace(
        P_total=float(H_interp['P'](*pt)),
        mu_B=float(H_interp['mu_B'](*pt)),
        mu_C=float(H_interp['mu_C'](*pt)),
        mu_S=float(H_interp['mu_S'](*pt)),
        mu_e=float(H_interp['mu_e'](*pt)),
        mu_nu=mu_nu_val,
        T=T,
    )
    # Y_C, Y_S, Y_e — needed by some solver paths (e.g. frozen)
    if 'Y_C' in H_interp:
        H.Y_C = float(H_interp['Y_C'](*pt))
    elif eq_type == 'fixed_yc':
        H.Y_C = pt[1]   # Y_C_H is the second element of pt
    if hasattr(H, 'Y_C'):
        H.Y_e = H.Y_C   # charge neutrality in H
    if 'Y_S' in H_interp:
        H.Y_S = float(H_interp['Y_S'](*pt))
    return H


def _Qs_from_interp(Q_interp, pt, eq_type, mu_nu_val, Y_L_H=None):
    """Evaluate Q* interpolators at a single point.

    Returns a SimpleNamespace with all fields needed by ``driving_force``.
    """
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
    """Compute scalar driving force and charge density from Q* result."""
    df = float(driving_force(Qs, H))
    if electric_charge_mode in ('lcn', 'gcn'):
        dnC = 0.0
    else:
        dnC = float((Qs.Y_C - Qs.Y_e) * Qs.n_B)
    return df, dnC


def compute_energy_barrier(
    H_interp,
    n_B_H,
    T,
    sigma,
    Q_interp=None,
    electric_charge_mode='gcn',
    params=None,
    flavor_mode='saddlepoint',
    quark_phase='unpaired',
    Delta0=None,
    Y_L_H=None,
    Y_C_H=None,
    R_values=None,
    include_photons=True,
    include_gluons=True,
    include_thermal_neutrinos=True,
    switching_mode='step',
    Rx=None,
    switching_width=None,
    Q_interp_unp=None,
    Q_interp_cfl=None,
):
    """
    Compute W(R) at a single physical point.

    Three computation paths for Q*:

    - **Path A** (``coulomb_minimize``): Q*(R) depends on R through the
      Coulomb correction. The solver is called at each R value.
      Requires ``params``.
    - **Path B** (``Q_interp`` provided): Q* is interpolated from a
      pre-computed table. Works for lcn/gcn/gcn_coulomb.
    - **Path C** (fallback): Q* is solved once at the physical point
      using ``get_solver_Qs``. Requires ``params``.

    Parameters
    ----------
    H_interp : dict
        Pre-built hadronic interpolators (from ``build_interpolators``).
    n_B_H : float
        Baryon number density (fm^-3).
    T : float
        Temperature (MeV).
    sigma : float
        Surface tension (MeV/fm^2).
    Q_interp : dict or None
        Pre-built Q* interpolators (from ``build_Qstar_interpolators``).
        Optional; used for lcn/gcn/gcn_coulomb. Ignored for
        ``coulomb_minimize``.
    electric_charge_mode : str
        'lcn', 'gcn', 'gcn_coulomb', or 'coulomb_minimize'.
    params : AlphaBagParams or None
        Quark EOS parameters. Required when ``Q_interp`` is None or
        when ``electric_charge_mode='coulomb_minimize'``.
    flavor_mode : str
        'frozen' or 'saddlepoint'. Used only in solver paths (A, C).
    quark_phase : str
        'unpaired', 'cfl', or 'unpCFL'. Used only in solver paths.
    Delta0 : float or None
        CFL pairing gap (MeV). Required if quark_phase='cfl' or 'unpCFL'.
    Y_L_H : float or None
        Lepton fraction. Required for 'trapped_neutrinos'.
    Y_C_H : float or None
        Charge fraction. Required for 'fixed_yc'.
    R_values : array-like or None
        Radius array (fm). If None, uses [0, 20] fm with 500 points.
    include_photons, include_gluons, include_thermal_neutrinos : bool
    switching_mode : str
        'step' or 'tanh'. Only used when quark_phase='unpCFL'.
    Rx : float or None
        Crossover radius (fm). Required when quark_phase='unpCFL'.
    switching_width : float or None
        Tanh transition width (fm). Required when switching_mode='tanh'.
    Q_interp_unp : dict or None
        Pre-built Q* interpolators for unpaired phase. For unpCFL.
    Q_interp_cfl : dict or None
        Pre-built Q* interpolators for CFL phase. For unpCFL.

    Returns
    -------
    EnergyBarrierResult
    """
    # ---- Build point tuple ----
    eq_type = H_interp['eq_type']
    if eq_type == 'beta_eq':
        pt = (n_B_H, T)
    elif eq_type == 'trapped_neutrinos':
        pt = (n_B_H, Y_L_H, T)
    elif eq_type == 'fixed_yc':
        pt = (n_B_H, Y_C_H, T)
    else:
        raise ValueError(f"Unsupported eq_type: '{eq_type}'")

    # ---- Evaluate H at the point ----
    H = _build_H_from_interp(H_interp, pt, eq_type)

    # ---- R array ----
    if R_values is None:
        R_values = np.linspace(0, 20.0, 500)
    R_values = np.asarray(R_values, dtype=float)
    n_R = len(R_values)

    # ---- Compute Delta_f(R) and delta_n_C(R) ----
    Delta_f = np.full(n_R, np.nan)
    delta_n_C = np.full(n_R, np.nan)

    if quark_phase == 'unpCFL':
        # --- Path unpCFL: blend unpaired and CFL with switching function ---
        if Rx is None:
            raise ValueError("Rx required for quark_phase='unpCFL'")
        if Delta0 is None:
            raise ValueError("Delta0 required for quark_phase='unpCFL'")

        S_func = get_switching_function(switching_mode, Rx, switching_width)
        common_solver_kw = dict(
            include_photons=include_photons,
            include_gluons=include_gluons,
            include_thermal_neutrinos=include_thermal_neutrinos,
        )

        if electric_charge_mode == 'coulomb_minimize':
            # Solve both phases at each R
            if params is None:
                raise ValueError("params required for coulomb_minimize")
            prev_guess_unp = None
            prev_guess_cfl = None
            for i, R in enumerate(R_values):
                if R == 0:
                    Delta_f[i] = 0.0
                    delta_n_C[i] = 0.0
                    continue

                Qs_unp = solve_saddlepoint_minimizecoulomb_at_R(
                    R, H, params, sigma, **common_solver_kw,
                    initial_guess=prev_guess_unp)
                Qs_cfl = solve_saddlepoint_minimizecoulomb_cfl_at_R(
                    R, H, params, Delta0, sigma, **common_solver_kw,
                    initial_guess=prev_guess_cfl)

                S_val = float(S_func(R))

                if Qs_unp is not None and Qs_cfl is not None:
                    df_unp = float(driving_force(Qs_unp, H))
                    df_cfl = float(driving_force(Qs_cfl, H))
                    dnC_unp = float((Qs_unp.Y_C - Qs_unp.Y_e) * Qs_unp.n_B)
                    dnC_cfl = float((Qs_cfl.Y_C - Qs_cfl.Y_e) * Qs_cfl.n_B)
                    Delta_f[i] = (1 - S_val) * df_unp + S_val * df_cfl
                    delta_n_C[i] = (1 - S_val) * dnC_unp + S_val * dnC_cfl
                    prev_guess_unp = np.array([Qs_unp.mu_u, Qs_unp.mu_d,
                                               Qs_unp.mu_s, Qs_unp.mu_e])
                    prev_guess_cfl = np.array([Qs_cfl.mu_u, Qs_cfl.mu_d,
                                               Qs_cfl.mu_s, Qs_cfl.mu_e])
                elif Qs_unp is not None:
                    Delta_f[i] = (1 - S_val) * float(driving_force(Qs_unp, H))
                    delta_n_C[i] = (1 - S_val) * float(
                        (Qs_unp.Y_C - Qs_unp.Y_e) * Qs_unp.n_B)
                    prev_guess_unp = np.array([Qs_unp.mu_u, Qs_unp.mu_d,
                                               Qs_unp.mu_s, Qs_unp.mu_e])
                elif Qs_cfl is not None:
                    Delta_f[i] = S_val * float(driving_force(Qs_cfl, H))
                    delta_n_C[i] = S_val * float(
                        (Qs_cfl.Y_C - Qs_cfl.Y_e) * Qs_cfl.n_B)
                    prev_guess_cfl = np.array([Qs_cfl.mu_u, Qs_cfl.mu_d,
                                               Qs_cfl.mu_s, Qs_cfl.mu_e])
        else:
            # Non-coulomb_minimize: solve each phase once, blend with S(R)
            df_unp_val = np.nan
            df_cfl_val = np.nan
            dnC_unp_val = 0.0
            dnC_cfl_val = 0.0

            # Get unpaired Q*
            if Q_interp_unp is not None:
                Qs_unp = _Qs_from_interp(Q_interp_unp, pt, eq_type,
                                          H.mu_nu, Y_L_H)
                df_unp_val, dnC_unp_val = _Delta_f_and_dnC_from_Qs(
                    Qs_unp, H, electric_charge_mode)
            elif params is not None:
                solver_unp = get_solver_Qs(
                    flavor_mode, electric_charge_mode, params,
                    quark_phase='unpaired', sigma=sigma, **common_solver_kw)
                Qs_unp_result = solver_unp(H)
                if Qs_unp_result is not None:
                    df_unp_val, dnC_unp_val = _Delta_f_and_dnC_from_Qs(
                        Qs_unp_result, H, electric_charge_mode)
            else:
                raise ValueError(
                    "Either Q_interp_unp or params must be provided for unpCFL")

            # Get CFL Q*
            if Q_interp_cfl is not None:
                Qs_cfl = _Qs_from_interp(Q_interp_cfl, pt, eq_type,
                                          H.mu_nu, Y_L_H)
                df_cfl_val, dnC_cfl_val = _Delta_f_and_dnC_from_Qs(
                    Qs_cfl, H, electric_charge_mode)
            elif params is not None:
                solver_cfl = get_solver_Qs(
                    flavor_mode, electric_charge_mode, params,
                    quark_phase='cfl', Delta0=Delta0, sigma=sigma,
                    **common_solver_kw)
                Qs_cfl_result = solver_cfl(H)
                if Qs_cfl_result is not None:
                    df_cfl_val, dnC_cfl_val = _Delta_f_and_dnC_from_Qs(
                        Qs_cfl_result, H, electric_charge_mode)
            else:
                raise ValueError(
                    "Either Q_interp_cfl or params must be provided for unpCFL")

            # Blend with switching function
            S_arr = S_func(R_values)
            Delta_f[:] = (1 - S_arr) * df_unp_val + S_arr * df_cfl_val
            delta_n_C[:] = (1 - S_arr) * dnC_unp_val + S_arr * dnC_cfl_val

    elif electric_charge_mode == 'coulomb_minimize':
        # --- Path A: R-dependent Q* via Coulomb solver ---
        if params is None:
            raise ValueError(
                "params required for electric_charge_mode='coulomb_minimize'")
        prev_guess = None
        for i, R in enumerate(R_values):
            if R == 0:
                Delta_f[i] = 0.0
                delta_n_C[i] = 0.0
                continue

            if quark_phase == 'cfl':
                Qs_R = solve_saddlepoint_minimizecoulomb_cfl_at_R(
                    R, H, params, Delta0, sigma,
                    include_photons, include_gluons,
                    include_thermal_neutrinos, initial_guess=prev_guess)
            else:
                Qs_R = solve_saddlepoint_minimizecoulomb_at_R(
                    R, H, params, sigma,
                    include_photons, include_gluons,
                    include_thermal_neutrinos, initial_guess=prev_guess)

            if Qs_R is None:
                continue

            Delta_f[i] = float(driving_force(Qs_R, H))
            delta_n_C[i] = float((Qs_R.Y_C - Qs_R.Y_e) * Qs_R.n_B)
            prev_guess = np.array([Qs_R.mu_u, Qs_R.mu_d, Qs_R.mu_s, Qs_R.mu_e])

    elif Q_interp is not None:
        # --- Path B: Q* from pre-built interpolators ---
        Qs = _Qs_from_interp(Q_interp, pt, eq_type, H.mu_nu, Y_L_H)
        df, dnC = _Delta_f_and_dnC_from_Qs(Qs, H, electric_charge_mode)
        Delta_f[:] = df
        delta_n_C[:] = dnC

    else:
        # --- Path C: fallback solver (single Q* solve) ---
        if params is None:
            raise ValueError(
                "Either Q_interp or params must be provided")
        solver = get_solver_Qs(
            flavor_mode, electric_charge_mode, params,
            quark_phase=quark_phase, Delta0=Delta0, sigma=sigma,
            include_photons=include_photons,
            include_gluons=include_gluons,
            include_thermal_neutrinos=include_thermal_neutrinos,
        )
        Qs_result = solver(H)
        if Qs_result is not None:
            df, dnC = _Delta_f_and_dnC_from_Qs(
                Qs_result, H, electric_charge_mode)
            Delta_f[:] = df
            delta_n_C[:] = dnC

    # ---- W(R) and components ----
    W_b = bulk_W(R_values, Delta_f)
    W_s = surface_W(R_values, sigma)
    W_c = coulomb_W(R_values, delta_n_C)
    W = work_of_formation(R_values, Delta_f, sigma, delta_n_C)

    return EnergyBarrierResult(
        R=R_values,
        W=W,
        W_bulk=W_b,
        W_surface=W_s,
        W_coulomb=W_c,
        Delta_f=Delta_f,
        delta_n_C=delta_n_C,
        sigma=sigma,
    )


# =============================================================================
# Nucleation temperature: T at which tau = tau_target
# =============================================================================
def _find_T_root(f, T_arr, T_guesses=None):
    """Find temperature T where f(T) = 0 (downward zero crossing).

    Scans grid points for bracketing, then refines with Brent's method.

    Parameters
    ----------
    f : callable
        Scalar function f(T) -> float.  Typically
        ``log10_tau(n_B_H, ..., T) - log10(tau_target)``.
        Root is where f crosses zero from above (f > 0 at lower T,
        f < 0 at higher T).
    T_arr : array-like
        Temperature grid used for bracketing scan.
    T_guesses : list of float, or None
        Warm-start guesses tried in order with secant method.
        Typically ``[T_extrapolated, T_previous]``.

    Returns
    -------
    T_root : float
        Root temperature, or NaN if not found.
    converged : bool
    residual : float
        |f(T_root)|, i.e. |log10(tau(T_root)) - log10(tau_target)|.
        NaN if not converged.
    """
    # Evaluate at grid points for bracketing
    vals = np.array([f(T) for T in T_arr])
    valid = np.isfinite(vals) & (T_arr > 0)
    if np.sum(valid) < 2:
        return np.nan, False, np.nan

    T_valid = T_arr[valid]
    vals_valid = vals[valid]

    # Fast path: secant method with warm-start guesses (before sign-change gate)
    if T_guesses is not None:
        eps = (T_valid[-1] - T_valid[0]) * 1e-4

        def _try_secant(x0, x1):
            try:
                sol = root_scalar(f, x0=x0, x1=x1, method='secant')
                if sol.converged and T_valid[0] <= sol.root <= T_valid[-1]:
                    if f(sol.root - eps) > 0 and f(sol.root + eps) < 0:
                        return sol.root, True, abs(f(sol.root))
            except (ValueError, RuntimeError):
                pass
            return None

        # 1) Secant from T_extrapolated
        # 2) Secant from T_previous
        for guess in T_guesses:
            result = _try_secant(guess, guess * 1.01)
            if result is not None:
                return result

        # 3) Secant with x0=T_extrapolated, x1=T_previous
        if len(T_guesses) >= 2:
            result = _try_secant(T_guesses[0], T_guesses[1])
            if result is not None:
                return result

    # Fallback: scan sign changes for downward crossings
    # (f goes from + to -, i.e. tau drops below tau_target)
    sign_changes = np.where(np.diff(np.sign(vals_valid)))[0]
    if len(sign_changes) == 0:
        return np.nan, False, np.nan

    downward = [sc for sc in sign_changes
                if vals_valid[sc] > 0 and vals_valid[sc + 1] < 0]
    if len(downward) == 0:
        return np.nan, False, np.nan

    # Pick the first downward crossing (lowest T)
    j = downward[0]
    T_lo = float(T_valid[j])
    T_hi = float(T_valid[j + 1])

    try:
        sol = root_scalar(f, bracket=[T_lo, T_hi], method='brentq')
        if sol.converged:
            return sol.root, True, abs(f(sol.root))
    except (ValueError, RuntimeError):
        pass

    return np.nan, False, np.nan


def compute_nucleation_temperature(
    log10_tau_fn,
    hadronic_grids,
    eq_type,
    tau_target=1.0,
    T_guess=None,
    verbose=False,
):
    """
    Find the temperature T at which tau = tau_target over the full grid.

    This function is general: it accepts any ``log10_tau`` interpolator
    (thermal or quantum) and root-finds along T for each (n_B_H, ...)
    point.

    Parameters
    ----------
    log10_tau_fn : callable
        Interpolator returning log10(tau) at a given point.
        Signature: f(n_B_H, T) for beta_eq, or
        f(n_B_H, Y_L_H, T) for trapped_neutrinos, etc.
    hadronic_grids : dict
        Grid arrays: must contain 'n_B_H' and 'T', plus 'Y_L_H'
        or 'Y_C_H' for multi-dimensional equilibrium types.
    eq_type : str
        'beta_eq', 'trapped_neutrinos', or 'fixed_yc'.
    tau_target : float
        Target nucleation time (s).
    T_guess : float or None
        Initial guess for T (MeV) at the first n_B point.
    verbose : bool

    Returns
    -------
    NucleationTemperatureResult
    """
    n_B_arr = hadronic_grids['n_B_H']
    T_arr = hadronic_grids['T']
    n_nB = len(n_B_arr)

    log_tau_target = np.log10(tau_target)

    # Determine non-T axes (everything except n_B_H and T)
    axes = GRID_AXES[eq_type]
    non_T_axes = [ax for ax in axes if ax not in ('n_B_H', 'T')]

    if len(non_T_axes) == 0:
        # ---- 1D: beta_eq ----
        T_nuc = np.full(n_nB, np.nan)
        conv = np.zeros(n_nB, dtype=bool)
        resid = np.full(n_nB, np.nan)
        prev_T = T_guess
        prev_nB = None
        prev_prev_T = None
        prev_prev_nB = None

        for i in range(n_nB):
            nB = float(n_B_arr[i])

            def f(T, _nB=nB):
                return log10_tau_fn(_nB, T) - log_tau_target

            # Build guess list: extrapolation first, then previous T
            T_guesses = []
            if (prev_prev_T is not None and prev_T is not None
                    and prev_prev_nB is not None and prev_nB is not None):
                dnB_prev = prev_nB - prev_prev_nB
                if abs(dnB_prev) > 0:
                    T_extrap = prev_T + (prev_T - prev_prev_T) * (nB - prev_nB) / dnB_prev
                    if T_extrap > 0:
                        T_guesses.append(T_extrap)
            if prev_T is not None:
                T_guesses.append(prev_T)

            T_root, ok, res = _find_T_root(f, T_arr, T_guesses or None)
            if ok:
                T_nuc[i] = T_root
                conv[i] = True
                resid[i] = res
                prev_prev_T = prev_T
                prev_prev_nB = prev_nB
                prev_T = T_root
                prev_nB = nB
            if verbose:
                guess_str = (f"{T_guesses[0]:.2f}" if T_guesses
                             else "None")
                if ok:
                    print(f"  n_B={n_B_arr[i]:.4f} -> T_nuc={T_root:.2f} MeV"
                          f"  (res={res:.2e}, T_guess={guess_str})")
                else:
                    print(f"  n_B={n_B_arr[i]:.4f} -> FAIL"
                          f"  (T_guess={guess_str})")

        out_grids = {'n_B_H': n_B_arr}

    elif len(non_T_axes) == 1:
        # ---- 2D: trapped_neutrinos or fixed_yc ----
        ax_name = non_T_axes[0]          # 'Y_L_H' or 'Y_C_H'
        ax_arr = hadronic_grids[ax_name]
        n_outer = len(ax_arr)

        T_nuc = np.full((n_nB, n_outer), np.nan)
        conv = np.zeros((n_nB, n_outer), dtype=bool)
        resid = np.full((n_nB, n_outer), np.nan)

        for k in range(n_outer):
            prev_T = T_guess
            prev_nB = None
            prev_prev_T = None
            prev_prev_nB = None
            outer_val = float(ax_arr[k])

            for i in range(n_nB):
                nB = float(n_B_arr[i])

                def f(T, _nB=nB, _ov=outer_val):
                    return log10_tau_fn(_nB, _ov, T) - log_tau_target

                # Build guess list: extrapolation first, then previous T
                T_guesses = []
                if (prev_prev_T is not None and prev_T is not None
                        and prev_prev_nB is not None and prev_nB is not None):
                    dnB_prev = prev_nB - prev_prev_nB
                    if abs(dnB_prev) > 0:
                        T_extrap = prev_T + (prev_T - prev_prev_T) * (nB - prev_nB) / dnB_prev
                        if T_extrap > 0:
                            T_guesses.append(T_extrap)
                if prev_T is not None:
                    T_guesses.append(prev_T)

                T_root, ok, res = _find_T_root(f, T_arr, T_guesses or None)
                if ok:
                    T_nuc[i, k] = T_root
                    conv[i, k] = True
                    resid[i, k] = res
                    prev_prev_T = prev_T
                    prev_prev_nB = prev_nB
                    prev_T = T_root
                    prev_nB = nB
                if verbose:
                    label = ax_name.replace('_H', '')
                    guess_str = (f"{T_guesses[0]:.2f}" if T_guesses
                                 else "None")
                    if ok:
                        print(f"  n_B={n_B_arr[i]:.4f}, "
                              f"{label}={outer_val:.3f}"
                              f" -> T_nuc={T_root:.2f} MeV"
                              f"  (res={res:.2e}, T_guess={guess_str})")
                    else:
                        print(f"  n_B={n_B_arr[i]:.4f}, "
                              f"{label}={outer_val:.3f}"
                              f" -> FAIL"
                              f"  (T_guess={guess_str})")

        out_grids = {'n_B_H': n_B_arr, ax_name: ax_arr}

    else:
        raise ValueError(f"Unsupported eq_type: '{eq_type}'")

    return NucleationTemperatureResult(
        hadronic_grids=out_grids,
        T_nuc=T_nuc,
        tau_target=tau_target,
        converged=conv,
        residual=resid,
        eq_type=eq_type,
    )


def compute_nucleation_density(nucleation_obs, tau_target=1.0, verbose=False):
    """Find n_B_H(T) at which tau = tau_target, for each T in the table grid.

    Root-finds along n_B at each fixed T grid point, using the raw
    table values with 1D cubic interpolation along n_B for sub-grid
    accuracy.  No multi-dimensional interpolator needed.

    Parameters
    ----------
    nucleation_obs : ThermalNucleationObservables
        Output of ``compute_thermal_nucleation_observables``.
    tau_target : float
        Target nucleation time (seconds).
    verbose : bool

    Returns
    -------
    SimpleNamespace
        T_arr, n_B_nuc, converged, residual, tau_target, eq_type.
        For 2D cases, also outer_name and outer_arr.
    """
    eq_type = nucleation_obs.eq_type
    grids = nucleation_obs.hadronic_grids
    axes = GRID_AXES[eq_type]

    n_B_arr = grids['n_B_H']
    T_arr = grids['T']
    n_T = len(T_arr)

    log_tau_target = np.log10(tau_target)

    # Compute log10(tau) from the raw table, clip inf/nan
    with np.errstate(divide='ignore', invalid='ignore'):
        log_tau_full = np.log10(nucleation_obs.tau)
    log_tau_full = np.clip(log_tau_full, -50, 100)
    log_tau_full = np.where(np.isnan(log_tau_full), 100.0, log_tau_full)

    non_T_axes = [ax for ax in axes if ax not in ('n_B_H', 'T')]

    def _find_nB_root(nB_valid, f_vals):
        """Find n_B where f crosses zero downward; refine with brentq."""
        sign_changes = np.where(np.diff(np.sign(f_vals)))[0]
        downward = [sc for sc in sign_changes
                    if f_vals[sc] > 0 and f_vals[sc + 1] < 0]
        if len(downward) == 0:
            return np.nan, False, np.nan
        idx = downward[0]
        nB_lo = float(nB_valid[idx])
        nB_hi = float(nB_valid[idx + 1])
        kind = 'cubic' if len(nB_valid) >= 4 else 'linear'
        f_interp = interp1d(nB_valid, f_vals, kind=kind)
        try:
            sol = root_scalar(f_interp, bracket=[nB_lo, nB_hi],
                              method='brentq')
            if sol.converged:
                return sol.root, True, abs(f_interp(sol.root))
        except (ValueError, RuntimeError):
            pass
        return np.nan, False, np.nan

    if len(non_T_axes) == 0:
        # ---- 1D: beta_eq ----
        # log_tau_full shape: (n_nB, n_T)
        n_B_nuc = np.full(n_T, np.nan)
        conv = np.zeros(n_T, dtype=bool)
        resid = np.full(n_T, np.nan)

        for j in range(n_T):
            T = float(T_arr[j])
            log_tau_slice = log_tau_full[:, j]
            conv_mask = nucleation_obs.converged[:, j]
            valid = conv_mask & np.isfinite(log_tau_slice)
            if np.sum(valid) < 2:
                if verbose:
                    print(f"  T={T:.2f} MeV -> SKIP (< 2 valid)")
                continue

            f_vals = log_tau_slice[valid] - log_tau_target
            nB_valid = n_B_arr[valid]

            nB_root, ok, res = _find_nB_root(nB_valid, f_vals)
            if ok:
                n_B_nuc[j] = nB_root
                conv[j] = True
                resid[j] = res
            if verbose:
                if ok:
                    print(f"  T={T:.2f} MeV -> n_B_nuc={nB_root:.4f} fm^-3"
                          f"  (res={res:.2e})")
                else:
                    print(f"  T={T:.2f} MeV -> no crossing")

        return SimpleNamespace(
            T_arr=T_arr,
            n_B_nuc=n_B_nuc,
            tau_target=tau_target,
            converged=conv,
            residual=resid,
            eq_type=eq_type,
        )

    elif len(non_T_axes) == 1:
        # ---- 2D: trapped_neutrinos / fixed_yc ----
        ax_name = non_T_axes[0]
        ax_arr = grids[ax_name]
        n_outer = len(ax_arr)
        # log_tau_full shape: (n_nB, n_outer, n_T)
        n_B_nuc = np.full((n_T, n_outer), np.nan)
        conv = np.zeros((n_T, n_outer), dtype=bool)
        resid = np.full((n_T, n_outer), np.nan)

        for j in range(n_T):
            for m in range(n_outer):
                log_tau_slice = log_tau_full[:, m, j]
                conv_mask = nucleation_obs.converged[:, m, j]
                valid = conv_mask & np.isfinite(log_tau_slice)
                if np.sum(valid) < 2:
                    continue

                f_vals = log_tau_slice[valid] - log_tau_target
                nB_valid = n_B_arr[valid]

                nB_root, ok, res = _find_nB_root(nB_valid, f_vals)
                if ok:
                    n_B_nuc[j, m] = nB_root
                    conv[j, m] = True
                    resid[j, m] = res
                if verbose:
                    label = ax_name.replace('_H', '')
                    if ok:
                        print(f"  T={T_arr[j]:.2f} MeV, "
                              f"{label}={ax_arr[m]:.3f}"
                              f" -> n_B_nuc={nB_root:.4f} fm^-3"
                              f"  (res={res:.2e})")
                    else:
                        print(f"  T={T_arr[j]:.2f} MeV, "
                              f"{label}={ax_arr[m]:.3f}"
                              f" -> no crossing")

        return SimpleNamespace(
            T_arr=T_arr,
            outer_name=ax_name,
            outer_arr=ax_arr,
            n_B_nuc=n_B_nuc,
            tau_target=tau_target,
            converged=conv,
            residual=resid,
            eq_type=eq_type,
        )

    else:
        raise ValueError(f"Unsupported eq_type: '{eq_type}'")


def build_nucleation_temperature_interpolator(result, kind='cubic'):
    """Build a smooth T_nuc interpolator from converged points only.

    Non-converged points (NaN gaps) are skipped; the interpolator
    bridges them with a continuous curve through the valid data.

    Parameters
    ----------
    result : NucleationTemperatureResult
        Output of ``compute_nucleation_temperature``.
    kind : str
        Interpolation kind passed to ``interp1d`` ('linear', 'cubic', etc.).

    Returns
    -------
    callable
        ``f(n_B_H)`` for beta_eq, or ``f(n_B_H, Y_L_H)`` for
        trapped_neutrinos / fixed_yc.
        Returns T_nuc (MeV). NaN outside the converged n_B range.
    """
    n_B_arr = result.hadronic_grids['n_B_H']
    eq_type = result.eq_type

    if result.T_nuc.ndim == 1:
        # ---- 1D: beta_eq ----
        mask = result.converged
        if np.sum(mask) < 2:
            return lambda n_B: np.nan

        f = interp1d(n_B_arr[mask], result.T_nuc[mask], kind=kind,
                     bounds_error=False, fill_value=np.nan)
        return lambda n_B: float(f(n_B))

    else:
        # ---- 2D: trapped_neutrinos / fixed_yc ----
        # T_nuc shape: (n_nB, n_outer)
        non_T_axes = [ax for ax in GRID_AXES[eq_type]
                      if ax not in ('n_B_H', 'T')]
        ax_name = non_T_axes[0]
        ax_arr = result.hadronic_grids[ax_name]
        n_outer = len(ax_arr)

        # Build a 1D interpolant per outer-axis slice
        slice_interps = []
        for k in range(n_outer):
            mask = result.converged[:, k]
            if np.sum(mask) < 2:
                slice_interps.append(None)
            else:
                slice_interps.append(
                    interp1d(n_B_arr[mask], result.T_nuc[mask, k],
                             kind=kind, bounds_error=False,
                             fill_value=np.nan))

        # Fill a dense (n_nB, n_outer) array from per-slice interpolants
        T_filled = np.full_like(result.T_nuc, np.nan)
        for k, fi in enumerate(slice_interps):
            if fi is not None:
                T_filled[:, k] = fi(n_B_arr)

        # Build 2D interpolator on the filled grid
        grid_tuple = (n_B_arr, ax_arr)
        rgi = RegularGridInterpolator(
            grid_tuple, T_filled, method='linear',
            bounds_error=False, fill_value=np.nan)

        return lambda n_B, outer: float(rgi((n_B, outer)))


###############################################################################
#                                                                             #
#  2. THERMAL — Grid-level thermal nucleation observables                     #
#                                                                             #
###############################################################################

# =============================================================================
# Output dataclass
# =============================================================================
@dataclass
class ThermalNucleationObservables:
    """Thermal nucleation observables computed over a hadronic grid.

    Attributes
    ----------
    eq_type : str
        Equilibrium type ('beta_eq', 'trapped_neutrinos', 'fixed_yc').
    hadronic_grids : dict
        Input grids (n_B_H, T, Y_L_H, etc.).
    flavor_mode : str
        Flavor mode used ('frozen' or 'saddlepoint').
    electric_charge_mode : str
        Electric charge mode used ('lcn', 'gcn', 'gcn_coulomb', 'coulomb_minimize').
    sigma : float
        Surface tension (MeV/fm^2).
    V : float
        System volume (fm^3).
    R_c : np.ndarray
        Critical radius (fm).
    W_c : np.ndarray
        Critical work / energy barrier (MeV).
    Gamma : np.ndarray
        Nucleation rate (fm^{-3} s^{-1}).
    tau : np.ndarray
        Nucleation time (s).
    converged : np.ndarray
        Boolean array: True where Q* converged and R_c, W_c are finite.
    Qstar_table : object
        Q* table used (for reference).
    """
    eq_type: str
    hadronic_grids: dict
    flavor_mode: str
    electric_charge_mode: str
    sigma: float
    V: float
    R_c: np.ndarray
    W_c: np.ndarray
    Gamma: np.ndarray
    tau: np.ndarray
    converged: np.ndarray
    Qstar_table: object = None



# =============================================================================
# Main function
# =============================================================================
def compute_thermal_nucleation_observables(
    hadronic_table,
    sigma,
    params=None,
    Qstar_table=None,
    V=4.18879e51,  # sphere with radius 100 m in fm^3
    quark_phase='unpaired',
    Delta0=None,
    flavor_mode='saddlepoint',
    electric_charge_mode='gcn',
    include_photons=True,
    include_gluons=True,
    include_thermal_neutrinos=True,
    xi_q=0.7,
    lambda_th=0.0,
    zeta_th=0.0,
    initial_guess=None,
    verbose=False,
    save_table=False,
    output_file=None,
    switching_mode='step', # only used for unpCFL
    Rx=None, # only used for unpCFL
    switching_width=None, # only used for unpCFL
    Qstar_table_unp=None, # only used for unpCFL
    Qstar_table_cfl=None, # only used for unpCFL
    params_unp=None, # only used for unpCFL (overrides params for unpaired)
    params_cfl=None, # only used for unpCFL (overrides params for CFL)
    R_max=20.0, # only used for unpCFL
    n_R=500,
):
    """
    Compute thermal nucleation observables for the hadron-to-quark phase transition.

    Parameters
    ----------
    hadronic_table : EOSTableData
        Hadronic phase conditions.
    sigma : float
        Surface tension (MeV/fm^2).
    params : AlphaBagParams or None
        Quark EOS parameters. Required if Qstar_table is None.
    Qstar_table : QstarTableData or None
        Pre-computed Q* table. If None, computed internally using params.
    V : float
        System volume (fm^3) for nucleation time.
    quark_phase : str
        'unpaired', 'cfl', or 'unpCFL'.
    Delta0 : float or None
        CFL pairing gap (MeV). Required if quark_phase='cfl' or 'unpCFL'.
    flavor_mode : str
        'frozen' or 'saddlepoint'.
    electric_charge_mode : str
        'lcn', 'gcn', 'gcn_coulomb', or 'coulomb_minimize'.
    include_photons, include_gluons, include_thermal_neutrinos : bool
    xi_q : float
        Quark correlation length (fm).
    lambda_th : float
        Thermal conductivity.
    zeta_th : float
        Bulk viscosity.
    initial_guess : array-like or None
    verbose : bool
    save_table : bool
        If True, export the result to a text file.
    output_file : str or None
        Output file path. Required if save_table is True.
    switching_mode : str
        'step' or 'tanh'. Only used when quark_phase='unpCFL'.
    Rx : float or None
        Crossover radius (fm). Required when quark_phase='unpCFL'.
    switching_width : float or None
        Tanh transition width delta (fm). Required when switching_mode='tanh'.
    Qstar_table_unp : QstarTableData or None
        Pre-computed unpaired Q* table. For unpCFL, avoids recomputation.
    Qstar_table_cfl : QstarTableData or None
        Pre-computed CFL Q* table. For unpCFL, avoids recomputation.
    params_unp : AlphaBagParams or None
        Quark EOS parameters for the unpaired phase. For unpCFL only.
        If None, falls back to ``params``.
    params_cfl : AlphaBagParams or None
        Quark EOS parameters for the CFL phase. For unpCFL only.
        If None, falls back to ``params``.
    R_max : float
        Maximum radius for W(R) evaluation in unpCFL (fm). Default 20.
    n_R : int
        Number of R grid points for W(R) evaluation in unpCFL. Default 500.

    Returns
    -------
    ThermalNucleationObservables
    """
    # ---- Validate inputs ----
    valid_flavor = ('frozen', 'saddlepoint')
    valid_charge = ('lcn', 'gcn', 'gcn_coulomb', 'coulomb_minimize')
    if flavor_mode not in valid_flavor:
        raise ValueError(
            f"Invalid flavor_mode: '{flavor_mode}'. Valid: {list(valid_flavor)}")
    if electric_charge_mode not in valid_charge:
        raise ValueError(
            f"Invalid electric_charge_mode: '{electric_charge_mode}'. "
            f"Valid: {list(valid_charge)}")
    if flavor_mode == 'frozen':
        if electric_charge_mode not in ('lcn', 'gcn'):
            raise ValueError(
                f"frozen flavor_mode only supports 'lcn' or 'gcn', "
                f"got '{electric_charge_mode}'")
        if quark_phase in ('cfl', 'unpCFL'):
            raise ValueError(
                f"frozen flavor_mode does not support quark_phase='{quark_phase}'")
    
    # if we use unpCFL mode, we need an ad hoc function
    if quark_phase == 'unpCFL':
        if Rx is None:
            raise ValueError("Rx required for quark_phase='unpCFL'")
        if Delta0 is None:
            raise ValueError("Delta0 required for quark_phase='unpCFL'")
        if switching_mode == 'tanh' and switching_width is None:
            raise ValueError("switching_width required for switching_mode='tanh'")
        return _compute_thermal_nucleation_unpCFL(
            hadronic_table=hadronic_table,
            sigma=sigma,
            params_unp=params_unp if params_unp is not None else params,
            params_cfl=params_cfl if params_cfl is not None else params,
            Delta0=Delta0,
            V=V,
            flavor_mode=flavor_mode,
            electric_charge_mode=electric_charge_mode,
            switching_mode=switching_mode,
            Rx=Rx,
            switching_width=switching_width,
            Qstar_table_unp=Qstar_table_unp,
            Qstar_table_cfl=Qstar_table_cfl,
            R_max=R_max,
            n_R=n_R,
            include_photons=include_photons,
            include_gluons=include_gluons,
            include_thermal_neutrinos=include_thermal_neutrinos,
            xi_q=xi_q,
            lambda_th=lambda_th,
            zeta_th=zeta_th,
            initial_guess=initial_guess,
            verbose=verbose,
            save_table=save_table,
            output_file=output_file,
        )

    # if not unpCFL, continue with the standard function

    # ---- Step 1: Compute Q* table if not already computed and passed ----
    if Qstar_table is None:
        if params is None:
            raise ValueError("params must be provided when Qstar_table is None")

        if verbose:
            print(f"Computing Q* table (flavor={flavor_mode}, "
                  f"charge={electric_charge_mode}, phase={quark_phase})...")


        Qstar_table = compute_Qstar_table(
            hadronic_table,
            flavor_mode=flavor_mode,
            electric_charge_mode=electric_charge_mode,
            params=params,
            quark_phase=quark_phase,
            Delta0=Delta0,
            sigma=sigma,
            include_photons=include_photons,
            include_gluons=include_gluons,
            include_thermal_neutrinos=include_thermal_neutrinos,
            initial_guess=initial_guess,
            verbose=verbose,
        )

    # ---- Step 2: Build phase data ----
    h_d = hadronic_table.data
    q_d = Qstar_table.data
    grids = hadronic_table.grids
    eq_type = hadronic_table.eq_type
    shape = q_d['P_total'].shape
    converged = q_d['converged']

    # Grid axes are 1D in EOSTableData; expand to full shape
    # for element-wise physics calculations.
    if eq_type == 'beta_eq':
        n_B_H, T_H = np.meshgrid(grids['n_B'], grids['T'], indexing='ij')
        mu_nu_H = np.zeros(shape)
        Y_nu_H = np.zeros(shape)
    elif eq_type == 'trapped_neutrinos':
        n_B_H, Y_L_H, T_H = np.meshgrid(
            grids['n_B'], grids['Y_L'], grids['T'], indexing='ij')
        mu_nu_H = h_d['mu_nu']
        Y_nu_H = Y_L_H - h_d['Y_C']
    elif eq_type == 'fixed_yc':
        n_B_H, _, T_H = np.meshgrid(
            grids['n_B'], grids['Y_C'], grids['T'], indexing='ij')
        mu_nu_H = np.zeros(shape)
        Y_nu_H = np.zeros(shape)

    # Hadronic phase — h_d plus grid axes and derived quantities
    H = SimpleNamespace(**{
        **h_d,
        'n_B': n_B_H, 'T': T_H,
        'Y_e': h_d['Y_C'],   # charge neutrality in H
        'mu_nu': mu_nu_H, 'Y_nu': Y_nu_H,
    })

    # Q* phase — q_d plus shared quantities
    Qs = SimpleNamespace(**{
        **q_d,
        'T': T_H,
        'mu_nu': mu_nu_H, 'Y_nu': Y_nu_H,
    })

    # ---- Step 3: Bulk driving force ----
    Delta_f_bulk = driving_force(Qs, H)

    # ---- Step 4: Critical radius and critical work ----
    R_c = q_d['R_c'].copy()

    if electric_charge_mode in ('lcn', 'gcn'):
        delta_n_C = np.zeros(shape)
        W_c = critical_work_noCoulomb(Delta_f_bulk, sigma)
    else:
        delta_n_C = (Qs.Y_C - Qs.Y_e) * Qs.n_B

        W_c = np.where(
            converged & ~np.isnan(R_c),
            work_of_formation(R_c, Delta_f_bulk, sigma, delta_n_C),
            np.nan,
        )

    # ---- Step 5: Nucleation rate and time ----
    Gamma = nucleation_rate(W_c, R_c, sigma, H.T, H, Qs, xi_q, lambda_th, zeta_th)
    tau = nucleation_time(Gamma, V)

    # ---- Step 6: Build result ----
    nuc_converged = converged & np.isfinite(R_c) & np.isfinite(W_c) & (R_c > 0)

    result = ThermalNucleationObservables(
        eq_type=hadronic_table.eq_type,
        hadronic_grids=Qstar_table.hadronic_grids,
        flavor_mode=flavor_mode,
        electric_charge_mode=electric_charge_mode,
        sigma=sigma,
        V=V,
        R_c=R_c,
        W_c=W_c,
        Gamma=Gamma,
        tau=tau,
        converged=nuc_converged,
        Qstar_table=Qstar_table,
    )

    if save_table:
        if output_file is None:
            output_file = f"thermal_nucleation_{hadronic_table.eq_type}.dat"
        export_thermal_nucleation_table(result, output_file)

    return result


# =============================================================================
# Interpolators for thermal nucleation observables
# =============================================================================
def build_thermal_nucleation_interpolators(nucleation_obs, method='linear'):
    """Build interpolation functions from a ThermalNucleationObservables result.

    Returns a dict of callables keyed by observable name.

    Usage (beta_eq)::

        interp = build_thermal_nucleation_interpolators(result)
        tau_val = interp['tau'](n_B_H, T)
        R_c_val = interp['R_c'](n_B_H, T)

    Usage (trapped_neutrinos)::

        interp = build_thermal_nucleation_interpolators(result)
        tau_val = interp['tau'](n_B_H, Y_L_H, T)

    Parameters
    ----------
    nucleation_obs : ThermalNucleationObservables
        Pre-computed thermal nucleation result.
    method : str
        Interpolation method ('linear', 'nearest', etc.).

    Returns
    -------
    dict
        Keys: 'tau', 'R_c', 'W_c', 'Gamma', 'log10_tau', 'log10_Gamma'.
        Values: callables with signature f(n_B_H, T) or f(n_B_H, Y_L_H, T).
    """
    eq_type = nucleation_obs.eq_type
    grids = nucleation_obs.hadronic_grids
    axes = GRID_AXES[eq_type]
    grid_tuple = tuple(grids[ax] for ax in axes)

    result = {}
    for name in ('tau', 'R_c', 'W_c', 'Gamma'):
        arr = getattr(nucleation_obs, name)
        interp = RegularGridInterpolator(
            grid_tuple, arr, method=method,
            bounds_error=False, fill_value=np.nan)
        result[name] = (lambda f: lambda *a: float(f(a)))(interp)

    # log10(tau) interpolator (often more useful than tau directly)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_tau = np.log10(nucleation_obs.tau)
    log_tau = np.clip(log_tau, -50, 100)
    interp_log = RegularGridInterpolator(
        grid_tuple, log_tau, method='cubic',
        bounds_error=False, fill_value=np.nan)
    result['log10_tau'] = (lambda f: lambda *a: float(f(a)))(interp_log)

    # log10(Gamma) interpolator
    with np.errstate(divide='ignore', invalid='ignore'):
        log_Gamma = np.log10(nucleation_obs.Gamma)
    interp_log_Gamma = RegularGridInterpolator(
        grid_tuple, log_Gamma, method=method,
        bounds_error=False, fill_value=np.nan)
    result['log10_Gamma'] = (lambda f: lambda *a: float(f(a)))(interp_log_Gamma)

    return result


# =============================================================================
# Export / load thermal nucleation tables
# =============================================================================
_THERMAL_DATA_KEYS = ['R_c', 'W_c', 'Gamma', 'tau', 'converged']


def export_thermal_nucleation_table(obs, output_file):
    """Export ThermalNucleationObservables to text file.

    Parameters
    ----------
    obs : ThermalNucleationObservables
    output_file : str
    """
    axes = GRID_AXES[obs.eq_type]
    grid_arrays = [obs.hadronic_grids[ax] for ax in axes]
    mesh = np.meshgrid(*grid_arrays, indexing='ij')
    input_cols = [m.ravel(order='F') for m in mesh]

    output_cols = [getattr(obs, k).ravel(order='F') for k in _THERMAL_DATA_KEYS]

    all_names = list(axes) + _THERMAL_DATA_KEYS
    all_cols = np.column_stack(input_cols + output_cols)

    header_lines = ["# Thermal nucleation observables"]
    header_lines.append(f"# eq_type: {obs.eq_type}")
    header_lines.append(f"# flavor_mode: {obs.flavor_mode}")
    header_lines.append(f"# electric_charge_mode: {obs.electric_charge_mode}")
    header_lines.append(f"# sigma: {obs.sigma} MeV/fm^2")
    header_lines.append(f"# V: {obs.V} fm^3")
    col_header = "  ".join(f"{name:>16s}" for name in all_names)
    header_lines.append(f"# {col_header}")
    meta = "\n".join(header_lines) + "\n"

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(meta)
    with open(output_file, 'ab') as f:
        np.savetxt(f, all_cols, fmt='%16.8e')
    print(f"  Saved thermal nucleation table ({all_cols.shape[0]} rows) -> {output_file}")


def load_thermal_nucleation_table(filepath):
    """Load ThermalNucleationObservables from text file.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    ThermalNucleationObservables
    """
    # Parse header metadata
    metadata = {}
    with open(filepath, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                break
            for key in ('eq_type', 'flavor_mode', 'electric_charge_mode',
                        'sigma', 'V'):
                tag = f"# {key}:"
                if line.startswith(tag):
                    val = line[len(tag):].strip()
                    # Strip units
                    for unit in ('MeV/fm^2', 'fm^3'):
                        val = val.replace(unit, '').strip()
                    metadata[key] = val

    eq_type = metadata['eq_type']
    axes = GRID_AXES[eq_type]
    n_axes = len(axes)

    raw = np.loadtxt(filepath)

    # Reconstruct 1D grids from unique values
    hadronic_grids = {}
    for i, ax in enumerate(axes):
        hadronic_grids[ax] = np.unique(raw[:, i])

    shape = tuple(len(hadronic_grids[ax]) for ax in axes)

    # Reshape data columns
    data = {}
    for j, key in enumerate(_THERMAL_DATA_KEYS):
        data[key] = raw[:, n_axes + j].reshape(shape, order='F')

    return ThermalNucleationObservables(
        eq_type=eq_type,
        hadronic_grids=hadronic_grids,
        flavor_mode=metadata['flavor_mode'],
        electric_charge_mode=metadata['electric_charge_mode'],
        sigma=float(metadata['sigma']),
        V=float(metadata['V']),
        R_c=data['R_c'],
        W_c=data['W_c'],
        Gamma=data['Gamma'],
        tau=data['tau'],
        converged=data['converged'].astype(bool),
    )


###############################################################################
#                                                                             #
#  3. QUANTUM — Grid-level quantum (WKB) nucleation observables              #
#                                                                             #
###############################################################################

# =============================================================================
# Output dataclass
# =============================================================================
@dataclass
class QuantumNucleationObservables:
    """Grid-level result of quantum (WKB) nucleation calculation.

    Attributes
    ----------
    eq_type : str
        Equilibrium type.
    hadronic_grids : dict
        Input grids (n_B_H, T, ...).
    sigma : float
        Surface tension (MeV/fm^2).
    N_c : float
        Number of nucleation centers.
    tau_qt : np.ndarray
        Quantum nucleation time (s). NaN where WKB failed.
    A : np.ndarray
        Tunneling action (dimensionless).
    E_0 : np.ndarray
        Ground-state energy (MeV).
    nu_0 : np.ndarray
        Small-oscillation frequency (s^-1).
    converged : np.ndarray
        Boolean: True where WKB succeeded.
    """
    eq_type: str
    hadronic_grids: dict
    sigma: float
    N_c: float
    tau_qt: np.ndarray
    A: np.ndarray
    E_0: np.ndarray
    nu_0: np.ndarray
    converged: np.ndarray


# =============================================================================
# Main function
# =============================================================================
def compute_quantum_nucleation_observables(
    hadronic_table,
    Qstar_table,
    sigma=30.0,
    electric_charge_mode='gcn',
    N_c=1e48,
    rho_H_func=None,
    verbose=False,
):
    """Compute quantum tunneling nucleation time over the full hadronic grid.

    Uses the WKB semiclassical approximation for tunneling through
    the potential barrier W(R) with effective inertia M(R).

    Parameters
    ----------
    hadronic_table : EOSTableData
        Hadronic phase conditions.
    Qstar_table : QstarTableData
        Pre-computed Q* table.
    sigma : float
        Surface tension (MeV/fm^2).
    electric_charge_mode : str
        'lcn', 'gcn', 'gcn_coulomb', or 'coulomb_minimize'.
    N_c : float
        Number of independent nucleation centers (default 10^48).
    rho_H_func : callable or None
        If None, uses rho_H = m_n * n_B_H.
        If provided, called as rho_H_func(n_B_H, T) -> float (MeV/fm^3).
    verbose : bool

    Returns
    -------
    QuantumNucleationObservables
    """
    from eos.general.physics_constants import m_neutron
    from nucleation.general_nucleation.quantum import (
        effective_inertia, quantum_nucleation_time,
    )

    h_d = hadronic_table.data
    q_d = Qstar_table.data
    grids = hadronic_table.grids
    eq_type = hadronic_table.eq_type
    shape = q_d['P_total'].shape
    qstar_converged = q_d['converged']

    # Grid axes are 1D in EOSTableData; expand to full shape
    # for element-wise physics calculations.
    if eq_type == 'beta_eq':
        n_B_H, T_H = np.meshgrid(grids['n_B'], grids['T'], indexing='ij')
        mu_nu_H = np.zeros(shape)
        Y_nu_H = np.zeros(shape)
    elif eq_type == 'trapped_neutrinos':
        n_B_H, Y_L_H, T_H = np.meshgrid(
            grids['n_B'], grids['Y_L'], grids['T'], indexing='ij')
        mu_nu_H = h_d['mu_nu']
        Y_nu_H = Y_L_H - h_d['Y_C']
    elif eq_type == 'fixed_yc':
        n_B_H, _, T_H = np.meshgrid(
            grids['n_B'], grids['Y_C'], grids['T'], indexing='ij')
        mu_nu_H = np.zeros(shape)
        Y_nu_H = np.zeros(shape)

    # Hadronic phase — h_d plus grid axes and derived quantities
    H = SimpleNamespace(**{
        **h_d,
        'n_B': n_B_H, 'T': T_H,
        'Y_e': h_d['Y_C'],   # charge neutrality in H
        'mu_nu': mu_nu_H, 'Y_nu': Y_nu_H,
    })

    # Q* phase — q_d plus shared quantities
    Qs = SimpleNamespace(**{
        **q_d,
        'T': T_H,
        'mu_nu': mu_nu_H, 'Y_nu': Y_nu_H,
    })

    # Bulk driving force
    Delta_f_full = driving_force(Qs, H)

    # Charge density for Coulomb modes
    if electric_charge_mode in ('lcn', 'gcn'):
        delta_n_C_full = np.zeros(shape)
    elif electric_charge_mode == 'gcn_coulomb':
        delta_n_C_full = (Qs.Y_C - Qs.Y_e) * Qs.n_B
    elif electric_charge_mode == 'coulomb_minimize':
        delta_n_C_full = (Qs.Y_C - Qs.Y_e) * Qs.n_B
    else:
        raise ValueError(f"Invalid electric_charge_mode: '{electric_charge_mode}'")

    # Output arrays
    tau_qt_out = np.full(shape, np.nan)
    A_out = np.full(shape, np.nan)
    E_0_out = np.full(shape, np.nan)
    nu_0_out = np.full(shape, np.nan)
    conv_out = np.zeros(shape, dtype=bool)

    # Iterate over all grid points
    total = np.prod(shape)
    done = 0

    it = np.nditer(Delta_f_full, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        it.iternext()

        if not qstar_converged[idx]:
            done += 1
            continue

        Delta_f = float(Delta_f_full[idx])
        if Delta_f <= 0:
            done += 1
            continue

        n_B_H_val = float(H.n_B[idx])
        n_B_Qs = float(Qs.n_B[idx])
        T_val = float(H.T[idx])
        delta_n_C = float(delta_n_C_full[idx])

        # Hadronic mass density
        if rho_H_func is not None:
            rho_H = rho_H_func(n_B_H_val, T_val)
        else:
            rho_H = m_neutron * n_B_H_val

        n_B_ratio = n_B_Qs / n_B_H_val if n_B_H_val > 0 else 1.0

        # Critical radius (for turning-point search)
        R_c = critical_radius_noCoulomb(Delta_f, sigma)
        if not np.isfinite(R_c) or R_c <= 0:
            done += 1
            continue

        # Build W(R) and M(R) closures for this grid point
        def W_func(R, _df=Delta_f, _s=sigma, _dnC=delta_n_C):
            return work_of_formation(R, _df, _s, _dnC)

        def M_func(R, _rho=rho_H, _ratio=n_B_ratio):
            return effective_inertia(R, _rho, _ratio)

        try:
            result = quantum_nucleation_time(W_func, M_func, R_c, N_c=N_c)
            tau_qt_out[idx] = result.tau_qt
            A_out[idx] = result.A
            E_0_out[idx] = result.E_0
            nu_0_out[idx] = result.nu_0
            conv_out[idx] = True
        except Exception:
            pass

        done += 1
        if verbose and done % max(1, total // 20) == 0:
            print(f"  Quantum nucleation: {done}/{total} points processed")

    if verbose:
        n_ok = np.sum(conv_out)
        print(f"  Quantum nucleation complete: {n_ok}/{total} converged")

    return QuantumNucleationObservables(
        eq_type=hadronic_table.eq_type,
        hadronic_grids=Qstar_table.hadronic_grids,
        sigma=sigma,
        N_c=N_c,
        tau_qt=tau_qt_out,
        A=A_out,
        E_0=E_0_out,
        nu_0=nu_0_out,
        converged=conv_out,
    )


###############################################################################
#                                                                             #
#  CFL + unp method                                                           #
#                                                                             #
###############################################################################


# =============================================================================
# unpCFL: R_c and W_c finders (step vs tanh)
# =============================================================================
def _find_Rc_Wc_step(R_c_unp, R_c_cfl, Rx, S_func,
                     Delta_f_unp, Delta_f_cfl,
                     delta_n_C_unp, delta_n_C_cfl, sigma,
                     Delta_f_unp_atRx=None, delta_n_C_unp_atRx=None):
    """Critical radius and barrier height for step switching.

    R_c = R_c_unp             if R_c_unp <= Rx  (and R_c_unp is finite)
    R_c = max(Rx, R_c_cfl)    otherwise

    When R_c = Rx (kink) and ``Delta_f_unp_atRx`` is provided
    (coulomb_minimize mode), uses Delta_f evaluated at R = Rx instead
    of the table value (which was evaluated at a different R_c).

    W_c = work_of_formation(R_c, blended Delta_f, sigma, blended delta_n_C)
    """
    unp_exists = np.isfinite(R_c_unp)
    R_c = np.where(unp_exists & (R_c_unp <= Rx),
                   R_c_unp,
                   np.maximum(Rx, R_c_cfl))
    S_at_Rc = S_func(R_c)

    # At kink (R_c = Rx, S=0): use Delta_f at Rx if available
    if Delta_f_unp_atRx is not None:
        at_kink = np.isclose(R_c, Rx) & (S_at_Rc < 0.5)
        Df_unp = np.where(at_kink, Delta_f_unp_atRx, Delta_f_unp)
        dnC_unp = np.where(at_kink, delta_n_C_unp_atRx, delta_n_C_unp)
    else:
        Df_unp = Delta_f_unp
        dnC_unp = delta_n_C_unp

    Delta_f_at_Rc = (1 - S_at_Rc) * Df_unp + S_at_Rc * Delta_f_cfl
    delta_n_C_at_Rc = (1 - S_at_Rc) * dnC_unp + S_at_Rc * delta_n_C_cfl
    W_c = work_of_formation(R_c, Delta_f_at_Rc, sigma, delta_n_C_at_Rc)
    return R_c, W_c, S_at_Rc


def _find_Rc_Wc_tanh(R_c_unp, R_c_cfl, Rx, S_func,
                     Delta_f_unp, Delta_f_cfl,
                     delta_n_C_unp, delta_n_C_cfl, sigma,
                     R_c_guess=None):
    """Critical radius and barrier height for tanh switching.

    Numerically maximises W(R) per grid point via minimize_scalar(-W),
    using the step-formula R_c as initial guess (unless R_c_guess is given).
    """
    shape = Delta_f_unp.shape
    if R_c_guess is None:
        R_c_guess = np.where(R_c_unp <= Rx, R_c_unp, np.maximum(Rx, R_c_cfl))

    R_c = np.full(shape, np.nan)
    W_c = np.full(shape, np.nan)
    flat_Df_unp = Delta_f_unp.ravel()
    flat_Df_cfl = Delta_f_cfl.ravel()
    flat_dnC_unp = delta_n_C_unp.ravel()
    flat_dnC_cfl = delta_n_C_cfl.ravel()
    flat_guess = R_c_guess.ravel()
    flat_Rc = R_c.ravel()
    flat_Wc = W_c.ravel()

    for k in range(flat_Rc.size):
        guess_k = flat_guess[k]
        if not np.isfinite(guess_k) or guess_k <= 0:
            continue
        df_u, df_c = flat_Df_unp[k], flat_Df_cfl[k]
        dnc_u, dnc_c = flat_dnC_unp[k], flat_dnC_cfl[k]

        def _neg_W(R, _df_u=df_u, _df_c=df_c,
                   _dnc_u=dnc_u, _dnc_c=dnc_c):
            S = S_func(R)
            Df = (1 - S) * _df_u + S * _df_c
            dnC = (1 - S) * _dnc_u + S * _dnc_c
            return -work_of_formation(R, Df, sigma, dnC)

        sol = minimize(lambda R: _neg_W(R[0]), x0=[guess_k])
        if sol.x[0] > 0:
            flat_Rc[k] = sol.x[0]
            flat_Wc[k] = -sol.fun

    return flat_Rc.reshape(shape), flat_Wc.reshape(shape)


# =============================================================================
# unpCFL: Q* at fixed R (coulomb_minimize)
# =============================================================================
def _compute_Qs_at_R(R, hadronic_table, params, sigma,
                     quark_phase='unpaired', Delta0=None,
                     include_photons=True, include_gluons=True,
                     include_thermal_neutrinos=True,
                     initial_guess=None, verbose=False):
    """Compute Q* at fixed radius R over the hadronic grid.

    Used for ``coulomb_minimize`` mode where Q* depends on R through
    ``coulomb_delta_mu_e(R, delta_n_C)``.

    Parameters
    ----------
    R : float or array_like
        Droplet radius (fm).  Can be a scalar (same R for all T)
        or a 1-D array with one element per temperature grid point
        (T-dependent R, e.g. Rx = hc / Delta(T)).
    hadronic_table : EOSTableData
    params : AlphaBagParams
    sigma : float
        Surface tension (MeV/fm^2).
    quark_phase : str
        'unpaired' or 'cfl'.
    Delta0 : float or None
        CFL pairing gap. Required if quark_phase='cfl'.

    Returns
    -------
    dict
        Arrays with same keys as QstarTableData.data.
    """
    from nucleation.energy_barrier.small_droplet.table import (
        _init_data, _build_H_from_table, _BASE_DATA_KEYS,
    )

    eq_type = hadronic_table.eq_type
    grids = hadronic_table.grids
    n_B_arr, T_arr = grids['n_B'], grids['T']
    solver_kw = dict(include_photons=include_photons,
                     include_gluons=include_gluons,
                     include_thermal_neutrinos=include_thermal_neutrinos)

    # Support R as scalar or 1-D array (one value per T)
    R_arr = np.atleast_1d(np.asarray(R, dtype=float))
    R_is_scalar = (R_arr.ndim == 1 and R_arr.size == 1)
    if R_is_scalar:
        R_arr = np.full(len(T_arr), R_arr[0])

    if eq_type == 'beta_eq':
        shape = (len(n_B_arr), len(T_arr))
        outer_arr = None
    elif eq_type == 'trapped_neutrinos':
        outer_arr = grids['Y_L']
        shape = (len(n_B_arr), len(outer_arr), len(T_arr))
    elif eq_type == 'fixed_yc':
        outer_arr = grids['Y_C']
        shape = (len(n_B_arr), len(outer_arr), len(T_arr))
    else:
        raise ValueError(f"Unsupported eq_type: '{eq_type}'")

    data = _init_data(shape)
    n_outer = len(outer_arr) if outer_arr is not None else 1
    guess = initial_guess

    for i_o in range(n_outer):
        for i_T in range(len(T_arr)):
            R_val = float(R_arr[i_T])
            row_conv = 0
            for i_nB in range(len(n_B_arr)):
                if eq_type == 'beta_eq':
                    idx = (i_nB, i_T)
                    gv = {'T': T_arr[i_T]}
                elif eq_type == 'trapped_neutrinos':
                    idx = (i_nB, i_o, i_T)
                    gv = {'T': T_arr[i_T], 'Y_L': outer_arr[i_o]}
                elif eq_type == 'fixed_yc':
                    idx = (i_nB, i_o, i_T)
                    gv = {'T': T_arr[i_T], 'Y_C': outer_arr[i_o]}

                H_pt = _build_H_from_table(hadronic_table, eq_type, idx, gv)

                if quark_phase == 'cfl':
                    result = solve_saddlepoint_minimizecoulomb_cfl_at_R(
                        R_val, H_pt, params, Delta0, sigma, **solver_kw,
                        initial_guess=guess)
                else:
                    result = solve_saddlepoint_minimizecoulomb_at_R(
                        R_val, H_pt, params, sigma, **solver_kw,
                        initial_guess=guess)

                if result is not None:
                    for key in _BASE_DATA_KEYS:
                        data[key][idx] = getattr(result, key)
                    data['converged'][idx] = True
                    data['R_c'][idx] = R_val
                    guess = np.array([result.mu_u, result.mu_d,
                                      result.mu_s, result.mu_e])
                    row_conv += 1

            if verbose:
                print(f"  Q* at R={R_val:.2f}: T={T_arr[i_T]:.1f} "
                      f"-> {row_conv}/{len(n_B_arr)} converged")

    return data


# =============================================================================
# unpCFL helper
# =============================================================================
def _compute_thermal_nucleation_unpCFL(
    hadronic_table,
    sigma,
    params_unp,
    params_cfl,
    Delta0,
    V,
    flavor_mode,
    electric_charge_mode,
    switching_mode,
    Rx,
    switching_width,
    Qstar_table_unp,
    Qstar_table_cfl,
    R_max,
    n_R,
    include_photons,
    include_gluons,
    include_thermal_neutrinos,
    xi_q,
    lambda_th,
    zeta_th,
    initial_guess,
    verbose,
    save_table,
    output_file,
):
    """Compute thermal nucleation observables for the unpCFL quark phase.

    Blends unpaired and CFL quark matter using a radius-dependent switching
    function S(R): PQ(R) = (1-S(R))*PQ_unp + S(R)*PQ_cfl.

    The two phases can have independent bag model parameters
    (``params_unp`` for unpaired, ``params_cfl`` for CFL).

    Called internally by ``compute_thermal_nucleation_observables`` when
    ``quark_phase='unpCFL'``.
    """
    common_table_kw = dict(
        include_photons=include_photons,
        include_gluons=include_gluons,
        include_thermal_neutrinos=include_thermal_neutrinos,
        initial_guess=initial_guess,
        verbose=verbose,
    )

    # ---- Step 1: Compute / accept both Q* tables ----
    if Qstar_table_unp is None:
        if params_unp is None:
            raise ValueError(
                "params_unp (or params) must be provided when Qstar_table_unp is None")
        if verbose:
            print("Computing unpaired Q* table...")
        Qstar_table_unp = compute_Qstar_table(
            hadronic_table,
            flavor_mode=flavor_mode,
            electric_charge_mode=electric_charge_mode,
            params=params_unp,
            quark_phase='unpaired',
            sigma=sigma,
            **common_table_kw,
        )

    if Qstar_table_cfl is None:
        if params_cfl is None:
            raise ValueError(
                "params_cfl (or params) must be provided when Qstar_table_cfl is None")
        if verbose:
            print("Computing CFL Q* table...")
        Qstar_table_cfl = compute_Qstar_table(
            hadronic_table,
            flavor_mode=flavor_mode,
            electric_charge_mode=electric_charge_mode,
            params=params_cfl,
            quark_phase='cfl',
            Delta0=Delta0,
            sigma=sigma,
            **common_table_kw,
        )

    # ---- Step 1b: Q* at R=Rx for coulomb_minimize ----
    if electric_charge_mode == 'coulomb_minimize':
        if verbose:
            print("Computing unpaired Q* at R=Rx...")
        q_d_unp_atRx = _compute_Qs_at_R(
            Rx, hadronic_table, params_unp, sigma,
            quark_phase='unpaired', **common_table_kw)
    else:
        q_d_unp_atRx = None

    # ---- Step 2: Build phase data ----
    h_d = hadronic_table.data
    q_d_unp = Qstar_table_unp.data
    q_d_cfl = Qstar_table_cfl.data
    grids = hadronic_table.grids
    eq_type = hadronic_table.eq_type
    shape = q_d_unp['P_total'].shape

    if eq_type == 'beta_eq':
        n_B_H, T_H = np.meshgrid(grids['n_B'], grids['T'], indexing='ij')
        mu_nu_H = np.zeros(shape)
        Y_nu_H = np.zeros(shape)
    elif eq_type == 'trapped_neutrinos':
        n_B_H, Y_L_H, T_H = np.meshgrid(
            grids['n_B'], grids['Y_L'], grids['T'], indexing='ij')
        mu_nu_H = h_d['mu_nu']
        Y_nu_H = Y_L_H - h_d['Y_C']
    elif eq_type == 'fixed_yc':
        n_B_H, _, T_H = np.meshgrid(
            grids['n_B'], grids['Y_C'], grids['T'], indexing='ij')
        mu_nu_H = np.zeros(shape)
        Y_nu_H = np.zeros(shape)

    H = SimpleNamespace(**{
        **h_d,
        'n_B': n_B_H, 'T': T_H,
        'Y_e': h_d['Y_C'],
        'mu_nu': mu_nu_H, 'Y_nu': Y_nu_H,
    })

    Qs_unp = SimpleNamespace(**{
        **q_d_unp,
        'T': T_H,
        'mu_nu': mu_nu_H, 'Y_nu': Y_nu_H,
    })
    Qs_cfl = SimpleNamespace(**{
        **q_d_cfl,
        'T': T_H,
        'mu_nu': mu_nu_H, 'Y_nu': Y_nu_H,
    })

    # ---- Step 3: Driving forces for both phases ----
    Delta_f_unp = driving_force(Qs_unp, H)
    Delta_f_cfl = driving_force(Qs_cfl, H)

    if electric_charge_mode in ('lcn', 'gcn'):
        delta_n_C_unp = np.zeros(shape)
        delta_n_C_cfl = np.zeros(shape)
    else:
        delta_n_C_unp = (Qs_unp.Y_C - Qs_unp.Y_e) * Qs_unp.n_B
        delta_n_C_cfl = (Qs_cfl.Y_C - Qs_cfl.Y_e) * Qs_cfl.n_B

    # ---- Step 3b: Driving forces at R=Rx (coulomb_minimize) ----
    if q_d_unp_atRx is not None:
        Qs_unp_atRx = SimpleNamespace(**{
            **q_d_unp_atRx,
            'T': T_H, 'mu_nu': mu_nu_H, 'Y_nu': Y_nu_H,
        })
        Delta_f_unp_atRx = driving_force(Qs_unp_atRx, H)
        delta_n_C_unp_atRx = (
            (Qs_unp_atRx.Y_C - Qs_unp_atRx.Y_e) * Qs_unp_atRx.n_B)
    else:
        Qs_unp_atRx = None
        Delta_f_unp_atRx = None
        delta_n_C_unp_atRx = None

    # ---- Step 4: Critical radius and barrier height ----
    S_func = get_switching_function(switching_mode, Rx, switching_width)
    R_c_unp = q_d_unp['R_c']
    R_c_cfl = q_d_cfl['R_c']
    common_kw = dict(
        R_c_unp=R_c_unp, R_c_cfl=R_c_cfl, Rx=Rx, S_func=S_func,
        Delta_f_unp=Delta_f_unp, Delta_f_cfl=Delta_f_cfl,
        delta_n_C_unp=delta_n_C_unp, delta_n_C_cfl=delta_n_C_cfl,
        sigma=sigma,
    )

    if switching_mode == 'step':
        extra_kw = {}

        if Delta_f_unp_atRx is not None:
            extra_kw['Delta_f_unp_atRx'] = Delta_f_unp_atRx
            extra_kw['delta_n_C_unp_atRx'] = delta_n_C_unp_atRx

        R_c, W_c, S_at_Rc = _find_Rc_Wc_step(**common_kw, **extra_kw)

    elif switching_mode == 'tanh':
        if Delta_f_unp_atRx is not None:
            raise ValueError("Coulomb_minimize mode not yet introduced for switching_mode == 'tanh'.")
        R_c, W_c = _find_Rc_Wc_tanh(**common_kw)
        S_at_Rc = None


    P_unp = Qs_unp.P_total
    e_unp = Qs_unp.e_total
    if Qs_unp_atRx is not None:
        at_kink = np.isclose(R_c, Rx) & (S_at_Rc < 0.5)
        P_unp = np.where(at_kink, Qs_unp_atRx.P_total, P_unp)
        e_unp = np.where(at_kink, Qs_unp_atRx.e_total, e_unp)

    Qs_mixed = SimpleNamespace(
        P_total=(1 - S_at_Rc) * P_unp + S_at_Rc * Qs_cfl.P_total,
        e_total=(1 - S_at_Rc) * e_unp + S_at_Rc * Qs_cfl.e_total,
    )

    Gamma = nucleation_rate(W_c, R_c, sigma, H.T, H, Qs_mixed,
                            xi_q, lambda_th, zeta_th)
    tau = nucleation_time(Gamma, V)

    # ---- Step 7: Build result ----
    nuc_converged = (converged & np.isfinite(R_c) & np.isfinite(W_c) & (R_c > 0))

    result = ThermalNucleationObservables(
        eq_type=hadronic_table.eq_type,
        hadronic_grids=Qstar_table_unp.hadronic_grids,
        flavor_mode=flavor_mode,
        electric_charge_mode=electric_charge_mode,
        sigma=sigma,
        V=V,
        R_c=R_c,
        W_c=W_c,
        Gamma=Gamma,
        tau=tau,
        converged=nuc_converged,
    )

    if save_table:
        if output_file is None:
            output_file = (f"thermal_nucleation_unpCFL_"
                           f"{hadronic_table.eq_type}.dat")
        export_thermal_nucleation_table(result, output_file)

    return result