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
from scipy.optimize import root_scalar
from scipy.interpolate import RegularGridInterpolator

from nucleation.energy_barrier.small_droplet.barrier import (
    driving_force,
    bulk_W, surface_W, coulomb_W,
    work_of_formation,
    critical_work_noCoulomb,
    critical_radius_noCoulomb,
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
    eq_type : str
        Equilibrium type.
    """
    hadronic_grids: dict
    T_nuc: np.ndarray
    tau_target: float
    converged: np.ndarray
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
        'unpaired' or 'cfl'. Used only in solver paths.
    Delta0 : float or None
        CFL pairing gap (MeV). Required if quark_phase='cfl'.
    Y_L_H : float or None
        Lepton fraction. Required for 'trapped_neutrinos'.
    Y_C_H : float or None
        Charge fraction. Required for 'fixed_yc'.
    R_values : array-like or None
        Radius array (fm). If None, uses [0, 20] fm with 500 points.
    include_photons, include_gluons, include_thermal_neutrinos : bool

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

    if electric_charge_mode == 'coulomb_minimize':
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
def _find_T_root(f, T_arr, prev_T):
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
    prev_T : float or None
        Warm-start guess from the previous density point.

    Returns
    -------
    T_root : float
        Root temperature, or NaN if not found.
    converged : bool
    """
    # Evaluate at grid points for bracketing
    vals = np.array([f(T) for T in T_arr])
    valid = np.isfinite(vals) & (T_arr > 0)
    if np.sum(valid) < 2:
        return np.nan, False

    T_valid = T_arr[valid]
    vals_valid = vals[valid]

    sign_changes = np.where(np.diff(np.sign(vals_valid)))[0]
    if len(sign_changes) == 0:
        return np.nan, False

    # Fast path: secant method with warm-start from previous solution
    if prev_T is not None:
        try:
            sol = root_scalar(f, x0=prev_T, method='secant',
                              x1=prev_T * 1.01)
            if sol.converged and T_valid[0] <= sol.root <= T_valid[-1]:
                # Verify it is a downward crossing (tau dropping below target)
                eps = (T_valid[-1] - T_valid[0]) * 1e-4
                if f(sol.root - eps) > 0 and f(sol.root + eps) < 0:
                    return sol.root, True
        except (ValueError, RuntimeError):
            pass

    # Fallback: scan sign changes for downward crossings
    # (f goes from + to -, i.e. tau drops below tau_target)
    downward = [sc for sc in sign_changes
                if vals_valid[sc] > 0 and vals_valid[sc + 1] < 0]
    if len(downward) == 0:
        return np.nan, False

    # Pick the first downward crossing (lowest T)
    j = downward[0]
    T_lo = float(T_valid[j])
    T_hi = float(T_valid[j + 1])

    try:
        sol = root_scalar(f, bracket=[T_lo, T_hi], method='brentq')
        if sol.converged:
            return sol.root, True
    except (ValueError, RuntimeError):
        pass

    return np.nan, False


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
        prev_T = T_guess

        for i in range(n_nB):
            nB = float(n_B_arr[i])

            def f(T, _nB=nB):
                return log10_tau_fn(_nB, T) - log_tau_target

            T_root, ok = _find_T_root(f, T_arr, prev_T)
            if ok:
                T_nuc[i] = T_root
                conv[i] = True
                prev_T = T_root
                if verbose:
                    print(f"  n_B={n_B_arr[i]:.4f} -> T_nuc={T_root:.2f} MeV")

        out_grids = {'n_B_H': n_B_arr}

    elif len(non_T_axes) == 1:
        # ---- 2D: trapped_neutrinos or fixed_yc ----
        ax_name = non_T_axes[0]          # 'Y_L_H' or 'Y_C_H'
        ax_arr = hadronic_grids[ax_name]
        n_outer = len(ax_arr)

        T_nuc = np.full((n_nB, n_outer), np.nan)
        conv = np.zeros((n_nB, n_outer), dtype=bool)

        for k in range(n_outer):
            prev_T = T_guess
            outer_val = float(ax_arr[k])

            for i in range(n_nB):
                nB = float(n_B_arr[i])

                def f(T, _nB=nB, _ov=outer_val):
                    return log10_tau_fn(_nB, _ov, T) - log_tau_target

                T_root, ok = _find_T_root(f, T_arr, prev_T)
                if ok:
                    T_nuc[i, k] = T_root
                    conv[i, k] = True
                    prev_T = T_root
                    if verbose:
                        label = ax_name.replace('_H', '')
                        print(f"  n_B={n_B_arr[i]:.4f}, "
                              f"{label}={outer_val:.3f}"
                              f" -> T_nuc={T_root:.2f} MeV")

        out_grids = {'n_B_H': n_B_arr, ax_name: ax_arr}

    else:
        raise ValueError(f"Unsupported eq_type: '{eq_type}'")

    return NucleationTemperatureResult(
        hadronic_grids=out_grids,
        T_nuc=T_nuc,
        tau_target=tau_target,
        converged=conv,
        eq_type=eq_type,
    )


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
        'unpaired' or 'cfl'.
    Delta0 : float or None
        CFL pairing gap (MeV). Required if quark_phase='cfl'.
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
        if quark_phase == 'cfl':
            raise ValueError(
                "frozen flavor_mode does not support quark_phase='cfl'")

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
    interp_log = RegularGridInterpolator(
        grid_tuple, log_tau, method=method,
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
_THERMAL_DATA_KEYS = ['R_c', 'W_c', 'Gamma', 'tau']


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
