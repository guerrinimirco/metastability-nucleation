"""
General Nucleation Observables
==============================

Energy barrier W(R) at a single physical point and nucleation
temperature computation (applicable to both thermal and quantum
nucleation).

For thermal nucleation (R_c, W_c, Gamma, tau) see ``thermal.py``.
For quantum nucleation (WKB tunneling) see ``quantum.py``.

Usage
-----
Energy barrier at a single point::

    >>> from nucleation.energy_barrier.small_droplet import compute_energy_barrier
    >>> barrier = compute_energy_barrier(
    ...     H_interp, Q_interp,
    ...     n_B_H=0.19, T=10.0, sigma=30.0,
    ...     electric_charge_mode='gcn',
    ...     Y_L_H=0.4,
    ... )
    >>> print(barrier.W, barrier.Delta_f)

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

import numpy as np
from types import SimpleNamespace
from dataclasses import dataclass
from scipy.optimize import root_scalar

from nucleation.energy_barrier.small_droplet.barrier import (
    driving_force,
    work_of_formation, bulk_W, surface_W, coulomb_W,
)


# =============================================================================
# Output dataclasses
# =============================================================================
@dataclass
class EnergyBarrierResult:
    """Energy barrier W(R) at a single physical point."""
    R: np.ndarray
    W: np.ndarray
    W_bulk: np.ndarray
    W_surface: np.ndarray
    W_coulomb: np.ndarray
    Delta_f: float
    delta_n_C: float
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
def compute_energy_barrier(
    H_interp,
    Q_interp,
    n_B_H,
    T,
    sigma,
    electric_charge_mode='gcn',
    Y_L_H=None,
    Y_C_H=None,
    R_values=None,
):
    """
    Compute W(R) at a single physical point using pre-built interpolators.

    Parameters
    ----------
    H_interp : dict
        Pre-built hadronic interpolators (from ``build_interpolators``).
    Q_interp : dict
        Pre-built Q* interpolators (from ``build_Qstar_interpolators``).
    n_B_H : float
        Baryon number density (fm^-3).
    T : float
        Temperature (MeV).
    sigma : float
        Surface tension (MeV/fm^2).
    electric_charge_mode : str
        'lcn', 'gcn', 'gcn_coulomb', or 'coulomb_minimize'.
    Y_L_H : float or None
        Lepton fraction. Required for 'trapped_neutrinos'.
    Y_C_H : float or None
        Charge fraction. Required for 'fixed_yc'.
    R_values : array-like or None
        Radius array (fm). If None, uses [0, 20] fm with 500 points.

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
    mu_nu_val = float(H_interp['mu_nu'](*pt)) if 'mu_nu' in H_interp else 0.0
    H = SimpleNamespace(
        P_total=float(H_interp['P'](*pt)),
        mu_B=float(H_interp['mu_B'](*pt)),
        mu_C=float(H_interp['mu_C'](*pt)),
        mu_S=float(H_interp['mu_S'](*pt)),
        mu_e=float(H_interp['mu_e'](*pt)),
        mu_nu=mu_nu_val,
    )

    # ---- Evaluate Qs at the point ----
    q_keys = ['n_B', 'P_total', 'mu_B', 'mu_C', 'mu_S', 'mu_e',
              'Y_C', 'Y_S', 'Y_e']
    q_vals = {k: float(Q_interp[k](*pt)) for k in q_keys}
    q_vals['mu_nu'] = mu_nu_val
    if eq_type == 'trapped_neutrinos':
        q_vals['Y_nu'] = Y_L_H - q_vals['Y_C']
    else:
        q_vals['Y_nu'] = 0.0
    Qs = SimpleNamespace(**q_vals)

    # ---- Driving force and charge density ----
    Delta_f = float(driving_force(Qs, H))

    if electric_charge_mode in ('lcn', 'gcn'):
        delta_n_C = 0.0
    else:
        delta_n_C = float((Qs.Y_C - Qs.Y_e) * Qs.n_B)

    # ---- R array ----
    if R_values is None:
        R_values = np.linspace(0, 20.0, 500)
    R_values = np.asarray(R_values, dtype=float)

    # ---- W(R) and components ----
    return EnergyBarrierResult(
        R=R_values,
        W=work_of_formation(R_values, Delta_f, sigma, delta_n_C),
        W_bulk=bulk_W(R_values, Delta_f),
        W_surface=surface_W(R_values, sigma),
        W_coulomb=coulomb_W(R_values, delta_n_C),
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
    from nucleation.energy_barrier.small_droplet.table import GRID_AXES

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
