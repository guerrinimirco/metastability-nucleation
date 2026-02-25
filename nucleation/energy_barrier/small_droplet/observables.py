"""
Nucleation Observables
======================

High-level functions that compute nucleation observables (R_c, W_c, Gamma, tau) for the hadron-to-quark phase transition.

Two independent parameters control the Q* solver:

    flavor_mode:
        frozen       : Flavor fractions frozen from the hadronic phase
        saddlepoint  : Flavor fractions minimized (saddlepoint approximation)

    electric_charge_mode:
        lcn              : Local charge neutrality, no Coulomb
        gcn              : Global charge neutrality, no Coulomb
        gcn_coulomb      : GCN solver (no Coulomb in equations),
                           Coulomb energy added to W(R) after, R_c from findroot
        coulomb_minimize : Full Coulomb minimization via solve_coulomb
                           (R is part of the unknowns)

Constraints:
    - frozen only supports 'lcn' and 'gcn' (no Coulomb modes, no CFL)
    - saddlepoint supports all four electric charge modes

Usage
-----
>>> from nucleation.energy_barrier.small_droplet import compute_nucleation_observables
>>> result = compute_nucleation_observables(
...     hadronic_table,
...     params=my_params,
...     sigma=30.0, V=1e39,
...     flavor_mode='saddlepoint',
...     electric_charge_mode='gcn',
... )
>>> print(result.R_c, result.tau)
"""

import numpy as np
from types import SimpleNamespace
from dataclasses import dataclass
from scipy.optimize import root_scalar

from nucleation.energy_barrier.small_droplet.barrier import (
    driving_force,
    critical_radius_noCoulomb, critical_work_noCoulomb,
    critical_radius_coulomb,
    work_of_formation, bulk_W, surface_W, coulomb_W,
)
from nucleation.general_nucleation.thermal import nucleation_rate, nucleation_time
from nucleation.energy_barrier.small_droplet.table import (
    compute_Qstar_table, build_Qstar_interpolators,
)

# =============================================================================
# Output dataclasses
# =============================================================================
@dataclass
class NucleationObservables:
    """Nucleation observables computed over a hadronic grid.

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
    R_c: float
    W_c: float


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
# Main function
# =============================================================================
def compute_nucleation_observables(
    hadronic_table,
    params=None,
    Qstar_table=None,
    sigma=30.0,
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
):
    """
    Compute nucleation observables for the hadron-to-quark phase transition.

    Parameters
    ----------
    hadronic_table : EOSTableData
        Hadronic phase conditions.
    params : AlphaBagParams or None
        Quark EOS parameters. Required if Qstar_table is None.
    Qstar_table : QstarTableData or None
        Pre-computed Q* table. If None, computed internally using params.
    sigma : float
        Surface tension (MeV/fm^2).
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

    Returns
    -------
    NucleationObservables
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
    # R_c is always stored in the Q* table (computed by compute_Qstar_table)
    R_c = q_d['R_c'].copy()

    if electric_charge_mode in ('lcn', 'gcn'):
        delta_n_C = np.zeros(shape)
    else:
        delta_n_C = (Qs.Y_C - Qs.Y_e) * Qs.n_B

    W_c = np.where(
        converged & ~np.isnan(R_c),
        work_of_formation(R_c, Delta_f_bulk, sigma, delta_n_C),
        np.nan,
    )

    # ---- Step 5: Nucleation rate and time ----
    Gamma = nucleation_rate(W_c, R_c, sigma, H.T, H, Qs,
                            xi_q, lambda_th, zeta_th)
    tau = nucleation_time(Gamma, V)

    # ---- Step 6: Build result ----
    return NucleationObservables(
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


# =============================================================================
# Energy barrier W(R) at a single physical point
# =============================================================================
def compute_energy_barrier(
    H_interp,
    Q_interp,
    n_B_H,
    T,
    Y_L_H=None,
    Y_C_H=None,
    sigma=30.0,
    electric_charge_mode='gcn',
    R_values=None,
    params=None,
    quark_phase='unpaired',
    Delta0=None,
    include_photons=True,
    include_gluons=True,
    include_thermal_neutrinos=True,
):
    """
    Compute W(R) at a single physical point (n_B_H, T) or (n_B_H, Y_L_H, T).

    Uses pre-built interpolators for both hadronic and Q* phases.
    For 'coulomb_minimize', solves the Q* equations at each R with
    Coulomb corrections, giving the true R-dependent W(R).

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
    Y_L_H : float or None
        Lepton fraction. Required for 'trapped_neutrinos'.
    Y_C_H : float or None
        Charge fraction. Required for 'fixed_yc'.
    sigma : float
        Surface tension (MeV/fm^2).
    electric_charge_mode : str
        'lcn', 'gcn', 'gcn_coulomb', or 'coulomb_minimize'.
    R_values : array-like or None
        Radius array (fm). If None, auto-generated from 0 to 5*R_c.
    params : AlphaBagParams or None
        Quark EOS parameters. Required for 'coulomb_minimize'.
    quark_phase : str
        'unpaired' or 'cfl'. Only used for 'coulomb_minimize'.
    Delta0 : float or None
        CFL pairing gap (MeV). Required if quark_phase='cfl'.
    include_photons, include_gluons, include_thermal_neutrinos : bool
        Only used for 'coulomb_minimize'.

    Returns
    -------
    EnergyBarrierResult
    """
    # ---- Interpolate H and Q* at the physical point ----
    H, Qs = _interpolate_at_point(
        H_interp, Q_interp, n_B_H, T,
        Y_L_H=Y_L_H, Y_C_H=Y_C_H)

    # Driving force at this point (scalar)
    Delta_f = float(driving_force(Qs, H))

    # ---- coulomb_minimize: solve Q* at each R ----
    if electric_charge_mode == 'coulomb_minimize':
        if params is None:
            raise ValueError(
                "params required for coulomb_minimize energy barrier")

        # Early exit for unfavorable driving force
        if Delta_f >= 0:
            R_arr = np.linspace(0, 20.0, 500) if R_values is None else np.asarray(R_values, dtype=float)
            return EnergyBarrierResult(
                R=R_arr,
                W=np.full_like(R_arr, np.nan),
                W_bulk=np.full_like(R_arr, np.nan),
                W_surface=surface_W(R_arr, sigma),
                W_coulomb=np.full_like(R_arr, np.nan),
                Delta_f=Delta_f,
                delta_n_C=0.0,
                sigma=sigma,
                R_c=np.nan,
                W_c=np.nan,
            )

        from nucleation.energy_barrier.small_droplet.solvers import (
            solve_saddlepoint_minimizecoulomb_at_R,
            solve_saddlepoint_minimizecoulomb_cfl_at_R,
        )

        R_c_val = float(Qs.R_c)

        if R_values is None:
            R_max = max(5.0 * R_c_val, 20.0) if np.isfinite(R_c_val) else 20.0
            R_values = np.linspace(0, R_max, 500)
        R_values = np.asarray(R_values, dtype=float)

        W_bulk = np.full_like(R_values, np.nan)
        W_surface = surface_W(R_values, sigma)
        W_coul = np.full_like(R_values, np.nan)

        guess = None
        dnC_at_Rc = 0.0
        for i, R in enumerate(R_values):
            if R <= 0:
                W_bulk[i] = 0.0
                W_coul[i] = 0.0
                continue

            if quark_phase == 'cfl':
                result = solve_saddlepoint_minimizecoulomb_cfl_at_R(
                    R, H, params, Delta0, sigma,
                    include_photons, include_gluons,
                    include_thermal_neutrinos, initial_guess=guess)
            else:
                result = solve_saddlepoint_minimizecoulomb_at_R(
                    R, H, params, sigma,
                    include_photons, include_gluons,
                    include_thermal_neutrinos, initial_guess=guess)

            if result is not None:
                dnC_R = (result.Y_C - result.Y_e) * result.n_B
                W_bulk[i] = bulk_W(R, result.P_total - H.P_total)
                W_coul[i] = float(coulomb_W(R, dnC_R))
                guess = np.array([
                    result.mu_u, result.mu_d, result.mu_s, result.mu_e])
                if abs(R - R_c_val) < (R_values[1] - R_values[0]):
                    dnC_at_Rc = dnC_R

        W_total = W_bulk + W_surface + W_coul
        W_c_val = float(np.nanmax(W_total)) if np.any(np.isfinite(W_total)) else np.nan

        return EnergyBarrierResult(
            R=R_values,
            W=W_total,
            W_bulk=W_bulk,
            W_surface=W_surface,
            W_coulomb=W_coul,
            Delta_f=Delta_f,
            delta_n_C=dnC_at_Rc,
            sigma=sigma,
            R_c=R_c_val,
            W_c=W_c_val,
        )

    # ---- Analytical modes: lcn, gcn, gcn_coulomb ----
    if electric_charge_mode in ('lcn', 'gcn'):
        dnC = 0.0
        R_c_val = float(critical_radius_noCoulomb(Delta_f, sigma))
        W_c_val = float(critical_work_noCoulomb(Delta_f, sigma))
    elif electric_charge_mode == 'gcn_coulomb':
        dnC = float((Qs.Y_C - Qs.Y_e) * Qs.n_B)
        R_c_val = float(critical_radius_coulomb(Delta_f, sigma, dnC))
        W_c_val = float(work_of_formation(R_c_val, Delta_f, sigma, dnC))
    else:
        raise ValueError(f"Invalid electric_charge_mode: '{electric_charge_mode}'")

    # Build R array
    if R_values is None:
        R_max = max(5.0 * R_c_val, 20.0) if np.isfinite(R_c_val) else 20.0
        R_values = np.linspace(0, R_max, 500)
    R_values = np.asarray(R_values, dtype=float)

    # Compute W(R) and its components
    W_bulk = bulk_W(R_values, Delta_f)
    W_surface = surface_W(R_values, sigma)
    W_coul = coulomb_W(R_values, dnC)
    W_total = W_bulk + W_surface + W_coul

    return EnergyBarrierResult(
        R=R_values,
        W=W_total,
        W_bulk=W_bulk,
        W_surface=W_surface,
        W_coulomb=W_coul,
        Delta_f=Delta_f,
        delta_n_C=dnC,
        sigma=sigma,
        R_c=R_c_val,
        W_c=W_c_val,
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
    nucleation_obs,
    tau_target=1.0,
    T_guess=None,
    verbose=False,
):
    """
    Find the temperature T at which tau = tau_target over the full grid.

    Uses ``build_nucleation_interpolators`` to build a ``log10_tau``
    interpolator, then root-finds along T for each (n_B_H, ...) point.

    Parameters
    ----------
    nucleation_obs : NucleationObservables
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

    grids = nucleation_obs.hadronic_grids
    eq_type = nucleation_obs.eq_type
    n_B_arr = grids['n_B_H']
    T_arr = grids['T']
    n_nB = len(n_B_arr)

    log_tau_target = np.log10(tau_target)

    # Build the log10_tau interpolator once
    interp = build_nucleation_interpolators(nucleation_obs)
    log10_tau_fn = interp['log10_tau']

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
        ax_arr = grids[ax_name]
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


# =============================================================================
# Interpolators for nucleation observables
# =============================================================================
def build_nucleation_interpolators(nucleation_obs, method='linear'):
    """Build interpolation functions from a NucleationObservables result.

    Returns a dict of callables keyed by observable name.

    Usage (beta_eq)::

        interp = build_nucleation_interpolators(result)
        tau_val = interp['tau'](n_B_H, T)
        R_c_val = interp['R_c'](n_B_H, T)

    Usage (trapped_neutrinos)::

        interp = build_nucleation_interpolators(result)
        tau_val = interp['tau'](n_B_H, Y_L_H, T)

    Parameters
    ----------
    nucleation_obs : NucleationObservables
        Pre-computed nucleation result.
    method : str
        Interpolation method ('linear', 'nearest', etc.).

    Returns
    -------
    dict
        Keys: 'tau', 'R_c', 'W_c', 'Gamma', 'log10_tau', 'log10_Gamma'.
        Values: callables with signature f(n_B_H, T) or f(n_B_H, Y_L_H, T).
    """
    from scipy.interpolate import RegularGridInterpolator
    from nucleation.energy_barrier.small_droplet.table import GRID_AXES

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
# Quantum nucleation (WKB tunneling) over the full grid
# =============================================================================
@dataclass
class QuantumNucleationGrid:
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


def compute_quantum_nucleation(
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
    QuantumNucleationGrid
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

        n_B_H = float(H.n_B[idx])
        n_B_Qs = float(Qs.n_B[idx])
        T_val = float(H.T[idx])
        delta_n_C = float(delta_n_C_full[idx])

        # Hadronic mass density
        if rho_H_func is not None:
            rho_H = rho_H_func(n_B_H, T_val)
        else:
            rho_H = m_neutron * n_B_H

        n_B_ratio = n_B_Qs / n_B_H if n_B_H > 0 else 1.0

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

    return QuantumNucleationGrid(
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
