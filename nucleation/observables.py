"""
Nucleation Observables
======================

High-level functions that compute nucleation observables (R_c, W_c, Gamma, tau)
for the hadron-to-quark phase transition.

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
>>> from nucleation import compute_nucleation_observables
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
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

from nucleation.physics import (
    driving_force,
    critical_radius, critical_work,
    critical_radius_coulomb,
    work_of_formation, bulk_W, surface_W, coulomb_W,
    nucleation_rate, nucleation_time,
)
from nucleation.table import compute_Qstar_table


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
    """Energy barrier W(R) at a single hadronic grid point."""
    R: np.ndarray
    W: np.ndarray
    W_bulk: np.ndarray
    W_surface: np.ndarray
    W_coulomb: np.ndarray
    Delta_F: float
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
# Helper: build H and Qs SimpleNamespace from table data
# =============================================================================
def _build_phase_namespaces(hadronic_table, q_d, shape):
    """Build array-based H and Qs SimpleNamespace objects from table data.

    These namespaces provide the interface expected by physics
    functions (driving_force, dynamical_prefactor, etc.).
    """
    h_d = hadronic_table.data
    grids = hadronic_table.grids
    eq_type = hadronic_table.eq_type

    n_B_arr = grids['n_B']
    T_arr = grids['T']

    if eq_type == 'beta_eq':
        n_B_mesh, T_mesh = np.meshgrid(n_B_arr, T_arr, indexing='ij')
        mu_nu_H = np.zeros(shape)
        Y_nu_H = np.zeros(shape)
    elif eq_type == 'trapped_neutrinos':
        Y_L_arr = grids['Y_L']
        meshes = np.meshgrid(n_B_arr, Y_L_arr, T_arr, indexing='ij')
        n_B_mesh, Y_L_mesh, T_mesh = meshes
        mu_nu_H = h_d['mu_nu']
        Y_C_H = h_d['Y_C']
        Y_nu_H = Y_L_mesh - Y_C_H
    elif eq_type == 'fixed_yc':
        Y_C_arr = grids['Y_C']
        meshes = np.meshgrid(n_B_arr, Y_C_arr, T_arr, indexing='ij')
        n_B_mesh, _, T_mesh = meshes
        mu_nu_H = np.zeros(shape)
        Y_nu_H = np.zeros(shape)
    else:
        raise ValueError(f"Unsupported eq_type: '{eq_type}'")

    Y_C_H_data = h_d['Y_C']

    H = SimpleNamespace(
        n_B=n_B_mesh,
        T=T_mesh,
        Y_C=Y_C_H_data,
        Y_S=h_d['Y_S'],
        Y_e=Y_C_H_data,       # charge neutrality in H: Y_e = Y_C
        Y_nu=Y_nu_H,
        mu_B=h_d['mu_B'],
        mu_C=h_d['mu_C'],
        mu_S=h_d['mu_S'],
        mu_e=h_d['mu_e'],
        mu_nu=mu_nu_H,
        P_total=h_d['P_total'],
        e_total=h_d['e_total'],
    )

    Qs = SimpleNamespace(
        n_B=q_d['n_B'],
        T=T_mesh,
        Y_C=q_d['Y_C'],
        Y_S=q_d['Y_S'],
        Y_e=q_d.get('Y_e', q_d['Y_C']),    # fallback for old tables
        Y_nu=Y_nu_H,                         # term vanishes (mu_nu shared)
        mu_B=q_d['mu_B'],
        mu_C=q_d['mu_C'],
        mu_S=q_d['mu_S'],
        mu_e=q_d['mu_e'],
        mu_nu=mu_nu_H,
        P_total=q_d['P_total'],
        e_total=q_d['e_total'],
    )

    return H, Qs


# =============================================================================
# Helper: dispatch Q* solver
# =============================================================================
def _compute_Qstar(hadronic_table, params, sigma,
                   quark_phase, Delta0,
                   flavor_mode, electric_charge_mode,
                   include_photons, include_gluons, include_thermal_neutrinos,
                   initial_guess, verbose,
                   pre_filter_mask=None):
    """Dispatch to appropriate Q* solver via build_solver_fn + generic grid loop."""
    from nucleation.solvers import build_solver_fn

    solver_fn, include_coulomb = build_solver_fn(
        flavor_mode=flavor_mode,
        electric_charge_mode=electric_charge_mode,
        params=params,
        quark_phase=quark_phase,
        Delta0=Delta0,
        sigma=sigma,
        include_photons=include_photons,
        include_gluons=include_gluons,
        include_thermal_neutrinos=include_thermal_neutrinos,
    )
    return compute_Qstar_table(
        hadronic_table, solver_fn,
        include_coulomb=include_coulomb,
        initial_guess=initial_guess,
        verbose=verbose,
        pre_filter_mask=pre_filter_mask,
    )


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

    # ---- Step 1: Compute or retrieve Q* table ----
    if Qstar_table is None:
        if params is None:
            raise ValueError("params must be provided when Qstar_table is None")

        # OPTIMIZATION: Two-pass approach for coulomb_minimize
        if electric_charge_mode == 'coulomb_minimize':
            if verbose:
                print(f"Computing Q* table (two-pass optimization: "
                      f"GCN pre-filter + coulomb_minimize)...")

            # PASS 1: Compute GCN table (fast, no R optimization)
            if verbose:
                print(f"  [1/2] Computing GCN Q* table (flavor={flavor_mode})...")
            gcn_table = _compute_Qstar(
                hadronic_table, params, sigma,
                quark_phase, Delta0,
                flavor_mode, 'gcn',  # Use GCN for pre-filtering
                include_photons, include_gluons, include_thermal_neutrinos,
                initial_guess, verbose,
            )

            # Extract Delta_F from GCN table to create filter mask
            gcn_data = gcn_table.data
            shape = gcn_data['P_total'].shape
            H_gcn, Qs_gcn = _build_phase_namespaces(hadronic_table, gcn_data, shape)

            # Compute driving force from GCN results
            Delta_F_gcn = driving_force(Qs_gcn, H_gcn)

            # Create boolean filter: skip coulomb_minimize where GCN shows Delta_F <= 0
            # Physics: coulomb_minimize always has Delta_F <= Delta_F_gcn
            # (Coulomb repulsion makes nucleation less favorable)
            pre_filter_mask = Delta_F_gcn <= 0

            if verbose:
                n_unfavorable = np.sum(pre_filter_mask)
                total = np.prod(shape)
                pct_skip = 100 * n_unfavorable / total
                print(f"  GCN pre-filter: {n_unfavorable}/{total} "
                      f"points unfavorable ({pct_skip:.1f}% will be skipped)")

            # PASS 2: Compute coulomb_minimize table (expensive, but filtered)
            if verbose:
                n_to_solve = total - n_unfavorable
                print(f"  [2/2] Computing coulomb_minimize Q* table "
                      f"({n_to_solve} points after filtering)...")
            Qstar_table = _compute_Qstar(
                hadronic_table, params, sigma,
                quark_phase, Delta0,
                flavor_mode, 'coulomb_minimize',
                include_photons, include_gluons, include_thermal_neutrinos,
                initial_guess, verbose,
                pre_filter_mask=pre_filter_mask,  # APPLY THE FILTER
            )
        else:
            # Standard single-pass for other modes (lcn, gcn, gcn_coulomb)
            if verbose:
                print(f"Computing Q* table (flavor={flavor_mode}, "
                      f"charge={electric_charge_mode}, phase={quark_phase})...")
            Qstar_table = _compute_Qstar(
                hadronic_table, params, sigma,
                quark_phase, Delta0,
                flavor_mode, electric_charge_mode,
                include_photons, include_gluons, include_thermal_neutrinos,
                initial_guess, verbose,
            )

    # ---- Step 2: Build phase namespaces ----
    q_d = Qstar_table.data
    shape = q_d['P_total'].shape
    H, Qs = _build_phase_namespaces(hadronic_table, q_d, shape)
    converged = q_d['converged']

    # ---- Step 3: Bulk driving force ----
    Delta_F_bulk = driving_force(Qs, H)
    Delta_F_masked = np.where(converged, Delta_F_bulk, np.nan)

    # ---- Step 4: Critical radius and critical work ----
    if electric_charge_mode in ('lcn', 'gcn'):
        R_c = critical_radius(Delta_F_masked, sigma)
        W_c = critical_work(Delta_F_masked, sigma)

    elif electric_charge_mode == 'gcn_coulomb':
        delta_n_C = (Qs.Y_C - Qs.Y_e) * Qs.n_B
        R_c = np.full(shape, np.nan)
        # np.nditer with 'multi_index' gives us the N-dimensional index
        # (e.g. (i_nB, i_T) or (i_nB, i_YL, i_T)) at each step,
        # regardless of the array's dimensionality.
        it = np.nditer(Delta_F_bulk, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            if converged[idx]:
                R_c[idx] = critical_radius_coulomb(
                    float(Delta_F_bulk[idx]), sigma, float(delta_n_C[idx]))
            it.iternext()

        W_c = np.where(
            converged & ~np.isnan(R_c),
            work_of_formation(R_c, Delta_F_bulk, sigma, delta_n_C),
            np.nan,
        )

    elif electric_charge_mode == 'coulomb_minimize':
        R_c = q_d['R_c'].copy()
        delta_n_C = (Qs.Y_C - Qs.Y_e) * Qs.n_B
        W_c = np.where(
            converged,
            work_of_formation(R_c, Delta_F_bulk, sigma, delta_n_C),
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
# Energy barrier W(R) at a specific grid point
# =============================================================================
def compute_energy_barrier(
    hadronic_table,
    Qstar_table,
    sigma=30.0,
    electric_charge_mode='gcn',
    idx=None,
    R_values=None,
    params=None,
    quark_phase='unpaired',
    Delta0=None,
    include_photons=True,
    include_gluons=True,
    include_thermal_neutrinos=True,
):
    """
    Compute W(R) at a specific hadronic grid point.

    For 'coulomb_minimize', solves the Q* equations at each R with
    Coulomb corrections, giving the true R-dependent W(R). This requires
    the quark EOS params.

    Parameters
    ----------
    hadronic_table : EOSTableData
    Qstar_table : QstarTableData
    sigma : float
        Surface tension (MeV/fm^2).
    electric_charge_mode : str
        'lcn', 'gcn', 'gcn_coulomb', or 'coulomb_minimize'.
    idx : tuple of int
        Grid index, e.g. (i_nB, i_T) for beta_eq
        or (i_nB, i_YL, i_T) for trapped.
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
    q_d = Qstar_table.data
    shape = q_d['P_total'].shape
    H, Qs = _build_phase_namespaces(hadronic_table, q_d, shape)
    converged = q_d['converged']

    if not converged[idx]:
        raise ValueError(f"Q* solver did not converge at index {idx}")

    # Driving force at this point
    Delta_F_full = driving_force(Qs, H)
    Delta_F = float(Delta_F_full[idx])

    # ---- coulomb_minimize: solve Q* at each R ----
    if electric_charge_mode == 'coulomb_minimize':
        if params is None:
            raise ValueError(
                "params required for coulomb_minimize energy barrier")

        # Early exit check for unfavorable driving force
        if Delta_F <= 0:
            # Return barrier result with NaNs - no favorable driving force
            R_values_default = np.linspace(0, 20.0, 500) if R_values is None else np.asarray(R_values, dtype=float)
            return EnergyBarrierResult(
                R=R_values_default,
                W=np.full_like(R_values_default, np.nan),
                W_bulk=np.full_like(R_values_default, np.nan),
                W_surface=surface_W(R_values_default, sigma),
                W_coulomb=np.full_like(R_values_default, np.nan),
                Delta_F=Delta_F,
                delta_n_C=0.0,
                sigma=sigma,
                R_c=np.nan,
                W_c=np.nan,
            )

        from nucleation.solvers import solve_coulomb_at_R, solve_coulomb_cfl_at_R

        R_c_val = float(q_d['R_c'][idx])

        if R_values is None:
            R_max = max(5.0 * R_c_val, 20.0) if np.isfinite(R_c_val) else 20.0
            R_values = np.linspace(0, R_max, 500)
        R_values = np.asarray(R_values, dtype=float)

        # Scalar H at this grid point
        H_point = SimpleNamespace(
            T=float(H.T[idx]),
            mu_B=float(H.mu_B[idx]),
            mu_C=float(H.mu_C[idx]),
            mu_S=float(H.mu_S[idx]),
            mu_e=float(H.mu_e[idx]),
            mu_nu=float(H.mu_nu[idx]),
            P_total=float(H.P_total[idx]),
        )

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
                result = solve_coulomb_cfl_at_R(
                    R, H_point, params, Delta0, sigma,
                    include_photons, include_gluons,
                    include_thermal_neutrinos, initial_guess=guess)
            else:
                result = solve_coulomb_at_R(
                    R, H_point, params, sigma,
                    include_photons, include_gluons,
                    include_thermal_neutrinos, initial_guess=guess)

            if result is not None:
                dnC_R = (result.Y_C - result.Y_e) * result.n_B
                W_bulk[i] = bulk_W(R, result.P_total - H_point.P_total)
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
            Delta_F=Delta_F,
            delta_n_C=dnC_at_Rc,
            sigma=sigma,
            R_c=R_c_val,
            W_c=W_c_val,
        )

    # ---- Analytical modes: lcn, gcn, gcn_coulomb ----
    if electric_charge_mode in ('lcn', 'gcn'):
        dnC = 0.0
        R_c_val = float(critical_radius(Delta_F, sigma))
        W_c_val = float(critical_work(Delta_F, sigma))
    elif electric_charge_mode == 'gcn_coulomb':
        dnC = float((Qs.Y_C[idx] - Qs.Y_e[idx]) * Qs.n_B[idx])
        R_c_val = float(critical_radius_coulomb(Delta_F, sigma, dnC))
        W_c_val = float(work_of_formation(R_c_val, Delta_F, sigma, dnC))
    else:
        raise ValueError(f"Invalid electric_charge_mode: '{electric_charge_mode}'")

    # Build R array
    if R_values is None:
        R_max = max(5.0 * R_c_val, 20.0) if np.isfinite(R_c_val) else 20.0
        R_values = np.linspace(0, R_max, 500)
    R_values = np.asarray(R_values, dtype=float)

    # Compute W(R) and its components
    W_bulk = bulk_W(R_values, Delta_F)
    W_surface = surface_W(R_values, sigma)
    W_coul = coulomb_W(R_values, dnC)
    W_total = W_bulk + W_surface + W_coul

    return EnergyBarrierResult(
        R=R_values,
        W=W_total,
        W_bulk=W_bulk,
        W_surface=W_surface,
        W_coulomb=W_coul,
        Delta_F=Delta_F,
        delta_n_C=dnC,
        sigma=sigma,
        R_c=R_c_val,
        W_c=W_c_val,
    )


# =============================================================================
# Nucleation temperature: T at which tau = tau_target
# =============================================================================
def _find_T_root_1d(T_arr, tau_slice, log_tau_target, prev_T):
    """Find T where log10(tau(T)) = log_tau_target along a 1D T slice."""
    with np.errstate(divide='ignore', invalid='ignore'):
        log_tau = np.log10(tau_slice)

    valid = np.isfinite(log_tau) & (T_arr > 0)
    if np.sum(valid) < 2:
        return np.nan, False

    T_valid = T_arr[valid]
    log_tau_valid = log_tau[valid]

    f_interp = interp1d(T_valid, log_tau_valid - log_tau_target,
                        kind='linear', bounds_error=False, fill_value=np.nan)

    diff = log_tau_valid - log_tau_target
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    if len(sign_changes) == 0:
        return np.nan, False

    # Fast path: if we have a previous solution, try root_scalar with
    # prev_T as initial guess (secant method, no bracket needed)
    if prev_T is not None:
        try:
            sol = root_scalar(f_interp, x0=prev_T, method='secant',
                              x1=prev_T * 1.01)
            if sol.converged and T_valid[0] <= sol.root <= T_valid[-1]:
                # Verify it's a downward crossing (tau dropping below target)
                eps = (T_valid[-1] - T_valid[0]) * 1e-4
                if f_interp(sol.root - eps) > 0 and f_interp(sol.root + eps) < 0:
                    return sol.root, True
        except (ValueError, RuntimeError):
            pass

    # Fallback: scan sign changes for downward crossings
    # (tau drops below tau_target, i.e. diff goes from + to -)
    downward = [sc for sc in sign_changes if diff[sc] > 0 and diff[sc + 1] < 0]
    if len(downward) == 0:
        return np.nan, False

    # Pick the first downward crossing (lowest T)
    j = downward[0]
    T_lo = float(T_valid[j])
    T_hi = float(T_valid[j + 1])

    try:
        sol = root_scalar(f_interp, bracket=[T_lo, T_hi], method='brentq')
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
    grids = nucleation_obs.hadronic_grids
    n_B_arr = grids['n_B_H']
    T_arr = grids['T']
    n_nB = len(n_B_arr)
    eq_type = nucleation_obs.eq_type

    log_tau_target = np.log10(tau_target)

    if eq_type == 'beta_eq':
        # tau shape: (n_nB, n_T)
        T_nuc = np.full(n_nB, np.nan)
        conv = np.zeros(n_nB, dtype=bool)
        prev_T = T_guess

        for i in range(n_nB):
            T_root, ok = _find_T_root_1d(
                T_arr, nucleation_obs.tau[i, :], log_tau_target, prev_T)
            if ok:
                T_nuc[i] = T_root
                conv[i] = True
                prev_T = T_root
                if verbose:
                    print(f"  n_B={n_B_arr[i]:.4f} -> T_nuc={T_root:.2f} MeV")

        out_grids = {'n_B_H': n_B_arr}

    elif eq_type == 'trapped_neutrinos':
        # tau shape: (n_nB, n_YL, n_T)
        YL_arr = grids['Y_L_H']
        n_YL = len(YL_arr)
        T_nuc = np.full((n_nB, n_YL), np.nan)
        conv = np.zeros((n_nB, n_YL), dtype=bool)

        for k in range(n_YL):
            prev_T = T_guess
            for i in range(n_nB):
                T_root, ok = _find_T_root_1d(
                    T_arr, nucleation_obs.tau[i, k, :], log_tau_target, prev_T)
                if ok:
                    T_nuc[i, k] = T_root
                    conv[i, k] = True
                    prev_T = T_root
                    if verbose:
                        print(f"  n_B={n_B_arr[i]:.4f}, Y_L={YL_arr[k]:.2f}"
                              f" -> T_nuc={T_root:.2f} MeV")

        out_grids = {'n_B_H': n_B_arr, 'Y_L_H': YL_arr}

    elif eq_type == 'fixed_yc':
        # tau shape: (n_nB, n_YC, n_T)
        YC_arr = grids['Y_C_H']
        n_YC = len(YC_arr)
        T_nuc = np.full((n_nB, n_YC), np.nan)
        conv = np.zeros((n_nB, n_YC), dtype=bool)

        for k in range(n_YC):
            prev_T = T_guess
            for i in range(n_nB):
                T_root, ok = _find_T_root_1d(
                    T_arr, nucleation_obs.tau[i, k, :], log_tau_target, prev_T)
                if ok:
                    T_nuc[i, k] = T_root
                    conv[i, k] = True
                    prev_T = T_root
                    if verbose:
                        print(f"  n_B={n_B_arr[i]:.4f}, Y_C={YC_arr[k]:.3f}"
                              f" -> T_nuc={T_root:.2f} MeV")

        out_grids = {'n_B_H': n_B_arr, 'Y_C_H': YC_arr}

    else:
        raise ValueError(f"Unknown eq_type: '{eq_type}'")

    return NucleationTemperatureResult(
        hadronic_grids=out_grids,
        T_nuc=T_nuc,
        tau_target=tau_target,
        converged=conv,
        eq_type=eq_type,
    )


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
    from nucleation.quantum import (
        effective_inertia, quantum_nucleation_time,
    )

    q_d = Qstar_table.data
    shape = q_d['P_total'].shape
    H, Qs = _build_phase_namespaces(hadronic_table, q_d, shape)
    qstar_converged = q_d['converged']

    # Bulk driving force
    Delta_F_full = driving_force(Qs, H)

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

    it = np.nditer(Delta_F_full, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        it.iternext()

        if not qstar_converged[idx]:
            done += 1
            continue

        Delta_F = float(Delta_F_full[idx])
        if Delta_F <= 0:
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
        R_c = critical_radius(Delta_F, sigma)
        if not np.isfinite(R_c) or R_c <= 0:
            done += 1
            continue

        # Build W(R) and M(R) closures for this grid point
        def W_func(R, _df=Delta_F, _s=sigma, _dnC=delta_n_C):
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
