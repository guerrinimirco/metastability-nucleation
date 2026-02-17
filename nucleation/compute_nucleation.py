"""
Nucleation Observables Wrapper
==============================

High-level function that computes nucleation observables (R_c, W_c, Gamma, tau)
for the hadron-to-quark phase transition.

Two independent parameters control the Q* solver:

    flavor_mode:
        frozen       : Flavor fractions frozen from the hadronic phase
        saddlepoint  : Flavor fractions minimized (saddlepoint approximation)

    electric_charge_mode:
        lcn              : Local charge neutrality, no Coulomb
        gcn              : Global charge neutrality, no Coulomb
        gcn_coulomb      : GCN solver (no Coulomb in equations),
                           Coulomb energy added post-hoc to W(R), R_c from findroot
        coulomb_minimize : Full Coulomb minimization via solve_Qstar_coulomb
                           (R is part of the unknowns)

Constraints:
    - frozen only supports 'lcn' and 'gcn' (no Coulomb modes, no CFL)
    - saddlepoint supports all four electric charge modes

Usage
-----
>>> from nucleation.compute_nucleation import compute_nucleation_observables
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

from nucleation.general.general_nucleation import (
    driving_force,
    critical_radius, critical_work,
    critical_radius_coulomb,
    work_of_formation,
    nucleation_rate, nucleation_time,
)


# =============================================================================
# Output dataclass
# =============================================================================
@dataclass
class NucleationObservables:
    """Nucleation observables computed over a hadronic grid.

    Attributes
    ----------
    eq_type : str
        Equilibrium type ('betaeq', 'trapped', 'fixedYC').
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
# Helper: build H and Qs SimpleNamespace from table data
# =============================================================================
def _build_phase_namespaces(hadronic_table, q_d, shape):
    """
    Build array-based H and Qs SimpleNamespace objects from table data.

    These namespaces provide the interface expected by general_nucleation
    functions (driving_force, dynamical_prefactor, etc.).

    Parameters
    ----------
    hadronic_table : EOSTableData
        Hadronic phase table.
    q_d : dict
        Q* data dictionary (from Qstar_table.data).
    shape : tuple
        Shape of the output arrays.

    Returns
    -------
    (H, Qs) : tuple of SimpleNamespace
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
                   initial_guess, verbose):
    """Dispatch to appropriate Q* solver based on flavor_mode and electric_charge_mode."""

    charge_neutrality = 'local' if electric_charge_mode == 'lcn' else 'global'

    if flavor_mode == 'frozen':
        from nucleation.Qstar.Qstar_frozen_solver import compute_Qstar_table
        return compute_Qstar_table(
            hadronic_table, params,
            electric_charge_neutrality=charge_neutrality,
            include_photons=include_photons,
            include_gluons=include_gluons,
            include_thermal_neutrinos=include_thermal_neutrinos,
            initial_guess=initial_guess,
            verbose=verbose,
        )

    # saddlepoint modes
    if electric_charge_mode in ('lcn', 'gcn', 'gcn_coulomb'):
        from nucleation.Qstar.Qstar_saddlepoint_solver import compute_Qstar_table
        return compute_Qstar_table(
            hadronic_table, params,
            electric_charge_neutrality=charge_neutrality,
            include_photons=include_photons,
            include_gluons=include_gluons,
            include_thermal_neutrinos=include_thermal_neutrinos,
            quark_phase=quark_phase,
            Delta0=Delta0,
            initial_guess=initial_guess,
            verbose=verbose,
        )

    # coulomb_minimize
    from nucleation.Qstar.Qstar_saddlepoint_coulomb_solver import (
        compute_Qstar_coulomb_table_betaeq,
        compute_Qstar_coulomb_table_trapped,
    )
    eq_type = hadronic_table.eq_type
    if eq_type == 'beta_eq':
        return compute_Qstar_coulomb_table_betaeq(
            hadronic_table, params, sigma,
            include_photons=include_photons,
            include_gluons=include_gluons,
            include_thermal_neutrinos=include_thermal_neutrinos,
            quark_phase=quark_phase,
            Delta0=Delta0,
            initial_guess=initial_guess,
            verbose=verbose,
        )
    elif eq_type == 'trapped_neutrinos':
        return compute_Qstar_coulomb_table_trapped(
            hadronic_table, params, sigma,
            include_photons=include_photons,
            include_gluons=include_gluons,
            include_thermal_neutrinos=include_thermal_neutrinos,
            quark_phase=quark_phase,
            Delta0=Delta0,
            initial_guess=initial_guess,
            verbose=verbose,
        )
    else:
        raise ValueError(
            f"coulomb_minimize not supported for eq_type '{eq_type}'. "
            f"Only 'beta_eq' and 'trapped_neutrinos' are supported.")


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

    Given hadronic conditions (from a table) and quark matter parameters,
    computes critical radius, energy barrier, nucleation rate, and
    nucleation time over the full grid.

    Parameters
    ----------
    hadronic_table : EOSTableData
        Hadronic phase conditions. Must contain grids ('n_B', 'T', ...)
        and data ('P_total', 'e_total', 'mu_B', 'mu_C', 'mu_S', 'mu_e', ...).
    params : AlphaBagParams or None
        Quark EOS parameters. Required if Qstar_table is None.
    Qstar_table : QstarTableData or None
        Pre-computed Q* table. If None, computed internally using params.
    sigma : float
        Surface tension (MeV/fm^2).
    V : float
        System volume (fm^3) for nucleation time. Default: sphere with R=100 m.
    quark_phase : str
        'unpaired' or 'cfl'.
    Delta0 : float or None
        CFL pairing gap (MeV). Required if quark_phase='cfl'.
    flavor_mode : str
        'frozen'       : Flavor fractions frozen from hadronic phase.
        'saddlepoint'  : Minimization of W with respect to flavor composition.
        frozen only supports 'lcn' and 'gcn', no CFL.
    electric_charge_mode : str
        'lcn'              : Local charge neutrality, no Coulomb.
        'gcn'              : Global charge neutrality, no Coulomb.
        'gcn_coulomb'      : GCN solver, Coulomb added post-hoc to W(R).
        'coulomb_minimize' : Full Coulomb minimization (R is unknown).
    include_photons, include_gluons, include_thermal_neutrinos : bool
        Include contributions in quark EOS.
    xi_q : float
        Quark correlation length (fm). Default 0.7.
    lambda_th : float
        Thermal conductivity. Default 0.
    zeta_th : float
        Bulk viscosity. Default 0.
    initial_guess : array-like or None
        Initial guess for Q* solver.
    verbose : bool
        Print convergence info.

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
        delta_n_C = q_d['delta_n_C']
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
    eq_type_map = {
        'beta_eq': 'betaeq',
        'trapped_neutrinos': 'trapped',
        'fixed_yc': 'fixedYC',
    }
    eq_type = eq_type_map.get(hadronic_table.eq_type, hadronic_table.eq_type)

    return NucleationObservables(
        eq_type=eq_type,
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
@dataclass
class EnergyBarrierResult:
    """Energy barrier W(R) at a single hadronic grid point.

    Attributes
    ----------
    R : np.ndarray
        Radius array (fm).
    W : np.ndarray
        Total work of formation W(R) (MeV).
    W_bulk : np.ndarray
        Bulk contribution -4/3 pi R^3 Delta_F (MeV).
    W_surface : np.ndarray
        Surface contribution 4 pi R^2 sigma (MeV).
    W_coulomb : np.ndarray
        Coulomb contribution (MeV). Zero for lcn/gcn modes.
    Delta_F : float
        Bulk driving force at this point (MeV/fm^3).
    delta_n_C : float
        Net charge density (fm^-3). Zero for lcn/gcn modes.
    sigma : float
        Surface tension (MeV/fm^2).
    R_c : float
        Critical radius (fm).
    W_c : float
        Critical work W(R_c) (MeV).
    """
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


def compute_energy_barrier(
    hadronic_table,
    Qstar_table,
    sigma=30.0,
    electric_charge_mode='gcn',
    idx=None,
    R_values=None,
):
    """
    Compute W(R) at a specific hadronic grid point.

    Parameters
    ----------
    hadronic_table : EOSTableData
        Hadronic phase table.
    Qstar_table : QstarTableData
        Pre-computed Q* table.
    sigma : float
        Surface tension (MeV/fm^2).
    electric_charge_mode : str
        'lcn', 'gcn', 'gcn_coulomb', or 'coulomb_minimize'.
    idx : tuple of int
        Grid index, e.g. (i_nB, i_T) for beta_eq
        or (i_nB, i_YL, i_T) for trapped.
    R_values : array-like or None
        Radius array (fm). If None, auto-generated from 0 to 5*R_c.

    Returns
    -------
    EnergyBarrierResult
    """
    from nucleation.general.general_nucleation import coulomb_W

    q_d = Qstar_table.data
    shape = q_d['P_total'].shape
    H, Qs = _build_phase_namespaces(hadronic_table, q_d, shape)
    converged = q_d['converged']

    if not converged[idx]:
        raise ValueError(f"Q* solver did not converge at index {idx}")

    # Driving force at this point
    Delta_F_full = driving_force(Qs, H)
    Delta_F = float(Delta_F_full[idx])

    # Charge density for Coulomb modes
    if electric_charge_mode in ('lcn', 'gcn'):
        dnC = 0.0
        R_c_val = float(critical_radius(Delta_F, sigma))
        W_c_val = float(critical_work(Delta_F, sigma))
    elif electric_charge_mode == 'gcn_coulomb':
        dnC = float((Qs.Y_C[idx] - Qs.Y_e[idx]) * Qs.n_B[idx])
        R_c_val = float(critical_radius_coulomb(Delta_F, sigma, dnC))
        W_c_val = float(work_of_formation(R_c_val, Delta_F, sigma, dnC))
    elif electric_charge_mode == 'coulomb_minimize':
        dnC = float(q_d['delta_n_C'][idx])
        R_c_val = float(q_d['R_c'][idx])
        W_c_val = float(work_of_formation(R_c_val, Delta_F, sigma, dnC))
    else:
        raise ValueError(f"Invalid electric_charge_mode: '{electric_charge_mode}'")

    # Build R array
    if R_values is None:
        R_max = max(5.0 * R_c_val, 20.0) if np.isfinite(R_c_val) else 20.0
        R_values = np.linspace(0, R_max, 500)
    R_values = np.asarray(R_values, dtype=float)

    # Compute W(R) and its components
    W_bulk = -4.0 / 3.0 * np.pi * R_values**3 * Delta_F
    W_surface = 4.0 * np.pi * R_values**2 * sigma
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
