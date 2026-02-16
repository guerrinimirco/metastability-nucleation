"""
Nucleation Observables Wrapper
==============================

High-level function that computes nucleation observables (R_c, W_c, Gamma, tau)
for the hadron-to-quark phase transition.

Supports four electric charge modes:

    lcn_standard      : Local charge neutrality (frozen solver), no Coulomb
    gcn_standard      : Global charge neutrality (saddlepoint solver), no Coulomb
    gcn_coulomb       : GCN saddlepoint solver (no Coulomb in equations),
                        Coulomb energy added post-hoc to W(R), R_c from findroot
    coulomb_minimize  : Full Coulomb minimization via solve_Qstar_coulomb
                        (R is part of the unknowns)

Usage
-----
>>> from nucleation.compute_nucleation import compute_nucleation_observables
>>> result = compute_nucleation_observables(
...     hadronic_table, params=my_params,
...     sigma=30.0, V=1e39,
...     electric_charge_mode='gcn_standard',
... )
>>> print(result.R_c, result.tau)
"""

import numpy as np
from types import SimpleNamespace
from scipy.optimize import brentq
from dataclasses import dataclass

from nucleation.Qstar.Qstar_saddlepoint_coulomb_solver import coulomb_W
from eos.general.physics_constants import alpha_EM, hc
from nucleation.general.general_nucleation import (
    statistical_prefactor, dynamical_prefactor,
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
    electric_charge_mode : str
        Mode used for computation.
    sigma : float
        Surface tension (MeV/fm²).
    V : float
        System volume (fm³).
    R_c : np.ndarray
        Critical radius (fm).
    W_c : np.ndarray
        Critical work / energy barrier (MeV).
    Gamma : np.ndarray
        Nucleation rate (fm⁻³ s⁻¹).
    tau : np.ndarray
        Nucleation time (s).
    Qstar_table : object
        Q* table used (for reference).
    """
    eq_type: str
    hadronic_grids: dict
    electric_charge_mode: str
    sigma: float
    V: float
    R_c: np.ndarray
    W_c: np.ndarray
    Gamma: np.ndarray
    tau: np.ndarray
    Qstar_table: object = None


# =============================================================================
# Helper: Coulomb post-hoc critical radius
# =============================================================================
def _critical_radius_coulomb_posthoc(Delta_F_bulk, sigma, delta_n_C):
    """
    Find critical radius including Coulomb correction (post-hoc).

    Solves dW/dR = 0 where W includes bulk + surface + Coulomb:

        -4 pi R^2 Delta_F + 8 pi R sigma
        - (16/3) pi^2 hbar*c alpha_em delta_n_C^2 R^4 = 0

    Parameters
    ----------
    Delta_F_bulk : float
        Bulk driving force (MeV/fm³).
    sigma : float
        Surface tension (MeV/fm²).
    delta_n_C : float
        Net charge density n_C_Q - n_e_Q (fm⁻³).

    Returns
    -------
    float
        Critical radius R_c (fm). NaN if no solution.
    """
    if Delta_F_bulk <= 0 or np.isnan(Delta_F_bulk):
        return np.nan

    def dW_dR(R):
        return (-4.0 * np.pi * R**2 * Delta_F_bulk
                + 8.0 * np.pi * R * sigma
                - (16.0 / 3.0) * np.pi**2 * hc * alpha_EM
                * delta_n_C**2 * R**4)

    R_standard = 2.0 * sigma / Delta_F_bulk
    try:
        R_c = brentq(dW_dR, 1e-6, 5.0 * R_standard)
    except ValueError:
        return np.nan
    return R_c


# =============================================================================
# Helper: build H and Qs SimpleNamespace from table data
# =============================================================================
def _build_phase_namespaces(hadronic_table, q_d, shape):
    """
    Build array-based H and Qs SimpleNamespace objects from table data.

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

    # For Q*, mu_nu is shared between phases (passed through in solver),
    # and Y_nu doesn't affect chem_sum since mu_nu_Qs = mu_nu_H.
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
        n_e=q_d.get('n_e', np.full(shape, np.nan)),
    )

    return H, Qs


# =============================================================================
# Helper: compute bulk driving force
# =============================================================================
def _compute_Delta_F_bulk(Qs, H):
    """
    Compute the general bulk driving force.

    Delta_F_bulk = (P_Qs - P_H) - n_B_Qs * sum_i Y_i^Qs (mu_i^Qs - mu_i^H)

    For the saddlepoint solver (GCN, no Coulomb), chem_sum = 0 and
    Delta_F_bulk = P_Qs - P_H.
    """
    Delta_P = Qs.P_total - H.P_total
    chem_sum = (
        1.0 * (Qs.mu_B - H.mu_B)
        + Qs.Y_C * (Qs.mu_C - H.mu_C)
        + Qs.Y_S * (Qs.mu_S - H.mu_S)
        + Qs.Y_e * (Qs.mu_e - H.mu_e)
        + Qs.Y_nu * (Qs.mu_nu - H.mu_nu)
    )
    return Delta_P - Qs.n_B * chem_sum


# =============================================================================
# Helper: dispatch Q* solver
# =============================================================================
def _compute_Qstar(hadronic_table, params, sigma,
                   quark_phase, Delta0, electric_charge_mode,
                   include_photons, include_gluons, include_thermal_neutrinos,
                   initial_guess, verbose):
    """Dispatch to appropriate Q* solver based on electric_charge_mode."""

    if electric_charge_mode == 'lcn_standard':
        if quark_phase == 'cfl':
            raise ValueError("lcn_standard does not support quark_phase='cfl'. "
                             "Use gcn_standard or coulomb_minimize.")
        from nucleation.Qstar.Qstar_frozen_solver import compute_Qstar_table
        return compute_Qstar_table(
            hadronic_table, params,
            electric_charge_neutrality='local',
            include_photons=include_photons,
            include_gluons=include_gluons,
            include_thermal_neutrinos=include_thermal_neutrinos,
            initial_guess=initial_guess,
            verbose=verbose,
        )

    elif electric_charge_mode in ('gcn_standard', 'gcn_coulomb'):
        from nucleation.Qstar.Qstar_saddlepoint_solver import compute_Qstar_table
        return compute_Qstar_table(
            hadronic_table, params,
            electric_charge_neutrality='global',
            include_photons=include_photons,
            include_gluons=include_gluons,
            include_thermal_neutrinos=include_thermal_neutrinos,
            quark_phase=quark_phase,
            Delta0=Delta0,
            initial_guess=initial_guess,
            verbose=verbose,
        )

    elif electric_charge_mode == 'coulomb_minimize':
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

    else:
        raise ValueError(f"Unknown electric_charge_mode: '{electric_charge_mode}'")


# =============================================================================
# Main function
# =============================================================================
def compute_nucleation_observables(
    hadronic_table,
    params=None,
    Qstar_table=None,
    sigma=30.0,
    V=1e39,
    quark_phase='unpaired',
    Delta0=None,
    electric_charge_mode='gcn_standard',
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
        Surface tension (MeV/fm²).
    V : float
        System volume (fm³) for nucleation time.
    quark_phase : str
        'unpaired' or 'cfl'.
    Delta0 : float or None
        CFL pairing gap (MeV). Required if quark_phase='cfl'.
    electric_charge_mode : str
        'lcn_standard'     : Local CN (frozen solver), no Coulomb.
        'gcn_standard'     : Global CN (saddlepoint), no Coulomb.
        'gcn_coulomb'      : GCN saddlepoint, Coulomb added post-hoc to W(R).
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
    valid_modes = ['lcn_standard', 'gcn_standard',
                   'gcn_coulomb', 'coulomb_minimize']
    if electric_charge_mode not in valid_modes:
        raise ValueError(
            f"Invalid electric_charge_mode: '{electric_charge_mode}'. "
            f"Valid: {valid_modes}")

    # ---- Step 1: Compute or retrieve Q* table ----
    if Qstar_table is None:
        if params is None:
            raise ValueError("params must be provided when Qstar_table is None")
        if verbose:
            print(f"Computing Q* table (mode={electric_charge_mode}, "
                  f"phase={quark_phase})...")
        Qstar_table = _compute_Qstar(
            hadronic_table, params, sigma,
            quark_phase, Delta0, electric_charge_mode,
            include_photons, include_gluons, include_thermal_neutrinos,
            initial_guess, verbose,
        )

    # ---- Step 2: Build phase namespaces ----
    q_d = Qstar_table.data
    shape = q_d['P_total'].shape
    H, Qs = _build_phase_namespaces(hadronic_table, q_d, shape)
    converged = q_d['converged']

    # ---- Step 3: Compute Delta_F_bulk ----
    Delta_F_bulk = _compute_Delta_F_bulk(Qs, H)

    # ---- Step 4: Compute R_c and W_c ----
    if electric_charge_mode in ('lcn_standard', 'gcn_standard'):
        # Standard CNT formulas (no Coulomb)
        R_c = np.where(
            (Delta_F_bulk > 0) & converged,
            2.0 * sigma / Delta_F_bulk,
            np.nan,
        )
        W_c = np.where(
            (Delta_F_bulk > 0) & converged,
            -4.0 / 3.0 * np.pi * R_c**3 * Delta_F_bulk
            + 4.0 * np.pi * R_c**2 * sigma,
            np.nan,
        )

    elif electric_charge_mode == 'gcn_coulomb':
        # Post-hoc Coulomb: R_c from findroot, W_c includes Coulomb term
        delta_n_C = Qs.Y_C * Qs.n_B - Qs.n_e
        R_c = np.full(shape, np.nan)
        W_c = np.full(shape, np.nan)

        it = np.nditer(Delta_F_bulk, flags=['multi_index'])
        while not it.finished:
            idx = it.multi_index
            if converged[idx]:
                df = float(Delta_F_bulk[idx])
                dnc = float(delta_n_C[idx])
                rc = _critical_radius_coulomb_posthoc(df, sigma, dnc)
                R_c[idx] = rc
                if not np.isnan(rc):
                    W_c[idx] = (
                        -4.0 / 3.0 * np.pi * rc**3 * df
                        + 4.0 * np.pi * rc**2 * sigma
                        + coulomb_W(rc, dnc)
                    )
            it.iternext()

    elif electric_charge_mode == 'coulomb_minimize':
        # R_c from Coulomb solver table
        R_c = q_d['R_c'].copy()
        delta_n_C = q_d['delta_n_C']
        W_c = np.where(
            converged,
            -4.0 / 3.0 * np.pi * R_c**3 * Delta_F_bulk
            + 4.0 * np.pi * R_c**2 * sigma
            + coulomb_W(R_c, delta_n_C),
            np.nan,
        )

    # ---- Step 5: Compute Gamma and tau ----
    T = H.T
    Omega_0 = statistical_prefactor(sigma, T, R_c, xi_q)
    kappa = dynamical_prefactor(sigma, R_c, H, Qs, T, lambda_th, zeta_th)
    Gamma = kappa / (2.0 * np.pi) * Omega_0 * np.exp(-W_c / T)
    tau = 1.0 / (V * Gamma)

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
        electric_charge_mode=electric_charge_mode,
        sigma=sigma,
        V=V,
        R_c=R_c,
        W_c=W_c,
        Gamma=Gamma,
        tau=tau,
        Qstar_table=Qstar_table,
    )
