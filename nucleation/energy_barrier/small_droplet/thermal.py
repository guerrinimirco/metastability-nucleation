"""
Thermal Nucleation Observables
==============================

Compute thermal nucleation observables (R_c, W_c, Gamma, tau) for the
hadron-to-quark phase transition over the full hadronic grid.

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
>>> from nucleation.energy_barrier.small_droplet import (
...     compute_thermal_nucleation_observables,
...     build_thermal_nucleation_interpolators,
... )
>>> obs = compute_thermal_nucleation_observables(
...     hadronic_table,
...     params=my_params,
...     sigma=30.0, V=1e39,
...     flavor_mode='saddlepoint',
...     electric_charge_mode='gcn',
... )
>>> print(obs.R_c, obs.tau)
>>>
>>> interp = build_thermal_nucleation_interpolators(obs)
>>> tau_val = interp['tau'](n_B_H, T)
"""

import numpy as np
from types import SimpleNamespace
from dataclasses import dataclass
from scipy.interpolate import RegularGridInterpolator

from nucleation.energy_barrier.small_droplet.barrier import (
    driving_force,
    critical_work_noCoulomb,
    work_of_formation,
)
from nucleation.general_nucleation.thermal import nucleation_rate, nucleation_time
from nucleation.energy_barrier.small_droplet.table import (
    compute_Qstar_table, GRID_AXES,
)


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
    return ThermalNucleationObservables(
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
