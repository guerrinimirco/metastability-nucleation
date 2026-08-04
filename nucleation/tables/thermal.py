"""
Thermal (Langer) nucleation observables
=======================================

Grid driver for the finite-temperature channel: the droplet escapes by thermal
activation OVER the barrier, at rate Gamma ~ exp(-W_c/T), so tau depends on the
barrier peak (R_c, W_c) alone.

Includes the unpCFL driver, where the barrier is the kinked composite of an
unpaired core and a CFL mantle and its global peak has to be located explicitly.

The .dat layout is shared with the quantum table: axis columns first, then the
data columns, ravelled in Fortran order, under a `# key: value` header.
"""

import os
import numpy as np
from types import SimpleNamespace
from dataclasses import dataclass
from scipy.interpolate import RegularGridInterpolator

from nucleation.barrier import (
    driving_force, work_of_formation, critical_work_noCoulomb,
    get_switching_function,
)
from nucleation.critical import _find_Rc_Wc_step, _find_Rc_Wc_tanh, _blend_phase
from nucleation.rates import nucleation_rate, nucleation_time
from nucleation.tables.grid import (
    GRID_AXES, _THERMAL_DATA_KEYS, expand_grid, _lambda_D_grid,
)
from nucleation.tables.qstar import compute_Qstar_table, _compute_Qs_at_R


@dataclass
class ThermalNucleationObservables:
    """Thermal nucleation observables over a hadronic grid."""
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


def compute_thermal_nucleation_observables(
        hadronic_table, sigma, params=None, Qstar_table=None,
        V=4.18879e51, quark_phase='unpaired', Delta0=None,
        flavor_mode='saddlepoint', electric_charge_mode='gcn',
        include_photons=True, include_gluons=True, include_thermal_neutrinos=True,
        xi_q=0.7, lambda_th=0.0, zeta_th=0.0, initial_guess=None, verbose=False,
        save_table=False, output_file=None, R_gcn_skip=None,
        switching_mode='step', Rx=None, switching_width=None,
        Qstar_table_unp=None, Qstar_table_cfl=None, Qstar_table_unp_atRx=None,
        params_unp=None, params_cfl=None, R_max=20.0, n_R=500):
    """Thermal nucleation observables (R_c, W_c, Gamma, tau) over the grid.

    Dispatches to the unpCFL helper for ``quark_phase='unpCFL'``; otherwise
    computes the Q* table (if not supplied), the driving force, R_c, W_c, and the
    Langer rate. 'screening' evaluates W_c with a Debye-screened Coulomb term
    (lambda_D from chi_e per grid point).
    """
    valid_flavor = ('frozen', 'saddlepoint')
    valid_charge = ('lcn', 'gcn', 'gcn_coulomb', 'coulomb_minimize', 'screening')
    if flavor_mode not in valid_flavor:
        raise ValueError(f"Invalid flavor_mode: '{flavor_mode}'.")
    if electric_charge_mode not in valid_charge:
        raise ValueError(f"Invalid electric_charge_mode: '{electric_charge_mode}'.")
    if flavor_mode == 'frozen':
        if electric_charge_mode not in ('lcn', 'gcn'):
            raise ValueError("frozen flavor_mode only supports 'lcn' or 'gcn'")
        if quark_phase in ('cfl', 'unpCFL'):
            raise ValueError(
                f"frozen flavor_mode does not support quark_phase='{quark_phase}'")

    if quark_phase == 'unpCFL':
        if Rx is None:
            raise ValueError("Rx required for quark_phase='unpCFL'")
        if Delta0 is None:
            raise ValueError("Delta0 required for quark_phase='unpCFL'")
        if switching_mode == 'tanh' and switching_width is None:
            raise ValueError("switching_width required for switching_mode='tanh'")
        return _compute_thermal_nucleation_unpCFL(
            hadronic_table=hadronic_table, sigma=sigma,
            params_unp=params_unp if params_unp is not None else params,
            params_cfl=params_cfl if params_cfl is not None else params,
            Delta0=Delta0, V=V, flavor_mode=flavor_mode,
            electric_charge_mode=electric_charge_mode,
            switching_mode=switching_mode, Rx=Rx, switching_width=switching_width,
            Qstar_table_unp=Qstar_table_unp, Qstar_table_cfl=Qstar_table_cfl,
            Qstar_table_unp_atRx=Qstar_table_unp_atRx,
            include_photons=include_photons, include_gluons=include_gluons,
            include_thermal_neutrinos=include_thermal_neutrinos, xi_q=xi_q,
            lambda_th=lambda_th, zeta_th=zeta_th, initial_guess=initial_guess,
            verbose=verbose, save_table=save_table, output_file=output_file,
            R_gcn_skip=R_gcn_skip)

    if Qstar_table is None:
        if params is None:
            raise ValueError("params must be provided when Qstar_table is None")
        if verbose:
            print(f"Computing Q* table (flavor={flavor_mode}, "
                  f"charge={electric_charge_mode}, phase={quark_phase})...")
        Qstar_table = compute_Qstar_table(
            hadronic_table, flavor_mode=flavor_mode,
            electric_charge_mode=electric_charge_mode, params=params,
            quark_phase=quark_phase, Delta0=Delta0, sigma=sigma,
            include_photons=include_photons, include_gluons=include_gluons,
            include_thermal_neutrinos=include_thermal_neutrinos,
            initial_guess=initial_guess, verbose=verbose, R_gcn_skip=R_gcn_skip)

    q_d = Qstar_table.data
    H, shape, _, T_H, mu_nu_H, Y_nu_H = expand_grid(hadronic_table)
    converged = q_d['converged']
    Qs = SimpleNamespace(**{**q_d, 'T': T_H, 'mu_nu': mu_nu_H, 'Y_nu': Y_nu_H})

    Delta_f_bulk = driving_force(Qs, H)
    R_c = q_d['R_c'].copy()

    if electric_charge_mode in ('lcn', 'gcn'):
        W_c = critical_work_noCoulomb(Delta_f_bulk, sigma)
    else:
        delta_n_C = (Qs.Y_C - Qs.Y_e) * Qs.n_B
        if electric_charge_mode == 'screening':
            lam = _lambda_D_grid(H.mu_e, T_H)
            W_c = np.where(
                converged & ~np.isnan(R_c),
                work_of_formation(R_c, Delta_f_bulk, sigma, delta_n_C, lambda_D=lam),
                np.nan)
        else:
            W_c = np.where(
                converged & ~np.isnan(R_c),
                work_of_formation(R_c, Delta_f_bulk, sigma, delta_n_C),
                np.nan)

    Gamma = nucleation_rate(W_c, R_c, sigma, H.T, H, Qs, xi_q, lambda_th, zeta_th)
    tau = nucleation_time(Gamma, V)
    nuc_converged = converged & np.isfinite(R_c) & np.isfinite(W_c) & (R_c > 0)

    result = ThermalNucleationObservables(
        eq_type=hadronic_table.eq_type, hadronic_grids=Qstar_table.hadronic_grids,
        flavor_mode=flavor_mode, electric_charge_mode=electric_charge_mode,
        sigma=sigma, V=V, R_c=R_c, W_c=W_c, Gamma=Gamma, tau=tau,
        converged=nuc_converged, Qstar_table=Qstar_table)
    if save_table:
        if output_file is None:
            output_file = f"thermal_nucleation_{hadronic_table.eq_type}.dat"
        export_thermal_nucleation_table(result, output_file)
    return result


def build_thermal_nucleation_interpolators(nucleation_obs, method='linear'):
    """Interpolation dict (tau, R_c, W_c, Gamma, log10_tau, log10_Gamma)."""
    eq_type = nucleation_obs.eq_type
    grids = nucleation_obs.hadronic_grids
    axes = GRID_AXES[eq_type]
    grid_tuple = tuple(grids[ax] for ax in axes)
    result = {}
    for name in ('tau', 'R_c', 'W_c', 'Gamma'):
        arr = getattr(nucleation_obs, name)
        interp = RegularGridInterpolator(
            grid_tuple, arr, method=method, bounds_error=False, fill_value=np.nan)
        result[name] = (lambda f: lambda *a: float(f(a)))(interp)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_tau = np.log10(nucleation_obs.tau)
    log_tau = np.clip(log_tau, -50, 100)
    log_tau_method = 'cubic' if np.all(np.isfinite(log_tau)) else 'linear'
    interp_log = RegularGridInterpolator(
        grid_tuple, log_tau, method=log_tau_method,
        bounds_error=False, fill_value=np.nan)
    result['log10_tau'] = (lambda f: lambda *a: float(f(a)))(interp_log)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_Gamma = np.log10(nucleation_obs.Gamma)
    interp_log_Gamma = RegularGridInterpolator(
        grid_tuple, log_Gamma, method=method, bounds_error=False, fill_value=np.nan)
    result['log10_Gamma'] = (lambda f: lambda *a: float(f(a)))(interp_log_Gamma)
    return result


def export_thermal_nucleation_table(obs, output_file):
    """Export thermal observables to a .dat file."""
    axes = GRID_AXES[obs.eq_type]
    grid_arrays = [obs.hadronic_grids[ax] for ax in axes]
    mesh = np.meshgrid(*grid_arrays, indexing='ij')
    input_cols = [m.ravel(order='F') for m in mesh]
    output_cols = [getattr(obs, k).ravel(order='F') for k in _THERMAL_DATA_KEYS]
    all_names = list(axes) + _THERMAL_DATA_KEYS
    all_cols = np.column_stack(input_cols + output_cols)
    header_lines = ["# Thermal nucleation observables",
                    f"# eq_type: {obs.eq_type}",
                    f"# flavor_mode: {obs.flavor_mode}",
                    f"# electric_charge_mode: {obs.electric_charge_mode}",
                    f"# sigma: {obs.sigma} MeV/fm^2",
                    f"# V: {obs.V} fm^3"]
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
    """Load thermal observables from a .dat file."""
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
                    for unit in ('MeV/fm^2', 'fm^3'):
                        val = val.replace(unit, '').strip()
                    metadata[key] = val
    eq_type = metadata['eq_type']
    axes = GRID_AXES[eq_type]
    n_axes = len(axes)
    raw = np.loadtxt(filepath)
    hadronic_grids = {}
    for i, ax in enumerate(axes):
        hadronic_grids[ax] = np.unique(raw[:, i])
    shape = tuple(len(hadronic_grids[ax]) for ax in axes)
    data = {}
    for j, key in enumerate(_THERMAL_DATA_KEYS):
        data[key] = raw[:, n_axes + j].reshape(shape, order='F')
    return ThermalNucleationObservables(
        eq_type=eq_type, hadronic_grids=hadronic_grids,
        flavor_mode=metadata['flavor_mode'],
        electric_charge_mode=metadata['electric_charge_mode'],
        sigma=float(metadata['sigma']), V=float(metadata['V']),
        R_c=data['R_c'], W_c=data['W_c'], Gamma=data['Gamma'], tau=data['tau'],
        converged=data['converged'].astype(bool))

def _compute_thermal_nucleation_unpCFL(
        hadronic_table, sigma, params_unp, params_cfl, Delta0, V, flavor_mode,
        electric_charge_mode, switching_mode, Rx, switching_width,
        Qstar_table_unp, Qstar_table_cfl, include_photons, include_gluons,
        include_thermal_neutrinos, xi_q, lambda_th, zeta_th, initial_guess,
        verbose, save_table, output_file, Qstar_table_unp_atRx=None,
        R_gcn_skip=None):
    """Thermal observables for the unpCFL quark phase (unpaired core + CFL mantle)."""
    common_table_kw = dict(
        include_photons=include_photons, include_gluons=include_gluons,
        include_thermal_neutrinos=include_thermal_neutrinos,
        initial_guess=initial_guess, verbose=verbose)

    if Qstar_table_unp is None:
        if params_unp is None:
            raise ValueError("params_unp (or params) required when Qstar_table_unp is None")
        if verbose:
            print("Computing unpaired Q* table...")
        Qstar_table_unp = compute_Qstar_table(
            hadronic_table, flavor_mode=flavor_mode,
            electric_charge_mode=electric_charge_mode, params=params_unp,
            quark_phase='unpaired', sigma=sigma, R_gcn_skip=R_gcn_skip,
            **common_table_kw)
    if Qstar_table_cfl is None:
        if params_cfl is None:
            raise ValueError("params_cfl (or params) required when Qstar_table_cfl is None")
        if verbose:
            print("Computing CFL Q* table...")
        Qstar_table_cfl = compute_Qstar_table(
            hadronic_table, flavor_mode=flavor_mode,
            electric_charge_mode=electric_charge_mode, params=params_cfl,
            quark_phase='cfl', Delta0=Delta0, sigma=sigma, R_gcn_skip=R_gcn_skip,
            **common_table_kw)

    if electric_charge_mode == 'coulomb_minimize':
        if Qstar_table_unp_atRx is not None:
            q_d_unp_atRx = Qstar_table_unp_atRx.data
        else:
            if verbose:
                print("Computing unpaired Q* at R=Rx...")
            q_d_unp_atRx = _compute_Qs_at_R(
                Rx, hadronic_table, params_unp, sigma, quark_phase='unpaired',
                **common_table_kw).data
    else:
        q_d_unp_atRx = None

    q_d_unp = Qstar_table_unp.data
    q_d_cfl = Qstar_table_cfl.data
    H, shape, _, T_H, mu_nu_H, Y_nu_H = expand_grid(hadronic_table)
    Qs_unp = SimpleNamespace(**{**q_d_unp, 'T': T_H, 'mu_nu': mu_nu_H, 'Y_nu': Y_nu_H})
    Qs_cfl = SimpleNamespace(**{**q_d_cfl, 'T': T_H, 'mu_nu': mu_nu_H, 'Y_nu': Y_nu_H})

    Delta_f_unp = driving_force(Qs_unp, H)
    Delta_f_cfl = driving_force(Qs_cfl, H)
    if electric_charge_mode in ('lcn', 'gcn'):
        delta_n_C_unp = np.zeros(shape)
        delta_n_C_cfl = np.zeros(shape)
    else:
        delta_n_C_unp = (Qs_unp.Y_C - Qs_unp.Y_e) * Qs_unp.n_B
        delta_n_C_cfl = (Qs_cfl.Y_C - Qs_cfl.Y_e) * Qs_cfl.n_B

    if q_d_unp_atRx is not None:
        Qs_unp_atRx = SimpleNamespace(
            **{**q_d_unp_atRx, 'T': T_H, 'mu_nu': mu_nu_H, 'Y_nu': Y_nu_H})
        Delta_f_unp_atRx = driving_force(Qs_unp_atRx, H)
        delta_n_C_unp_atRx = (Qs_unp_atRx.Y_C - Qs_unp_atRx.Y_e) * Qs_unp_atRx.n_B
    else:
        Qs_unp_atRx = None
        Delta_f_unp_atRx = None
        delta_n_C_unp_atRx = None

    R_c_unp = q_d_unp['R_c']
    R_c_cfl = q_d_cfl['R_c']
    common_kw = dict(R_c_unp=R_c_unp, R_c_cfl=R_c_cfl, Rx=Rx,
                     Delta_f_unp=Delta_f_unp, Delta_f_cfl=Delta_f_cfl,
                     delta_n_C_unp=delta_n_C_unp, delta_n_C_cfl=delta_n_C_cfl,
                     sigma=sigma)

    if switching_mode == 'step':
        extra_kw = {}
        if Delta_f_unp_atRx is not None:
            extra_kw['Delta_f_unp_atRx'] = Delta_f_unp_atRx
            extra_kw['delta_n_C_unp_atRx'] = delta_n_C_unp_atRx
        R_c, W_c, S_at_Rc = _find_Rc_Wc_step(**common_kw, **extra_kw)
        if verbose:
            W_at_kink = work_of_formation(
                Rx, Delta_f_unp_atRx if Delta_f_unp_atRx is not None else Delta_f_unp,
                sigma, delta_n_C_unp_atRx if delta_n_C_unp_atRx is not None else delta_n_C_unp)
            W_at_unp = np.where(np.isfinite(R_c_unp) & (R_c_unp <= Rx),
                                work_of_formation(R_c_unp, Delta_f_unp, sigma, delta_n_C_unp),
                                np.nan)
            W_at_cfl = np.where(np.isfinite(R_c_cfl) & (R_c_cfl > Rx),
                                work_of_formation(R_c_cfl, Delta_f_cfl, sigma, delta_n_C_cfl),
                                np.nan)
            W_max = np.fmax(np.fmax(W_at_unp, W_at_kink), W_at_cfl)
            mismatch = np.isfinite(W_max) & np.isfinite(W_c) & (W_max > W_c * 1.01)
            if int(np.sum(mismatch)) > 0:
                print(f"  WARNING: {int(np.sum(mismatch))} grid points where the "
                      f"selected R_c does not give the highest W barrier.")
    elif switching_mode == 'tanh':
        if Delta_f_unp_atRx is not None:
            raise NotImplementedError(
                "coulomb_minimize not implemented for switching_mode='tanh'.")
        S_func = get_switching_function(switching_mode, Rx, switching_width)
        R_c, W_c = _find_Rc_Wc_tanh(S_func=S_func, **common_kw)
        S_at_Rc = S_func(R_c)

    if Qs_unp_atRx is not None:
        at_kink = np.isclose(R_c, Rx) & (S_at_Rc < 0.5)
    else:
        at_kink = None

    P_unp = Qs_unp.P_total
    e_unp = Qs_unp.e_total
    if at_kink is not None:
        P_unp = np.where(at_kink, Qs_unp_atRx.P_total, P_unp)
        e_unp = np.where(at_kink, Qs_unp_atRx.e_total, e_unp)
    Qs_mixed = SimpleNamespace(
        P_total=_blend_phase(S_at_Rc, P_unp, Qs_cfl.P_total),
        e_total=_blend_phase(S_at_Rc, e_unp, Qs_cfl.e_total))

    Gamma = nucleation_rate(W_c, R_c, sigma, H.T, H, Qs_mixed, xi_q, lambda_th, zeta_th)
    tau = nucleation_time(Gamma, V)

    if switching_mode == 'step':
        converged = np.where(S_at_Rc < 0.5, q_d_unp['converged'], q_d_cfl['converged'])
        if at_kink is not None:
            converged = np.where(at_kink, q_d_unp_atRx['converged'], converged)
    else:
        converged = q_d_unp['converged'] & q_d_cfl['converged']
    nuc_converged = converged & np.isfinite(R_c) & np.isfinite(W_c) & (R_c > 0)

    result = ThermalNucleationObservables(
        eq_type=hadronic_table.eq_type, hadronic_grids=Qstar_table_unp.hadronic_grids,
        flavor_mode=flavor_mode, electric_charge_mode=electric_charge_mode,
        sigma=sigma, V=V, R_c=R_c, W_c=W_c, Gamma=Gamma, tau=tau,
        converged=nuc_converged)
    if save_table:
        if output_file is None:
            output_file = f"thermal_nucleation_unpCFL_{hadronic_table.eq_type}.dat"
        export_thermal_nucleation_table(result, output_file)
    return result
