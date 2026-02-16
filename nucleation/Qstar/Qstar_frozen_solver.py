"""
Q* Computation: Flavor-Frozen Nucleation using AlphaBag EOS
===================================================================

Given initial hadronic conditions, find the quark droplet Q* thermodynamics

Three equilibrium modes of the initial hadronic phase:
  - 'beta_eq':  μ_ν = 0, neutrino untrapped
  - 'trapped':  μ_ν = μ_ν_H, trapped neutrino, global L conservation
  - 'fixed_YC': global C conservation
"""

import os
import numpy as np
from types import SimpleNamespace
from scipy.optimize import root
from eos.alphabag.eos import compute_alphabag_total_thermo_from_mu
from eos.alphabag.thermodynamics_quarks import compute_alphabag_thermo_from_mu
from eos.general.thermodynamics_leptons import electron_thermo, neutrino_thermo
from dataclasses import dataclass


# =============================================================================
# Q* table data container
# =============================================================================
QSTAR_GRID_AXES = {
    'betaeq':  ['n_B_H', 'T'],
    'trapped': ['n_B_H', 'Y_L_H', 'T'],
    'fixedYC': ['n_B_H', 'Y_C_H', 'T'],
}


@dataclass
class QstarFrozenTableData:
    """Q* flavor-frozen nucleation table.

    Maps hadronic conditions (n_B_H, T, ...) to Q* quark droplet thermodynamics.
    """
    eq_type: str            # 'betaeq', 'trapped', 'fixedYC'
    hadronic_grids: dict    # input grids: {n_B_H: array, T: array, ...}
    data: dict              # Q* output:  {n_B, mu_u, ..., P_total, ...}
    filepath: str = ""

    def __repr__(self):
        shapes = [f"{k}={len(v)}" for k, v in self.hadronic_grids.items()]
        return f"QstarFrozenTableData(eq_type='{self.eq_type}', grids=({', '.join(shapes)}))"


# =============================================================================
# Build interpolators from Q* table
# =============================================================================
def build_Qstar_interpolators(table, method='linear'):
    """
    Build interpolation functions from a QstarFrozenTableData.

    Returns dict with convenience functions for each Q* quantity,
    keyed by hadronic input coordinates.

    Usage (betaeq)::
        Q = build_Qstar_interpolators(Q_betaeq_table)
        P_Qs = Q['P'](n_B_H, T)
        mu_B_Qs = Q['mu_B'](n_B_H, T)

    Usage (trapped)::
        Q = build_Qstar_interpolators(Q_trapped_table)
        P_Qs = Q['P'](n_B_H, Y_L_H, T)
    """
    from scipy.interpolate import RegularGridInterpolator

    axes = QSTAR_GRID_AXES[table.eq_type]
    grid_tuple = tuple(table.hadronic_grids[ax] for ax in axes)

    interpolators = {}
    for name, arr in table.data.items():
        if name == 'converged':
            continue
        interpolators[name] = RegularGridInterpolator(
            grid_tuple, arr, method=method,
            bounds_error=False, fill_value=np.nan
        )

    result = {
        'interpolators': interpolators,
        'hadronic_grids': table.hadronic_grids,
        'axes': axes,
        'eq_type': table.eq_type,
    }

    # Convenience: result[key](*args) for each stored quantity
    n = len(axes)
    for key in interpolators:
        result[key] = (lambda k: lambda *a: interpolators[k](a))(key)

    # Derived conserved-charge chemical potentials
    result['mu_B'] = lambda *a: interpolators['mu_u'](a) + 2 * interpolators['mu_d'](a)
    result['mu_C'] = lambda *a: interpolators['mu_u'](a) - interpolators['mu_d'](a)
    result['mu_S'] = lambda *a: interpolators['mu_s'](a) - interpolators['mu_d'](a)

    # Short aliases
    result['P']   = result['P_total']
    result['eps'] = result['e_total']
    result['s']   = result['s_total']
    result['f']   = result['f_total']

    return result


# =============================================================================
# Load Q* table from file
# =============================================================================
_GRID_KEY_MAP = {'n_B_H': 'n_B_H', 'Y_L_H': 'Y_L_H', 'Y_C_H': 'Y_C_H', 'T': 'T'}


def load_Qstar_table(filepath):
    """
    Load a Q* flavor-frozen table from a .dat file saved by _export_table.

    Backward-compatible: handles files that lack the Y_e column
    (exported before that field was added).

    Parameters
    ----------
    filepath : str
        Path to the .dat file.

    Returns
    -------
    QstarFrozenTableData
    """
    eq_type = None
    col_names = None
    with open(filepath) as f:
        for line in f:
            if not line.startswith('#'):
                break
            if 'hadronic_eq_type:' in line:
                eq_type = line.split('hadronic_eq_type:')[1].strip()
            col_names = line.lstrip('#').split()

    if eq_type is None or col_names is None:
        raise ValueError(f"Cannot parse header in {filepath}")

    axes = QSTAR_GRID_AXES[eq_type]
    n_axes = len(axes)

    raw = np.loadtxt(filepath)

    hadronic_grids = {}
    shape = []
    for i, ax in enumerate(axes):
        vals = np.unique(raw[:, i])
        hadronic_grids[_GRID_KEY_MAP[ax]] = vals
        shape.append(len(vals))
    shape = tuple(shape)

    data_col_names = col_names[n_axes:]
    data = {}
    for j, cname in enumerate(data_col_names):
        key = cname.replace('_Qs', '')
        col_flat = raw[:, n_axes + j]
        arr = col_flat.reshape(shape, order='F')
        if key == 'converged':
            arr = arr.astype(bool)
        data[key] = arr

    # Backward compat: reconstruct Y_e if absent
    if 'Y_e' not in data and 'Y_C' in data:
        data['Y_e'] = data['Y_C'].copy()

    return QstarFrozenTableData(
        eq_type=eq_type,
        hadronic_grids=hadronic_grids,
        data=data,
        filepath=filepath,
    )


# =============================================================================
# Qstar computation for given n_B_H, Y_C_H, Y_S_H, T_H
# =============================================================================
def solve_Qstar(H, params, electric_charge_neutrality,
                include_photons, include_gluons,
                include_thermal_neutrinos, initial_guess):
    """
    Parameters
    ----------
    H : object
        Hadronic state with attributes:
        n_B_H, T_H, Y_C_H, Y_S_H, Y_e_H, Y_nu_H, mu_B_H, mu_C_H, mu_S_H, mu_e_H, mu_nu_H.
    params : AlphaBagParams
        AlphaBag EOS parameters.
    electric_charge_neutrality : str
        'local'  : Y_C_Qs - Y_e_Qs = 0 in Q* (locally charge neutral droplet)
        'global' : mu_e_Qs = mu_e_H
    initial_guess : initial guess for [mu_u, mu_d, mu_s, mu_e] (MeV).
    equations:
    frozen C: Y_C(μ_u, μ_d, μ_s) = Y_C_H 
    frozen S: Y_S(μ_u, μ_d, μ_s) = Y_S_H 
    saddle point: μ_B_H + Y_C_Qs·μ_C_H + Y_S_Qs·μ_S_H + Y_e_Qs·μ_e_H + Y_ν_Qs·μ_ν_H = μ_B_Qs + Y_C_Qs·μ_C_Qs + Y_S_Qs·μ_S_Qs + Y_e_Qs·μ_e_Qs + Y_ν_Qs·μ_ν_Qs
    if electric_charge_neutrality == "local": Y_C_Qs(μ_u, μ_d, μ_s) - Y_e_Qs(μ_e) = 0
    if electric_charge_neutrality == "global": μ_e_Qs(μ_u, μ_d, μ_s) = μ_e_H
    global neutrino trapping [or no neutrino]: μ_ν_Qs = μ_ν_H [ = 0 for no neutrino]
        

    Returns
    -------
    AlphaBagEOSResult or None
    """

    def equations(x):

        mu_u, mu_d, mu_s, mu_e = x
        Qs = compute_alphabag_thermo_from_mu(mu_u, mu_d, mu_s, H.T, params)
        e_Qs = electron_thermo(mu_e, H.T)

        C_frozen_equation = Qs.Y_C - H.Y_C
        S_frozen_equation = Qs.Y_S - H.Y_S

        # Minimization wrt n_e_Qs: local or global electric charge neutrality
        if electric_charge_neutrality == "local":
            electron_charge_neutrality_equation =  Qs.n_C - e_Qs.n
            Y_e_Qs = e_Qs.n / Qs.n_B
        elif electric_charge_neutrality == "global":
            electron_charge_neutrality_equation =  mu_e - H.mu_e
            Y_e_Qs = H.Y_e

        saddle_point_nBQs_equation = (Qs.mu_B-H.mu_B) + Qs.Y_C * (Qs.mu_C-H.mu_C) + Qs.Y_S * (Qs.mu_S-H.mu_S) + Y_e_Qs * (mu_e-H.mu_e)
        
        return [
            C_frozen_equation,
            S_frozen_equation,
            electron_charge_neutrality_equation,
            saddle_point_nBQs_equation
        ]

    # Physics-based guess from hadronic chemical potentials
    mu_d_h = (H.mu_B - H.mu_C) / 3.0
    mu_u_h = (H.mu_B + 2.0 * H.mu_C) / 3.0
    mu_s_h = mu_d_h + H.mu_S
    h_guess = np.array([mu_u_h, mu_d_h, mu_s_h, H.mu_e])

    if initial_guess is None:
        initial_guess = h_guess

    # Try with provided guess (propagated from previous grid point, or H-based)
    sol = root(equations, initial_guess, method='hybr')
    if not sol.success:
        sol = root(equations, initial_guess, method='lm')

    # Fallback: if propagated guess failed, retry with H-based guess
    if not sol.success and not np.allclose(initial_guess, h_guess):
        sol = root(equations, h_guess, method='hybr')
        if not sol.success:
            sol = root(equations, h_guess, method='lm')

    if not sol.success:
        return None

    mu_u, mu_d, mu_s, mu_e = sol.x

    return compute_alphabag_total_thermo_from_mu(
        mu_u, mu_d, mu_s, mu_e, H.T, params,
        include_photons=include_photons,
        include_gluons=include_gluons,
        include_thermal_neutrinos=include_thermal_neutrinos,
        mu_nu=H.mu_nu
    )


# =============================================================================
# Wrapper: compute Q* table for any equilibrium mode
# =============================================================================
def compute_Qstar_table(hadronic_table, params,
                        electric_charge_neutrality,
                        include_photons=True, include_gluons=True,
                        include_thermal_neutrinos=True,
                        initial_guess=None, verbose=False,
                        save_table=False, output_file=None):
    """
    Parameters
    ----------
    hadronic_table : EOSTableData
        Table from load_eos_table(..., eq_type).
        eq_type determines which solver is used:
        'beta_eq'           -> 2D (n_B, T)
        'trapped_neutrinos' -> 3D (n_B, Y_L, T)
        'fixed_yc'          -> 3D (n_B, Y_C, T)
    save_table : bool
        If True, export results to a text file.
    output_file : str or None
        Path for the exported file.  Defaults to 'Qstar_<eq_type>.dat'.

    Returns
    -------
    QstarFrozenTableData
    """
    solvers = {
        'beta_eq': compute_Qstar_table_betaeq,
        'trapped_neutrinos': compute_Qstar_table_trapped,
        'fixed_yc': compute_Qstar_table_fixedYC,
    }
    if hadronic_table.eq_type not in solvers:
        raise ValueError(f"Unsupported eq_type: '{hadronic_table.eq_type}'. "
                         f"Valid: {list(solvers.keys())}")

    result = solvers[hadronic_table.eq_type](
        hadronic_table, params, electric_charge_neutrality,
        include_photons=include_photons,
        include_gluons=include_gluons,
        include_thermal_neutrinos=include_thermal_neutrinos,
        initial_guess=initial_guess,
        verbose=verbose,
    )

    if save_table:
        if output_file is None:
            output_file = f"Qstar_{result.eq_type}.dat"
        _export_table(result, params, electric_charge_neutrality, output_file)

    return result


# =============================================================================
# Grid computation: beta-equilibrium (2D: n_B_H, T)
# =============================================================================
def compute_Qstar_table_betaeq(hadronic_table, params,
                               electric_charge_neutrality,
                               include_photons=True, include_gluons=True,
                               include_thermal_neutrinos=True,
                               initial_guess=None, verbose=False):
    """
    Parameters
    ----------
    hadronic_table : EOSTableData
        Table from load_eos_table(..., 'beta_eq').
        grids: 'n_B', 'T'
        data:  'Y_C', 'Y_S', 'mu_B', 'mu_C', 'mu_S', 'mu_e'

    Returns
    -------
    QstarFrozenTableData
    """
    n_B_arr = hadronic_table.grids['n_B']
    T_arr = hadronic_table.grids['T']
    d = hadronic_table.data
    shape = (len(n_B_arr), len(T_arr))
    data = _init_data(shape)
    row_start_guess = initial_guess
    computed = 0
    total = shape[0] * shape[1]

    for i_T in range(len(T_arr)):
        current_guess = row_start_guess
        row_converged = 0
        for i_nB in range(len(n_B_arr)):
            idx = (i_nB, i_T)

            Y_C_H = d['Y_C'][i_nB, i_T]
            H = SimpleNamespace(
                T=T_arr[i_T],
                Y_C=Y_C_H,
                Y_S=d['Y_S'][i_nB, i_T],
                Y_e=Y_C_H,   # charge neutrality in H: Y_e = Y_C
                Y_nu=0.0,
                mu_B=d['mu_B'][i_nB, i_T],
                mu_C=d['mu_C'][i_nB, i_T],
                mu_S=d['mu_S'][i_nB, i_T],
                mu_e=d['mu_e'][i_nB, i_T],
                mu_nu=0.0,
            )

            Qs = solve_Qstar(H, params, electric_charge_neutrality,
                             include_photons, include_gluons,
                             include_thermal_neutrinos, current_guess)

            computed += 1
            current_guess = _store_result(data, idx, Qs, current_guess)
            if Qs is not None:
                row_converged += 1
                if i_nB == 0:
                    row_start_guess = current_guess

        if verbose:
            print(f"  [{i_T+1}/{len(T_arr)}] T={T_arr[i_T]:.1f} -> {row_converged}/{len(n_B_arr)} converged")

    hadronic_grids = {'n_B_H': n_B_arr, 'T': T_arr}
    return QstarFrozenTableData(eq_type='betaeq', hadronic_grids=hadronic_grids, data=data)


# =============================================================================
# Grid computation: trapped neutrinos (3D: n_B_H, Y_L_H, T)
# =============================================================================
def compute_Qstar_table_trapped(hadronic_table, params,
                                electric_charge_neutrality,
                                include_photons=True, include_gluons=True,
                                include_thermal_neutrinos=True,
                                initial_guess=None, verbose=False):
    """
    Parameters
    ----------
    hadronic_table : EOSTableData
        Table from load_eos_table(..., 'trapped_neutrinos').
        grids: 'n_B', 'Y_L', 'T'
        data:  'Y_C', 'Y_S', 'mu_B', 'mu_C', 'mu_S', 'mu_e', 'mu_nu'

    Returns
    -------
    QstarFrozenTableData
    """
    n_B_arr = hadronic_table.grids['n_B']
    Y_L_arr = hadronic_table.grids['Y_L']
    T_arr = hadronic_table.grids['T']
    d = hadronic_table.data
    shape = (len(n_B_arr), len(Y_L_arr), len(T_arr))
    data = _init_data(shape)
    row_start_guess = initial_guess
    computed = 0
    total = shape[0] * shape[1] * shape[2]

    for i_YL in range(len(Y_L_arr)):
        for i_T in range(len(T_arr)):
            current_guess = row_start_guess
            row_converged = 0
            for i_nB in range(len(n_B_arr)):
                idx = (i_nB, i_YL, i_T)

                Y_C_H = d['Y_C'][i_nB, i_YL, i_T]
                Y_L_H = Y_L_arr[i_YL]
                H = SimpleNamespace(
                    T=T_arr[i_T],
                    Y_C=Y_C_H,
                    Y_S=d['Y_S'][i_nB, i_YL, i_T],
                    Y_e=Y_C_H,           # charge neutrality in H: Y_e = Y_C
                    Y_nu=Y_L_H - Y_C_H,  # Y_nu = Y_L - Y_e = Y_L - Y_C
                    mu_B=d['mu_B'][i_nB, i_YL, i_T],
                    mu_C=d['mu_C'][i_nB, i_YL, i_T],
                    mu_S=d['mu_S'][i_nB, i_YL, i_T],
                    mu_e=d['mu_e'][i_nB, i_YL, i_T],
                    mu_nu=d['mu_nu'][i_nB, i_YL, i_T],
                )

                Qs = solve_Qstar(H, params, electric_charge_neutrality,
                                 include_photons, include_gluons,
                                 include_thermal_neutrinos, current_guess)

                computed += 1
                current_guess = _store_result(data, idx, Qs, current_guess)
                if Qs is not None:
                    row_converged += 1
                    if i_nB == 0:
                        row_start_guess = current_guess

            if verbose:
                step = i_YL * len(T_arr) + i_T + 1
                n_steps = len(Y_L_arr) * len(T_arr)
                print(f"  [{step}/{n_steps}] Y_L_H={Y_L_H:.2f}, T={T_arr[i_T]:.1f} -> {row_converged}/{len(n_B_arr)} converged")

    hadronic_grids = {'n_B_H': n_B_arr, 'Y_L_H': Y_L_arr, 'T': T_arr}
    return QstarFrozenTableData(eq_type='trapped', hadronic_grids=hadronic_grids, data=data)


# =============================================================================
# Grid computation: fixed Y_C (3D: n_B_H, Y_C_H, T)
# =============================================================================
def compute_Qstar_table_fixedYC(hadronic_table, params,
                                electric_charge_neutrality,
                                include_photons=True, include_gluons=True,
                                include_thermal_neutrinos=True,
                                initial_guess=None, verbose=False):
    """
    Parameters
    ----------
    hadronic_table : EOSTableData
        Table from load_eos_table(..., 'fixed_yc').
        grids: 'n_B', 'Y_C', 'T'
        data:  'Y_S', 'mu_B', 'mu_C', 'mu_S', 'mu_e'

    Returns
    -------
    QstarFrozenTableData
    """
    n_B_arr = hadronic_table.grids['n_B']
    Y_C_arr = hadronic_table.grids['Y_C']
    T_arr = hadronic_table.grids['T']
    d = hadronic_table.data
    shape = (len(n_B_arr), len(Y_C_arr), len(T_arr))
    data = _init_data(shape)
    row_start_guess = initial_guess
    computed = 0
    total = shape[0] * shape[1] * shape[2]

    for i_YC in range(len(Y_C_arr)):
        for i_T in range(len(T_arr)):
            current_guess = row_start_guess
            row_converged = 0
            for i_nB in range(len(n_B_arr)):
                idx = (i_nB, i_YC, i_T)

                H = SimpleNamespace(
                    T=T_arr[i_T],
                    Y_C=Y_C_arr[i_YC],
                    Y_S=d['Y_S'][i_nB, i_YC, i_T],
                    Y_e=Y_C_arr[i_YC],   # charge neutrality in H: Y_e = Y_C
                    Y_nu=0.0,
                    mu_B=d['mu_B'][i_nB, i_YC, i_T],
                    mu_C=d['mu_C'][i_nB, i_YC, i_T],
                    mu_S=d['mu_S'][i_nB, i_YC, i_T],
                    mu_e=d['mu_e'][i_nB, i_YC, i_T],
                    mu_nu=0.0,
                )

                Qs = solve_Qstar(H, params, electric_charge_neutrality,
                                 include_photons, include_gluons,
                                 include_thermal_neutrinos, current_guess)

                computed += 1
                current_guess = _store_result(data, idx, Qs, current_guess)
                if Qs is not None:
                    row_converged += 1
                    if i_nB == 0:
                        row_start_guess = current_guess

            if verbose:
                step = i_YC * len(T_arr) + i_T + 1
                n_steps = len(Y_C_arr) * len(T_arr)
                print(f"  [{step}/{n_steps}] Y_C_H={Y_C_arr[i_YC]:.2f}, T={T_arr[i_T]:.1f} -> {row_converged}/{len(n_B_arr)} converged")

    hadronic_grids = {'n_B_H': n_B_arr, 'Y_C_H': Y_C_arr, 'T': T_arr}
    return QstarFrozenTableData(eq_type='fixedYC', hadronic_grids=hadronic_grids, data=data)


# =============================================================================
# Helpers
# =============================================================================
def _init_data(shape):
    return {
        'n_B': np.full(shape, np.nan),
        'mu_B': np.full(shape, np.nan),
        'mu_C': np.full(shape, np.nan),
        'mu_S': np.full(shape, np.nan),
        'mu_u': np.full(shape, np.nan),
        'mu_d': np.full(shape, np.nan),
        'mu_s': np.full(shape, np.nan),
        'mu_e': np.full(shape, np.nan),
        'Y_C': np.full(shape, np.nan),
        'Y_S': np.full(shape, np.nan),
        'Y_u': np.full(shape, np.nan),
        'Y_d': np.full(shape, np.nan),
        'Y_s': np.full(shape, np.nan),
        'Y_e': np.full(shape, np.nan),
        'P_total': np.full(shape, np.nan),
        'e_total': np.full(shape, np.nan),
        's_total': np.full(shape, np.nan),
        'f_total': np.full(shape, np.nan),
        'converged': np.zeros(shape, dtype=bool),
    }


def _store_result(data, idx, Qs, current_guess):
    if Qs is not None:
        data['n_B'][idx] = Qs.n_B
        data['mu_B'][idx] = Qs.mu_B
        data['mu_C'][idx] = Qs.mu_C
        data['mu_S'][idx] = Qs.mu_S
        data['mu_u'][idx] = Qs.mu_u
        data['mu_d'][idx] = Qs.mu_d
        data['mu_s'][idx] = Qs.mu_s
        data['mu_e'][idx] = Qs.mu_e
        data['Y_C'][idx] = Qs.Y_C
        data['Y_S'][idx] = Qs.Y_S
        data['Y_u'][idx] = Qs.Y_u
        data['Y_d'][idx] = Qs.Y_d
        data['Y_s'][idx] = Qs.Y_s
        data['Y_e'][idx] = Qs.Y_e
        data['P_total'][idx] = Qs.P_total
        data['e_total'][idx] = Qs.e_total
        data['s_total'][idx] = Qs.s_total
        data['f_total'][idx] = Qs.f_total
        data['converged'][idx] = True
        current_guess = np.array([Qs.mu_u, Qs.mu_d, Qs.mu_s, Qs.mu_e])
    return current_guess



def _export_table(result, params, electric_charge_neutrality, output_file):
    """Export Q* results to a text file with input grid + output columns."""
    axes = QSTAR_GRID_AXES[result.eq_type]
    grid_arrays = [result.hadronic_grids[ax] for ax in axes]
    mesh = np.meshgrid(*grid_arrays, indexing='ij')
    # n_B varies fastest (inner loop), T slowest (outer loop)
    input_cols = [m.ravel(order='F') for m in mesh]

    # Output columns (Q* thermodynamic quantities)
    data_keys = ['n_B', 'mu_B', 'mu_C', 'mu_S', 'mu_u', 'mu_d', 'mu_s', 'mu_e',
                 'Y_C', 'Y_S', 'Y_u', 'Y_d', 'Y_s', 'Y_e',
                 'P_total', 'e_total', 's_total', 'f_total', 'converged']
    output_names = [k + '_Qs' for k in data_keys]
    output_cols = [result.data[k].ravel(order='F') for k in data_keys]

    all_names = list(axes) + output_names
    all_cols  = np.column_stack(input_cols + output_cols)

    # Metadata header
    col_header = "  ".join(f"{name:>14s}" for name in all_names)
    meta = (
        f"# Q* flavor-frozen nucleation table\n"
        f"# hadronic_eq_type: {result.eq_type}\n"
        f"# electric_charge_neutrality: {electric_charge_neutrality}\n"
        f"# quark_eos: {params.name}\n"
        f"# m_u={params.m_u} MeV  m_d={params.m_d} MeV  m_s={params.m_s} MeV"
        f"  alpha={params.alpha}  B4={params.B4} MeV\n"
        f"# {col_header}\n"
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(meta)
    with open(output_file, 'ab') as f:
        np.savetxt(f, all_cols, fmt='%16.8e')
    print(f"  Saved Q* table ({all_cols.shape[0]} rows) -> {output_file}")
