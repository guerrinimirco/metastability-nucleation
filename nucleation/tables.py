"""
Grid tables & I/O
=================

Grid-level drivers that sweep the critical-droplet physics over a hadronic
(n_B_H, [Y_L_H | Y_C_H], T) grid, plus the .dat readers/writers. Everything
here is "the same single-point physics, many times, warm-started", built on
``nucleation.composition`` (Q* solves), ``nucleation.barrier`` (R_c / W_c),
``nucleation.critical`` (unpCFL peak logic) and ``nucleation.rates`` (Gamma, tau).

Public API
----------
    QstarTableData, GRID_AXES
    compute_Qstar_table, build_Qstar_interpolators, load_Qstar_table, export_table
    ThermalNucleationObservables, compute_thermal_nucleation_observables
    build_thermal_nucleation_interpolators
    export_thermal_nucleation_table, load_thermal_nucleation_table
    QuantumNucleationObservables, compute_quantum_nucleation_observables

Grid W_c convention: NaN where a point did not converge / has no critical
droplet (this is what round-trips through the .dat files). The engine's +inf
"H-stable" convention lives in ``nucleation.critical`` / sigma_crit only.
"""

import os
import numpy as np
from types import SimpleNamespace
from dataclasses import dataclass
from scipy.interpolate import RegularGridInterpolator

from nucleation.barrier import (
    driving_force, bulk_W, surface_W, coulomb_W, coulomb_screened_W,
    work_of_formation, critical_work_noCoulomb, critical_radius_noCoulomb,
    critical_radius_coulomb, critical_radius_coulomb_screened,
    electron_susceptibility, debye_length, get_switching_function,
)
from nucleation.composition import (
    get_solver_Qs, R_GCN_SKIP_DEFAULT,
    solve_coulomb_minimize_at_R, solve_coulomb_minimize_cfl_at_R,
)
from nucleation.critical import _find_Rc_Wc_step, _find_Rc_Wc_tanh, _blend_phase
from nucleation.rates import nucleation_rate, nucleation_time


# =============================================================================
# Constants
# =============================================================================
GRID_AXES = {
    'beta_eq':           ['n_B_H', 'T'],
    'trapped_neutrinos': ['n_B_H', 'Y_L_H', 'T'],
    'fixed_yc':          ['n_B_H', 'Y_C_H', 'T'],
}

# Backward compat for old .dat files
_EQ_TYPE_ALIASES = {
    'betaeq': 'beta_eq',
    'trapped': 'trapped_neutrinos',
    'fixedYC': 'fixed_yc',
}

_BASE_DATA_KEYS = [
    'n_B', 'mu_B', 'mu_C', 'mu_S', 'mu_u', 'mu_d', 'mu_s', 'mu_e',
    'Y_C', 'Y_S', 'Y_u', 'Y_d', 'Y_s', 'Y_e',
    'P_total', 'e_total', 's_total', 'f_total',
]
_COULOMB_EXTRA_KEYS = ['R_c']
_THERMAL_DATA_KEYS = ['R_c', 'W_c', 'Gamma', 'tau', 'converged']
# Quantum (WKB) grid columns. Same on-disk shape as the thermal table: axis
# columns first, then these, Fortran-order ravelled.
_QUANTUM_DATA_KEYS = ['R_c', 'W_c', 'tau_qt', 'A', 'E_0', 'nu_0',
                      'R_inner', 'R_outer', 'm_0', 'E_min', 'converged']

# charge modes whose Coulomb term uses the droplet net charge delta_n_C
_COULOMB_MODES = ('gcn_coulomb', 'coulomb_minimize', 'screening')


# =============================================================================
# Unified Q* container
# =============================================================================
@dataclass
class QstarTableData:
    """Q* nucleation table: hadronic conditions -> Q* droplet thermodynamics."""
    eq_type: str
    hadronic_grids: dict
    data: dict
    filepath: str = ""

    def __repr__(self):
        shapes = [f"{k}={len(v)}" for k, v in self.hadronic_grids.items()]
        return f"QstarTableData(eq_type='{self.eq_type}', grids=({', '.join(shapes)})"


# =============================================================================
# Grid iteration / expansion helpers  (internal, dedup the eq_type boilerplate)
# =============================================================================
def _grid_layout(hadronic_table):
    """(shape, grid_out, outer_key, outer_arr, n_B_arr, T_arr) for an eq_type."""
    eq_type = hadronic_table.eq_type
    grids = hadronic_table.grids
    n_B_arr, T_arr = grids['n_B'], grids['T']
    if eq_type == 'beta_eq':
        return ((len(n_B_arr), len(T_arr)),
                {'n_B_H': n_B_arr, 'T': T_arr}, None, None, n_B_arr, T_arr)
    if eq_type == 'trapped_neutrinos':
        outer = grids['Y_L']
        return ((len(n_B_arr), len(outer), len(T_arr)),
                {'n_B_H': n_B_arr, 'Y_L_H': outer, 'T': T_arr},
                'Y_L', outer, n_B_arr, T_arr)
    if eq_type == 'fixed_yc':
        outer = grids['Y_C']
        return ((len(n_B_arr), len(outer), len(T_arr)),
                {'n_B_H': n_B_arr, 'Y_C_H': outer, 'T': T_arr},
                'Y_C', outer, n_B_arr, T_arr)
    raise ValueError(f"Unsupported eq_type: '{hadronic_table.eq_type}'")


def expand_grid(hadronic_table):
    """Build the full-shape hadronic namespace H (+ mesh axes) once.

    Replaces the meshgrid + SimpleNamespace block formerly repeated across the
    thermal / quantum / unpCFL drivers. Returns (H, shape, grid_out, T_H,
    mu_nu_H, Y_nu_H).
    """
    h_d = hadronic_table.data
    grids = hadronic_table.grids
    eq_type = hadronic_table.eq_type
    shape, grid_out, _, _, _, _ = _grid_layout(hadronic_table)

    if eq_type == 'beta_eq':
        n_B_H, T_H = np.meshgrid(grids['n_B'], grids['T'], indexing='ij')
        mu_nu_H = np.zeros(shape)
        Y_nu_H = np.zeros(shape)
    elif eq_type == 'trapped_neutrinos':
        n_B_H, Y_L_H, T_H = np.meshgrid(
            grids['n_B'], grids['Y_L'], grids['T'], indexing='ij')
        mu_nu_H = h_d['mu_nu']
        Y_nu_H = Y_L_H - h_d['Y_C']
    else:  # fixed_yc
        n_B_H, _, T_H = np.meshgrid(
            grids['n_B'], grids['Y_C'], grids['T'], indexing='ij')
        mu_nu_H = np.zeros(shape)
        Y_nu_H = np.zeros(shape)

    H = SimpleNamespace(**{
        **h_d, 'n_B': n_B_H, 'T': T_H, 'Y_e': h_d['Y_C'],
        'mu_nu': mu_nu_H, 'Y_nu': Y_nu_H})
    return H, shape, grid_out, T_H, mu_nu_H, Y_nu_H


def _lambda_D_grid(mu_e_arr, T_arr):
    """Debye length lambda_D at each grid point (screening mode).

    electron_thermo is scalar, so we loop; two EOS calls/point, negligible.
    """
    mu_e_arr = np.asarray(mu_e_arr, dtype=float)
    T_arr = np.asarray(T_arr, dtype=float)
    out = np.full(mu_e_arr.shape, np.nan)
    it = np.nditer(mu_e_arr, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        mu = float(mu_e_arr[idx])
        T = float(T_arr[idx])
        if np.isfinite(mu) and np.isfinite(T) and T > 0:
            chi = electron_susceptibility(mu, T)
            if chi > 0:
                out[idx] = debye_length(chi)
        it.iternext()
    return out


# =============================================================================
# Data storage helpers  (internal)
# =============================================================================
def _init_data(shape):
    """NaN-filled output arrays for a Q* table computation."""
    data = {k: np.full(shape, np.nan) for k in _BASE_DATA_KEYS}
    data['converged'] = np.zeros(shape, dtype=bool)
    data['R_c'] = np.full(shape, np.nan)
    return data


def _store_result(data, idx, result, current_guess):
    """Store one solver output; return updated warm-start guess."""
    if result is None:
        return current_guess
    if isinstance(result, tuple):        # coulomb_minimize: (result, R_c)
        eos_result, R_c = result
        data['R_c'][idx] = R_c
        guess_extra = [R_c]
    else:
        eos_result = result
        guess_extra = []
    for key in _BASE_DATA_KEYS:
        data[key][idx] = getattr(eos_result, key)
    data['converged'][idx] = True
    return np.array(
        [eos_result.mu_u, eos_result.mu_d, eos_result.mu_s, eos_result.mu_e]
        + guess_extra)


def _compute_and_store_Rc(data, idx, result, H, electric_charge_mode, sigma,
                          lambda_D=None):
    """Derive R_c analytically for non-coulomb_minimize modes and store it."""
    Qs_point = SimpleNamespace(
        P_total=result.P_total, n_B=result.n_B, mu_B=result.mu_B,
        mu_C=result.mu_C, mu_S=result.mu_S, mu_e=result.mu_e, mu_nu=H.mu_nu,
        Y_C=result.Y_C, Y_S=result.Y_S,
        Y_e=getattr(result, 'Y_e', result.Y_C), Y_nu=H.Y_nu)
    Delta_f = driving_force(Qs_point, H)
    if Delta_f >= 0:
        return                            # H not metastable, R_c stays NaN
    if electric_charge_mode in ('lcn', 'gcn'):
        data['R_c'][idx] = float(critical_radius_noCoulomb(Delta_f, sigma))
    elif electric_charge_mode == 'gcn_coulomb':
        dnC = (result.Y_C - getattr(result, 'Y_e', result.Y_C)) * result.n_B
        data['R_c'][idx] = float(critical_radius_coulomb(Delta_f, sigma, dnC))
    elif electric_charge_mode == 'screening':
        dnC = (result.Y_C - getattr(result, 'Y_e', result.Y_C)) * result.n_B
        data['R_c'][idx] = float(
            critical_radius_coulomb_screened(Delta_f, sigma, dnC, lambda_D))


def _build_H_from_table(hadronic_table, eq_type, idx, grid_values):
    """Hadronic SimpleNamespace at a single grid point."""
    d = hadronic_table.data
    T = grid_values['T']
    if eq_type == 'beta_eq':
        Y_C_H = d['Y_C'][idx]
        Y_nu = mu_nu = 0.0
    elif eq_type == 'trapped_neutrinos':
        Y_C_H = d['Y_C'][idx]
        Y_nu = grid_values['Y_L'] - Y_C_H
        mu_nu = d['mu_nu'][idx]
    elif eq_type == 'fixed_yc':
        Y_C_H = grid_values['Y_C']
        Y_nu = mu_nu = 0.0
    else:
        raise ValueError(f"Unsupported eq_type: '{eq_type}'")
    H = SimpleNamespace(
        T=T, Y_C=Y_C_H, Y_S=d['Y_S'][idx], Y_e=Y_C_H, Y_nu=Y_nu,
        mu_B=d['mu_B'][idx], mu_C=d['mu_C'][idx], mu_S=d['mu_S'][idx],
        mu_e=d['mu_e'][idx], mu_nu=mu_nu, P_total=d['P_total'][idx])
    return H


# =============================================================================
# Generic Q* grid computation  (public)
# =============================================================================
def compute_Qstar_table(hadronic_table, flavor_mode, electric_charge_mode,
                        params, sigma, quark_phase='unpaired', Delta0=None,
                        initial_guess=None, verbose=False,
                        save_table=False, output_file=None,
                        include_photons=True, include_gluons=True,
                        include_thermal_neutrinos=True, R_gcn_skip=None):
    """Compute the Q* table over the hadronic grid for any eq_type.

    Warm-started (n_B fastest inside each (outer, T) row). For non-coulomb_minimize
    modes R_c is derived analytically per point; coulomb_minimize co-solves R.
    'screening' uses the GCN composition with a Debye-screened critical radius.
    """
    if R_gcn_skip is None:
        R_gcn_skip = R_GCN_SKIP_DEFAULT

    solver_fn = get_solver_Qs(
        flavor_mode=flavor_mode, electric_charge_mode=electric_charge_mode,
        params=params, quark_phase=quark_phase, Delta0=Delta0, sigma=sigma,
        include_photons=include_photons, include_gluons=include_gluons,
        include_thermal_neutrinos=include_thermal_neutrinos,
        R_gcn_skip=R_gcn_skip)
    include_coulomb = (electric_charge_mode == 'coulomb_minimize')

    eq_type = hadronic_table.eq_type
    shape, grid_out, outer_key, outer_arr, n_B_arr, T_arr = _grid_layout(
        hadronic_table)

    data = _init_data(shape)
    row_start_guess = initial_guess
    n_outer = len(outer_arr) if outer_arr is not None else 1

    for i_outer in range(n_outer):
        for i_T in range(len(T_arr)):
            current_guess = row_start_guess
            row_converged = 0
            for i_nB in range(len(n_B_arr)):
                if eq_type == 'beta_eq':
                    idx = (i_nB, i_T)
                    gv = {'T': T_arr[i_T]}
                elif eq_type == 'trapped_neutrinos':
                    idx = (i_nB, i_outer, i_T)
                    gv = {'T': T_arr[i_T], 'Y_L': outer_arr[i_outer]}
                else:  # fixed_yc
                    idx = (i_nB, i_outer, i_T)
                    gv = {'T': T_arr[i_T], 'Y_C': outer_arr[i_outer]}

                H = _build_H_from_table(hadronic_table, eq_type, idx, gv)
                result = solver_fn(H, initial_guess=current_guess)
                current_guess = _store_result(data, idx, result, current_guess)
                if result is not None:
                    if not include_coulomb and sigma is not None:
                        lam = None
                        if electric_charge_mode == 'screening':
                            chi = electron_susceptibility(H.mu_e, H.T)
                            lam = debye_length(chi) if chi > 0 else np.nan
                        _compute_and_store_Rc(
                            data, idx, result, H, electric_charge_mode, sigma,
                            lambda_D=lam)
                    row_converged += 1
                    if i_nB == 0:
                        row_start_guess = current_guess
            if verbose:
                if outer_arr is not None:
                    step = i_outer * len(T_arr) + i_T + 1
                    n_steps = n_outer * len(T_arr)
                    label = f"{outer_key}_H={outer_arr[i_outer]:.2f}, "
                else:
                    step = i_T + 1
                    n_steps = len(T_arr)
                    label = ""
                print(f"  [{step}/{n_steps}] {label}T={T_arr[i_T]:.1f} "
                      f"-> {row_converged}/{len(n_B_arr)} converged")

    table = QstarTableData(eq_type=eq_type, hadronic_grids=grid_out, data=data)
    if save_table:
        if output_file is None:
            output_file = f"Qstar_{eq_type}.dat"
        charge_neutrality = 'local' if electric_charge_mode == 'lcn' else 'global'
        export_table(table, params, output_file,
                     charge_neutrality=charge_neutrality, sigma=sigma)
    return table


# =============================================================================
# Q* interpolation + I/O  (public)
# =============================================================================
def build_Qstar_interpolators(table, method='linear'):
    """Interpolation dict for a Q* table (keys: n_B, P, mu_*, Y_*, ...)."""
    axes = GRID_AXES[table.eq_type]
    grid_tuple = tuple(table.hadronic_grids[ax] for ax in axes)
    interpolators = {}
    for name, arr in table.data.items():
        if name == 'converged':
            continue
        interpolators[name] = RegularGridInterpolator(
            grid_tuple, arr, method=method, bounds_error=False, fill_value=np.nan)
    result = {
        'interpolators': interpolators,
        'hadronic_grids': table.hadronic_grids,
        'axes': axes,
        'eq_type': table.eq_type,
    }
    for key in interpolators:
        result[key] = (lambda k: lambda *a: interpolators[k](a))(key)
    result['mu_B'] = lambda *a: interpolators['mu_u'](a) + 2 * interpolators['mu_d'](a)
    result['mu_C'] = lambda *a: interpolators['mu_u'](a) - interpolators['mu_d'](a)
    result['mu_S'] = lambda *a: interpolators['mu_s'](a) - interpolators['mu_d'](a)
    result['P'] = result['P_total']
    result['eps'] = result['e_total']
    result['s'] = result['s_total']
    result['f'] = result['f_total']
    return result


def load_Qstar_table(filepath):
    """Load a Q* table from a .dat file (auto-detects columns, old eq_type names)."""
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
    eq_type = _EQ_TYPE_ALIASES.get(eq_type, eq_type)
    axes = GRID_AXES[eq_type]
    n_axes = len(axes)
    raw = np.loadtxt(filepath)
    hadronic_grids = {}
    shape = []
    for i, ax in enumerate(axes):
        vals = np.unique(raw[:, i])
        hadronic_grids[ax] = vals
        shape.append(len(vals))
    shape = tuple(shape)
    data_col_names = col_names[n_axes:]
    data = {}
    for j, cname in enumerate(data_col_names):
        key = cname.replace('_Qs', '')
        arr = raw[:, n_axes + j].reshape(shape, order='F')
        if key == 'converged':
            arr = arr.astype(bool)
        data[key] = arr
    if 'Y_e' not in data and 'Y_C' in data:
        data['Y_e'] = data['Y_C'].copy()
    return QstarTableData(eq_type=eq_type, hadronic_grids=hadronic_grids,
                          data=data, filepath=filepath)


def export_table(table, params, output_file, charge_neutrality=None, sigma=None):
    """Export a Q* table to a .dat file (Fortran-order flatten, metadata header)."""
    axes = GRID_AXES[table.eq_type]
    grid_arrays = [table.hadronic_grids[ax] for ax in axes]
    mesh = np.meshgrid(*grid_arrays, indexing='ij')
    input_cols = [m.ravel(order='F') for m in mesh]
    data_keys = [k for k in _BASE_DATA_KEYS + _COULOMB_EXTRA_KEYS
                 if k in table.data]
    data_keys.append('converged')
    output_names = [k + '_Qs' for k in data_keys]
    output_cols = [table.data[k].ravel(order='F') for k in data_keys]
    all_names = list(axes) + output_names
    all_cols = np.column_stack(input_cols + output_cols)
    header_lines = ["# Q* nucleation table",
                    f"# hadronic_eq_type: {table.eq_type}"]
    if charge_neutrality:
        header_lines.append(f"# electric_charge_neutrality: {charge_neutrality}")
    if sigma is not None:
        header_lines.append(f"# sigma: {sigma} MeV/fm^2")
    header_lines.append(
        f"# quark_eos: {params.name}\n"
        f"# m_u={params.m_u} MeV  m_d={params.m_d} MeV  m_s={params.m_s} MeV"
        f"  alpha={params.alpha}  B4={params.B4} MeV")
    col_header = "  ".join(f"{name:>14s}" for name in all_names)
    header_lines.append(f"# {col_header}")
    meta = "\n".join(header_lines) + "\n"
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(meta)
    with open(output_file, 'ab') as f:
        np.savetxt(f, all_cols, fmt='%16.8e')
    print(f"  Saved Q* table ({all_cols.shape[0]} rows) -> {output_file}")


###############################################################################
#                                                                             #
#  THERMAL nucleation observables (grid)                                      #
#                                                                             #
###############################################################################
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


###############################################################################
#                                                                             #
#  QUANTUM nucleation observables (grid)                                      #
#                                                                             #
###############################################################################
@dataclass
class QuantumNucleationObservables:
    """Grid-level quantum (WKB) nucleation observables.

    Mirrors ``ThermalNucleationObservables`` field-for-field where the two
    overlap, so the same downstream code (curves, interpolators, .dat I/O) works
    on either. In particular ``tau`` aliases ``tau_qt``: the thermal tau is
    1/(V*Gamma) while the quantum one is 1/(N_c*nu_0*exp(-A)), but both are "the
    time to nucleate one droplet in the star", which is what a tau=tau_target
    locus is asking for.
    """
    eq_type: str
    hadronic_grids: dict
    flavor_mode: str
    electric_charge_mode: str
    quark_phase: str
    sigma: float
    N_c: float
    R_c: np.ndarray          # barrier peak [fm] -- from the SAME solve as thermal
    W_c: np.ndarray          # barrier height [MeV], for comparison with W_c/T
    tau_qt: np.ndarray       # tunnelling time [s]
    A: np.ndarray            # WKB exponent at E_0 [dimensionless]
    E_0: np.ndarray          # ground-state energy [MeV]
    nu_0: np.ndarray         # small-oscillation frequency [s^-1]
    R_inner: np.ndarray      # inner turning point [fm]
    R_outer: np.ndarray      # outer turning point [fm]
    m_0: np.ndarray          # number of positive-energy bound levels
    E_min: np.ndarray        # lower edge of the positive-energy band [MeV]
    converged: np.ndarray
    Qstar_table: object = None

    @property
    def tau(self):
        """Alias so quantum observables drop into the tau=tau_target finders."""
        return self.tau_qt


def compute_quantum_nucleation_observables(
        hadronic_table, sigma=30.0, params=None, Qstar_table=None,
        N_c=1e48, quark_phase='unpaired', Delta0=None,
        flavor_mode='saddlepoint', electric_charge_mode='gcn',
        include_photons=True, include_gluons=True, include_thermal_neutrinos=True,
        rho_H_func=None, initial_guess=None, verbose=False,
        save_table=False, output_file=None, R_gcn_skip=None,
        switching_mode='step', Rx=None, switching_width=None,
        Qstar_table_unp=None, Qstar_table_cfl=None,
        params_unp=None, params_cfl=None):
    """Quantum tunnelling nucleation time tau_qt over the hadronic grid (WKB).

    The zero-temperature counterpart of
    ``compute_thermal_nucleation_observables``, and deliberately built from the
    same pieces: the SAME Q* table, the SAME driving force, and the SAME barrier
    peak R_c. Only the escape mechanism differs -- Langer's thermal activation
    over the barrier is replaced by tunnelling through it, so tau depends on the
    whole shape of W(R) rather than on its peak alone.

    Physics: a droplet of radius R carries an effective inertia M(R) from the
    hadronic flow it must displace (``rates.effective_inertia``). Bohr-Sommerfeld
    quantization of the sub-barrier motion fixes the ground-state energy E_0;
    the WKB exponent A(E_0) across the classically forbidden region then gives
    tau_qt = 1 / (N_c nu_0 exp(-A)).

    Mechanics:
      sigma         : surface tension [MeV/fm^2] -- sets the barrier scale.
      params        : alpha-Bag parameters; required when Qstar_table is None.
      Qstar_table   : reuse a previously computed Q* table (skips the solve).
      N_c           : number of independent nucleation centres in the star.
      quark_phase   : 'unpaired', 'cfl' or 'unpCFL' (needs Rx and Delta0).
      electric_charge_mode : 'lcn'/'gcn' (no Coulomb), 'gcn_coulomb',
                      'coulomb_minimize' (R_c from the self-consistent 5-eq
                      solve) or 'screening' (Debye-screened Coulomb).
      rho_H_func    : rho_H(n_B, T) [MeV/fm^3]; defaults to m_neutron * n_B.
      switching_mode/Rx/switching_width : unpCFL barrier shape, as thermal.

    Returns a ``QuantumNucleationObservables``. Points with no critical droplet
    (Delta_f >= 0, i.e. the hadronic phase is stable) are left NaN / not
    converged.
    """
    from eos.general.physics_constants import m_neutron
    from nucleation.rates import effective_inertia, quantum_nucleation_time

    valid_flavor = ('frozen', 'saddlepoint')
    if flavor_mode not in valid_flavor:
        raise ValueError(f"Invalid flavor_mode: '{flavor_mode}'.")
    if electric_charge_mode not in ('lcn', 'gcn') + _COULOMB_MODES:
        raise ValueError(f"Invalid electric_charge_mode: '{electric_charge_mode}'")
    if quark_phase not in ('unpaired', 'cfl', 'unpCFL'):
        raise ValueError(f"Invalid quark_phase: '{quark_phase}'")
    if quark_phase == 'unpCFL':
        if Rx is None:
            raise ValueError("Rx required for quark_phase='unpCFL'")
        if Delta0 is None:
            raise ValueError("Delta0 required for quark_phase='unpCFL'")
        if switching_mode == 'tanh' and switching_width is None:
            raise ValueError("switching_width required for switching_mode='tanh'")

    common_kw = dict(
        include_photons=include_photons, include_gluons=include_gluons,
        include_thermal_neutrinos=include_thermal_neutrinos,
        initial_guess=initial_guess, verbose=verbose)

    H, shape, _, T_H, mu_nu_H, Y_nu_H = expand_grid(hadronic_table)
    # Debye length is a property of the ambient hadronic matter, so it is
    # computed once per grid point regardless of phase.
    lam_grid = (_lambda_D_grid(H.mu_e, T_H)
                if electric_charge_mode == 'screening' else None)

    def _phase_fields(table, phase, pars, D0):
        """(Q* table, Delta_f, delta_n_C, R_c) for one quark phase."""
        if table is None:
            if pars is None:
                raise ValueError(
                    f"params required to build the {phase} Q* table")
            if verbose:
                print(f"Computing {phase} Q* table "
                      f"(flavor={flavor_mode}, charge={electric_charge_mode})...")
            table = compute_Qstar_table(
                hadronic_table, flavor_mode=flavor_mode,
                electric_charge_mode=electric_charge_mode, params=pars,
                quark_phase=phase, Delta0=D0, sigma=sigma,
                R_gcn_skip=R_gcn_skip, **common_kw)
        d = table.data
        Qs = SimpleNamespace(**{**d, 'T': T_H, 'mu_nu': mu_nu_H, 'Y_nu': Y_nu_H})
        df = driving_force(Qs, H)
        dnC = (np.zeros(shape) if electric_charge_mode in ('lcn', 'gcn')
               else (Qs.Y_C - Qs.Y_e) * Qs.n_B)
        return table, Qs, df, dnC, d['R_c']

    if quark_phase == 'unpCFL':
        tab_unp, Qs_unp, df_unp, dnC_unp, Rc_unp = _phase_fields(
            Qstar_table_unp, 'unpaired',
            params_unp if params_unp is not None else params, None)
        tab_cfl, Qs_cfl, df_cfl, dnC_cfl, Rc_cfl = _phase_fields(
            Qstar_table_cfl, 'cfl',
            params_cfl if params_cfl is not None else params, Delta0)
        # Barrier peak of the kinked composite -- identical logic to the thermal
        # driver, so thermal and quantum quote the same R_c at the same point.
        peak_kw = dict(R_c_unp=Rc_unp, R_c_cfl=Rc_cfl, Rx=Rx,
                       Delta_f_unp=df_unp, Delta_f_cfl=df_cfl,
                       delta_n_C_unp=dnC_unp, delta_n_C_cfl=dnC_cfl, sigma=sigma)
        if switching_mode == 'step':
            R_c_grid, W_c_grid, _ = _find_Rc_Wc_step(**peak_kw)
        elif switching_mode == 'tanh':
            S_func = get_switching_function(switching_mode, Rx, switching_width)
            R_c_grid, W_c_grid = _find_Rc_Wc_tanh(S_func=S_func, **peak_kw)
        else:
            raise ValueError(f"Invalid switching_mode: '{switching_mode}'")
        S_of_R = get_switching_function(switching_mode, Rx, switching_width)
        qstar_converged = tab_unp.data['converged'] & tab_cfl.data['converged']
        grids_src, Qs_for_ratio = tab_unp, Qs_unp
        primary_table = tab_unp
    else:
        tab, Qs, df, dnC, Rc = _phase_fields(
            Qstar_table, quark_phase, params, Delta0)
        qstar_converged = tab.data['converged']
        # R_c: for the Coulomb modes it comes from the self-consistent solve
        # (barrier.critical_radius_coulomb*), NOT the closed form -- using the
        # closed form there would put the turning points around the wrong peak.
        if electric_charge_mode in ('lcn', 'gcn'):
            R_c_grid = critical_radius_noCoulomb(df, sigma)
            W_c_grid = critical_work_noCoulomb(df, sigma)
        else:
            R_c_grid = Rc.copy()
            W_c_grid = np.where(
                qstar_converged & np.isfinite(R_c_grid),
                work_of_formation(R_c_grid, df, sigma, dnC, lambda_D=lam_grid),
                np.nan)
        grids_src, Qs_for_ratio = tab, Qs
        primary_table = tab

    out = {k: np.full(shape, np.nan) for k in
           ('tau_qt', 'A', 'E_0', 'nu_0', 'R_inner', 'R_outer', 'm_0', 'E_min')}
    conv_out = np.zeros(shape, dtype=bool)
    total = int(np.prod(shape))
    done = 0

    it = np.nditer(np.zeros(shape), flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        it.iternext()
        done += 1
        if verbose and done % max(1, total // 20) == 0:
            print(f"  Quantum nucleation: {done}/{total} points processed")
        if not qstar_converged[idx]:
            continue
        R_c = float(R_c_grid[idx])
        # Delta_f < 0 <=> the hadronic phase is METASTABLE <=> a critical
        # droplet exists (barrier.driving_force sign convention). The guard used
        # to read `Delta_f <= 0: continue`, which kept exactly the points with no
        # droplet and dropped every real one, so the driver converged 0/N always.
        if not np.isfinite(R_c) or R_c <= 0:
            continue

        n_B_H_val = float(H.n_B[idx])
        T_val = float(H.T[idx])
        if n_B_H_val <= 0:
            continue
        rho_H = (rho_H_func(n_B_H_val, T_val) if rho_H_func is not None
                 else m_neutron * n_B_H_val)
        n_B_ratio = float(Qs_for_ratio.n_B[idx]) / n_B_H_val
        lam = None if lam_grid is None else float(lam_grid[idx])

        if quark_phase == 'unpCFL':
            def W_func(R, _u=(float(df_unp[idx]), float(dnC_unp[idx])),
                       _c=(float(df_cfl[idx]), float(dnC_cfl[idx])),
                       _s=sigma, _lam=lam, _S=S_of_R):
                # Blend the two single-phase barriers with the switching
                # function, exactly as the thermal unpCFL driver does, so the
                # WKB integral sees the kinked composite rather than one phase.
                w_u = work_of_formation(R, _u[0], _s, _u[1], lambda_D=_lam)
                w_c = work_of_formation(R, _c[0], _s, _c[1], lambda_D=_lam)
                s = _S(R)
                return (1.0 - s) * w_u + s * w_c
        else:
            def W_func(R, _df=float(df[idx]), _s=sigma,
                       _dnC=float(dnC[idx]), _lam=lam):
                return work_of_formation(R, _df, _s, _dnC, lambda_D=_lam)

        def M_func(R, _rho=rho_H, _ratio=n_B_ratio):
            return effective_inertia(R, _rho, _ratio)

        try:
            res = quantum_nucleation_time(W_func, M_func, R_c, N_c=N_c)
        except Exception:
            # A point where the WKB pipeline finds no two real turning points
            # is genuinely non-tunnelling, not an error worth aborting the grid.
            continue
        out['tau_qt'][idx] = res.tau_qt
        out['A'][idx] = res.A
        out['E_0'][idx] = res.E_0
        out['nu_0'][idx] = res.nu_0
        out['R_inner'][idx] = res.R_inner
        out['R_outer'][idx] = res.R_outer
        out['m_0'][idx] = res.m_0
        out['E_min'][idx] = res.E_min
        conv_out[idx] = True

    if verbose:
        print(f"  Quantum nucleation complete: "
              f"{int(np.sum(conv_out))}/{total} converged")

    result = QuantumNucleationObservables(
        eq_type=hadronic_table.eq_type,
        hadronic_grids=grids_src.hadronic_grids,
        flavor_mode=flavor_mode, electric_charge_mode=electric_charge_mode,
        quark_phase=quark_phase, sigma=sigma, N_c=N_c,
        R_c=R_c_grid, W_c=W_c_grid, converged=conv_out,
        Qstar_table=primary_table, **out)
    if save_table:
        if output_file is None:
            output_file = f"quantum_nucleation_{hadronic_table.eq_type}.dat"
        export_quantum_nucleation_table(result, output_file)
    return result


def build_quantum_nucleation_interpolators(nucleation_obs, method='linear'):
    """Interpolation dict (tau_qt, A, E_0, nu_0, R_c, W_c, log10_tau_qt).

    ``log10_tau_qt`` is also exposed as ``log10_tau`` so the same key works for
    thermal and quantum observables -- that is what lets
    ``compute_nucleation_temperature`` accept either without branching.
    """
    grids = nucleation_obs.hadronic_grids
    axes = GRID_AXES[nucleation_obs.eq_type]
    grid_tuple = tuple(grids[ax] for ax in axes)
    result = {}
    for name in ('tau_qt', 'A', 'E_0', 'nu_0', 'R_c', 'W_c'):
        interp = RegularGridInterpolator(
            grid_tuple, getattr(nucleation_obs, name), method=method,
            bounds_error=False, fill_value=np.nan)
        result[name] = (lambda f: lambda *a: float(f(a)))(interp)
    # Same clipping as the thermal builder: tau spans hundreds of decades, so
    # interpolate log10(tau) and keep non-finite entries out of the spline.
    with np.errstate(divide='ignore', invalid='ignore'):
        log_tau = np.log10(nucleation_obs.tau_qt)
    log_tau = np.clip(log_tau, -50, 100)
    log_tau = np.where(np.isnan(log_tau), 100.0, log_tau)
    interp_log = RegularGridInterpolator(
        grid_tuple, log_tau, method=method, bounds_error=False, fill_value=np.nan)
    fn = (lambda f: lambda *a: float(f(a)))(interp_log)
    result['log10_tau_qt'] = fn
    result['log10_tau'] = fn
    return result


def export_quantum_nucleation_table(obs, output_file):
    """Export quantum observables to a .dat file (same layout as thermal)."""
    axes = GRID_AXES[obs.eq_type]
    mesh = np.meshgrid(*[obs.hadronic_grids[ax] for ax in axes], indexing='ij')
    input_cols = [m.ravel(order='F') for m in mesh]
    output_cols = [np.asarray(getattr(obs, k), dtype=float).ravel(order='F')
                   for k in _QUANTUM_DATA_KEYS]
    all_names = list(axes) + _QUANTUM_DATA_KEYS
    all_cols = np.column_stack(input_cols + output_cols)
    header_lines = ["# Quantum (WKB) nucleation observables",
                    f"# eq_type: {obs.eq_type}",
                    f"# flavor_mode: {obs.flavor_mode}",
                    f"# electric_charge_mode: {obs.electric_charge_mode}",
                    f"# quark_phase: {obs.quark_phase}",
                    f"# sigma: {obs.sigma} MeV/fm^2",
                    f"# N_c: {obs.N_c}"]
    header_lines.append("# " + "  ".join(f"{n:>16s}" for n in all_names))
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write("\n".join(header_lines) + "\n")
    with open(output_file, 'ab') as f:
        np.savetxt(f, all_cols, fmt='%16.8e')
    print(f"  Saved quantum nucleation table ({all_cols.shape[0]} rows) "
          f"-> {output_file}")


def load_quantum_nucleation_table(filepath):
    """Load quantum observables from a .dat file."""
    metadata = {}
    with open(filepath, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                break
            for key in ('eq_type', 'flavor_mode', 'electric_charge_mode',
                        'quark_phase', 'sigma', 'N_c'):
                tag = f"# {key}:"
                if line.startswith(tag):
                    val = line[len(tag):].strip()
                    for unit in ('MeV/fm^2', 'fm^3'):
                        val = val.replace(unit, '').strip()
                    metadata[key] = val
    eq_type = _EQ_TYPE_ALIASES.get(metadata['eq_type'], metadata['eq_type'])
    axes = GRID_AXES[eq_type]
    n_axes = len(axes)
    raw = np.loadtxt(filepath)
    hadronic_grids = {ax: np.unique(raw[:, i]) for i, ax in enumerate(axes)}
    shape = tuple(len(hadronic_grids[ax]) for ax in axes)
    data = {k: raw[:, n_axes + j].reshape(shape, order='F')
            for j, k in enumerate(_QUANTUM_DATA_KEYS)}
    data['converged'] = data['converged'].astype(bool)
    return QuantumNucleationObservables(
        eq_type=eq_type, hadronic_grids=hadronic_grids,
        flavor_mode=metadata['flavor_mode'],
        electric_charge_mode=metadata['electric_charge_mode'],
        quark_phase=metadata.get('quark_phase', 'unpaired'),
        sigma=float(metadata['sigma']), N_c=float(metadata['N_c']), **data)


###############################################################################
#                                                                             #
#  unpCFL thermal driver (grid)                                               #
#                                                                             #
###############################################################################
def _compute_Qs_at_R(R, hadronic_table, params, sigma, quark_phase='unpaired',
                     Delta0=None, include_photons=True, include_gluons=True,
                     include_thermal_neutrinos=True, initial_guess=None,
                     verbose=False, save_table=False, output_file=None):
    """Compute coulomb_minimize Q* at fixed radius R over the hadronic grid.

    R may be scalar, per-T (length len(T)), or a full-grid array; non-finite
    entries (Rx=inf above T_c) are skipped. Used for the unpCFL kink correction.
    """
    eq_type = hadronic_table.eq_type
    grids = hadronic_table.grids
    n_B_arr, T_arr = grids['n_B'], grids['T']
    solver_kw = dict(include_photons=include_photons, include_gluons=include_gluons,
                     include_thermal_neutrinos=include_thermal_neutrinos)
    shape, grid_out, _, outer_arr, _, _ = _grid_layout(hadronic_table)
    R_full = np.broadcast_to(np.asarray(R, dtype=float), shape)
    data = _init_data(shape)
    n_outer = len(outer_arr) if outer_arr is not None else 1
    guess = initial_guess

    for i_o in range(n_outer):
        for i_T in range(len(T_arr)):
            row_conv = 0
            for i_nB in range(len(n_B_arr)):
                if eq_type == 'beta_eq':
                    idx = (i_nB, i_T)
                    gv = {'T': T_arr[i_T]}
                elif eq_type == 'trapped_neutrinos':
                    idx = (i_nB, i_o, i_T)
                    gv = {'T': T_arr[i_T], 'Y_L': outer_arr[i_o]}
                else:
                    idx = (i_nB, i_o, i_T)
                    gv = {'T': T_arr[i_T], 'Y_C': outer_arr[i_o]}
                R_val = float(R_full[idx])
                if not np.isfinite(R_val):
                    continue
                H_pt = _build_H_from_table(hadronic_table, eq_type, idx, gv)
                if quark_phase == 'cfl':
                    result = solve_coulomb_minimize_cfl_at_R(
                        R_val, H_pt, params, Delta0, sigma, **solver_kw,
                        initial_guess=guess)
                else:
                    result = solve_coulomb_minimize_at_R(
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
                print(f"  Q* at R(T={T_arr[i_T]:.1f}) "
                      f"-> {row_conv}/{len(n_B_arr)} converged")

    table = QstarTableData(eq_type=eq_type, hadronic_grids=grid_out, data=data)
    if save_table:
        if output_file is None:
            output_file = f"Qstar_atR_{eq_type}.dat"
        export_table(table, params, output_file,
                     charge_neutrality='global', sigma=sigma)
    return table


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
