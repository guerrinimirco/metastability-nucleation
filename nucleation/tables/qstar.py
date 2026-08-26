"""
Q* droplet tables
=================

Sweep the droplet-composition solve over a hadronic (n_B_H, [Y], T) grid,
warm-starting along n_B so each point begins from its neighbour's answer, and
read/write the result as a .dat table.

A Q* table is the expensive, reusable intermediate: the thermal and quantum
drivers both consume one rather than re-solving compositions themselves.
"""

import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from nucleation.barrier import electron_susceptibility, debye_length
from nucleation.composition import (
    get_solver_Qs, R_GCN_SKIP_DEFAULT,
    solve_coulomb_minimize_at_R, solve_coulomb_minimize_cfl_at_R,
)
from nucleation.tables.grid import (
    GRID_AXES, _EQ_TYPE_ALIASES, _BASE_DATA_KEYS, _COULOMB_EXTRA_KEYS,
    QstarTableData, _grid_layout, _init_data, _store_result,
    _compute_and_store_Rc, _build_H_from_table,
)


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

