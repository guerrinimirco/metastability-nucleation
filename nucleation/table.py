"""
Q* Table Infrastructure
=======================

Unified data container, generic grid-loop computation, file I/O,
and interpolation for Q* nucleation tables.

Works with any solver (frozen, saddlepoint, coulomb) and any
equilibrium type (beta_eq, trapped_neutrinos, fixed_yc).
"""

import os
import numpy as np
from types import SimpleNamespace
from dataclasses import dataclass


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

_COULOMB_EXTRA_KEYS = ['R_c', 'delta_n_C']


# =============================================================================
# Unified data container
# =============================================================================
@dataclass
class QstarTableData:
    """Q* nucleation table.

    Maps hadronic conditions (n_B_H, T, ...) to Q* quark droplet thermodynamics.
    Optionally includes Coulomb fields (R_c, delta_n_C).
    """
    eq_type: str            # 'beta_eq', 'trapped_neutrinos', 'fixed_yc'
    hadronic_grids: dict    # {n_B_H: array, T: array, ...}
    data: dict              # {n_B, mu_u, ..., P_total, converged, ...}
    filepath: str = ""

    @property
    def has_coulomb(self):
        return 'R_c' in self.data

    def __repr__(self):
        shapes = [f"{k}={len(v)}" for k, v in self.hadronic_grids.items()]
        tag = " +Coulomb" if self.has_coulomb else ""
        return f"QstarTableData(eq_type='{self.eq_type}', grids=({', '.join(shapes)}){tag})"


# =============================================================================
# Data initialization and storage
# =============================================================================
def _init_data(shape, include_coulomb=False):
    """Initialize NaN-filled output arrays for Q* table computation."""
    data = {k: np.full(shape, np.nan) for k in _BASE_DATA_KEYS}
    data['converged'] = np.zeros(shape, dtype=bool)
    if include_coulomb:
        for k in _COULOMB_EXTRA_KEYS:
            data[k] = np.full(shape, np.nan)
    return data


def _store_result(data, idx, result, current_guess):
    """Store solver output into data arrays and return updated guess.

    Parameters
    ----------
    result : EOS result object, (EOS result, R_c) tuple, or None
    """
    if result is None:
        return current_guess

    # Unpack Coulomb tuple
    if isinstance(result, tuple):
        eos_result, R_c = result
        data['R_c'][idx] = R_c
        data['delta_n_C'][idx] = (eos_result.Y_C - eos_result.Y_e) * eos_result.n_B
        guess_extra = [R_c]
    else:
        eos_result = result
        guess_extra = []

    for key in _BASE_DATA_KEYS:
        data[key][idx] = getattr(eos_result, key)
    data['converged'][idx] = True

    current_guess = np.array(
        [eos_result.mu_u, eos_result.mu_d, eos_result.mu_s, eos_result.mu_e]
        + guess_extra
    )
    return current_guess


# =============================================================================
# Build hadronic state at a single grid point
# =============================================================================
def _build_H_at_point(hadronic_table, eq_type, idx, grid_values):
    """Build hadronic SimpleNamespace at a single grid point.

    Parameters
    ----------
    hadronic_table : EOSTableData
    eq_type : str
    idx : tuple of int
        Full index into the data arrays.
    grid_values : dict
        {T: float, Y_L: float (optional), Y_C: float (optional)}
    """
    d = hadronic_table.data
    T = grid_values['T']

    if eq_type == 'beta_eq':
        Y_C_H = d['Y_C'][idx]
        Y_nu = 0.0
        mu_nu = 0.0
    elif eq_type == 'trapped_neutrinos':
        Y_C_H = d['Y_C'][idx]
        Y_nu = grid_values['Y_L'] - Y_C_H
        mu_nu = d['mu_nu'][idx]
    elif eq_type == 'fixed_yc':
        Y_C_H = grid_values['Y_C']
        Y_nu = 0.0
        mu_nu = 0.0
    else:
        raise ValueError(f"Unsupported eq_type: '{eq_type}'")

    H = SimpleNamespace(
        T=T,
        Y_C=Y_C_H,
        Y_S=d['Y_S'][idx],
        Y_e=Y_C_H,        # charge neutrality in H: Y_e = Y_C
        Y_nu=Y_nu,
        mu_B=d['mu_B'][idx],
        mu_C=d['mu_C'][idx],
        mu_S=d['mu_S'][idx],
        mu_e=d['mu_e'][idx],
        mu_nu=mu_nu,
        P_total=d['P_total'][idx],
    )
    return H


# =============================================================================
# Generic grid computation
# =============================================================================
def compute_Qstar_table(hadronic_table, solver_fn, include_coulomb=False,
                        initial_guess=None, verbose=False,
                        save_table=False, output_file=None,
                        export_params=None, export_charge_neutrality=None,
                        export_sigma=None):
    """Compute Q* table over the hadronic grid for any eq_type.

    Parameters
    ----------
    hadronic_table : EOSTableData
    solver_fn : callable
        Signature: solver_fn(H, initial_guess=...) -> result or None.
        Result is an EOS result object or (result, R_c) tuple for Coulomb.
    include_coulomb : bool
        If True, data dict includes R_c and delta_n_C.
    initial_guess : array-like or None
    verbose : bool
    save_table : bool
    output_file : str or None
    export_params : AlphaBagParams or None
        Required if save_table=True.
    export_charge_neutrality : str or None
    export_sigma : float or None

    Returns
    -------
    QstarTableData
    """
    eq_type = hadronic_table.eq_type
    grids = hadronic_table.grids
    n_B_arr = grids['n_B']
    T_arr = grids['T']

    # Determine outer axes and shape
    if eq_type == 'beta_eq':
        outer_key = None
        outer_arr = None
        shape = (len(n_B_arr), len(T_arr))
        grid_out = {'n_B_H': n_B_arr, 'T': T_arr}
    elif eq_type == 'trapped_neutrinos':
        outer_key = 'Y_L'
        outer_arr = grids['Y_L']
        shape = (len(n_B_arr), len(outer_arr), len(T_arr))
        grid_out = {'n_B_H': n_B_arr, 'Y_L_H': outer_arr, 'T': T_arr}
    elif eq_type == 'fixed_yc':
        outer_key = 'Y_C'
        outer_arr = grids['Y_C']
        shape = (len(n_B_arr), len(outer_arr), len(T_arr))
        grid_out = {'n_B_H': n_B_arr, 'Y_C_H': outer_arr, 'T': T_arr}
    else:
        raise ValueError(f"Unsupported eq_type: '{eq_type}'")

    data = _init_data(shape, include_coulomb=include_coulomb)
    row_start_guess = initial_guess

    # Iterate: outer axes (Y_L or Y_C if present) -> T -> n_B
    n_outer = len(outer_arr) if outer_arr is not None else 1

    for i_outer in range(n_outer):
        for i_T in range(len(T_arr)):
            current_guess = row_start_guess
            row_converged = 0

            for i_nB in range(len(n_B_arr)):
                # Build index and grid values for this point
                if eq_type == 'beta_eq':
                    idx = (i_nB, i_T)
                    gv = {'T': T_arr[i_T]}
                elif eq_type == 'trapped_neutrinos':
                    idx = (i_nB, i_outer, i_T)
                    gv = {'T': T_arr[i_T], 'Y_L': outer_arr[i_outer]}
                elif eq_type == 'fixed_yc':
                    idx = (i_nB, i_outer, i_T)
                    gv = {'T': T_arr[i_T], 'Y_C': outer_arr[i_outer]}

                H = _build_H_at_point(hadronic_table, eq_type, idx, gv)
                result = solver_fn(H, initial_guess=current_guess)

                current_guess = _store_result(data, idx, result, current_guess)
                if result is not None:
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
        if export_params is None:
            raise ValueError("export_params required when save_table=True")
        if output_file is None:
            output_file = f"Qstar_{eq_type}.dat"
        export_table(table, export_params, output_file,
                     charge_neutrality=export_charge_neutrality,
                     sigma=export_sigma)

    return table


# =============================================================================
# Interpolation
# =============================================================================
def build_Qstar_interpolators(table, method='linear'):
    """Build interpolation functions from a QstarTableData.

    Returns dict with convenience functions for each Q* quantity,
    keyed by hadronic input coordinates.

    Usage (beta_eq)::
        Q = build_Qstar_interpolators(table)
        P_Qs = Q['P'](n_B_H, T)
        mu_B_Qs = Q['mu_B'](n_B_H, T)

    Usage (trapped_neutrinos)::
        Q = build_Qstar_interpolators(table)
        P_Qs = Q['P'](n_B_H, Y_L_H, T)
    """
    from scipy.interpolate import RegularGridInterpolator

    axes = GRID_AXES[table.eq_type]
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
# File I/O
# =============================================================================
def load_Qstar_table(filepath):
    """Load a Q* table from a .dat file.

    Auto-detects Coulomb columns (R_c, delta_n_C) and normalizes
    old eq_type names (betaeq -> beta_eq, etc.).

    Parameters
    ----------
    filepath : str
        Path to the .dat file.

    Returns
    -------
    QstarTableData
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

    # Normalize old eq_type names
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
        col_flat = raw[:, n_axes + j]
        arr = col_flat.reshape(shape, order='F')
        if key == 'converged':
            arr = arr.astype(bool)
        data[key] = arr

    # Backward compat: reconstruct Y_e if absent
    if 'Y_e' not in data and 'Y_C' in data:
        data['Y_e'] = data['Y_C'].copy()

    return QstarTableData(
        eq_type=eq_type,
        hadronic_grids=hadronic_grids,
        data=data,
        filepath=filepath,
    )


def export_table(table, params, output_file,
                 charge_neutrality=None, sigma=None):
    """Export QstarTableData to text file.

    Parameters
    ----------
    table : QstarTableData
    params : AlphaBagParams
    output_file : str
    charge_neutrality : str or None
    sigma : float or None
    """
    axes = GRID_AXES[table.eq_type]
    grid_arrays = [table.hadronic_grids[ax] for ax in axes]
    mesh = np.meshgrid(*grid_arrays, indexing='ij')
    input_cols = [m.ravel(order='F') for m in mesh]

    # Determine data keys from what's actually in the table
    data_keys = [k for k in _BASE_DATA_KEYS if k in table.data]
    if table.has_coulomb:
        data_keys += [k for k in _COULOMB_EXTRA_KEYS if k in table.data]
    data_keys.append('converged')

    output_names = [k + '_Qs' for k in data_keys]
    output_cols = [table.data[k].ravel(order='F') for k in data_keys]

    all_names = list(axes) + output_names
    all_cols = np.column_stack(input_cols + output_cols)

    # Metadata header
    header_lines = [f"# Q* nucleation table"]
    header_lines.append(f"# hadronic_eq_type: {table.eq_type}")
    if charge_neutrality:
        header_lines.append(f"# electric_charge_neutrality: {charge_neutrality}")
    if sigma is not None:
        header_lines.append(f"# sigma: {sigma} MeV/fm^2")
    header_lines.append(
        f"# quark_eos: {params.name}\n"
        f"# m_u={params.m_u} MeV  m_d={params.m_d} MeV  m_s={params.m_s} MeV"
        f"  alpha={params.alpha}  B4={params.B4} MeV"
    )
    col_header = "  ".join(f"{name:>14s}" for name in all_names)
    header_lines.append(f"# {col_header}")
    meta = "\n".join(header_lines) + "\n"

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(meta)
    with open(output_file, 'ab') as f:
        np.savetxt(f, all_cols, fmt='%16.8e')
    print(f"  Saved Q* table ({all_cols.shape[0]} rows) -> {output_file}")
