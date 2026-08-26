"""
Grid layout and per-point storage
=================================

The bookkeeping every grid driver shares: which axes an equilibrium type has,
how to expand a hadronic table into full-shape arrays, and where a solver result
goes once computed. No physics decisions live here -- only shape.

This module imports nothing from the other `tables` submodules, so it is the
bottom of the package and safe to import from anywhere (including
`nucleation.critical`, which needs `_BASE_DATA_KEYS`).
"""

import numpy as np
from types import SimpleNamespace
from dataclasses import dataclass

from nucleation.barrier import (
    driving_force, critical_radius_noCoulomb,
    critical_radius_coulomb, critical_radius_coulomb_screened,
    electron_susceptibility, debye_length,
)


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
        outer = grids['Y_Le']
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
            grids['n_B'], grids['Y_Le'], grids['T'], indexing='ij')
        mu_nu_H = h_d['mu_nue']
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
        mu_nu = d['mu_nue'][idx]
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
