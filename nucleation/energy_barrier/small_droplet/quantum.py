"""
Quantum Nucleation Observables
==============================

Compute quantum (WKB) tunneling nucleation observables over the full
hadronic grid for the hadron-to-quark phase transition.

At each grid point the function builds the energy barrier W(R) and
effective inertia M(R), then solves the WKB tunneling integral to
obtain the quantum nucleation time tau_qt.

Usage
-----
>>> from nucleation.energy_barrier.small_droplet import (
...     compute_quantum_nucleation_observables,
... )
>>> qobs = compute_quantum_nucleation_observables(
...     hadronic_table,
...     Qstar_table,
...     sigma=30.0,
...     electric_charge_mode='gcn',
... )
>>> print(qobs.tau_qt, qobs.A)
"""

import numpy as np
from types import SimpleNamespace
from dataclasses import dataclass

from nucleation.energy_barrier.small_droplet.barrier import (
    driving_force,
    critical_radius_noCoulomb,
    work_of_formation,
)


# =============================================================================
# Output dataclass
# =============================================================================
@dataclass
class QuantumNucleationObservables:
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


# =============================================================================
# Main function
# =============================================================================
def compute_quantum_nucleation_observables(
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
    QuantumNucleationObservables
    """
    from eos.general.physics_constants import m_neutron
    from nucleation.general_nucleation.quantum import (
        effective_inertia, quantum_nucleation_time,
    )

    h_d = hadronic_table.data
    q_d = Qstar_table.data
    grids = hadronic_table.grids
    eq_type = hadronic_table.eq_type
    shape = q_d['P_total'].shape
    qstar_converged = q_d['converged']

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

    # Bulk driving force
    Delta_f_full = driving_force(Qs, H)

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

    it = np.nditer(Delta_f_full, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        it.iternext()

        if not qstar_converged[idx]:
            done += 1
            continue

        Delta_f = float(Delta_f_full[idx])
        if Delta_f <= 0:
            done += 1
            continue

        n_B_H_val = float(H.n_B[idx])
        n_B_Qs = float(Qs.n_B[idx])
        T_val = float(H.T[idx])
        delta_n_C = float(delta_n_C_full[idx])

        # Hadronic mass density
        if rho_H_func is not None:
            rho_H = rho_H_func(n_B_H_val, T_val)
        else:
            rho_H = m_neutron * n_B_H_val

        n_B_ratio = n_B_Qs / n_B_H_val if n_B_H_val > 0 else 1.0

        # Critical radius (for turning-point search)
        R_c = critical_radius_noCoulomb(Delta_f, sigma)
        if not np.isfinite(R_c) or R_c <= 0:
            done += 1
            continue

        # Build W(R) and M(R) closures for this grid point
        def W_func(R, _df=Delta_f, _s=sigma, _dnC=delta_n_C):
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

    return QuantumNucleationObservables(
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
