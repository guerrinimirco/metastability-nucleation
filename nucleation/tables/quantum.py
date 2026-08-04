"""
Quantum (WKB) nucleation observables
====================================

Grid driver for the zero-temperature channel: the droplet tunnels THROUGH the
barrier rather than being activated over it, so tau depends on the whole shape
of W(R), not just its peak. See docs/nucleation_physics.md section 7.

Deliberately built from the same pieces as the thermal driver -- same Q* table,
same driving force, same R_c and W_c -- so the two channels can be compared
point by point and the shorter time taken as physical.
"""

import os
import numpy as np
from types import SimpleNamespace
from dataclasses import dataclass
from scipy.interpolate import RegularGridInterpolator

from nucleation.barrier import (
    driving_force, work_of_formation, critical_work_noCoulomb,
    critical_radius_noCoulomb, get_switching_function,
)
from nucleation.critical import _find_Rc_Wc_step, _find_Rc_Wc_tanh
from nucleation.tables.grid import (
    GRID_AXES, _EQ_TYPE_ALIASES, _QUANTUM_DATA_KEYS, _COULOMB_MODES,
    expand_grid, _lambda_D_grid,
)
from nucleation.tables.qstar import compute_Qstar_table


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

