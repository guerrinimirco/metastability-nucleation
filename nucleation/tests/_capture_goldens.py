"""Capture golden reference values from the CURRENT (pre-rewrite) code.

Run ONCE before the rewrite to freeze a regression baseline:

    python3.14 nucleation/tests/_capture_goldens.py

Writes JSON files into nucleation/tests/golden/. The pytest suite then
re-runs the same computations against the rewritten modules and asserts
agreement to a tight tolerance.

Design choices
--------------
* Hadronic points come from the committed trapped SFHO table so they are
  physically consistent and reproducible without regenerating any EOS table.
* The CFL coulomb_minimize mode is KNOWN BUGGY (mu_B condition instead of
  mu_B+mu_S). Its output is stored separately in ``old_buggy_cfl_reference.json``
  and is NOT part of the regression set: after the fix the value changes on
  purpose, and the post-fix value is captured by the test itself.
"""
import json
import os

import numpy as np

from eos.sfho.compute_tables import (
    load_eos_table as load_h, build_interpolators as build_h,
)
from eos.alphabag.parameters import get_alphabag_custom

from nucleation.energy_barrier.small_droplet.solvers import (
    get_solver_Qs,
    solve_saddlepoint_minimizecoulomb,
    solve_saddlepoint_minimizecoulomb_cfl,
)
from nucleation.energy_barrier.small_droplet.barrier import (
    driving_force, critical_radius_noCoulomb, critical_work_noCoulomb,
    work_of_formation, critical_radius_coulomb,
)
from nucleation.energy_barrier.small_droplet.observables import compute_energy_barrier
from nucleation.general_nucleation.thermal import nucleation_rate, nucleation_time
from nucleation.general_nucleation.quantum import (
    effective_inertia, quantum_nucleation_time,
)
from eos.general.physics_constants import m_neutron

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, 'golden')
H_TABLE = os.path.join(
    HERE, '..', '..', 'output', 'tables_Hphase',
    'eos_hadronic_trapped_sfho_2famphi_xsd115.dat')

# Bag parameters used throughout (match B4145/D80/a0.16/ms100 committed tables).
PARAMS = get_alphabag_custom(alpha=0.15707963267948966, B4=145.0, m_s=100.0)
DELTA0 = 80.0
SIGMA = 30.0

# Two trapped hadronic points (n_B, Y_L, T). Chosen dense enough that the
# hadronic phase is metastable (Delta_f < 0) for most quark modes.
POINTS = [(0.8, 0.25, 20.0), (1.1, 0.10, 30.0)]

Q_KEYS = ['n_B', 'mu_B', 'mu_C', 'mu_S', 'mu_u', 'mu_d', 'mu_s', 'mu_e',
          'Y_C', 'Y_S', 'Y_u', 'Y_d', 'Y_s', 'Y_e', 'P_total', 'e_total']


def build_H(H_interp, pt):
    """SimpleNamespace hadronic state at (n_B, Y_L, T)."""
    from types import SimpleNamespace
    nB, YL, T = pt
    H = SimpleNamespace(
        n_B=float(nB), T=float(T),
        P_total=float(H_interp['P'](*pt)), e_total=float(H_interp['eps'](*pt)),
        mu_B=float(H_interp['mu_B'](*pt)), mu_C=float(H_interp['mu_C'](*pt)),
        mu_S=float(H_interp['mu_S'](*pt)), mu_e=float(H_interp['mu_e'](*pt)),
        mu_nu=float(H_interp['mu_nu'](*pt)),
        Y_C=float(H_interp['Y_C'](*pt)), Y_S=float(H_interp['Y_S'](*pt)))
    H.Y_e = H.Y_C
    H.Y_nu = float(YL) - H.Y_C
    return H


def qs_dict(Qs):
    return {k: float(getattr(Qs, k)) for k in Q_KEYS}


def capture():
    os.makedirs(GOLDEN, exist_ok=True)
    t = load_h(H_TABLE, 'trapped_neutrinos')
    H_interp = build_h(t)

    combos = [
        ('frozen', 'lcn', 'unpaired'),
        ('frozen', 'gcn', 'unpaired'),
        ('saddlepoint', 'lcn', 'unpaired'),
        ('saddlepoint', 'gcn', 'unpaired'),
        ('saddlepoint', 'gcn_coulomb', 'unpaired'),
        ('saddlepoint', 'lcn', 'cfl'),
        ('saddlepoint', 'gcn', 'cfl'),
        ('saddlepoint', 'gcn_coulomb', 'cfl'),
    ]

    regression = {'points': POINTS, 'params': dict(
        alpha=PARAMS.alpha, B4=PARAMS.B4, m_s=PARAMS.m_s), 'sigma': SIGMA,
        'Delta0': DELTA0, 'cases': []}

    for pt in POINTS:
        H = build_H(H_interp, pt)
        for flavor, charge, phase in combos:
            kw = dict(quark_phase=phase, Delta0=DELTA0 if phase == 'cfl' else None,
                      sigma=SIGMA)
            Qs = get_solver_Qs(flavor, charge, PARAMS, **kw)(H)
            entry = dict(pt=pt, flavor=flavor, charge=charge, phase=phase)
            if Qs is None:
                entry['converged'] = False
                regression['cases'].append(entry)
                continue
            Df = float(driving_force(Qs, H))
            entry['converged'] = True
            entry['Qs'] = qs_dict(Qs)
            entry['Delta_f'] = Df
            if charge in ('lcn', 'gcn'):
                entry['R_c'] = (float(critical_radius_noCoulomb(Df, SIGMA))
                                if Df < 0 else None)
                entry['W_c'] = (float(critical_work_noCoulomb(Df, SIGMA))
                                if Df < 0 else None)
            else:  # gcn_coulomb
                dnC = float((Qs.Y_C - Qs.Y_e) * Qs.n_B)
                Rc = (float(critical_radius_coulomb(Df, SIGMA, dnC))
                      if Df < 0 else None)
                entry['R_c'] = Rc
                entry['W_c'] = (float(work_of_formation(Rc, Df, SIGMA, dnC))
                                if Rc is not None else None)
            regression['cases'].append(entry)

    # coulomb_minimize UNPAIRED (5-eq self-consistent solver): regression-safe
    cmin = {'unpaired': []}
    for pt in POINTS:
        H = build_H(H_interp, pt)
        out = solve_saddlepoint_minimizecoulomb(
            H, PARAMS, SIGMA, include_photons=True, include_gluons=True,
            include_thermal_neutrinos=True)
        if out is None:
            cmin['unpaired'].append(dict(pt=pt, converged=False))
        else:
            Qs, Rc = out
            Df = float(driving_force(Qs, H))
            dnC = float((Qs.Y_C - Qs.Y_e) * Qs.n_B)
            cmin['unpaired'].append(dict(
                pt=pt, converged=True, Qs=qs_dict(Qs), R_c=float(Rc),
                Delta_f=Df, W_c=float(work_of_formation(Rc, Df, SIGMA, dnC))))
    regression['coulomb_minimize'] = cmin

    # W(R) vectors from compute_energy_barrier (paths B and C)
    R_vec = list(np.linspace(0.0, 20.0, 50))
    wr = []
    for pt in POINTS:
        nB, YL, T = pt
        for charge in ('lcn', 'gcn', 'gcn_coulomb', 'coulomb_minimize'):
            eb = compute_energy_barrier(
                H_interp, nB, T, SIGMA, electric_charge_mode=charge,
                params=PARAMS, flavor_mode='saddlepoint', quark_phase='unpaired',
                Y_L_H=YL, R_values=np.asarray(R_vec))
            wr.append(dict(pt=pt, charge=charge,
                           W=[None if not np.isfinite(x) else float(x)
                              for x in eb.W]))
    regression['energy_barrier_R'] = R_vec
    regression['energy_barrier'] = wr

    # Rates + quantum at one converged unpaired gcn point
    H = build_H(H_interp, POINTS[0])
    Qs = get_solver_Qs('saddlepoint', 'gcn', PARAMS, quark_phase='unpaired')(H)
    Df = float(driving_force(Qs, H))
    Rc = float(critical_radius_noCoulomb(Df, SIGMA))
    Wc = float(critical_work_noCoulomb(Df, SIGMA))
    V = 4.18879e51
    Gamma = float(nucleation_rate(Wc, Rc, SIGMA, H.T, H, Qs))
    tau = float(nucleation_time(Gamma, V))
    rho_H = m_neutron * H.n_B
    ratio = Qs.n_B / H.n_B

    def W_func(R):
        return work_of_formation(R, Df, SIGMA, 0.0)

    def M_func(R):
        return effective_inertia(R, rho_H, ratio)

    qn = quantum_nucleation_time(W_func, M_func, Rc, N_c=1e48)
    regression['rates'] = dict(pt=POINTS[0], R_c=Rc, W_c=Wc, Gamma=Gamma, tau=tau,
                               A=float(qn.A), E_0=float(qn.E_0),
                               nu_0=float(qn.nu_0), tau_qt=float(qn.tau_qt))

    with open(os.path.join(GOLDEN, 'regression.json'), 'w') as f:
        json.dump(regression, f, indent=2)
    print(f"wrote {os.path.join(GOLDEN, 'regression.json')} "
          f"({len(regression['cases'])} solver cases)")

    # KNOWN-BUGGY CFL coulomb_minimize, archived separately.
    buggy = {'points': POINTS, 'cfl': []}
    for pt in POINTS:
        H = build_H(H_interp, pt)
        out = solve_saddlepoint_minimizecoulomb_cfl(
            H, PARAMS, DELTA0, SIGMA, include_photons=True, include_gluons=True,
            include_thermal_neutrinos=True)
        if out is None:
            buggy['cfl'].append(dict(pt=pt, converged=False))
        else:
            Qs, Rc = out
            buggy['cfl'].append(dict(pt=pt, converged=True, R_c=float(Rc),
                                     Qs=qs_dict(Qs)))
    with open(os.path.join(GOLDEN, 'old_buggy_cfl_reference.json'), 'w') as f:
        json.dump(buggy, f, indent=2)
    print(f"wrote {os.path.join(GOLDEN, 'old_buggy_cfl_reference.json')}")


if __name__ == '__main__':
    capture()
