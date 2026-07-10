"""Nucleation-curve finders on a computed thermal grid."""
import os

import numpy as np
import pytest

from nucleation.tables import compute_thermal_nucleation_observables
from nucleation.curves import compute_nucleation_density, nucleation_curve

HERE = os.path.dirname(os.path.abspath(__file__))


def _thermal_obs(params):
    from eos.sfho.compute_tables import load_eos_table as load_h
    path = os.path.join(HERE, '..', '..', 'output', 'tables_Hphase',
                        'eos_hadronic_trapped_sfho_2famphi_xsd115.dat')
    if not os.path.exists(path):
        pytest.skip("hadronic table not present")
    import copy
    th = load_h(path, 'trapped_neutrinos')
    sub = copy.copy(th)
    inB, iYL, iT = range(20, 120), range(0, 2), range(0, 40)
    sub.grids = {'n_B': th.grids['n_B'][20:120], 'Y_L': th.grids['Y_L'][0:2],
                 'T': th.grids['T'][0:40]}
    sub.data = {k: (v[np.ix_(inB, iYL, iT)] if getattr(v, 'ndim', 0) == 3 else v)
                for k, v in th.data.items()}
    return compute_thermal_nucleation_observables(
        sub, sigma=30.0, params=params, flavor_mode='saddlepoint',
        electric_charge_mode='gcn', quark_phase='unpaired')


def test_density_scans_produce_curve(params):
    obs = _thermal_obs(params)
    r_T = compute_nucleation_density(obs, tau_target=1e-3, scan='T')
    r_n = compute_nucleation_density(obs, tau_target=1e-3, scan='n_B')
    assert np.isfinite(r_n.T_nuc).sum() > 0
    # nucleation_curve slices the n_B scan into plottable arrays
    nB, T = nucleation_curve(r_n, iYL=0)
    assert nB.shape == T.shape
    assert np.isfinite(T).sum() > 0
