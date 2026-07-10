"""Grid tables: .dat round-trip, old-file load, screening smoke."""
import os

import numpy as np
import pytest

from nucleation.tables import (
    load_Qstar_table, export_table, compute_Qstar_table,
    build_Qstar_interpolators,
)

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, 'data', 'Qstar_fixture_gcn_cfl.dat')


def test_old_file_load():
    if not os.path.exists(FIXTURE):
        pytest.skip("fixture missing")
    t = load_Qstar_table(FIXTURE)
    assert t.eq_type == 'trapped_neutrinos'
    assert int(t.data['converged'].sum()) > 0
    assert set(t.hadronic_grids) == {'n_B_H', 'Y_L_H', 'T'}


def test_roundtrip(tmp_path, params):
    if not os.path.exists(FIXTURE):
        pytest.skip("fixture missing")
    t = load_Qstar_table(FIXTURE)
    out = str(tmp_path / 'rt.dat')
    export_table(t, params, out, charge_neutrality='global', sigma=100.0)
    t2 = load_Qstar_table(out)
    for k, arr in t.data.items():
        if arr.dtype == float:
            assert np.allclose(t2.data[k], arr, equal_nan=True)
        else:
            assert np.array_equal(t2.data[k], arr)


def _small_H(H_table_subgrid_needed=True):
    from eos.sfho.compute_tables import load_eos_table as load_h
    path = os.path.join(HERE, '..', '..', 'output', 'tables_Hphase',
                        'eos_hadronic_trapped_sfho_2famphi_xsd115.dat')
    if not os.path.exists(path):
        pytest.skip("hadronic table not present")
    import copy
    th = load_h(path, 'trapped_neutrinos')
    sub = copy.copy(th)
    inB, iYL, iT = range(40, 46), range(0, 2), range(10, 13)
    sub.grids = {'n_B': th.grids['n_B'][40:46], 'Y_L': th.grids['Y_L'][0:2],
                 'T': th.grids['T'][10:13]}
    sub.data = {k: (v[np.ix_(inB, iYL, iT)] if getattr(v, 'ndim', 0) == 3 else v)
                for k, v in th.data.items()}
    return sub


def test_qstar_table_and_interp(params):
    sub = _small_H()
    t = compute_Qstar_table(sub, 'saddlepoint', 'gcn', params, sigma=30.0,
                            quark_phase='unpaired')
    assert int(t.data['converged'].sum()) > 0
    Q = build_Qstar_interpolators(t)
    # interpolator reproduces a node value
    nB = t.hadronic_grids['n_B_H'][0]
    YL = t.hadronic_grids['Y_L_H'][0]
    T = t.hadronic_grids['T'][0]
    assert float(Q['P'](nB, YL, T)) == pytest.approx(
        t.data['P_total'][0, 0, 0], rel=1e-6)


def test_screening_table_smoke(params):
    sub = _small_H()
    t_scr = compute_Qstar_table(sub, 'saddlepoint', 'screening', params,
                                sigma=30.0, quark_phase='unpaired')
    t_gc = compute_Qstar_table(sub, 'saddlepoint', 'gcn_coulomb', params,
                               sigma=30.0, quark_phase='unpaired')
    # screening R_c finite wherever gcn_coulomb R_c is finite
    both = np.isfinite(t_scr.data['R_c']) & np.isfinite(t_gc.data['R_c'])
    assert both.sum() == int(np.isfinite(t_gc.data['R_c']).sum())
