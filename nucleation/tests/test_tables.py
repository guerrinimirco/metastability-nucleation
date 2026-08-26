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


def _small_H(H_table):
    """A few-point corner of the fixture table -- enough to exercise the grid
    driver without paying for the whole grid in every table test."""
    import copy
    sub = copy.copy(H_table)
    inB, iYL, iT = range(8, 14), range(0, 2), range(4, 7)
    sub.grids = {'n_B': H_table.grids['n_B'][8:14],
                 'Y_Le': H_table.grids['Y_Le'][0:2],
                 'T': H_table.grids['T'][4:7]}
    sub.data = {k: (v[np.ix_(inB, iYL, iT)] if getattr(v, 'ndim', 0) == 3 else v)
                for k, v in H_table.data.items()}
    return sub


def test_qstar_table_and_interp(params, H_table):
    sub = _small_H(H_table)
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


def test_screening_table_smoke(params, H_table):
    sub = _small_H(H_table)
    t_scr = compute_Qstar_table(sub, 'saddlepoint', 'screening', params,
                                sigma=30.0, quark_phase='unpaired')
    t_gc = compute_Qstar_table(sub, 'saddlepoint', 'gcn_coulomb', params,
                               sigma=30.0, quark_phase='unpaired')
    # screening R_c finite wherever gcn_coulomb R_c is finite
    both = np.isfinite(t_scr.data['R_c']) & np.isfinite(t_gc.data['R_c'])
    assert both.sum() == int(np.isfinite(t_gc.data['R_c']).sum())
