"""Shared fixtures for the nucleation test suite.

The hadronic points come from the committed trapped SFHO table, so tests need no
EOS-table regeneration. Golden reference values live in ``golden/regression.json``
(captured from the pre-rewrite code by ``_capture_goldens.py``).
"""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, 'golden')
DATA = os.path.join(HERE, 'data')
H_TABLE = os.path.join(
    HERE, '..', '..', 'output', 'tables_Hphase',
    'eos_hadronic_trapped_sfho_2famphi_xsd115.dat')


@pytest.fixture(scope='session')
def regression():
    with open(os.path.join(GOLDEN, 'regression.json')) as f:
        return json.load(f)


@pytest.fixture(scope='session')
def buggy_cfl():
    with open(os.path.join(GOLDEN, 'old_buggy_cfl_reference.json')) as f:
        return json.load(f)


@pytest.fixture(scope='session')
def params():
    from eos.alphabag.parameters import get_alphabag_custom
    return get_alphabag_custom(alpha=0.15707963267948966, B4=145.0, m_s=100.0)


@pytest.fixture(scope='session')
def H_interp():
    if not os.path.exists(H_TABLE):
        pytest.skip("hadronic table not present")
    from eos.sfho.compute_tables import (
        load_eos_table as load_h, build_interpolators as build_h)
    return build_h(load_h(H_TABLE, 'trapped_neutrinos'))


@pytest.fixture(scope='session')
def build_H(H_interp):
    def _build(pt):
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
    return _build
