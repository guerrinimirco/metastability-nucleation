"""Shared fixtures for the nucleation test suite.

The hadronic points come from a COMMITTED trapped-SFHo fixture table, so the
suite runs on a fresh clone with no EoS-table regeneration and no dependence on
anything under ``output/``.

Why the fixture grid looks irregular
------------------------------------
``eos.sfho.build_interpolators`` is MULTILINEAR, so the value at an off-node
point depends only on its two BRACKETING nodes per axis. The fixture therefore
ships exactly those bracket pairs for every golden test point, plus a coarse
sweep so tau(n_B, T) still has a genuine downward crossing for the
nucleation-curve tests. That reproduces ``golden/regression.json`` bit-for-bit
at 936 points (247 KB) instead of the full 300x7x52 production table (37 MB).
Adding nodes is safe because every bracket pair is adjacent in the production
grid, so nothing can land strictly between a pair.

Regenerate with ``python -m nucleation.tests.make_fixture``.

Golden reference values live in ``golden/regression.json``; treat them as a
TRIPWIRE, not a target -- never loosen a tolerance to make one pass.
"""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, 'golden')
DATA = os.path.join(HERE, 'data')
H_TABLE = os.path.join(DATA, 'eos_hadronic_trapped_fixture.dat')


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
    # The fixture table is COMMITTED, so a missing file means a broken checkout,
    # not an environment we should quietly skip. Skipping here is what let the
    # whole suite go green-but-hollow when output/ disappeared.
    assert os.path.exists(H_TABLE), (
        f"missing committed fixture {H_TABLE}\n"
        f"regenerate with: python -m nucleation.tests.make_fixture")
    from eos.sfho.compute_tables import (
        load_eos_table as load_h, build_interpolators as build_h)
    return build_h(load_h(H_TABLE, 'trapped_neutrinos'))


@pytest.fixture(scope='session')
def H_table():
    """The raw fixture table (grids + data), for the grid-driver tests."""
    assert os.path.exists(H_TABLE), (
        f"missing committed fixture {H_TABLE}\n"
        f"regenerate with: python -m nucleation.tests.make_fixture")
    from eos.sfho.compute_tables import load_eos_table as load_h
    return load_h(H_TABLE, 'trapped_neutrinos')


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
