"""Critical-droplet engine: analytic paths, screening limit, conventions, unpCFL."""
import numpy as np
import pytest

from nucleation.critical import critical_droplet, compute_energy_barrier

SIGMA = 30.0
DELTA0 = 80.0


def test_engine_matches_golden(regression, params, build_H):
    for c in regression['cases']:
        if not c['converged'] or c.get('R_c') is None:
            continue
        H = build_H(tuple(c['pt']))
        cd = critical_droplet(
            H, sigma=SIGMA, params=params, flavor_mode=c['flavor'],
            electric_charge_mode=c['charge'], quark_phase=c['phase'],
            Delta0=DELTA0 if c['phase'] == 'cfl' else None)
        assert cd.R_c == pytest.approx(c['R_c'], rel=1e-8)
        assert cd.W_c == pytest.approx(c['W_c'], rel=1e-8)


def test_energy_barrier_matches_golden(regression, params, H_interp):
    R = np.asarray(regression['energy_barrier_R'])
    for c in regression['energy_barrier']:
        nB, YL, T = c['pt']
        eb = compute_energy_barrier(
            H_interp, nB, T, SIGMA, electric_charge_mode=c['charge'],
            params=params, flavor_mode='saddlepoint', quark_phase='unpaired',
            Y_L_H=YL, R_values=R)
        gold = np.array([np.nan if x is None else x for x in c['W']])
        m = np.isfinite(gold) & np.isfinite(eb.W)
        assert np.all(np.isnan(gold) == np.isnan(eb.W))
        # Scaled to the barrier height, not absolute. W(R) runs from 0 at R = 0
        # to ~1e6 MeV at the top of the curve, so a single absolute bound is in
        # practice a bound on the largest entry alone: the former 1e-9 sat at
        # 4e-16 of it, below one ulp of a double, and so asserted bit-identity.
        # A relative bound is what "reproduces the barrier" actually means, and
        # 1e-12 is four orders tighter than the rel=1e-8 this suite accepts on
        # W_c itself. Pointwise relative is deliberately NOT used: the curve
        # passes through zero at R = 0, where relative error is meaningless.
        assert np.max(np.abs(gold[m] - eb.W[m])) < 1e-12 * np.max(np.abs(gold[m]))


def test_screening_reduces_to_gcn_coulomb(params, build_H):
    H = build_H((0.8, 0.25, 20.0))
    cd_gc = critical_droplet(H, sigma=SIGMA, params=params,
                             electric_charge_mode='gcn_coulomb')
    cd_big = critical_droplet(H, sigma=SIGMA, params=params,
                              electric_charge_mode='screening', lambda_D=1e8)
    assert cd_big.R_c == pytest.approx(cd_gc.R_c, rel=1e-6)
    assert cd_big.W_c == pytest.approx(cd_gc.W_c, rel=1e-6)


def test_screening_weakens_barrier(params, build_H):
    H = build_H((0.8, 0.25, 20.0))
    cd_gc = critical_droplet(H, sigma=SIGMA, params=params,
                             electric_charge_mode='gcn_coulomb')
    cd_scr = critical_droplet(H, sigma=SIGMA, params=params,
                              electric_charge_mode='screening')
    assert np.isfinite(cd_scr.lambda_D) and cd_scr.lambda_D > 0
    # screening removes positive Coulomb energy -> smaller (or equal) barrier
    assert cd_scr.W_c <= cd_gc.W_c + 1e-9


def test_stable_H_convention(params, build_H):
    # low density -> hadronic phase stable (Delta_f >= 0)
    H = build_H((0.3, 0.25, 20.0))
    cd = critical_droplet(H, sigma=SIGMA, params=params, flavor_mode='frozen',
                          electric_charge_mode='gcn', quark_phase='unpaired')
    assert np.isnan(cd.R_c) and np.isposinf(cd.W_c) and cd.Qs is not None


def test_unpcfl_limits(params, build_H):
    H = build_H((0.9, 0.25, 25.0))
    # Rx -> 0 : CFL everywhere
    cd_cfl = critical_droplet(H, sigma=SIGMA, params=params,
                              electric_charge_mode='gcn', quark_phase='cfl',
                              Delta0=DELTA0)
    cd_rx0 = critical_droplet(H, sigma=SIGMA, params=params,
                              electric_charge_mode='gcn', quark_phase='unpCFL',
                              Delta0=DELTA0, Rx=1e-6)
    assert cd_rx0.R_c == pytest.approx(cd_cfl.R_c, rel=1e-6)
    # Rx -> inf : unpaired everywhere
    cd_unp = critical_droplet(H, sigma=SIGMA, params=params,
                              electric_charge_mode='gcn', quark_phase='unpaired')
    cd_rxinf = critical_droplet(H, sigma=SIGMA, params=params,
                                electric_charge_mode='gcn', quark_phase='unpCFL',
                                Delta0=DELTA0, Rx=1e6)
    assert cd_rxinf.R_c == pytest.approx(cd_unp.R_c, rel=1e-6)
