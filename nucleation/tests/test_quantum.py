"""Quantum (WKB) nucleation: grid driver, phases, charge modes, .dat round-trip.

The first test here is the one whose absence let the grid driver ship broken:
its guard read ``if Delta_f <= 0: continue``, which is inverted relative to the
package convention (``barrier.driving_force`` is NEGATIVE when the hadronic
phase is metastable). It therefore skipped exactly the points that HAVE a
critical droplet and kept the ones that do not, converging 0/N on every grid
ever computed. A single "some point converges" assertion catches that.
"""
import numpy as np
import pytest

from nucleation.tables import (
    compute_Qstar_table, compute_quantum_nucleation_observables,
    build_quantum_nucleation_interpolators,
    export_quantum_nucleation_table, load_quantum_nucleation_table,
    compute_thermal_nucleation_observables,
)

SIGMA = 30.0


@pytest.fixture(scope='module')
def sub_H(H_table):
    """A dense-and-warm corner of the fixture grid: where droplets nucleate.

    Delta_f < 0 needs high n_B; the WKB pipeline additionally needs a W(R) with
    two real turning points, which the low-density end does not provide.
    """
    import copy
    sub = copy.copy(H_table)
    inB, iYL, iT = range(12, 18), range(2, 4), range(3, 6)
    sub.grids = {'n_B': H_table.grids['n_B'][12:18],
                 'Y_L': H_table.grids['Y_L'][2:4],
                 'T': H_table.grids['T'][3:6]}
    sub.data = {k: (v[np.ix_(inB, iYL, iT)] if getattr(v, 'ndim', 0) == 3 else v)
                for k, v in H_table.data.items()}
    return sub


@pytest.fixture(scope='module')
def qobs(sub_H, params):
    return compute_quantum_nucleation_observables(
        sub_H, sigma=SIGMA, params=params, quark_phase='unpaired',
        flavor_mode='saddlepoint', electric_charge_mode='gcn')


# =============================================================================
# The regression that the missing test hid
# =============================================================================
def test_metastable_points_converge(qobs):
    """Points WITH a critical droplet must converge and give a finite tau_qt."""
    n_conv = int(qobs.converged.sum())
    assert n_conv > 0, (
        "quantum grid converged 0 points -- the Delta_f sign guard is inverted "
        "again (metastable H has driving_force < 0)")
    assert np.all(np.isfinite(qobs.tau_qt[qobs.converged]))
    assert np.all(qobs.tau_qt[qobs.converged] > 0)
    assert np.all(np.isfinite(qobs.A[qobs.converged]))


def test_stable_points_do_not_converge(qobs):
    """Where no critical droplet exists (R_c NaN), nothing is reported."""
    no_droplet = ~np.isfinite(qobs.R_c)
    if no_droplet.any():
        assert not qobs.converged[no_droplet].any()
        assert np.all(np.isnan(qobs.tau_qt[no_droplet]))


def test_turning_points_bracket_the_barrier_peak(qobs):
    """R_inner < R_c < R_outer: the tunnelling path must straddle the peak."""
    m = qobs.converged
    assert m.any()
    assert np.all(qobs.R_inner[m] < qobs.R_c[m])
    assert np.all(qobs.R_c[m] < qobs.R_outer[m])


def test_shares_the_thermal_barrier(sub_H, params, qobs):
    """Quantum and thermal must quote the SAME R_c / W_c at the same point.

    They differ only in the escape mechanism; if the barriers disagree, one of
    the two is computing its own inconsistent critical droplet.
    """
    tobs = compute_thermal_nucleation_observables(
        sub_H, sigma=SIGMA, params=params, quark_phase='unpaired',
        flavor_mode='saddlepoint', electric_charge_mode='gcn')
    both = qobs.converged & tobs.converged
    assert both.any()
    assert np.allclose(qobs.R_c[both], tobs.R_c[both], rtol=1e-10)
    assert np.allclose(qobs.W_c[both], tobs.W_c[both], rtol=1e-10)


# =============================================================================
# Charge modes and phases actually reach the physics
# =============================================================================
def test_coulomb_minimize_uses_the_selfconsistent_Rc(sub_H, params):
    """R_c must come from the 5-eq solve, not the closed-form no-Coulomb one.

    The old driver always called critical_radius_noCoulomb, which put the WKB
    turning points around the wrong peak for every Coulomb mode.
    """
    from nucleation.barrier import critical_radius_noCoulomb, driving_force
    from nucleation.tables import expand_grid
    from types import SimpleNamespace

    qt = compute_Qstar_table(sub_H, 'saddlepoint', 'coulomb_minimize', params,
                             sigma=SIGMA, quark_phase='unpaired')
    obs = compute_quantum_nucleation_observables(
        sub_H, sigma=SIGMA, params=params, Qstar_table=qt,
        quark_phase='unpaired', flavor_mode='saddlepoint',
        electric_charge_mode='coulomb_minimize')

    assert np.allclose(obs.R_c, qt.data['R_c'], equal_nan=True)

    H, shape, _, T_H, mu_nu_H, Y_nu_H = expand_grid(sub_H)
    Qs = SimpleNamespace(**{**qt.data, 'T': T_H, 'mu_nu': mu_nu_H,
                            'Y_nu': Y_nu_H})
    R_noc = critical_radius_noCoulomb(driving_force(Qs, H), SIGMA)
    m = obs.converged & np.isfinite(R_noc)
    assert m.any()
    assert not np.allclose(obs.R_c[m], R_noc[m]), \
        "coulomb_minimize R_c equals the no-Coulomb closed form -- not self-consistent"


def test_cfl_differs_from_unpaired(sub_H, params):
    """A CFL droplet is a different droplet: its barrier must differ."""
    unp = compute_quantum_nucleation_observables(
        sub_H, sigma=SIGMA, params=params, quark_phase='unpaired',
        flavor_mode='saddlepoint', electric_charge_mode='gcn')
    cfl = compute_quantum_nucleation_observables(
        sub_H, sigma=SIGMA, params=params, quark_phase='cfl', Delta0=80.0,
        flavor_mode='saddlepoint', electric_charge_mode='gcn')
    both = unp.converged & cfl.converged
    assert both.any(), "no point converged in both phases"
    assert not np.allclose(unp.R_c[both], cfl.R_c[both])


def test_rejects_bad_arguments(sub_H, params):
    with pytest.raises(ValueError):
        compute_quantum_nucleation_observables(
            sub_H, sigma=SIGMA, params=params, quark_phase='nonsense')
    with pytest.raises(ValueError):
        compute_quantum_nucleation_observables(       # unpCFL without Rx
            sub_H, sigma=SIGMA, params=params, quark_phase='unpCFL',
            Delta0=80.0)
    with pytest.raises(ValueError):
        compute_quantum_nucleation_observables(
            sub_H, sigma=SIGMA, params=params, electric_charge_mode='bogus')


# =============================================================================
# I/O and interpolators -- parity with the thermal path
# =============================================================================
def test_export_load_roundtrip(qobs, tmp_path):
    out = str(tmp_path / 'q.dat')
    export_quantum_nucleation_table(qobs, out)
    back = load_quantum_nucleation_table(out)

    assert back.eq_type == qobs.eq_type
    assert back.quark_phase == qobs.quark_phase
    assert back.electric_charge_mode == qobs.electric_charge_mode
    assert back.sigma == pytest.approx(qobs.sigma)
    assert back.N_c == pytest.approx(qobs.N_c)
    assert np.array_equal(back.converged, qobs.converged)
    for k in ('R_c', 'W_c', 'tau_qt', 'A', 'E_0', 'nu_0', 'R_inner', 'R_outer'):
        assert np.allclose(getattr(back, k), getattr(qobs, k),
                           equal_nan=True, rtol=1e-7), f"{k} did not round-trip"


def test_tau_alias_and_interpolators(qobs):
    # `tau` aliases `tau_qt` so quantum observables drop into the tau=tau_target
    # finders with no branching downstream.
    assert qobs.tau is qobs.tau_qt

    itp = build_quantum_nucleation_interpolators(qobs)
    assert 'log10_tau' in itp and 'log10_tau_qt' in itp
    g = qobs.hadronic_grids
    node = (g['n_B_H'][1], g['Y_L_H'][0], g['T'][1])
    val = itp['tau_qt'](*node)
    assert np.isfinite(val) or np.isnan(val)        # must not raise
    assert np.isfinite(itp['log10_tau'](*node))


def test_curve_finder_accepts_quantum_observables(qobs):
    """compute_nucleation_density reads obs.tau -- the alias must make it work."""
    from nucleation.conditions import compute_nucleation_density
    res = compute_nucleation_density(qobs, tau_target=1e-3, scan='n_B')
    assert hasattr(res, 'T_nuc')                     # ran without AttributeError
