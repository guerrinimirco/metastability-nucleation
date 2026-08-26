"""Langer thermal rate + WKB quantum time regression."""
import pytest

from nucleation.barrier import (
    driving_force, critical_radius_noCoulomb, critical_work_noCoulomb,
    work_of_formation)
from nucleation.composition import get_solver_Qs
from nucleation.rates import (
    nucleation_rate, nucleation_time, effective_inertia, quantum_nucleation_time)

SIGMA = 30.0


def test_thermal_and_quantum_golden(regression, params, build_H):
    from eos.general.physics_constants import m_neutron
    g = regression['rates']
    H = build_H(tuple(g['pt']))
    Qs = get_solver_Qs('saddlepoint', 'gcn', params, quark_phase='unpaired')(H)
    Df = float(driving_force(Qs, H))
    Rc = float(critical_radius_noCoulomb(Df, SIGMA))
    Wc = float(critical_work_noCoulomb(Df, SIGMA))
    assert Rc == pytest.approx(g['R_c'], rel=1e-9)
    assert Wc == pytest.approx(g['W_c'], rel=1e-9)

    V = 4.18879e51
    Gamma = float(nucleation_rate(Wc, Rc, SIGMA, H.T, H, Qs))
    tau = float(nucleation_time(Gamma, V))
    assert Gamma == pytest.approx(g['Gamma'], rel=1e-8)
    assert tau == pytest.approx(g['tau'], rel=1e-8)

    rho_H = m_neutron * H.n_B
    ratio = Qs.n_B / H.n_B
    qn = quantum_nucleation_time(
        lambda R: work_of_formation(R, Df, SIGMA, 0.0),
        lambda R: effective_inertia(R, rho_H, ratio), Rc, N_c=1e48)
    assert qn.A == pytest.approx(g['A'], rel=1e-6)
    assert qn.tau_qt == pytest.approx(g['tau_qt'], rel=1e-6)
