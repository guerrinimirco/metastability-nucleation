"""Q* composition solvers: regression on unaffected modes + the CFL fix."""
import numpy as np
import pytest

from nucleation.composition import (
    get_solver_Qs, solve_coulomb_minimize,
    solve_coulomb_minimize_cfl, solve_coulomb_minimize_cfl_at_R,
    solve_coulomb_minimize_at_R,
)
from nucleation.barrier import driving_force, work_of_formation

SIGMA = 30.0
DELTA0 = 80.0


def test_regression_solver_cases(regression, params, build_H):
    """Every unaffected (flavor, charge, phase) Q* matches the pre-rewrite golden."""
    for c in regression['cases']:
        if not c['converged']:
            continue
        H = build_H(tuple(c['pt']))
        kw = dict(quark_phase=c['phase'],
                  Delta0=DELTA0 if c['phase'] == 'cfl' else None, sigma=SIGMA)
        Qs = get_solver_Qs(c['flavor'], c['charge'], params, **kw)(H)
        assert Qs is not None
        for k, v in c['Qs'].items():
            if abs(v) > 1e-12:
                assert getattr(Qs, k) == pytest.approx(v, rel=1e-9), \
                    f"{c['flavor']}/{c['charge']}/{c['phase']} {k}"


def test_coulomb_minimize_unpaired_regression(regression, params, build_H):
    for c in regression['coulomb_minimize']['unpaired']:
        H = build_H(tuple(c['pt']))
        out = solve_coulomb_minimize(H, params, SIGMA, True, True, True)
        assert out is not None
        Qs, Rc = out
        assert Rc == pytest.approx(c['R_c'], rel=1e-8)


def _argmax_W_cfl(H, params, Rc_hint):
    """argmax over R of W(R) built from the fixed-R CFL solver."""
    R = np.linspace(max(0.05, 0.3 * Rc_hint), 3 * Rc_hint, 400)
    W = np.full_like(R, np.nan)
    guess = None
    for i, r in enumerate(R):
        q = solve_coulomb_minimize_cfl_at_R(r, H, params, DELTA0, SIGMA,
                                            True, True, True, initial_guess=guess)
        if q is None:
            continue
        guess = np.array([q.mu_u, q.mu_d, q.mu_s, q.mu_e])
        Df = float(driving_force(q, H))
        dnC = (q.Y_C - q.Y_e) * q.n_B
        W[i] = work_of_formation(r, Df, SIGMA, dnC)
    return R[np.nanargmax(W)]


def test_cfl_fix_solver_equals_argmax(params, build_H):
    """CFL kill-test: the 5-eq self-consistent R_c must equal argmax_R W(R).

    This is exactly the invariant the old mu_B (instead of mu_B+mu_S) condition
    violated (2.55 vs 2.89 fm at one point).
    """
    for pt in [(0.8, 0.25, 20.0), (1.1, 0.10, 30.0)]:
        H = build_H(pt)
        out = solve_coulomb_minimize_cfl(H, params, DELTA0, SIGMA, True, True, True)
        assert out is not None
        _, Rc5 = out
        Rc_argmax = _argmax_W_cfl(H, params, Rc5)
        assert Rc5 == pytest.approx(Rc_argmax, abs=0.02), \
            f"pt={pt}: 5eq {Rc5:.4f} vs argmax {Rc_argmax:.4f}"


def test_cfl_fix_changed_from_buggy(buggy_cfl, params, build_H):
    """The fixed CFL R_c differs from the archived buggy value (fix took effect)."""
    for c in buggy_cfl['cfl']:
        if not c['converged']:
            continue
        H = build_H(tuple(c['pt']))
        out = solve_coulomb_minimize_cfl(H, params, DELTA0, SIGMA, True, True, True)
        assert out is not None
        _, Rc_new = out
        assert abs(Rc_new - c['R_c']) > 1e-3    # meaningfully different


def test_unpaired_solver_equals_argmax(params, build_H):
    """Regression guard: unpaired already satisfied solver == argmax."""
    H = build_H((0.9, 0.25, 25.0))
    out = solve_coulomb_minimize(H, params, SIGMA, True, True, True)
    _, Rc5 = out
    R = np.linspace(0.3 * Rc5, 3 * Rc5, 400)
    W = np.full_like(R, np.nan)
    guess = None
    for i, r in enumerate(R):
        q = solve_coulomb_minimize_at_R(r, H, params, SIGMA, True, True, True,
                                        initial_guess=guess)
        if q is None:
            continue
        guess = np.array([q.mu_u, q.mu_d, q.mu_s, q.mu_e])
        Df = float(driving_force(q, H))
        dnC = (q.Y_C - q.Y_e) * q.n_B
        W[i] = work_of_formation(r, Df, SIGMA, dnC)
    assert Rc5 == pytest.approx(R[np.nanargmax(W)], abs=0.02)
