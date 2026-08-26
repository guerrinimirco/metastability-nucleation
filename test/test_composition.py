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

# Tolerance on the quantities the CFL flavour lock forces to zero. This is the
# tightest claim the code makes about them, not a chosen number: the lock is a
# pair of RESIDUAL ROWS of a root find (`Qs.Y_C` and `Qs.Y_S - 1` in
# solve_saddlepoint_cfl), and robust_root() accepts a solution at
# max|residual| < 1e-8. Asserting anything tighter would fail a correctly
# converged point.
LOCK_TOL = 1e-8


def test_regression_solver_cases(regression, params, build_H):
    """Every unaffected (flavor, charge, phase) Q* matches the pre-rewrite golden.

    The quantities the CFL lock forces to zero are checked AGAINST ZERO rather
    than against the golden. There the golden records only where the root find
    happened to stop, so a relative comparison pins the floating-point
    association of the quark block rather than any physics -- and duly broke on
    a one-ulp reassociation in `eos.alphabag` that moved no physical quantity.
    Asserting the lock is both the stronger claim and the stable one.
    """
    for c in regression['cases']:
        if not c['converged']:
            continue
        H = build_H(tuple(c['pt']))
        kw = dict(quark_phase=c['phase'],
                  Delta0=DELTA0 if c['phase'] == 'cfl' else None, sigma=SIGMA)
        Qs = get_solver_Qs(c['flavor'], c['charge'], params, **kw)(H)
        assert Qs is not None
        tag = f"{c['flavor']}/{c['charge']}/{c['phase']}"

        locked = set()
        if c['phase'] == 'cfl':
            # Y_C = 0 and Y_S = +1 identically in the locked phase, in every
            # charge mode. mu_C = mu_u - mu_d is a LINEAR image of the first
            # (equal densities at equal masses => equal potentials), so it
            # inherits the residual gate once scaled by mu_B.
            assert abs(Qs.Y_C) < LOCK_TOL, f"{tag}: Y_C not locked to zero"
            assert abs(Qs.mu_C) < LOCK_TOL * abs(Qs.mu_B), \
                f"{tag}: mu_C not locked to zero"
            locked |= {'Y_C', 'mu_C'}
            if c['charge'] == 'lcn':
                # CFL is electrically neutral by the lock, so a LOCALLY neutral
                # droplet needs no electrons at all. (Under global neutrality it
                # is charged and mu_e = mu_e^H is a real ~300 MeV sea, which is
                # why this is keyed on the charge mode.)
                assert abs(Qs.Y_e) < LOCK_TOL, f"{tag}: Y_e not zero"
                # mu_e is a NONLINEAR image of Y_e -- n_e ~ mu_e T^2 at these
                # temperatures -- so no bound on it follows from the gate: a
                # point converging to Y_e at the gate would carry mu_e five
                # orders above where it sits now and still be correct. Y_e
                # above is the assertion; mu_e adds nothing to it.
                locked |= {'Y_e', 'mu_e'}

        for k, v in c['Qs'].items():
            if k in locked:
                continue
            assert getattr(Qs, k) == pytest.approx(v, rel=1e-9), f"{tag} {k}"


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
