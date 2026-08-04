"""Nucleation-curve finders on a computed thermal grid."""
import numpy as np
import pytest

from nucleation.tables import compute_thermal_nucleation_observables
from nucleation.conditions import compute_nucleation_density, nucleation_curve


# sigma for the locus tests. NOT arbitrary: at sigma=30 the barrier is so low
# that tau < tau_target at EVERY temperature on the fixture grid, so there is no
# downward crossing to find and the locus collapses to a single point. sigma=80
# puts the crossing inside the grid at all 18 densities, which is what makes the
# T_of_nB / nB_of_T round-trip assertions meaningful rather than skipped.
LOCUS_SIGMA = 80.0


@pytest.fixture(scope='module')
def thermal_obs(params, H_table):
    """A real thermal grid: tau(n_B, Y_L, T) over the committed fixture table.

    Uses the whole fixture (936 points) rather than an index slice -- the old
    index slice silently depended on the shape of the production table.
    """
    return compute_thermal_nucleation_observables(
        H_table, sigma=LOCUS_SIGMA, params=params, flavor_mode='saddlepoint',
        electric_charge_mode='gcn', quark_phase='unpaired')


def test_density_scans_produce_curve(thermal_obs):
    obs = thermal_obs
    r_T = compute_nucleation_density(obs, tau_target=1e-3, scan='T')
    r_n = compute_nucleation_density(obs, tau_target=1e-3, scan='n_B')
    assert np.isfinite(r_n.T_nuc).sum() > 0
    # nucleation_curve slices the n_B scan into plottable arrays
    nB, T = nucleation_curve(r_n, iYL=0)
    assert nB.shape == T.shape
    assert np.isfinite(T).sum() > 0


# =============================================================================
# The single-point API
# =============================================================================
def test_nucleation_point_reports_a_whole_state(H_interp, params):
    from nucleation import nucleation_point

    pt = nucleation_point(H_interp, n_B_H=0.9, T=25.0, sigma=30.0,
                          params=params, Y_L_H=0.25)
    assert pt.H is not None and pt.Qs is not None
    assert np.isfinite(pt.R_star) and pt.R_star > 0
    assert np.isfinite(pt.W_star) and pt.W_star > 0
    assert np.isfinite(pt.tau) and pt.tau > 0
    # derived quantities agree with their definitions
    assert pt.W_over_T == pytest.approx(pt.W_star / pt.T)
    assert pt.N_B_star == pytest.approx(
        4.0 / 3.0 * np.pi * pt.R_star ** 3 * float(pt.Qs.n_B))
    assert pt.nucleates(tau_target=1e300) is True
    assert pt.nucleates(tau_target=0.0) is False


def test_nucleation_point_matches_the_analysis_adapter(H_interp, params):
    """The core point API and nuc_an.tau_pt must be the SAME number.

    They are two entry points onto one engine; if they drift, the notebook and
    the sigma_crit scan are quoting different physics.
    """
    from nucleation import nucleation_point, hadronic_point
    from nucleation.analysis import tau_pt, critical_droplet_pt, NucConfig

    nuc = NucConfig()
    H = hadronic_point(H_interp, 0.9, 0.25, 25.0)
    pt = nucleation_point(H, 0.9, 25.0, 30.0, params=params, V=nuc.V)
    tau_adapter = tau_pt(30.0, H, 25.0, 'saddlepoint', 'gcn', 'unpaired', {},
                         params, 80.0, nuc)
    R_c, W_c, _ = critical_droplet_pt(
        30.0, H, 25.0, 'saddlepoint', 'gcn', 'unpaired', {}, params, 80.0, nuc)
    assert pt.tau == pytest.approx(tau_adapter, rel=1e-12)
    assert pt.R_star == pytest.approx(R_c, rel=1e-12)
    assert pt.W_star == pytest.approx(W_c, rel=1e-12)


def test_stable_hadronic_point_never_nucleates(H_interp, params):
    """Where H is stable the engine reports W*=+inf, so tau must be +inf."""
    from nucleation import nucleation_point

    pt = nucleation_point(H_interp, n_B_H=0.3, T=20.0, sigma=30.0,
                          params=params, Y_L_H=0.25, flavor_mode='frozen')
    assert np.isposinf(pt.W_star)
    assert np.isposinf(pt.tau)
    assert not pt.nucleates(tau_target=1e300)


# =============================================================================
# The locus facade
# =============================================================================
def test_condition_from_table_matches_the_raw_finder(thermal_obs):
    from nucleation import NucleationCondition
    from nucleation.conditions import compute_nucleation_density, nucleation_curve

    cond = NucleationCondition.from_table(thermal_obs, tau_target=1e-3)
    raw = compute_nucleation_density(thermal_obs, tau_target=1e-3, scan='n_B')
    nB_raw, T_raw = nucleation_curve(raw, iYL=0)

    nB_c, T_c = cond.curve(Y_L=thermal_obs.hadronic_grids['Y_L_H'][0])
    m = np.isfinite(T_raw)
    assert np.allclose(T_c, T_raw[m], equal_nan=True)
    assert np.allclose(nB_c, nB_raw[m])


def test_T_of_nB_and_its_inverse_are_consistent(thermal_obs):
    """nB_of_T must invert T_of_nB -- they are one locus, not two curves."""
    from nucleation import NucleationCondition

    cond = NucleationCondition.from_table(thermal_obs, tau_target=1e-3)
    YL = thermal_obs.hadronic_grids['Y_L_H'][0]
    nB, T = cond.curve(Y_L=YL)
    assert nB.size >= 4, "locus collapsed -- see LOCUS_SIGMA"
    nB_mid = float(nB[len(nB) // 2])
    T_mid = cond.T_of_nB(nB_mid, Y_L=YL)
    assert np.isfinite(T_mid)
    back = cond.nB_of_T(T_mid, Y_L=YL)
    if np.isfinite(back):
        assert back == pytest.approx(nB_mid, rel=0.05)


def test_scalar_in_scalar_out_array_in_array_out(thermal_obs):
    from nucleation import NucleationCondition

    cond = NucleationCondition.from_table(thermal_obs, tau_target=1e-3)
    YL = thermal_obs.hadronic_grids['Y_L_H'][0]
    nB, _ = cond.curve(Y_L=YL)
    assert nB.size >= 2, "locus collapsed -- see LOCUS_SIGMA"
    one = cond.T_of_nB(float(nB[0]), Y_L=YL)
    many = cond.T_of_nB(nB[:2], Y_L=YL)
    assert np.isscalar(one) or np.ndim(one) == 0
    assert np.shape(many) == (2,)


def test_T_nuc_shorthand_agrees_with_the_object(thermal_obs):
    from nucleation import NucleationCondition, T_nuc

    cond = NucleationCondition.from_table(thermal_obs, tau_target=1e-3)
    YL = thermal_obs.hadronic_grids['Y_L_H'][0]
    nB, _ = cond.curve(Y_L=YL)
    assert nB.size >= 2, "locus collapsed -- see LOCUS_SIGMA"
    x = float(nB[len(nB) // 2])
    assert T_nuc(thermal_obs, x, YL, tau_target=1e-3) == pytest.approx(
        cond.T_of_nB(x, Y_L=YL), nan_ok=True)


def test_from_point_solver_needs_no_table(H_interp, params):
    """The constructor that answers 'at what T does THIS parameter set nucleate?'
    without building a Q* table first."""
    from nucleation import NucleationCondition

    T_grid = np.linspace(5.0, 80.0, 16)
    n_B_grid = np.array([0.8, 0.9, 1.0])
    cond = NucleationCondition.from_point_solver(
        H_interp, T_grid, n_B_grid, sigma=LOCUS_SIGMA, params=params,
        tau_target=1e-3, Y_L_H=0.25)
    assert cond.T_nuc.shape == n_B_grid.shape
    assert repr(cond).startswith('NucleationCondition')
    # any converged point must genuinely sit on the locus
    from nucleation import tau_at
    for nB, T in zip(n_B_grid[cond.converged], cond.T_nuc[cond.converged]):
        tau = tau_at(H_interp, float(nB), float(T), LOCUS_SIGMA,
                     params=params, Y_L_H=0.25)
        assert np.log10(tau) == pytest.approx(np.log10(1e-3), abs=0.15)


def test_condition_accepts_quantum_observables(H_table, params):
    """Thermal and quantum both expose .tau, so the facade takes either."""
    import copy
    from nucleation import NucleationCondition
    from nucleation.tables import compute_quantum_nucleation_observables

    sub = copy.copy(H_table)
    inB, iYL, iT = range(12, 18), range(2, 4), range(3, 7)
    sub.grids = {'n_B': H_table.grids['n_B'][12:18],
                 'Y_L': H_table.grids['Y_L'][2:4],
                 'T': H_table.grids['T'][3:7]}
    sub.data = {k: (v[np.ix_(inB, iYL, iT)] if getattr(v, 'ndim', 0) == 3 else v)
                for k, v in H_table.data.items()}
    qobs = compute_quantum_nucleation_observables(
        sub, sigma=30.0, params=params, quark_phase='unpaired')
    cond = NucleationCondition.from_table(qobs, tau_target=1e-3)
    assert cond.T_nuc.shape[0] == len(sub.grids['n_B'])
