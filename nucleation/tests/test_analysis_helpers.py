"""Helpers extracted from the notebook into nucleation.analysis.

These are deliberately testable on hand-made arrays with no EoS and no TOV
solve, which is most of why they were worth extracting: in the notebook they
were unreachable except by running the figure that used them.
"""
import numpy as np
import pytest

from nucleation.analysis import (
    TOV_COL, stable_branch, max_masses, branch_interp, snapshot_key,
    nearest_trapped_sequence, nucleates, pick_cell_by_Mmax, to_latex_tabular,
    resample_curve, REGIME,
)
from nucleation.analysis.figure import (
    symmetric_vlim, resample_profiles, quantile_bins, summary_points,
)


def _seq(M):
    """A TOV sequence with a prescribed gravitational-mass column."""
    M = np.asarray(M, dtype=float)
    n = M.size
    out = np.zeros((n, 6))
    out[:, TOV_COL['e_c']] = np.linspace(200, 2000, n)
    out[:, TOV_COL['n_Bc']] = np.linspace(0.3, 1.5, n)
    out[:, TOV_COL['R']] = np.linspace(12.0, 10.0, n)
    out[:, TOV_COL['M']] = M
    out[:, TOV_COL['M_B']] = M * 1.15
    return out


# =============================================================================
# stellar
# =============================================================================
def test_stable_branch_stops_at_the_first_maximum():
    """Rows past M_max are unstable to radial collapse and are not real stars."""
    seq = _seq([1.0, 1.5, 2.0, 2.1, 1.9, 1.4])
    st = stable_branch(seq)
    assert st.shape[0] == 4
    assert st[-1, TOV_COL['M']] == pytest.approx(2.1)


def test_max_masses():
    seq = _seq([1.0, 1.5, 2.1, 1.9])
    M_max, MB_max = max_masses(seq)
    assert M_max == pytest.approx(2.1)
    assert MB_max == pytest.approx(2.1 * 1.15)


def test_branch_interp_survives_duplicate_abscissae():
    """The reason three copies of this function existed in the notebook.

    The cold CFL quark-star sequence can repeat a baryonic mass; scipy's cubic
    spline rejects duplicate x outright, so the naive version raised and a
    second, then third, variant was written instead of fixing it.
    """
    seq = _seq([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    seq[3, TOV_COL['M_B']] = seq[2, TOV_COL['M_B']]        # exact duplicate
    f = branch_interp(seq, 'M_B', 'M')
    val = f(float(seq[1, TOV_COL['M_B']]))
    assert np.isfinite(val)


def test_branch_interp_is_degenerate_safe():
    """Too few usable points returns NaN rather than raising -- one bad
    parameter cell must not abort a whole figure."""
    f = branch_interp(_seq([1.0, 0.9]), 'M_B', 'M')
    assert np.isnan(f(1.0)) or np.isfinite(f(1.0))         # must not raise


def test_snapshot_key_is_one_rounding_policy():
    """The notebook had three key policies; they agreed only by luck."""
    assert snapshot_key(0.35000000000000003, 1.5) == snapshot_key(0.35, 1.5)
    assert snapshot_key(0.35, 1.5) == (0.35, 1.5)


def test_nearest_trapped_sequence_picks_the_closest():
    store = {(0.25, 2.0): _seq([1.0, 2.0]), (0.35, 1.5): _seq([1.0, 2.2])}
    got = nearest_trapped_sequence(store, 0.34, 1.55)
    assert got[-1, TOV_COL['M']] == pytest.approx(2.2)


# =============================================================================
# outcomes
# =============================================================================
def test_nucleates_conventions():
    assert nucleates(100.0, 50.0) is True         # sigma below threshold
    assert nucleates(100.0, 150.0) is False       # sigma above threshold
    assert nucleates(np.inf, 1e9) is True         # +inf = always nucleates
    assert nucleates(np.nan, 1.0) is False        # failed solve is not "yes"


def test_pick_cell_by_Mmax_takes_the_lowest_sigma_crit_match():
    """Anchoring to an observed M_max should give the CONSERVATIVE cell."""
    sig = np.array([[[10.0, 200.0], [50.0, 30.0]]])       # (1, 2, 2)
    mmax = np.array([[[2.20, 2.20], [2.60, 2.19]]])
    pset, M, sc = pick_cell_by_Mmax(
        sig, mmax, alpha_slices=[0.1], B4_grid=[145.0, 150.0],
        Delta0_grid=[80.0, 100.0], target=2.20, tol=0.02, m_s=100.0)
    assert sc == pytest.approx(10.0)                      # not 200 or 30
    assert M == pytest.approx(2.20)
    assert pset['m_s'] == 100.0

    none, M2, sc2 = pick_cell_by_Mmax(
        sig, mmax, [0.1], [145.0, 150.0], [80.0, 100.0],
        target=3.5, tol=0.01, m_s=100.0)
    assert none is None and np.isnan(M2) and np.isnan(sc2)


def test_to_latex_tabular_renders_nan_as_a_dash():
    import pandas as pd
    df = pd.DataFrame([{'type': 'NS', 'M_rem': 1.4, 'E_conv_e53': np.nan}])
    tex = to_latex_tabular(df)
    assert r'\begin{tabular}' in tex and r'\end{tabular}' in tex
    assert '--' in tex                                    # the NaN cell
    assert r'E\_conv\_e53' in tex                         # underscores escaped


# =============================================================================
# stability
# =============================================================================
def test_resample_curve_falls_back_when_no_root_exists():
    target = np.linspace(0, 1, 5)
    out = resample_curve(np.full(3, np.nan), np.linspace(0, 1, 3), target, 150.0)
    assert np.allclose(out, 150.0)
    out2 = resample_curve(None, np.linspace(0, 1, 3), target, 42.0)
    assert np.allclose(out2, 42.0)


def test_resample_curve_interpolates_partial_curves():
    src = np.array([0.0, 0.5, 1.0])
    curve = np.array([140.0, np.nan, 160.0])
    out = resample_curve(curve, src, np.array([0.0, 0.5, 1.0]), 0.0)
    assert out[1] == pytest.approx(150.0)          # bridges the NaN


# =============================================================================
# figure primitives
# =============================================================================
def test_symmetric_vlim_ignores_outliers():
    """A handful of extreme cells must not flatten every other panel."""
    field = np.concatenate([np.full(99, 1.0), [1000.0]])
    assert symmetric_vlim([field], pct=90) == pytest.approx(1.0, abs=0.2)
    assert symmetric_vlim([field], override=7.5) == 7.5
    assert symmetric_vlim([]) == 1.0


def test_resample_profiles_is_nan_outside_each_curve_span():
    """A short curve must not prop up the envelope where it does not reach."""
    curves = [({'M': np.array([1.0, 2.0]), 'R': np.array([12.0, 11.0])}, 50.0),
              ({'M': np.array([1.0, 1.5]), 'R': np.array([12.5, 12.0])}, 80.0)]
    grid = np.array([1.0, 1.5, 2.0])
    prof = resample_profiles(curves, 'M', 'R', grid)
    assert prof.shape == (2, 3)
    assert np.isfinite(prof[0]).all()
    assert np.isnan(prof[1, 2])                    # second curve stops at 1.5


def test_quantile_bins_are_equal_count():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    curves = [({'M': np.array([1.0, 2.0]), 'R': np.array([12.0, 11.0])},
               float(s)) for s in range(1, 13)]
    bins = quantile_bins(curves, 3, plt.get_cmap('viridis'), Normalize(1, 12))
    assert len(bins) == 3
    assert [len(b[2]) for b in bins] == [4, 4, 4]  # equal COUNT, not width


def test_summary_points_radius_definitions_differ():
    M = np.array([1.0, 1.4, 1.8, 2.0, 1.9])
    R = np.array([13.0, 12.5, 11.5, 11.0, 10.0])
    curves = [({'M': M, 'R': R}, 50.0)]
    r14 = summary_points(curves, 'R_1.4')[0]
    rmm = summary_points(curves, 'R_Mmax')[0]
    rmx = summary_points(curves, 'R_max')[0]
    assert r14[0] == pytest.approx(2.0)            # M_max is the same for all
    assert r14[1] == pytest.approx(12.5)           # radius at 1.4 M_sun
    assert rmm[1] == pytest.approx(11.0)           # radius at M_max
    assert rmx[1] == pytest.approx(13.0)           # largest radius anywhere
    with pytest.raises(ValueError):
        summary_points(curves, 'nonsense')


def test_regime_codes_are_distinct():
    assert len(set(REGIME.values())) == len(REGIME)
    assert REGIME['none'] == -1


# =============================================================================
# Drawing primitives: they must actually render
# =============================================================================
@pytest.fixture
def ax():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, a = plt.subplots()
    yield a
    plt.close(fig)


def test_map_layers_render(ax):
    from nucleation.analysis.figure import (sigma_map, diverging_map, iso_lines,
                                            reject_outlines, regime_outlines)
    from nucleation.analysis import REASON_CODE

    B4 = np.linspace(130, 180, 12)
    D0 = np.linspace(0, 200, 10)
    X, Y = np.meshgrid(B4, D0)
    field = np.hypot(X - 155, Y - 100)
    field[0, 0] = np.nan                     # a not-viable cell

    assert sigma_map(ax, X, Y, field) is not None
    iso_lines(ax, X, Y, field, [20.0, 40.0], 'white', fmt='%.0f')
    assert diverging_map(ax, X, Y, field - 40.0, vlim=30.0) is not None

    reason = np.full(X.shape, REASON_CODE['OK'])
    reason[:3, :4] = REASON_CODE['mmax']
    reason[7:, 8:] = REASON_CODE['rehadr']    # exercises the leader-arrow path
    reject_outlines(ax, X, Y, reason)

    regime = np.zeros(X.shape, dtype=int)
    regime[5:, :] = 2
    regime_outlines(ax, X, Y, regime)


def test_mark_helpers_render(ax):
    from nucleation.analysis.figure import (mass_marks, label_with_arrows,
                                            isentrope_mass_markers)

    seq = _seq([0.9, 1.1, 1.3, 1.5, 1.7, 1.75, 1.6])
    yv = np.linspace(10.0, 12.0, seq.shape[0])
    mass_marks(ax, seq, yv, 'k')
    mass_marks(ax, seq, yv, 'k', label=False, dot_arrow=True)

    label_with_arrows(ax, [(1.0, 2.0), (1.2, 2.1)], 'PSR test', 'k')
    label_with_arrows(ax, [(np.nan, np.nan)], 'skipped', 'k')   # must not raise

    isentrope_mass_markers(ax, seq, lambda nb: 30.0, 'k', n_sat=0.16)


def test_band_skips_thin_columns(ax):
    """A percentile over one or two curves is noise drawn as if it were a band."""
    from nucleation.analysis.figure import band

    prof = np.array([[1.0, 2.0, np.nan],
                     [1.1, 2.1, np.nan],
                     [1.2, 2.2, 5.0]])
    grid = np.array([0.0, 1.0, 2.0])
    band(ax, prof, grid, 'k', label='x', min_n=3)
    # the third column is backed by 1 curve -> dropped from the drawn band
    xs = ax.lines[0].get_xdata()
    assert len(xs) == 2


def test_replay_accepted_can_return_the_parameters(monkeypatch):
    """with_params must pair each curve with the cell it came from.

    The pairing is the whole point: the accept ordering and the max_curves
    stride live inside replay_accepted, so a caller reconstructing (alpha, B4,
    Delta0) outside would silently drift out of step and mis-colour a plot with
    no error anywhere.
    """
    import nucleation.analysis.replay as rp

    # One accepted cell per (Delta0, B4) corner, with a known sigma_crit each.
    sig = np.full((1, 2, 2), np.nan)
    sig[0, 0, 0], sig[0, 1, 1] = 10.0, 20.0
    alpha, B4, D0 = [0.1], np.array([140.0, 150.0]), np.array([50.0, 60.0])

    # Stand in for the EoS+TOV solve: identify the curve by the params it got.
    monkeypatch.setattr(rp, 'replay_cfl',
                        lambda a, b, d, cfg: dict(mu=[a], P=[b], R=[d], M=[1.0]))

    out = rp.replay_accepted(sig, alpha, B4, D0, cfg=None, n_jobs=1,
                             verbose=False, with_params=True)
    assert [(s, p) for _, s, p in out] == [(10.0, (0.1, 140.0, 50.0)),
                                           (20.0, (0.1, 150.0, 60.0))]
    # each curve carries the parameters it was actually solved at
    assert [(c['P'][0], c['R'][0]) for c, _, _ in out] == [(140.0, 50.0),
                                                           (150.0, 60.0)]
    # default stays a 2-tuple, so every existing caller is untouched
    assert all(len(t) == 2 for t in
               rp.replay_accepted(sig, alpha, B4, D0, cfg=None, n_jobs=1,
                                  verbose=False))
