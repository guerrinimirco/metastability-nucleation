"""sigma_crit per-point layer: engine conventions + sigma_target_pt robustness."""
import numpy as np
import pytest

from nucleation.analysis.sigma_crit import (
    critical_droplet_pt, tau_pt, sigma_target_pt, rehad_flags, NucConfig)

SIGMA = 30.0
DELTA0 = 80.0


def test_critical_droplet_pt_conventions(params, build_H):
    nuc = NucConfig()
    # genuine droplet
    H = build_H((0.9, 0.25, 25.0))
    R_c, W_c, Qs = critical_droplet_pt(
        SIGMA, H, 25.0, 'saddlepoint', 'gcn', 'unpaired', {}, params, DELTA0, nuc)
    assert np.isfinite(R_c) and np.isfinite(W_c) and Qs is not None
    # stable H: (nan, +inf, Qs)
    Hs = build_H((0.3, 0.25, 20.0))
    R_c, W_c, Qs = critical_droplet_pt(
        SIGMA, Hs, 20.0, 'frozen', 'gcn', 'unpaired', {}, params, DELTA0, nuc)
    assert np.isnan(R_c) and np.isposinf(W_c) and Qs is not None


def test_tau_pt_runs(params, build_H):
    nuc = NucConfig()
    H = build_H((0.9, 0.25, 25.0))
    tau = tau_pt(SIGMA, H, 25.0, 'saddlepoint', 'coulomb_minimize', 'cfl', {},
                 params, DELTA0, nuc)
    assert (np.isfinite(tau) and tau > 0) or np.isposinf(tau)


def test_sigma_target_pt_synthetic(monkeypatch):
    """sigma_target_pt on a controlled tau(sigma): clean crossing / gaps / +inf."""
    import nucleation.analysis.sigma_crit as sc
    nuc = NucConfig(sig_lo=1.0, sig_hi=300.0, n_sigma_scan=40, tau_target=1e-3)
    lo = np.log10(nuc.tau_target)

    # 1) clean monotone crossing at sigma ~ 150 (tau rises through target)
    def tau_clean(s, *a, **k):
        return 10.0 ** (lo + (s - 150.0) / 50.0)
    monkeypatch.setattr(sc, 'tau_pt', tau_clean)
    sig = sigma_target_pt(None, None, 'f', 'c', 'p', None, None, nuc)
    assert sig == pytest.approx(150.0, abs=2.0)

    # 2) always below target -> nucleates for all sigma -> +inf
    monkeypatch.setattr(sc, 'tau_pt', lambda s, *a, **k: 10.0 ** (lo - 5))
    assert np.isposinf(sigma_target_pt(None, None, 'f', 'c', 'p', None, None, nuc))

    # 3) never nucleates (tau always above target) -> NaN
    monkeypatch.setattr(sc, 'tau_pt', lambda s, *a, **k: 10.0 ** (lo + 5))
    assert np.isnan(sigma_target_pt(None, None, 'f', 'c', 'p', None, None, nuc))


def test_rehad_flags_synthetic():
    """rehad_flags on controlled ΔP(n_BH) profiles (no EoS needed)."""
    n = np.linspace(0.3, 1.5, 13)

    # monotone increasing, crosses 0 once -> both flags True, cross at the 0-point
    d = n - 0.9
    no_re, strong, cross = rehad_flags(n, d)
    assert no_re and strong
    assert cross == pytest.approx(0.9, abs=n[1] - n[0])

    # crosses positive then dips back negative -> no_rehad False, not strong
    d2 = np.array([-1, -0.5, 0.2, 0.5, -0.1, 0.3, 1, 2, 3, 4, 5, 6, 7], float)
    no_re2, strong2, _ = rehad_flags(n, d2)
    assert not no_re2 and not strong2

    # never reaches 0 -> no crossing: no_rehad False, cross NaN
    no_re3, _, cross3 = rehad_flags(n, np.full_like(n, -1.0))
    assert not no_re3 and np.isnan(cross3)

    # NaNs (failed solves) dropped before evaluation
    d4 = n - 0.9
    d4[2] = np.nan
    assert rehad_flags(n, d4)[0]


def test_rehad_pressure_profile_smoke(H_interp, params):
    """rehad_pressure_profile runs on the real trapped EoS and returns ΔP."""
    from nucleation.analysis.sigma_crit import rehad_pressure_profile
    n_grid = np.linspace(0.4, 1.2, 6) * 0.16   # ~2.5..7.5 n_sat span in fm^-3
    n, dP = rehad_pressure_profile(
        H_interp, params, n_grid, Y_L_H=0.25, T=25.0,
        flavor_mode='saddlepoint', electric_charge_mode='gcn')
    assert n.shape == dP.shape == n_grid.shape
    assert np.isfinite(dP).any()   # at least some Q* solves converge
