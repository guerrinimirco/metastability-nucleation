"""Barrier components + Debye screening form factor."""
import numpy as np
import pytest

from nucleation import barrier as b


def test_f_screening_limits():
    assert b.f_screening(0.0) == pytest.approx(1.0)
    assert b.df_screening(0.0) == pytest.approx(-5.0 / 6.0)
    # monotonically decreasing
    xs = np.linspace(0.01, 8, 200)
    fs = np.array([b.f_screening(x) for x in xs])
    assert np.all(np.diff(fs) < 0)


def test_series_closed_continuity():
    # series (used for x<0.5) and closed form must agree in the crossover band,
    # so f_screening has no jump at the branch point.
    from math import cosh, sinh, exp

    def f_closed(x):
        return 5.0 / (2.0 * x**5) * (
            x**3 - 3.0 * (x + 1.0) * (x * cosh(x) - sinh(x)) * exp(-x))

    # agreement at the same x proves the piecewise f_screening has no jump.
    for x in (0.3, 0.45, 0.5, 0.55, 0.7):
        assert b._f_series(x) == pytest.approx(f_closed(x), abs=1e-9)


def test_unscreened_limit():
    R = np.linspace(0.1, 5, 40)
    dnC, Df, sig = 0.05, -50.0, 30.0
    w_none = b.work_of_formation(R, Df, sig, dnC, lambda_D=None)
    w_big = b.work_of_formation(R, Df, sig, dnC, lambda_D=1e9)
    assert np.max(np.abs(w_none - w_big)) < 1e-6
    # screened pressure -> coulomb_P as lambda_D -> inf
    assert b.coulomb_screened_P(2.0, dnC, 1e9) == pytest.approx(
        b.coulomb_P(2.0, dnC), rel=1e-5)


def test_screened_pressure_is_minus_dEdV():
    # P_Coul = -dE/dV = -(1/4piR^2) dE/dR  =>  dE/dR = -4 pi R^2 P_Coul
    dnC, lam, R = 0.05, 3.0, 2.0
    dR = 1e-5
    dEdR = (b.coulomb_screened_W(R + dR, dnC, lam)
            - b.coulomb_screened_W(R - dR, dnC, lam)) / (2 * dR)
    assert dEdR == pytest.approx(
        -4 * np.pi * R**2 * b.coulomb_screened_P(R, dnC, lam), rel=1e-5)


def test_analytic_fast_paths():
    Df, sig = -50.0, 30.0
    assert float(b.critical_radius_noCoulomb(Df, sig)) == pytest.approx(-2 * sig / Df)
    assert float(b.critical_work_noCoulomb(Df, sig)) == pytest.approx(
        16 * np.pi * sig**3 / (3 * Df**2))
    assert np.isnan(b.critical_radius_noCoulomb(5.0, sig))  # Df>=0 -> no barrier
