r"""
Nucleation barrier -- W(R) components
=====================================

Pure, EOS-agnostic building blocks of the work of formation of a spherical
quark droplet of radius R inside hadronic matter (thin-wall CNT):

    W(R) = W_bulk + W_surface + W_Coulomb
         = (4/3) pi R^3 Delta_f + 4 pi R^2 sigma + E_Coulomb(R)

Everything here is a function of the *reduced* inputs (R, Delta_f, sigma,
delta_n_C, lambda_D) only -- no phase solve, no EOS call except the single
electron-susceptibility helper. The droplet composition (hence Delta_f and
delta_n_C) is produced by ``nucleation.composition``; assembling and maximizing
W(R) is done by ``nucleation.critical``.

Public API (general -- valid for ANY Delta_f)
---------------------------------------------
    driving_force(Qs, H)                          -> Delta_f  [MeV/fm^3]
    bulk_W, surface_W, coulomb_W                  -> W components  [MeV]
    coulomb_screened_W(R, dnC, lambda_D)          -> screened Coulomb energy [MeV]
    work_of_formation(R, Delta_f, sigma, dnC=0, lambda_D=None)  -> W(R) [MeV]
    switching_step / switching_tanh / get_switching_function    (unpCFL blend)

Public API (analytic FAST PATHS -- closed forms, no root find)
--------------------------------------------------------------
    critical_radius_noCoulomb(Delta_f, sigma)     -> R_c = -2 sigma / Delta_f
    critical_work_noCoulomb(Delta_f, sigma)       -> W_c = 16 pi sigma^3 / (3 Delta_f^2)

Public API (numeric critical radius with Coulomb)
-------------------------------------------------
    critical_radius_coulomb(Delta_f, sigma, dnC)                 (unscreened)
    critical_radius_coulomb_screened(Delta_f, sigma, dnC, lambda_D)

Coulomb helpers (see the "two Coulomb conventions" note at the bottom)
---------------------------------------------------------------------
    coulomb_delta_mu_e, coulomb_delta_P    -> used INSIDE the coulomb_minimize solve
    coulomb_P, coulomb_screened_P          -> P_Coul = -dE_Coul/dV (a-posteriori modes)
    f_screening / df_screening             -> Debye screening form factor f(R/lambda_D)
    electron_susceptibility, debye_length  -> lambda_D from chi_e = dn_e/dmu_e|_T
"""

from fractions import Fraction
from math import cosh, sinh, exp, factorial

import numpy as np
from scipy.optimize import root, brentq

from eos.general.physics_constants import alpha_EM, hc
from eos.general.thermodynamics_leptons import electron_thermo


# =============================================================================
# Bulk driving force  (public, general)
# =============================================================================
def driving_force(Qs, H):
    """Bulk driving force Delta_f = f_Qs - f_H per unit droplet volume.

    Delta_f = -(P_Qs - P_H) + n_B^Qs sum_i Y_i^Qs (mu_i^Qs - mu_i^H),  i=B,C,S,e,nu

    (Y_B = 1 by definition). The hadronic phase is metastable -- droplet growth
    is favoured -- when Delta_f < 0. Qs must expose P_total, n_B, mu_{B,C,S,e,nu}
    and Y_{C,S,e,nu}; H must expose P_total and mu_{B,C,S,e,nu}.
    """
    Delta_P = Qs.P_total - H.P_total
    chem_sum = (
        1.0 * (Qs.mu_B - H.mu_B)
        + Qs.Y_C * (Qs.mu_C - H.mu_C)
        + Qs.Y_S * (Qs.mu_S - H.mu_S)
        + Qs.Y_e * (Qs.mu_e - H.mu_e)
        + Qs.Y_nu * (Qs.mu_nu - H.mu_nu)
    )
    return -Delta_P + Qs.n_B * chem_sum


# =============================================================================
# Bulk and surface work  (public, general)
# =============================================================================
def bulk_W(R, Delta_f_bulk):
    """Bulk (volume) work  W_bulk = (4/3) pi R^3 Delta_f  [MeV]."""
    return 4.0 / 3.0 * np.pi * R**3 * Delta_f_bulk


def surface_W(R, sigma):
    """Surface work  W_surface = 4 pi R^2 sigma  [MeV]."""
    return 4.0 * np.pi * R**2 * sigma


# =============================================================================
# Coulomb energy / pressure (uniform-charge sphere)  (public, general)
# =============================================================================
def coulomb_W(R, delta_n_C):
    """Unscreened Coulomb energy of a uniformly charged droplet.

    W_Coulomb = (16/15) pi^2 hc alpha delta_n_C^2 R^5   [MeV]

    delta_n_C = (Y_C^Qs - Y_e^Qs) n_B^Qs is the droplet net charge density.
    """
    return (16.0 / 15.0) * np.pi**2 * hc * alpha_EM * delta_n_C**2 * R**5


def coulomb_P(R, delta_n_C):
    """Thermodynamic Coulomb pressure  P_Coul = -dE_Coul/dV|_n  [MeV/fm^3].

    P_Coul = -(4/3) alpha pi hc R^2 delta_n_C^2

    This is the a-posteriori Coulomb pressure used by gcn_coulomb / screening
    (NOT the coulomb_minimize delta_P; see the note at the bottom).
    """
    return -(4.0 / 3.0) * alpha_EM * np.pi * hc * R**2 * delta_n_C**2


def coulomb_delta_mu_e(R, delta_n_C):
    """Coulomb shift of mu_e that enters the coulomb_minimize composition solve.

    delta_mu_e = (8/5) alpha pi hc R^2 delta_n_C

    (Step 1a of the coulomb_minimize derivation: minimizing W incl. E_Coul wrt
    Y_e gives mu_e^Qs = mu_e^H + delta_mu_e.)
    """
    return (8.0 / 5.0) * alpha_EM * np.pi * hc * R**2 * delta_n_C


def coulomb_delta_P(R, delta_n_C):
    """Coulomb term in the coulomb_minimize pressure balance (dW/dR = 0).

    delta_P = (4/15) alpha pi hc R^2 delta_n_C^2

    Enters as  P_Qs + delta_P - 2 sigma/R = P_H. It is NOT -dE_Coul/dV: it also
    carries the electron chem-sum contribution (see the note at the bottom).
    """
    return (4.0 / 15.0) * alpha_EM * np.pi * hc * R**2 * delta_n_C**2


# =============================================================================
# Work of formation at arbitrary radius  (public, general)
# =============================================================================
def work_of_formation(R, Delta_f_bulk, sigma, delta_n_C=0.0, lambda_D=None):
    """Total work of formation W(R) [MeV].

    W(R) = (4/3) pi R^3 Delta_f + 4 pi R^2 sigma + E_Coulomb(R)

    * delta_n_C = 0            -> no Coulomb (lcn / gcn).
    * delta_n_C != 0, lambda_D None -> unscreened Coulomb (gcn_coulomb / coulomb_minimize).
    * delta_n_C != 0, finite lambda_D -> Debye-screened Coulomb (screening mode).

    lambda_D=None reproduces the unscreened result bit-for-bit.
    """
    if lambda_D is None:
        W_coul = coulomb_W(R, delta_n_C)
    else:
        W_coul = coulomb_screened_W(R, delta_n_C, lambda_D)
    return bulk_W(R, Delta_f_bulk) + surface_W(R, sigma) + W_coul


# =============================================================================
# Analytic fast paths (no Coulomb)  (public)
# =============================================================================
def critical_radius_noCoulomb(Delta_f_bulk, sigma):
    """Closed-form critical radius  R_c = -2 sigma / Delta_f  [fm].

    NaN where Delta_f >= 0 (no barrier maximum: droplet always shrinks).
    """
    Delta_f_bulk = np.asarray(Delta_f_bulk, dtype=float)
    return np.where(Delta_f_bulk < 0, -2.0 * sigma / Delta_f_bulk, np.nan)


def critical_work_noCoulomb(Delta_f_bulk, sigma):
    """Closed-form barrier height  W_c = 16 pi sigma^3 / (3 Delta_f^2)  [MeV].

    NaN where Delta_f >= 0.
    """
    Delta_f_bulk = np.asarray(Delta_f_bulk, dtype=float)
    return np.where(
        Delta_f_bulk < 0,
        16.0 * np.pi * sigma**3 / (3.0 * Delta_f_bulk**2),
        np.nan,
    )


# =============================================================================
# Numeric critical radius with (unscreened) Coulomb  (public)
# =============================================================================
def critical_radius_coulomb(Delta_f_bulk, sigma, delta_n_C, initial_guess=None):
    """Critical radius from dW/dR = 0 with the unscreened a-posteriori Coulomb.

    Solves  Delta_f + 2 sigma/R - coulomb_P(R) = 0  (i.e. the maximum of
    bulk + surface + coulomb_W). Scalar inputs; NaN if Delta_f >= 0 or no root.
    """
    if Delta_f_bulk >= 0 or np.isnan(Delta_f_bulk):
        return np.nan

    def dW_dR(R):
        return Delta_f_bulk + 2.0 * sigma / R - coulomb_P(R, delta_n_C)

    if initial_guess is None:
        initial_guess = -2.0 * sigma / Delta_f_bulk
    sol = root(dW_dR, initial_guess)
    return float(sol.x[0]) if sol.success else np.nan


# =============================================================================
# Debye screening form factor  (internal building block)
# =============================================================================
# f(x) = (5 / 2 x^5) [ x^3 - 3 (x+1)(x cosh x - sinh x) e^{-x} ], x = R / lambda_D.
#
# Numerically this is a 3-fold catastrophic cancellation for small x (leading
# O(1), O(x^2), O(x^3) terms cancel to leave O(x^5)); the physically relevant
# regime is exactly R << lambda_D. So we use an exact Maclaurin series for
# x < _X_SERIES and the (stable-there) closed form above it. The series is
# f(x) = sum_k (5/2) g_{k+5} x^k, with g the Maclaurin coefficients of
#     g(x) = x^3 - (3/2)(x^2 - 1) - (3/2)(x+1)^2 e^{-2x}
# (an algebraically simplified, still-cancelling closed form). We generate the
# rational coefficients once with `fractions` so there is nothing to mistype.
_X_SERIES = 0.5          # crossover; validated series<->closed agree to ~1e-13 here
_N_SERIES = 20           # terms (machine-accurate for x < ~1)


def _screening_series_coeffs(n_terms):
    """Exact Maclaurin coefficients f_k (k=0..n_terms-1) of f(x); f_0 = 1."""
    # e_k = (-2)^k / k!  ;  c_n = e_n + 2 e_{n-1} + e_{n-2}  (coeffs of (x+1)^2 e^{-2x})
    # g_n = -(3/2) c_n for n >= 5 (g_0..g_4 vanish identically) ; f_k = (5/2) g_{k+5}.
    e = [Fraction((-2) ** k, factorial(k)) for k in range(n_terms + 5 + 1)]

    def c(n):
        r = e[n]
        if n - 1 >= 0:
            r += 2 * e[n - 1]
        if n - 2 >= 0:
            r += e[n - 2]
        return r

    f = [Fraction(5, 2) * (-Fraction(3, 2) * c(k + 5)) for k in range(n_terms)]
    df = [f[k] * k for k in range(n_terms)]   # term-by-term derivative coeffs
    return [float(v) for v in f], [float(v) for v in df]


_F_COEFFS, _DF_COEFFS = _screening_series_coeffs(_N_SERIES)


def _f_series(x):
    """Series value of f(x) (Horner over the exact Maclaurin coefficients)."""
    acc = 0.0
    for coef in reversed(_F_COEFFS):
        acc = acc * x + coef
    return acc


def _df_series(x):
    """Series value of f'(x): sum_k k f_k x^{k-1}."""
    # _DF_COEFFS[k] = k f_k is the coeff of x^k in x f'(x); divide by x once.
    acc = 0.0
    for coef in reversed(_DF_COEFFS):
        acc = acc * x + coef
    return acc / x if x != 0 else 0.0   # df(0) = f_1 = -5/6, handled by caller


def f_screening(x):
    """Debye screening form factor f(x), x = R/lambda_D.

    f(0) = 1 (unscreened limit), monotonically decreasing; f -> (5/2)(1/x)^2 as
    x -> inf (strong screening). Uses an exact series for x < 0.5 to avoid the
    small-x cancellation of the closed form.
    """
    if x < _X_SERIES:
        return _f_series(x)
    return 5.0 / (2.0 * x**5) * (x**3 - 3.0 * (x + 1.0)
                                 * (x * cosh(x) - sinh(x)) * exp(-x))


def df_screening(x):
    """Derivative f'(x) of the screening form factor.

    Closed form  f'(x) = 15[(x-1) + (x+1)e^{-2x}] / (2 x^4) - 5 f(x)/x  suffers a
    5/x - 5/x cancellation at small x, so we use the exact series for x < 0.5.
    """
    if x < _X_SERIES:
        if x == 0.0:
            return _DF_COEFFS[1]   # f'(0) = f_1 = -5/6
        return _df_series(x)
    return (15.0 * ((x - 1.0) + (x + 1.0) * exp(-2.0 * x)) / (2.0 * x**4)
            - 5.0 * f_screening(x) / x)


# =============================================================================
# Screened Coulomb energy / pressure  (public, general)
# =============================================================================
def coulomb_screened_W(R, delta_n_C, lambda_D):
    """Debye-screened Coulomb energy  W = coulomb_W(R) * f(R/lambda_D)  [MeV]."""
    x = R / lambda_D
    fx = f_screening(x) if np.isscalar(x) else np.vectorize(f_screening)(x)
    return coulomb_W(R, delta_n_C) * fx


def coulomb_screened_P(R, delta_n_C, lambda_D):
    """Debye-screened a-posteriori Coulomb pressure  [MeV/fm^3].

    P_Coul = -(4/3) pi hc alpha delta_n_C^2 R^2 [ f(x) + (1/5) x f'(x) ],
    x = R/lambda_D. Reduces to coulomb_P as x -> 0 (f->1, f'->-5/6).
    """
    x = R / lambda_D
    if np.isscalar(x):
        bracket = f_screening(x) + 0.2 * x * df_screening(x)
    else:
        bracket = (np.vectorize(f_screening)(x)
                   + 0.2 * x * np.vectorize(df_screening)(x))
    return -(4.0 / 3.0) * np.pi * hc * alpha_EM * delta_n_C**2 * R**2 * bracket


def critical_radius_coulomb_screened(Delta_f_bulk, sigma, delta_n_C, lambda_D,
                                     initial_guess=None):
    """Critical radius from dW/dR = 0 with the Debye-screened Coulomb.

    Solves  Delta_f + 2 sigma/R - coulomb_screened_P(R) = 0. Scalar inputs; NaN
    if Delta_f >= 0. Falls back to a bracketed brentq around the no-Coulomb R_c
    if the local Newton solve fails near the high-sigma saddle-node.
    """
    if Delta_f_bulk >= 0 or np.isnan(Delta_f_bulk):
        return np.nan
    R0 = -2.0 * sigma / Delta_f_bulk
    if initial_guess is None:
        initial_guess = R0

    def dW_dR(R):
        return Delta_f_bulk + 2.0 * sigma / R - coulomb_screened_P(
            R, delta_n_C, lambda_D)

    sol = root(dW_dR, initial_guess)
    if sol.success and sol.x[0] > 0:
        return float(sol.x[0])
    # Fallback: bracket around the no-Coulomb R_c (screening only weakens Coulomb,
    # so the true R_c stays close to R0).
    try:
        return float(brentq(dW_dR, 0.5 * R0, 5.0 * R0))
    except (ValueError, RuntimeError):
        return np.nan


# =============================================================================
# Debye length from the electron susceptibility  (public)
# =============================================================================
def electron_susceptibility(mu_e, T, rel_step=1e-4):
    """chi_e = dn_e/dmu_e|_T  [fm^-3 MeV^-1] by central difference.

    The electron EOS (eos.general.thermodynamics_leptons) has no analytic
    susceptibility, so we differentiate electron_thermo numerically with a step
    h = rel_step * max(|mu_e|, T, 1 MeV).
    """
    h = rel_step * max(abs(mu_e), T, 1.0)
    n_plus = electron_thermo(mu_e + h, T).n
    n_minus = electron_thermo(mu_e - h, T).n
    return (n_plus - n_minus) / (2.0 * h)


def debye_length(chi):
    """Debye screening length  lambda_D = 1 / sqrt(4 pi alpha hc chi)  [fm]."""
    return 1.0 / np.sqrt(4.0 * np.pi * alpha_EM * hc * chi)


# =============================================================================
# Switching functions for unpCFL (radius-dependent phase blend)  (public)
# =============================================================================
def switching_step(R, Rx):
    """Step blend: 0 (unpaired) for R <= Rx, 1 (CFL) for R > Rx."""
    return np.where(R > Rx, 1.0, 0.0)


def switching_tanh(R, Rx, delta):
    """Smooth blend S(R) = 0.5 (1 + tanh((R - Rx)/delta)): ~0 unpaired, ~1 CFL."""
    return 0.5 * (1.0 + np.tanh((R - Rx) / delta))


def get_switching_function(mode, Rx, delta=None):
    """Return a callable S(R) for the given switching mode ('step' or 'tanh')."""
    if mode == 'step':
        return lambda R: switching_step(R, Rx)
    if mode == 'tanh':
        if delta is None:
            raise ValueError("delta required for tanh switching")
        return lambda R: switching_tanh(R, Rx, delta)
    raise ValueError(f"Unknown switching mode: '{mode}'")


r"""
Note -- two Coulomb "pressures" (why coulomb_delta_P != coulomb_P)
=================================================================
Two charge prescriptions both add Coulomb energy but differ in how the droplet
composition responds, and hence in the R-balance that fixes R_c:

* gcn_coulomb / screening (A-POSTERIORI): the composition is the plain GCN one
  (Coulomb ignored in the minimization). Then W = -4/3 pi R^3 DP + surface +
  coulomb_W, and dW/dR = 0 gives the *thermodynamic* Coulomb pressure
  P_Coul = -dE_Coul/dV = coulomb_P (<0).

* coulomb_minimize (SELF-CONSISTENT): E_Coul is inside the minimization, so the
  composition itself shifts (mu_e^Qs = mu_e^H + coulomb_delta_mu_e). That extra
  electron chem-sum term does NOT vanish, and the R-balance becomes
      P_Qs + coulomb_delta_P - 2 sigma/R = P_H,   coulomb_delta_P = +(4/15)...,
  related to the thermodynamic one by
      coulomb_delta_P = coulomb_P + (8/5) alpha pi hc R^2 delta_n_C^2.
  The +(8/5)... piece is the electron contribution (coulomb_delta_mu_e * Y_e).

So `screening` is architecturally an a-posteriori (coulomb_P-family) mode and
must never be combined with the coulomb_delta_P machinery.
"""
