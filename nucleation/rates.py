"""
Nucleation rates
================

Turns a critical droplet (R_c, W_c, and the surrounding thermodynamic state)
into a nucleation *rate* and *waiting time*, by two independent mechanisms:

* **Thermal** activation over the barrier (Langer's theory) -- dominant at
  high temperature.
* **Quantum** tunnelling through the barrier (relativistic WKB) -- dominant at
  low temperature.

This module is pure "rate physics": it takes W(R) / R_c as given (built by
``nucleation.critical``) and never solves for the droplet composition itself.

Public API (used by tables / sigma_crit / the notebook)
-------------------------------------------------------
    nucleation_rate(W_c, R_c, sigma, T, H, Qs, ...)   -> Gamma  [fm^-3 s^-1]
    nucleation_time(Gamma, V)                          -> tau    [s]
    effective_inertia(R, rho_H, n_B_ratio)             -> M(R)   [MeV]
    quantum_nucleation_time(W_func, M_func, R_c, N_c)  -> QuantumNucleationResult

Internal helpers (building blocks; exposed for testing / diagnostics)
---------------------------------------------------------------------
    shear_viscosity, statistical_prefactor, dynamical_prefactor
    find_E_min, find_turning_points, oscillation_action, tunneling_action,
    oscillation_frequency, ground_state_energy

References
----------
J. S. Langer, Annals Phys. 54 (1969); Physica 73.1 (1974).
L. P. Csernai & J. I. Kapusta, Phys. Rev. D 46 (1992).
P. Danielewicz, Phys. Lett. B 146 (1984)  [transport coefficients].
I. M. Lifshitz & Yu. Kagan (1972); Iida & Sato (1997, 1998)  [relativistic WKB].
Thesis chapter 4 (Sec. thermal / quantum nucleation rate).
"""

import numpy as np
from dataclasses import dataclass
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, brentq

from eos.general.physics_constants import hc

# =============================================================================
# Constants
# =============================================================================
N0 = 0.16    # nuclear saturation density (fm^-3) for the shear-viscosity fit
XI_Q = 0.7   # quark correlation length (fm) in the statistical prefactor
HBAR_MEV_S = 6.582119569e-22   # hbar in MeV*s (frequency MeV -> s^-1)


###############################################################################
#                                                                             #
#  THERMAL (Langer)                                                           #
#                                                                             #
###############################################################################

# =============================================================================
# Transport: shear viscosity  (internal)
# =============================================================================
def shear_viscosity(n_B_H, T, n0=N0):
    """Shear viscosity of hadronic matter (Danielewicz 1984 fit).

    eta = 7.6e26 (n_B / n0)^2 T^{-2}   [MeV / (fm s)]

    Physics: the dynamical prefactor is dominated by shear viscosity (thermal
    conductivity and bulk viscosity are set to zero by default), so this fit
    controls how fast the droplet grows at the saddle.
    """
    return 7.6e26 * (n_B_H / n0)**2 / T**2


# =============================================================================
# Statistical prefactor  (internal)
# =============================================================================
def statistical_prefactor(sigma, T, R_c, xi_q=XI_Q):
    """Statistical (curvature-of-saddle) prefactor Omega_0.

    Omega_0 = 2 / (3 sqrt(3)) (sigma / T)^{3/2} (R_c / xi_q)^4    [fm^-3]

    Measures the phase-space volume of the saddle-point region.
    """
    return 2.0 / (3.0 * np.sqrt(3.0)) * (sigma / T)**1.5 * (R_c / xi_q)**4


# =============================================================================
# Dynamical prefactor  (internal)
# =============================================================================
def dynamical_prefactor(sigma, R_c, H, Qs, T, lambda_th=0.0, zeta_th=0.0):
    """Dynamical (droplet growth-rate) prefactor kappa.

    kappa = 2 sigma / (R_c^3 (Delta omega)^2) * [lambda T + 2 (4/3 eta + zeta)]

    where Delta omega = w_Qs - w_H is the enthalpy-density jump at the saddle,
    eta the shear viscosity, and lambda / zeta the (default-zero) thermal
    conductivity / bulk viscosity.
    """
    eta = shear_viscosity(H.n_B, T)
    Delta_omega = (Qs.P_total + Qs.e_total) - (H.P_total + H.e_total)
    return (2.0 * sigma / (R_c**3 * Delta_omega**2)
            * (lambda_th * T + 2.0 * (4.0 / 3.0 * eta + zeta_th)))


# =============================================================================
# Thermal nucleation rate / time  (public)
# =============================================================================
def nucleation_rate(W_c, R_c, sigma, T, H, Qs,
                    xi_q=XI_Q, lambda_th=0.0, zeta_th=0.0):
    """Thermal nucleation rate per unit volume.

    Gamma = kappa / (2 pi) * Omega_0 * exp(-W_c / T)   [fm^-3 s^-1]

    The exponential exp(-W_c/T) dominates by many orders of magnitude; the
    prefactor (kappa*Omega_0) varies mildly. W_c and R_c are supplied by the
    critical-droplet engine; here we only assemble the Langer rate.
    """
    Omega_0 = statistical_prefactor(sigma, T, R_c, xi_q)
    kappa = dynamical_prefactor(sigma, R_c, H, Qs, T, lambda_th, zeta_th)
    return kappa / (2.0 * np.pi) * Omega_0 * np.exp(-W_c / T)


def nucleation_time(Gamma, V):
    """Nucleation waiting time tau = 1 / (V * Gamma)  [s].

    Expected time for the FIRST critical droplet to appear in a system of
    volume V (not a droplet lifetime nor a growth time). Gamma = 0 (barrier so
    high the rate underflows) gives tau = +inf, as intended.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        return 1.0 / (V * Gamma)


###############################################################################
#                                                                             #
#  QUANTUM (relativistic WKB)                                                 #
#                                                                             #
###############################################################################

# =============================================================================
# Output dataclass
# =============================================================================
@dataclass
class QuantumNucleationResult:
    """Result of a WKB quantum-nucleation calculation at a single point.

    Attributes: E_0 (ground-state energy, MeV), m_0 (number of positive-energy
    bound levels), E_min (lower edge of the positive-energy continuum, MeV),
    R_inner / R_outer (turning points, fm), A (tunnelling action, dimensionless),
    nu_0 (small-oscillation frequency, s^-1), tau_qt (nucleation time, s),
    N_c (nucleation centres used).
    """
    E_0: float
    m_0: int
    E_min: float
    R_inner: float
    R_outer: float
    A: float
    nu_0: float
    tau_qt: float
    N_c: float


# =============================================================================
# Effective inertia  (public)
# =============================================================================
def effective_inertia(R, rho_H, n_B_ratio):
    """Effective inertia of the growing droplet (irrotational-flow model).

    M(R) = 4 pi rho_H (1 - n_B^Qs / n_B^H)^2 R^3   [MeV]

    Comes from the hydrodynamic flow the surrounding hadronic matter must
    perform as R changes at (quasi-)mechanical equilibrium.
    """
    return 4.0 * np.pi * rho_H * (1.0 - n_B_ratio)**2 * R**3


# =============================================================================
# Continuum edge, turning points, actions  (internal)
# =============================================================================
def find_E_min(W_func, M_func, R_max=100.0):
    """E_min = max_R [W(R) - 2 M(R)]: lower edge of the positive-energy band.

    ``R_max`` MUST be the barrier peak R_c, not an arbitrary large radius. The
    band edge belongs to the droplet's BOUND sub-barrier oscillation, which
    lives entirely at R < R_inner < R_c (``oscillation_action`` integrates
    0 -> R_inner), so a maximum located beyond the peak is not a band edge at
    all.

    This matters as soon as the Coulomb term is on. Without it W(R) falls
    monotonically past R_c and the maximum is interior either way. With it the
    electrostatic energy grows as +R^5 and eventually beats the -R^3 bulk gain,
    so W(R) turns back UP at large R: an unbounded search then locks onto that
    outer Coulomb wall and returns a meaningless E_min (~1e9 MeV instead of
    ~10 MeV), after which no turning point exists below R_c and the whole WKB
    pipeline raises. That is why every Coulomb-mode quantum point used to fail.
    """
    def neg_diff(R):
        return -(W_func(R) - 2.0 * M_func(R))
    result = minimize_scalar(neg_diff, bounds=(1e-6, R_max), method='bounded')
    return -result.fun


def find_turning_points(E, W_func, R_c, R_max=1000.0):
    """Inner (R < R_c) and outer (R > R_c) roots of W(R) = E.

    The outer root is the FIRST crossing beyond the peak, located by walking
    outward until W drops below E and only then bracketing. Bracketing
    [R_c, R_max] directly is wrong whenever the Coulomb term is on: W(R) falls
    through E, bottoms out, then climbs back above E as +R^5, so W(R_max) and
    W(R_c) are both above E and brentq rejects the bracket ("f(a) and f(b) must
    have different signs") even though a turning point plainly exists.
    """
    def f(R):
        return W_func(R) - E
    R_inner = brentq(f, 1e-6, R_c)

    # Walk outward geometrically for the first R with W(R) < E.
    R_lo, R_hi = R_c, R_c * 1.5
    while R_hi < R_max:
        if f(R_hi) < 0:
            return R_inner, brentq(f, R_lo, R_hi)
        R_lo, R_hi = R_hi, R_hi * 1.5
    raise ValueError(
        f"no outer turning point below R_max={R_max} for E={E:.6g}")


def oscillation_action(E, W_func, M_func, R_inner):
    """Bohr-Sommerfeld action I(E) of the sub-barrier bound motion.

    I(E) = (2/hc) integral_0^{R_inner} sqrt([2M + E - W][E - W]) dR
    """
    def integrand(R):
        W = W_func(R)
        M = M_func(R)
        product = (2.0 * M + E - W) * (E - W)
        return np.sqrt(product) if product > 0 else 0.0
    result, _ = quad(integrand, 0, R_inner, limit=100)
    return 2.0 / hc * result


def tunneling_action(E, W_func, M_func, R_inner, R_outer):
    """WKB tunnelling exponent A(E) across the classically forbidden region.

    A(E) = (2/hc) integral_{R_inner}^{R_outer} sqrt([2M + E - W][W - E]) dR
    """
    def integrand(R):
        W = W_func(R)
        M = M_func(R)
        product = (2.0 * M + E - W) * (W - E)
        return np.sqrt(product) if product > 0 else 0.0
    result, _ = quad(integrand, R_inner, R_outer, limit=100)
    return 2.0 / hc * result


def oscillation_frequency(E, W_func, M_func, R_inner):
    """Small-oscillation frequency nu_0 = (dI/dE)^{-1}  [s^-1]."""
    def integrand(R):
        W = W_func(R)
        M = M_func(R)
        product = (2.0 * M + E - W) * (E - W)
        if product <= 0:
            return 0.0
        return (M + E - W) / np.sqrt(product)
    dIdE, _ = quad(integrand, 0, R_inner, limit=100)
    dIdE *= 2.0 / hc
    if dIdE <= 0:
        return 0.0
    nu_0_MeV = 1.0 / dIdE
    return nu_0_MeV / HBAR_MEV_S


def ground_state_energy(W_func, M_func, R_c):
    """Ground-state energy E_0 via Bohr-Sommerfeld quantization.

    1. E_min = max(W - 2M); 2. R_inner at E_min; 3. m_0 = floor(I(E_min)/2pi + 1/4);
    4. solve I(E_0) = 2pi(m_0 + 3/4). Returns (E_0, R_inner, R_outer, m_0, E_min).
    """
    # Search the band edge only BELOW the barrier peak -- see find_E_min. With a
    # Coulomb term W(R) rises again as +R^5 at large R, so searching past R_c
    # finds that outer wall instead of the band edge.
    E_min_val = find_E_min(W_func, M_func, R_max=R_c)
    R_inner_Emin, _ = find_turning_points(
        E_min_val, W_func, R_c, R_max=max(10.0 * R_c, 1000.0))
    I_Emin = oscillation_action(E_min_val, W_func, M_func, R_inner_Emin)
    m_0 = int(np.floor(I_Emin / (2.0 * np.pi) + 0.25))
    target = 2.0 * np.pi * (m_0 + 0.75)

    def bs_residual(E):
        try:
            R_in, _ = find_turning_points(
                E, W_func, R_c, R_max=max(10.0 * R_c, 1000.0))
            return oscillation_action(E, W_func, M_func, R_in) - target
        except (ValueError, RuntimeError):
            return -target
    E_lo = max(W_func(0.01), 1e-3)
    E_hi = E_min_val
    try:
        E_0 = brentq(bs_residual, E_lo, E_hi, xtol=1e-6, rtol=1e-8)
    except ValueError:
        E_0 = E_min_val * 0.99
    R_inner, R_outer = find_turning_points(
        E_0, W_func, R_c, R_max=max(10.0 * R_c, 1000.0))
    return E_0, R_inner, R_outer, m_0, E_min_val


# =============================================================================
# Full WKB pipeline  (public)
# =============================================================================
def quantum_nucleation_time(W_func, M_func, R_c, N_c=1e48):
    """Quantum tunnelling nucleation time tau_qt via the full WKB pipeline.

    tau_qt = 1 / (N_c nu_0 exp(-A(E_0))). Takes W(R) and M(R) as callables
    (built by the critical-droplet engine) and R_c; returns a
    ``QuantumNucleationResult``.
    """
    E_0, R_inner, R_outer, m_0, E_min_val = ground_state_energy(
        W_func, M_func, R_c)
    A = tunneling_action(E_0, W_func, M_func, R_inner, R_outer)
    nu_0 = oscillation_frequency(E_0, W_func, M_func, R_inner)
    if nu_0 > 0 and np.isfinite(A):
        tau_qt = 1.0 / (N_c * nu_0 * np.exp(-A))
    else:
        tau_qt = np.inf
    return QuantumNucleationResult(
        E_0=E_0, m_0=m_0, E_min=E_min_val, R_inner=R_inner, R_outer=R_outer,
        A=A, nu_0=nu_0, tau_qt=tau_qt, N_c=N_c)
