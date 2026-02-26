"""
Quantum Nucleation (WKB Tunneling)
===================================

Semiclassical WKB calculation of the quantum tunneling nucleation rate
for the hadron-to-quark phase transition at low temperatures.

The quark droplet coordinate R_Q is modeled as a relativistic particle
moving in the effective potential W(R) with radius-dependent inertia M(R).
The tunneling rate is computed via the WKB approximation.

References:
    Eqs. (4.141)-(4.149) of the thesis.

Functions
---------
    effective_inertia       : M(R) from hadronic mass density
    find_E_min              : Lower edge of positive-energy continuum
    find_turning_points     : Classical turning points W(R) = E
    oscillation_action      : Bohr-Sommerfeld integral I(E)
    tunneling_action        : WKB tunneling exponent A(E)
    oscillation_frequency   : Small-oscillation frequency nu_0
    ground_state_energy     : Bohr-Sommerfeld quantization for E_0
    quantum_nucleation_time : Full WKB pipeline -> tau_qt
"""

import numpy as np
from dataclasses import dataclass
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, brentq

from eos.general.physics_constants import hc

# hbar in MeV*s  (for converting frequency to s^-1)
HBAR_MEV_S = 6.582119569e-22


# =============================================================================
# Output dataclass
# =============================================================================
@dataclass
class QuantumNucleationResult:
    """Result of WKB quantum nucleation calculation at a single point.

    Attributes
    ----------
    E_0 : float
        Ground-state energy (MeV).
    m_0 : int
        Number of positive-energy bound levels below the continuum.
    E_min : float
        Lower edge of the positive-energy continuum (MeV).
    R_inner : float
        Inner turning point (fm), where W(R_inner) = E_0, R_inner < R_c.
    R_outer : float
        Outer turning point (fm), where W(R_outer) = E_0, R_outer > R_c.
    A : float
        Tunneling action (dimensionless).
    nu_0 : float
        Small-oscillation frequency (s^-1).
    tau_qt : float
        Quantum nucleation time (s).
    N_c : float
        Number of nucleation centers used.
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
# Effective inertia  (Eq. 4.142)
# =============================================================================
def effective_inertia(R, rho_H, n_B_ratio):
    """Effective inertia of the quark droplet.

    M(R) = 4 pi rho_H (1 - n_B^Qs / n_B^H)^2 R^3

    Parameters
    ----------
    R : float or array
        Droplet radius (fm).
    rho_H : float
        Hadronic mass density (MeV/fm^3).
    n_B_ratio : float
        n_B^Qs / n_B^H (dimensionless).

    Returns
    -------
    float or array
        M(R) (MeV).
    """
    return 4.0 * np.pi * rho_H * (1.0 - n_B_ratio)**2 * R**3


# =============================================================================
# Lower edge of the continuum  (related to Eq. 4.147)
# =============================================================================
def find_E_min(W_func, M_func, R_max=100.0):
    """Find E_min = max_R [W(R) - 2 M(R)].

    This is the maximum of the function W(R) - 2*M(R), which sets the
    lower edge of the positive-energy continuum band.

    Parameters
    ----------
    W_func : callable
        W(R) -> float (MeV).
    M_func : callable
        M(R) -> float (MeV).
    R_max : float
        Upper bound for search (fm).

    Returns
    -------
    float
        E_min (MeV).
    """
    def neg_diff(R):
        return -(W_func(R) - 2.0 * M_func(R))

    result = minimize_scalar(neg_diff, bounds=(1e-6, R_max), method='bounded')
    return -result.fun


# =============================================================================
# Turning points
# =============================================================================
def find_turning_points(E, W_func, R_c, R_max=1000.0):
    """Find inner and outer turning points where W(R) = E.

    Parameters
    ----------
    E : float
        Energy level (MeV).
    W_func : callable
        W(R) -> float (MeV).
    R_c : float
        Critical radius (fm). Used to separate inner / outer roots.
    R_max : float
        Upper bound for outer root search (fm).

    Returns
    -------
    R_inner, R_outer : float
        Inner (R < R_c) and outer (R > R_c) turning points (fm).
    """
    def f(R):
        return W_func(R) - E

    # Inner turning point: between 0 and R_c
    R_inner = brentq(f, 1e-6, R_c)

    # Outer turning point: between R_c and R_max
    R_outer = brentq(f, R_c, R_max)

    return R_inner, R_outer


# =============================================================================
# Oscillation action I(E)  (Eq. 4.146)
# =============================================================================
def oscillation_action(E, W_func, M_func, R_inner):
    """Bohr-Sommerfeld oscillation action I(E).

    I(E) = (2/hc) * integral_0^{R_inner} sqrt([2M(R) + E - W(R)] [E - W(R)]) dR

    The integrand is real in the classically allowed region [0, R_inner]
    where E > W(R) (near R=0 where W -> 0).

    Parameters
    ----------
    E : float
        Energy (MeV).
    W_func, M_func : callable
        W(R) and M(R) functions.
    R_inner : float
        Inner turning point (fm).

    Returns
    -------
    float
        I(E) (dimensionless).
    """
    def integrand(R):
        W = W_func(R)
        M = M_func(R)
        factor1 = 2.0 * M + E - W
        factor2 = E - W
        product = factor1 * factor2
        if product <= 0:
            return 0.0
        return np.sqrt(product)

    result, _ = quad(integrand, 0, R_inner, limit=100)
    return 2.0 / hc * result


# =============================================================================
# Tunneling action A(E)  (Eq. 4.144)
# =============================================================================
def tunneling_action(E, W_func, M_func, R_inner, R_outer):
    """WKB tunneling action A(E).

    A(E) = (2/hc) * integral_{R_inner}^{R_outer}
             sqrt([2M(R) + E - W(R)] [W(R) - E]) dR

    The integrand is real in the classically forbidden region [R_inner, R_outer]
    where W(R) > E.

    Parameters
    ----------
    E : float
        Energy (MeV).
    W_func, M_func : callable
        W(R) and M(R) functions.
    R_inner, R_outer : float
        Turning points (fm).

    Returns
    -------
    float
        A(E) (dimensionless).
    """
    def integrand(R):
        W = W_func(R)
        M = M_func(R)
        factor1 = 2.0 * M + E - W
        factor2 = W - E
        product = factor1 * factor2
        if product <= 0:
            return 0.0
        return np.sqrt(product)

    result, _ = quad(integrand, R_inner, R_outer, limit=100)
    return 2.0 / hc * result


# =============================================================================
# Small-oscillation frequency  (Eq. 4.148)
# =============================================================================
def oscillation_frequency(E, W_func, M_func, R_inner):
    """Small-oscillation frequency nu_0 = (dI/dE)^{-1}.

    Computes dI/dE by integrating the derivative of the I(E) integrand
    with respect to E:

    dI/dE = (2/hc) * integral_0^{R_inner}
              [M(R) + E - W(R)] / sqrt([2M(R) + E - W(R)] [E - W(R)]) dR

    Parameters
    ----------
    E : float
        Energy (MeV).
    W_func, M_func : callable
        W(R) and M(R) functions.
    R_inner : float
        Inner turning point (fm).

    Returns
    -------
    float
        nu_0 (s^-1).
    """
    def integrand(R):
        W = W_func(R)
        M = M_func(R)
        factor1 = 2.0 * M + E - W
        factor2 = E - W
        product = factor1 * factor2
        if product <= 0:
            return 0.0
        numerator = M + E - W
        return numerator / np.sqrt(product)

    dIdE, _ = quad(integrand, 0, R_inner, limit=100)
    dIdE *= 2.0 / hc

    if dIdE <= 0:
        return 0.0

    # nu_0 has units of MeV (since I is dimensionless and E is in MeV)
    # Convert to s^-1: nu_0 [s^-1] = nu_0 [MeV] / hbar [MeV*s]
    nu_0_MeV = 1.0 / dIdE
    return nu_0_MeV / HBAR_MEV_S


# =============================================================================
# Ground-state energy via Bohr-Sommerfeld  (Eqs. 4.145-4.147)
# =============================================================================
def ground_state_energy(W_func, M_func, R_c):
    """Find ground-state energy E_0 via Bohr-Sommerfeld quantization.

    1. Compute E_min = max(W - 2M)
    2. Find R_inner at E = E_min
    3. Compute m_0 = floor(I(E_min)/(2pi) + 1/4)
    4. Solve I(E_0) = 2pi*(m_0 + 3/4) for E_0

    Parameters
    ----------
    W_func, M_func : callable
        W(R) and M(R) functions.
    R_c : float
        Critical radius (fm).

    Returns
    -------
    E_0 : float
        Ground-state energy (MeV).
    R_inner : float
        Inner turning point at E_0 (fm).
    R_outer : float
        Outer turning point at E_0 (fm).
    m_0 : int
        Number of bound levels.
    E_min_val : float
        Lower edge of continuum (MeV).
    """
    # Step 1: E_min
    E_min_val = find_E_min(W_func, M_func, R_max=max(10.0 * R_c, 100.0))

    # Step 2: Inner turning point at E_min
    W_c = W_func(R_c)
    R_inner_Emin, _ = find_turning_points(E_min_val, W_func, R_c,
                                           R_max=max(10.0 * R_c, 1000.0))

    # Step 3: m_0
    I_Emin = oscillation_action(E_min_val, W_func, M_func, R_inner_Emin)
    m_0 = int(np.floor(I_Emin / (2.0 * np.pi) + 0.25))

    # Step 4: Bohr-Sommerfeld condition I(E_0) = 2pi*(m_0 + 3/4)
    target = 2.0 * np.pi * (m_0 + 0.75)

    # E_0 lies between W(0) = 0 and E_min
    # Search for E where I(E) = target
    def bs_residual(E):
        try:
            R_in, _ = find_turning_points(E, W_func, R_c,
                                           R_max=max(10.0 * R_c, 1000.0))
            I_val = oscillation_action(E, W_func, M_func, R_in)
            return I_val - target
        except (ValueError, RuntimeError):
            return -target  # If turning point not found, return large negative

    # Bracket: at E small, I ~ 0 < target; at E_min, I(E_min) >= target
    # Use a small positive E as lower bound
    E_lo = max(W_func(0.01), 1e-3)
    E_hi = E_min_val

    try:
        E_0 = brentq(bs_residual, E_lo, E_hi, xtol=1e-6, rtol=1e-8)
    except ValueError:
        # If bracketing fails, try E_0 close to E_min
        E_0 = E_min_val * 0.99

    R_inner, R_outer = find_turning_points(E_0, W_func, R_c,
                                            R_max=max(10.0 * R_c, 1000.0))

    return E_0, R_inner, R_outer, m_0, E_min_val


# =============================================================================
# Full WKB pipeline  (Eq. 4.149)
# =============================================================================
def quantum_nucleation_time(W_func, M_func, R_c, N_c=1e48):
    """Compute quantum tunneling nucleation time tau_qt.

    Full WKB pipeline:
    1. Find ground-state energy E_0 via Bohr-Sommerfeld
    2. Compute tunneling action A(E_0)
    3. Compute oscillation frequency nu_0
    4. tau_qt = 1 / (N_c * nu_0 * exp(-A))

    Parameters
    ----------
    W_func : callable
        Work of formation W(R) -> float (MeV).
    M_func : callable
        Effective inertia M(R) -> float (MeV).
    R_c : float
        Critical radius (fm).
    N_c : float
        Number of independent nucleation centers.

    Returns
    -------
    QuantumNucleationResult
    """
    # Step 1: Ground-state energy
    E_0, R_inner, R_outer, m_0, E_min_val = ground_state_energy(
        W_func, M_func, R_c)

    # Step 2: Tunneling action
    A = tunneling_action(E_0, W_func, M_func, R_inner, R_outer)

    # Step 3: Oscillation frequency
    nu_0 = oscillation_frequency(E_0, W_func, M_func, R_inner)

    # Step 4: Nucleation time
    if nu_0 > 0 and np.isfinite(A):
        tau_qt = 1.0 / (N_c * nu_0 * np.exp(-A))
    else:
        tau_qt = np.inf

    return QuantumNucleationResult(
        E_0=E_0,
        m_0=m_0,
        E_min=E_min_val,
        R_inner=R_inner,
        R_outer=R_outer,
        A=A,
        nu_0=nu_0,
        tau_qt=tau_qt,
        N_c=N_c,
    )
