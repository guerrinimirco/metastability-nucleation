"""
Classical Nucleation Theory (CNT) for the hadron-to-quark phase transition.

Composable functions for computing nucleation observables:

    driving_force         -> Delta_F_bulk
    critical_radius       -> R_c = 2 sigma / Delta_F
    critical_work         -> W_c = 16 pi sigma^3 / (3 Delta_F^2)
    critical_radius_coulomb -> R_c with Coulomb (numerical root-finding)
    work_of_formation     -> W(R) at arbitrary radius
    nucleation_rate       -> Gamma = kappa/(2pi) Omega_0 exp(-W_c/T)
    nucleation_time       -> tau = 1/(V Gamma)

Coulomb correction functions (uniform-charge-sphere model):

    coulomb_delta_mu_e    -> Coulomb shift to electron chemical potential
    coulomb_delta_P       -> Coulomb correction to pressure balance
    coulomb_P             -> Coulomb contribution to pressure
    coulomb_W             -> Coulomb contribution to work of formation
"""

import numpy as np
from scipy.optimize import brentq
from eos.general.physics_constants import alpha_EM, hc


# =============================================================================
# Constants
# =============================================================================
N0 = 0.16    # nuclear saturation density (fm^-3)
XI_Q = 0.7   # quark correlation length (fm)


# =============================================================================
# Coulomb correction terms (uniform-charge-sphere model)
# =============================================================================
def coulomb_delta_mu_e(R, delta_n_C):
    """
    Coulomb correction to electron chemical potential.

    delta_mu_e = (8/5) alpha_em pi hbar*c R^2 delta_n_C

    Parameters
    ----------
    R : float or array
        Droplet radius (fm).
    delta_n_C : float or array
        Net charge density n_C_Q - n_e_Q (fm^-3).

    Returns
    -------
    float or array
        Correction to mu_e (MeV).
    """
    return (8.0 / 5.0) * alpha_EM * np.pi * hc * R**2 * delta_n_C


def coulomb_delta_P(R, delta_n_C):
    """
    Coulomb correction to pressure balance (appears in dW/dR = 0).

    delta_P = (4/15) alpha_em pi hbar*c R^2 delta_n_C^2

    Parameters
    ----------
    R : float or array
        Droplet radius (fm).
    delta_n_C : float or array
        Net charge density (fm^-3).

    Returns
    -------
    float or array
        Pressure correction (MeV/fm^3).
    """
    return (4.0 / 15.0) * alpha_EM * np.pi * hc * R**2 * delta_n_C**2


def coulomb_P(R, delta_n_C):
    """
    Coulomb contribution to pressure: -dE_Coul/dV at fixed n_i.

    P_Coul = -(4/3) alpha_em pi hbar*c R^2 delta_n_C^2

    Not used in the standard nucleation pipeline. Provided for
    thermodynamic consistency checks.

    Parameters
    ----------
    R : float or array
        Droplet radius (fm).
    delta_n_C : float or array
        Net charge density (fm^-3).

    Returns
    -------
    float or array
        Coulomb pressure contribution (MeV/fm^3).
    """
    return -(4.0 / 3.0) * alpha_EM * np.pi * hc * R**2 * delta_n_C**2


def coulomb_W(R, delta_n_C):
    """
    Coulomb contribution to the work of formation.

    W_Coulomb = -(16/15) pi^2 hbar*c alpha_em delta_n_C^2 R^5

    Parameters
    ----------
    R : float or array
        Droplet radius (fm).
    delta_n_C : float or array
        Net charge density (fm^-3).

    Returns
    -------
    float or array
        Coulomb contribution to W (MeV).
    """
    return -(16.0 / 15.0) * np.pi**2 * hc * alpha_EM * delta_n_C**2 * R**5


# =============================================================================
# Bulk driving force (single source of truth)
# =============================================================================
def driving_force(Qs, H):
    """
    Bulk driving force for nucleation.

    Delta_F_bulk = (P_Qs - P_H) - n_B_Qs * sum_i Y_i^Qs (mu_i^Qs - mu_i^H)

    where i = B, C, S, e, nu and Y_B = 1 by definition.

    For the saddlepoint solver with GCN (no Coulomb), the equilibrium
    conditions make the chemical-potential sum vanish, so
    Delta_F_bulk = P_Qs - P_H.

    Parameters
    ----------
    Qs : namespace
        Q* phase state. Required attributes:
        P_total, n_B, mu_B, mu_C, mu_S, mu_e, mu_nu, Y_C, Y_S, Y_e, Y_nu.
    H : namespace
        Hadronic phase state. Required attributes:
        P_total, mu_B, mu_C, mu_S, mu_e, mu_nu.

    Returns
    -------
    float or ndarray
        Bulk driving force (MeV/fm^3). Positive when quark phase is favored.
    """
    Delta_P = Qs.P_total - H.P_total
    chem_sum = (
        1.0 * (Qs.mu_B - H.mu_B)
        + Qs.Y_C * (Qs.mu_C - H.mu_C)
        + Qs.Y_S * (Qs.mu_S - H.mu_S)
        + Qs.Y_e * (Qs.mu_e - H.mu_e)
        + Qs.Y_nu * (Qs.mu_nu - H.mu_nu)
    )
    return Delta_P - Qs.n_B * chem_sum


# =============================================================================
# Critical radius (standard CNT, no Coulomb)
# =============================================================================
def critical_radius(Delta_F_bulk, sigma):
    """
    Critical droplet radius (standard CNT, no Coulomb).

    R_c = 2 sigma / Delta_F_bulk

    Parameters
    ----------
    Delta_F_bulk : float or ndarray
        Bulk driving force (MeV/fm^3).
    sigma : float
        Surface tension (MeV/fm^2).

    Returns
    -------
    float or ndarray
        Critical radius R_c (fm). NaN where Delta_F_bulk <= 0.
    """
    Delta_F_bulk = np.asarray(Delta_F_bulk, dtype=float)
    return np.where(Delta_F_bulk > 0, 2.0 * sigma / Delta_F_bulk, np.nan)


# =============================================================================
# Critical work (standard CNT, no Coulomb)
# =============================================================================
def critical_work(Delta_F_bulk, sigma):
    """
    Energy barrier at the critical radius (standard CNT, no Coulomb).

    W_c = 16 pi sigma^3 / (3 Delta_F_bulk^2)

    Parameters
    ----------
    Delta_F_bulk : float or ndarray
        Bulk driving force (MeV/fm^3).
    sigma : float
        Surface tension (MeV/fm^2).

    Returns
    -------
    float or ndarray
        Critical work W_c (MeV). NaN where Delta_F_bulk <= 0.
    """
    Delta_F_bulk = np.asarray(Delta_F_bulk, dtype=float)
    return np.where(
        Delta_F_bulk > 0,
        16.0 * np.pi * sigma**3 / (3.0 * Delta_F_bulk**2),
        np.nan,
    )


# =============================================================================
# Critical radius with Coulomb correction (post-hoc)
# =============================================================================
def critical_radius_coulomb(Delta_F_bulk, sigma, delta_n_C):
    """
    Critical radius including Coulomb correction.

    Solves dW/dR = 0 where W = bulk + surface + Coulomb:

        -4 pi R^2 Delta_F + 8 pi R sigma
        - (16/3) pi^2 hbar*c alpha_em delta_n_C^2 R^4 = 0

    Parameters
    ----------
    Delta_F_bulk : float
        Bulk driving force (MeV/fm^3). Scalar.
    sigma : float
        Surface tension (MeV/fm^2).
    delta_n_C : float
        Net charge density n_C_Q - n_e_Q (fm^-3). Scalar.

    Returns
    -------
    float
        Critical radius R_c (fm). NaN if no solution.
    """
    if Delta_F_bulk <= 0 or np.isnan(Delta_F_bulk):
        return np.nan

    def dW_dR(R):
        return (-4.0 * np.pi * R**2 * Delta_F_bulk
                + 8.0 * np.pi * R * sigma
                - (16.0 / 3.0) * np.pi**2 * hc * alpha_EM
                * delta_n_C**2 * R**4)

    R_standard = 2.0 * sigma / Delta_F_bulk
    try:
        R_c = brentq(dW_dR, 1e-6, 5.0 * R_standard)
    except ValueError:
        return np.nan
    return R_c


# =============================================================================
# Work of formation at arbitrary radius
# =============================================================================
def work_of_formation(R, Delta_F_bulk, sigma, delta_n_C=0.0):
    """
    Work of formation of a quark droplet at radius R.

    W(R) = -4/3 pi R^3 Delta_F_bulk + 4 pi R^2 sigma + W_Coulomb(R)

    When delta_n_C = 0, the Coulomb term vanishes (standard CNT).

    Parameters
    ----------
    R : float or ndarray
        Droplet radius (fm).
    Delta_F_bulk : float or ndarray
        Bulk driving force (MeV/fm^3).
    sigma : float
        Surface tension (MeV/fm^2).
    delta_n_C : float or ndarray, optional
        Net charge density (fm^-3). Default 0 (no Coulomb).

    Returns
    -------
    float or ndarray
        Work of formation W(R) (MeV).
    """
    W_bulk = -4.0 / 3.0 * np.pi * R**3 * Delta_F_bulk
    W_surface = 4.0 * np.pi * R**2 * sigma
    W_coul = coulomb_W(R, delta_n_C)
    return W_bulk + W_surface + W_coul


# =============================================================================
# Transport: shear viscosity
# =============================================================================
def shear_viscosity(n_B_H, T):
    """
    Shear viscosity of hadronic matter.

    eta = 7.6e26 (n_B / n0)^2 T^{-2}   [MeV / (fm s)]

    Eq. (4.139).

    Parameters
    ----------
    n_B_H : float or array
        Baryon number density (fm^-3).
    T : float or array
        Temperature (MeV).

    Returns
    -------
    float or array
        Shear viscosity eta (MeV / (fm s)).
    """
    return 7.6e26 * (n_B_H / N0)**2 / T**2


# =============================================================================
# Statistical prefactor
# =============================================================================
def statistical_prefactor(sigma, T, R_c, xi_q=XI_Q):
    """
    Statistical prefactor Omega_0.

    Omega_0 = 2 / (3 sqrt(3)) (sigma / T)^{3/2} (R_c / xi_q)^4

    Eq. (4.137).

    Parameters
    ----------
    sigma : float
        Surface tension (MeV/fm^2).
    T : float or array
        Temperature (MeV).
    R_c : float or array
        Critical radius (fm).
    xi_q : float, optional
        Quark correlation length (fm). Default 0.7.

    Returns
    -------
    float or array
        Omega_0 (fm^{-3}).
    """
    return 2.0 / (3.0 * np.sqrt(3.0)) * (sigma / T)**1.5 * (R_c / xi_q)**4


# =============================================================================
# Dynamical prefactor
# =============================================================================
def dynamical_prefactor(sigma, R_c, H, Qs, T,
                        lambda_th=0.0, zeta_th=0.0):
    """
    Dynamical prefactor kappa.

    kappa = 2 sigma / (R_c^3 (Delta omega)^2)
            * [lambda T + 2 (4/3 eta + zeta)]

    Eq. (4.138).  Default: lambda = zeta = 0 (only shear viscosity).

    Parameters
    ----------
    sigma : float
        Surface tension (MeV/fm^2).
    R_c : float or array
        Critical radius (fm).
    H : namespace
        Hadronic phase state. Required: n_B, P_total, e_total.
    Qs : namespace
        Q* phase state. Required: P_total, e_total.
    T : float or array
        Temperature (MeV).
    lambda_th : float, optional
        Thermal conductivity. Default 0.
    zeta_th : float, optional
        Bulk viscosity. Default 0.

    Returns
    -------
    float or array
        kappa (s^{-1}).
    """
    eta = shear_viscosity(H.n_B, T)
    Delta_omega = (Qs.P_total + Qs.e_total) - (H.P_total + H.e_total)
    return (2.0 * sigma / (R_c**3 * Delta_omega**2)
            * (lambda_th * T + 2.0 * (4.0 / 3.0 * eta + zeta_th)))


# =============================================================================
# Nucleation rate
# =============================================================================
def nucleation_rate(W_c, R_c, sigma, T, H, Qs,
                    xi_q=XI_Q, lambda_th=0.0, zeta_th=0.0):
    """
    Thermal nucleation rate per unit volume.

    Gamma = kappa / (2 pi) * Omega_0 * exp(-W_c / T)

    Parameters
    ----------
    W_c : float or ndarray
        Critical work / energy barrier (MeV). Pre-computed.
    R_c : float or ndarray
        Critical radius (fm). Pre-computed.
    sigma : float
        Surface tension (MeV/fm^2).
    T : float or ndarray
        Temperature (MeV).
    H : namespace
        Hadronic phase state. Required: n_B, P_total, e_total.
    Qs : namespace
        Q* phase state. Required: P_total, e_total.
    xi_q : float, optional
        Quark correlation length (fm). Default 0.7.
    lambda_th : float, optional
        Thermal conductivity. Default 0.
    zeta_th : float, optional
        Bulk viscosity. Default 0.

    Returns
    -------
    float or ndarray
        Nucleation rate Gamma (fm^{-3} s^{-1}).
    """
    Omega_0 = statistical_prefactor(sigma, T, R_c, xi_q)
    kappa = dynamical_prefactor(sigma, R_c, H, Qs, T, lambda_th, zeta_th)
    return kappa / (2.0 * np.pi) * Omega_0 * np.exp(-W_c / T)


# =============================================================================
# Nucleation time
# =============================================================================
def nucleation_time(Gamma, V):
    """
    Nucleation waiting time.

    tau = 1 / (V * Gamma)

    Parameters
    ----------
    Gamma : float or ndarray
        Nucleation rate (fm^{-3} s^{-1}).
    V : float
        System volume (fm^3).

    Returns
    -------
    float or ndarray
        Nucleation time tau (s).
    """
    return 1.0 / (V * Gamma)
