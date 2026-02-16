"""
General nucleation functions for the hadron-to-quark phase transition.

Computes critical droplet properties and thermal nucleation rates
using Classical Nucleation Theory (CNT).

All functions take two phase objects (Qs, H) that must provide the
following attributes:

    Required:  P_total, e_total, n_B, T,
               mu_B, mu_C, mu_S, mu_e
    Optional:  mu_nu  (default 0)
               Y_C, Y_S, Y_e, Y_nu

Functions
---------
work_of_formation     : W(R) at given radius (includes driving force, Eq. 4.35)
critical_radius       : R_c
critical_work         : W*(R_c) energy barrier
shear_viscosity       : eta(n_B, T)
statistical_prefactor : Omega_0
dynamical_prefactor   : kappa
nucleation_rate       : Gamma_th per unit volume
nucleation_time       : tau_th = 1/(V Gamma)
"""

import numpy as np


# =============================================================================
# Constants
# =============================================================================
N0 = 0.16    # nuclear saturation density (fm^-3)
XI_Q = 0.7   # quark correlation length (fm)


# =============================================================================
# Work of formation 
# =============================================================================
def work_of_formation(R, Qs, H, sigma, include_coulomb_energy=False):
    """
    Work of formation of a quark droplet of radius R.

    W(R) = -4/3 pi R^3 Delta_F_bulk + 4 pi R^2 sigma + W_Coulomb

    Assuming a small droplet with respect to the total accessible volume and
    assuming the quark free energy can be described as bulk + finite size effects:
    
    Delta_F_bulk = (P_Qs - P_H) - n_B_Qs * sum_i Y_i^Qs (mu_i^Qs - mu_i^H)

    with i = B, C, S, e, nu  and  Y_B = 1 by definition.

    For the saddlepoint and frozen flavor solvers the saddle-point condition on n_B
    makes the chemical-potential sum vanish, so Delta_F = P_Qs - P_H.

    ----------
    R : float or array
        Droplet radius (fm).
    Qs : object
        Q* phase state with attributes:
        n_B, T, mu_B, mu_C, mu_S, mu_e, mu_nu, Y_C, Y_S, Y_e, Y_nu, P_total, e_total, s_total
    H : object
        Hadronic phase state with attributes:
        n_B, T, mu_B, mu_C, mu_S, mu_e, mu_nu, Y_C, Y_S, Y_e, Y_nu, P_total, e_total, s_total
    sigma : float
        Surface tension (MeV/fm^2).

    Returns
    -------
    float or array
        Work of formation W (MeV).
    """
    Delta_P = Qs.P_total - H.P_total
    chem_sum = (
        1.0 * (Qs.mu_B - H.mu_B)
        + Qs.Y_C * (Qs.mu_C - H.mu_C)
        + Qs.Y_S * (Qs.mu_S - H.mu_S)
        + Qs.Y_e * (Qs.mu_e - H.mu_e)
        + Qs.Y_nu * (Qs.mu_nu - H.mu_nu)
    )
    
    W_bulk = -4.0 / 3.0 * np.pi * R**3 * (Delta_P - Qs.n_B * chem_sum)

    W_surface = 4.0 * np.pi * R**2 * sigma

    if include_coulomb_energy:
        W_Coulomb = W_coulomb_energy(R, Qs, H)
    else:
        W_Coulomb = 0.0

    return W_bulk + W_surface + W_Coulomb


# =============================================================================
# Critical radius
# =============================================================================
def critical_radius_standard(Qs, H, sigma):
    """
    Critical droplet radius.

    R_c = 2 sigma / Delta_F_bulk

    where Delta_F_bulk is the general bulk driving force.

    Only defined when Delta_F_bulk > 0 (metastable region).

    ----------
    Qs : object
        Q* phase state with attributes:
        n_B, T, mu_B, mu_C, mu_S, mu_e, mu_nu, Y_C, Y_S, Y_e, Y_nu, P_total, e_total, s_total
    H : object
        Hadronic phase state with attributes:
        n_B, T, mu_B, mu_C, mu_S, mu_e, mu_nu, Y_C, Y_S, Y_e, Y_nu, P_total, e_total, s_total
    sigma : float
        Surface tension (MeV/fm^2).

    Returns
    -------
    float or array
        Critical radius R_c (fm). NaN where Delta_F_bulk <= 0.
    """
    Delta_P = Qs.P_total - H.P_total
    chem_sum = (
        1.0 * (Qs.mu_B - H.mu_B)
        + Qs.Y_C * (Qs.mu_C - H.mu_C)
        + Qs.Y_S * (Qs.mu_S - H.mu_S)
        + Qs.Y_e * (Qs.mu_e - H.mu_e)
        + Qs.Y_nu * (Qs.mu_nu - H.mu_nu)
    )
    Delta_F_bulk = np.asarray(Delta_P - Qs.n_B * chem_sum, dtype=float)
    return np.where(Delta_F_bulk > 0, 2.0 * sigma / Delta_F_bulk, np.nan)


# =============================================================================
# Critical work (energy barrier, without Coulomb)
# =============================================================================
def critical_work_standard(Qs, H, sigma):
    """
    Work of formation at the critical radius (without Coulomb).

    W_c = 16 pi sigma^3 / (3 Delta_F_bulk^2)

    where Delta_F_bulk is the general bulk driving force (Eq. 4.35).

    Only defined when Delta_F_bulk > 0 (metastable region).

    ----------
    Qs : object
        Q* phase state with attributes:
        n_B, T, mu_B, mu_C, mu_S, mu_e, mu_nu, Y_C, Y_S, Y_e, Y_nu, P_total, e_total, s_total
    H : object
        Hadronic phase state with attributes:
        n_B, T, mu_B, mu_C, mu_S, mu_e, mu_nu, Y_C, Y_S, Y_e, Y_nu, P_total, e_total, s_total
    sigma : float
        Surface tension (MeV/fm^2).

    Returns
    -------
    float or array
        Critical work W_c (MeV). NaN where Delta_F_bulk <= 0.
    """
    Delta_P = Qs.P_total - H.P_total
    chem_sum = (
        1.0 * (Qs.mu_B - H.mu_B)
        + Qs.Y_C * (Qs.mu_C - H.mu_C)
        + Qs.Y_S * (Qs.mu_S - H.mu_S)
        + Qs.Y_e * (Qs.mu_e - H.mu_e)
        + Qs.Y_nu * (Qs.mu_nu - H.mu_nu)
    )
    Delta_F_bulk = np.asarray(Delta_P - Qs.n_B * chem_sum, dtype=float)
    return np.where(Delta_F_bulk > 0, 16.0 * np.pi * sigma**3 / (3.0 * Delta_F_bulk**2), np.nan)


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
    H : object
        Hadronic phase state with attributes:
        n_B, T, mu_B, mu_C, mu_S, mu_e, mu_nu, Y_C, Y_S, Y_e, Y_nu, P_total, e_total, s_total
    Qs : object
        Q* phase state with attributes:
        n_B, T, mu_B, mu_C, mu_S, mu_e, mu_nu, Y_C, Y_S, Y_e, Y_nu, P_total, e_total, s_total
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
def nucleation_rate(Qs, H, sigma,
                    xi_q=XI_Q, lambda_th=0.0, zeta_th=0.0, type_coulomb="no"):
    """
    Thermal nucleation rate per unit volume.

    Gamma = kappa / (2 pi) * Omega_0 * exp(-W_c / T)

    Parameters
    ----------
    Qs : object
        Q* phase state (P_total, e_total, n_B, mu_B, ...).
    H : object
        Hadronic phase state (P_total, e_total, n_B, T, mu_B, ...).
    sigma : float
        Surface tension (MeV/fm^2).
    xi_q : float, optional
        Quark correlation length (fm). Default 0.7.
    lambda_th : float, optional
        Thermal conductivity. Default 0.
    zeta_th : float, optional
        Bulk viscosity. Default 0.
    type_coulomb : str, optional
        Type of Coulomb interaction: "no" (no Coulomb), "only_in_W" (only in W), "minimization" (in W and in chemical minimization), "screening" (electron screening).

    Returns
    -------
    float or array
        Nucleation rate Gamma (fm^{-3} s^{-1}). NaN where Delta_F <= 0.
    """
    R_c = critical_radius(Qs, H, sigma, type_coulomb=type_coulomb) 
    W_c = critical_work(Qs, H, sigma, type_coulomb=type_coulomb)

    Omega_0 = statistical_prefactor(sigma, H.T, R_c, xi_q)
    kappa = dynamical_prefactor(sigma, R_c, H, Qs, H.T,
                                lambda_th, zeta_th)

    return kappa / (2.0 * np.pi) * Omega_0 * np.exp(-W_c / H.T)


# =============================================================================
# Nucleation time
# =============================================================================
def nucleation_time(Qs, H, sigma, V,
                    xi_q=XI_Q, lambda_th=0.0, zeta_th=0.0, type_coulomb="no"):
    """
    Nucleation waiting time.

    tau = 1 / (V * Gamma)

    Parameters
    ----------
    Qs, H, sigma :
        Same as nucleation_rate.
    V : float
        System volume (fm^3).
    xi_q, lambda_th, zeta_th :
        Same as nucleation_rate.
    type_coulomb : str, optional
        Type of Coulomb interaction: "no" (no Coulomb), "only_in_W" (only in W), "minimization" (in W and in chemical minimization), "screening" (electron screening).

    Returns
    -------
    float or array
        Nucleation time tau (s). NaN where rate is undefined.
    """
    Gamma = nucleation_rate(Qs, H, sigma, xi_q, lambda_th, zeta_th, type_coulomb=type_coulomb)
    return 1.0 / (V * Gamma)
