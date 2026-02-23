"""
Thermal Nucleation
==================

Thermal homogeneous nucleation rate following Langer's theory for the hadron-to-quark phase transition.

J. S. Langer, Annals Phys. 54 (1969)
J. S. Langer, Physica 73.1 (1974)
L. P. Csernai and J. I. Kapusta, Phys. Rev. D 46 (1992)
J. I. Kapusta, A. P. Vischer, and R. Venugopalan, Phys. Rev. C 51 (1995)

The nucleation rate per unit volume is:

    Gamma = kappa / (2 pi) * Omega_0 * exp(-W_c / T)

where Omega_0 is the statistical prefactor (related to the
curvature of W around the saddle point) and kappa is the
dynamical prefactor (set by dissipation via shear/bulk viscosity
and thermal conductivity).

Functions:
    shear_viscosity        : eta(n_B, T) from Danielewicz (1984)
    statistical_prefactor  : Omega_0(sigma, T, R_c)
    dynamical_prefactor    : kappa(sigma, R_c, H, Qs, T)
    nucleation_rate        : Gamma(W_c, R_c, sigma, T, H, Qs)
    nucleation_time        : tau = 1 / (V * Gamma)
"""

# =============================================================================
# Constants
# =============================================================================
N0 = 0.16    # nuclear saturation density (fm^-3) for shear viscosity.
XI_Q = 0.7   # quark correlation length (fm)




# =============================================================================
# Transport: shear viscosity
# =============================================================================
def shear_viscosity(n_B_H, T, n0=N0):
    """
    Shear viscosity of hadronic matter.
    P. Danielewicz, Phys. Lett. B 146 (1984)

    eta = 7.6e26 (n_B / n0)^2 T^{-2}   [MeV / (fm s)]

    Parameters
    ----------
    n_B_H : float or array
        Baryon number density (fm^-3).
    T : float or array
        Temperature (MeV).
    n0: nuclear saturation density

    Returns
    -------
    float or array
        Shear viscosity eta (MeV / (fm s)).
    """
    return 7.6e26 * (n_B_H / n0)**2 / T**2


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
