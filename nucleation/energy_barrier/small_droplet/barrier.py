"""
Nucleation Barrier
==================

Work of formation W(R) for a quark droplet nucleating inside
hadronic matter, and related quantities (critical radius, critical work).

It applies when the critical quark droplet is much smaller than the total volume.

W(R) = W_bulk + W_surface + W_Coulomb

Components:
    bulk_W          : volume energy  4/3 pi R^3 Delta_f_bulk
    surface_W       : surface energy 4 pi R^2 sigma
    coulomb_W       : Coulomb energy (uniform-charge-sphere model)

Coulomb corrections to the minimization solvers:
    coulomb_delta_mu_e : shift of mu_e from electrostatic screening
    coulomb_delta_P    : pressure-balance correction from dW/dR = 0
    coulomb_P          : thermodynamic Coulomb pressure -dE_Coul/dV

Critical quantities:
    critical_radius_noCoulomb  : R_c = -2 sigma / Delta_f_bulk
    critical_work_noCoulomb    : W_c = 16 pi sigma^3 / (3 Delta_f_bulk^2)
    critical_radius_coulomb    : R_c from numerical dW/dR = 0

Driving force:
    driving_force : Delta_f_bulk = -(P_Qs - P_H) + n_B sum_i Y_i (dmu_i)
"""

import numpy as np
from scipy.optimize import root
from eos.general.physics_constants import alpha_EM, hc



def bulk_W(R, Delta_f_bulk):
    """Bulk (volume) contribution to the work of formation.

    W_bulk = 4/3 pi R^3 Delta_f_bulk

    Parameters
    ----------
    R : float or array
        Droplet radius (fm).
    Delta_f_bulk : float or array
        Bulk driving force (MeV/fm^3).

    Returns
    -------
    float or array
        Bulk contribution to W (MeV).
    """
    return 4.0 / 3.0 * np.pi * R**3 * Delta_f_bulk


def surface_W(R, sigma):
    """Surface contribution to the work of formation.

    W_surface = 4 pi R^2 sigma

    Parameters
    ----------
    R : float or array
        Droplet radius (fm).
    sigma : float
        Surface tension (MeV/fm^2).

    Returns
    -------
    float or array
        Surface contribution to W (MeV).
    """
    return 4.0 * np.pi * R**2 * sigma


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
    Coulomb correction to pressure balance (appears in dW/dR = 0)
    We first minimized W with respect to R and only afterwards imposed the relations that minimize W with respect to the remaining
    variables. 
    Namely, this delta_P is the correction to the pressure balance that arises from the Coulomb interaction, not -dE_coul/dV.
    PQ+delta_P-2sigma/R=PH is the pressure balance equation.

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

    W_Coulomb = (16/15) pi^2 hbar*c alpha_em delta_n_C^2 R^5

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
    return (16.0 / 15.0) * np.pi**2 * hc * alpha_EM * delta_n_C**2 * R**5


def coulomb_delta_W(R, delta_n_C):
    """
    This is the correction to the work of formation due to the Coulomb interaction in the minimize_coulomb case. 
    See "Note on the relation between coulomb_delta_P and coulomb_P" for details.

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
# Work of formation at arbitrary radius
# =============================================================================
def work_of_formation(R, Delta_f_bulk, sigma, delta_n_C=0.0):
    """
    Work of formation of a quark droplet at radius R.

    W(R) = 4/3 pi R^3 Delta_f_bulk + 4 pi R^2 sigma + W_Coulomb(R)

    When delta_n_C = 0, the Coulomb term vanishes.

    Parameters
    ----------
    R : float or ndarray
        Droplet radius (fm).
    Delta_f_bulk : float or ndarray
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
    return bulk_W(R, Delta_f_bulk) + surface_W(R, sigma) + coulomb_W(R, delta_n_C)


# =============================================================================
# Critical radius (no Coulomb)
# =============================================================================
def critical_radius_noCoulomb(Delta_f_bulk, sigma):
    """
    Critical droplet radius (standard CNT, no Coulomb).

    R_c = - 2 sigma / Delta_f_bulk

    Parameters
    ----------
    Delta_f_bulk : float or ndarray
        Bulk driving force (MeV/fm^3).
    sigma : float
        Surface tension (MeV/fm^2).

    Returns
    -------
    float or ndarray
        Critical radius R_c (fm). NaN where Delta_f_bulk <= 0.
    """
    Delta_f_bulk = np.asarray(Delta_f_bulk, dtype=float)
    return np.where(Delta_f_bulk < 0, -2.0 * sigma / Delta_f_bulk, np.nan)


# =============================================================================
# Critical work (no Coulomb)
# =============================================================================
def critical_work_noCoulomb(Delta_f_bulk, sigma):
    """
    Energy barrier at the critical radius

    W_c = 16 pi sigma^3 / (3 Delta_f_bulk^2)

    Parameters
    ----------
    Delta_f_bulk : float or ndarray
        Bulk driving force (MeV/fm^3).
    sigma : float
        Surface tension (MeV/fm^2).

    Returns
    -------
    float or ndarray
        Critical work W_c (MeV). NaN where Delta_f_bulk >= 0.
    """
    Delta_f_bulk = np.asarray(Delta_f_bulk, dtype=float)
    return np.where(
        Delta_f_bulk < 0,
        16.0 * np.pi * sigma**3 / (3.0 * Delta_f_bulk**2),
        np.nan,
    )


# =============================================================================
# Critical radius with Coulomb correction 
# =============================================================================
def critical_radius_coulomb(Delta_f_bulk, sigma, delta_n_C, initial_guess=None):
    """
    Critical radius including Coulomb correction. 

    Solves dW/dR = 0 where W = bulk + surface + Coulomb

    Parameters
    ----------
    Delta_f_bulk : float
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
    if Delta_f_bulk >= 0 or np.isnan(Delta_f_bulk):
        return np.nan

    def dW_dR(R):
        return (4.0 * np.pi * R**2 * Delta_f_bulk
                + 8.0 * np.pi * R * sigma
                + (16.0 / 3.0) * np.pi**2 * hc * alpha_EM * delta_n_C**2 * R**4)
    
    # note that defining P_Coul=-dE/dVQ|_ni = - 1/(4 Pi R^2) * dE/dR|_ni
    # P_Coul = - 4/3 * pi * hc * alpha_EM * delta_n_C^2 * R^2 that is coulomb_P().

    if initial_guess is None:
        initial_guess = - 2.0 * sigma / Delta_f_bulk
    sol = root(dW_dR, initial_guess)
    if sol.success:
        return float(sol.x[0])
    return np.nan



# =============================================================================
# Bulk driving force
# =============================================================================
def driving_force(Qs, H):
    """
    Bulk driving force for nucleation if the Q* droplet is much smaller than the total volume.

    #### H is metastable if f_Qs < f_H, i.e. Delta_f_bulk < 0. ####

    Delta_f_bulk = F_Qs - F_H = -(P_Qs - P_H) + n_B_Qs * sum_i Y_i^Qs (mu_i^Qs - mu_i^H)

    where i = B, C, S, e, nu and Y_B = 1 by definition.

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
    return -Delta_P + Qs.n_B * chem_sum


# =============================================================================
# Switching functions for unpCFL (radius-dependent phase blending)
# =============================================================================
def switching_step(R, Rx):
    """Step switching function for unpCFL phase blending.

    S(R) = 0 for R < Rx  (unpaired)
    S(R) = 1 for R >= Rx (CFL)

    Parameters
    ----------
    R : float or array
        Droplet radius (fm).
    Rx : float
        Crossover radius (fm).

    Returns
    -------
    float or array
        Switching function value in [0, 1].
    """
    return np.where(R > Rx, 1.0, 0.0)


def switching_tanh(R, Rx, delta):
    """Smooth tanh switching function for unpCFL phase blending.

    S(R) = 0.5 * (1 + tanh((R - Rx) / delta))

    Goes from ~0 at R << Rx (unpaired) to ~1 at R >> Rx (CFL).

    Parameters
    ----------
    R : float or array
        Droplet radius (fm).
    Rx : float
        Crossover radius (fm).
    delta : float
        Transition width (fm).

    Returns
    -------
    float or array
        Switching function value in [0, 1].
    """
    return 0.5 * (1.0 + np.tanh((R - Rx) / delta))


def get_switching_function(mode, Rx, delta=None):
    """Return a switching callable S(R) for the given mode.

    Parameters
    ----------
    mode : str
        'step' or 'tanh'.
    Rx : float
        Crossover radius (fm).
    delta : float or None
        Transition width (fm). Required for 'tanh'.

    Returns
    -------
    callable
        S(R) -> float or array.
    """
    if mode == 'step':
        return lambda R: switching_step(R, Rx)
    elif mode == 'tanh':
        if delta is None:
            raise ValueError("delta required for tanh switching")
        return lambda R: switching_tanh(R, Rx, delta)
    else:
        raise ValueError(f"Unknown switching mode: '{mode}'")


"""
Note on the relation between coulomb_delta_P and coulomb_P
==========================================================

Notation:
    Df      = Delta_f_bulk = driving_force(Qs, H)
    DP      = P_Qs - P_H
    dnC     = delta_n_C
    alpha   = alpha_EM
    hc      = hbar * c

Without Coulomb
------------------
W minimization wrt Y_i implies Df = -DP (chem_sum vanishes). Thus:

    W = -4/3 pi R^3 DP + 4 pi R^2 sigma

    dW/dR = 0  =>  R_c = 2 sigma / DP

With Coulomb
---------------
The full W before simplification is:

    W = 4/3 pi R^3 Df + 4 pi R^2 sigma + 16/15 pi^2 hc alpha dnC^2 R^5

Minimizing W wrt Y_i yields the condition (from dW/dY_e = 0):

    Df = -DP - 8/5 pi hc alpha dnC^2 R^2

Note: the extra term comes from the electron contribution to the
chemical sum in the bulk driving force (coulomb_delta_mu_e).
Unlike the no-Coulomb case, chem_sum does NOT vanish.

Substituting into W and simplifying:

    W = -4/3 pi R^3 DP + 4 pi R^2 sigma - 16/15 pi^2 hc alpha dnC^2 R^5

(the last term is the net of the Coulomb W and the bulk chem_sum contribution)

Starting from dW/dR of the general W:

    dW/dR = 4 pi R^2 Df + 8 pi R sigma + 16/3 pi^2 hc alpha dnC^2 R^4

Substituting Df = -DP - 8/5 pi hc alpha dnC^2 R^2:

    dW/dR = 4 pi R^2 (-DP - 8/5 pi hc alpha dnC^2 R^2)
            + 8 pi R sigma
            + 16/3 pi^2 hc alpha dnC^2 R^4
          = -4 pi R^2 DP + 8 pi R sigma - 16/15 pi^2 hc alpha dnC^2 R^4

Setting dW/dR = 0 and dividing by 4 pi R^2:

    -DP + 2 sigma / R - 4/15 pi hc alpha dnC^2 R^2 = 0

Rearranging:

    P_Qs + 4/15 pi hc alpha dnC^2 R^2 - 2 sigma / R = P_H
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            this is coulomb_delta_P(), NOT coulomb_P()

The relation between the two:

    coulomb_delta_P = coulomb_P + 8/5 pi hc alpha dnC^2 R^2
                    = -4/3 pi hc alpha dnC^2 R^2 + 8/5 pi hc alpha dnC^2 R^2
                    = 4/15 pi hc alpha dnC^2 R^2

where the 8/5 pi hc alpha dnC^2 R^2 term originates from the chem_sum
(specifically, from coulomb_delta_mu_e * Y_e in the bulk driving force).
"""


