"""
Q* Solvers
==========

Equation systems for finding the quark droplet Q* thermodynamics.

Three solver families:

    Frozen       : Flavor fractions frozen from hadronic phase
    Saddlepoint  : Minimization of W w.r.t. flavor composition (+ CFL variant)
    Coulomb      : Full Coulomb minimization with R as unknown (+ CFL variant)

Shared utilities:

    hadronic_guess  : Physics-based initial guess from hadronic chemical potentials
    robust_root     : 4-step fallback root-finding strategy
"""

import numpy as np
from scipy.optimize import root
from eos.alphabag.eos import (
    compute_alphabag_total_thermo_from_mu,
    compute_cfl_total_thermo_from_mu,
)
from eos.alphabag.thermodynamics_quarks import (
    compute_alphabag_thermo_from_mu,
    compute_cfl_thermo_from_mu,
)
from eos.general.thermodynamics_leptons import electron_thermo
from nucleation.physics import coulomb_delta_mu_e, coulomb_delta_P, coulomb_W


# =============================================================================
# Shared utilities
# =============================================================================
def hadronic_guess(H):
    """Physics-based initial guess [mu_u, mu_d, mu_s, mu_e] from hadronic state."""
    mu_d_h = (H.mu_B - H.mu_C) / 3.0
    mu_u_h = (H.mu_B + 2.0 * H.mu_C) / 3.0
    mu_s_h = mu_d_h + H.mu_S
    return np.array([mu_u_h, mu_d_h, mu_s_h, H.mu_e])


def robust_root(equations, guess, h_guess):
    """Solve nonlinear system with fallback strategy.

    Strategy:
    1. hybr with given guess
    2. lm with given guess
    3. hybr with hadronic guess (if different from given guess)
    4. lm with hadronic guess

    Returns scipy OptimizeResult or None.
    """
    sol = root(equations, guess, method='hybr')
    if sol.success:
        return sol

    sol = root(equations, guess, method='lm')
    if sol.success:
        return sol

    if not np.allclose(guess[:len(h_guess)], h_guess):
        sol = root(equations, h_guess, method='hybr')
        if sol.success:
            return sol
        sol = root(equations, h_guess, method='lm')
        if sol.success:
            return sol

    return None


# =============================================================================
# Frozen solver
# =============================================================================
def solve_frozen(H, params, charge_neutrality,
                 include_photons, include_gluons,
                 include_thermal_neutrinos, initial_guess=None):
    """Solve for Q* with frozen flavor fractions.

    4 unknowns: [mu_u, mu_d, mu_s, mu_e]
    4 equations:
      1. Y_C_Qs = Y_C_H                    (frozen C)
      2. Y_S_Qs = Y_S_H                    (frozen S)
      3. charge neutrality (local or global)
      4. saddlepoint on n_B

    Returns AlphaBagEOSResult or None.
    """
    def equations(x):
        mu_u, mu_d, mu_s, mu_e = x
        Qs = compute_alphabag_thermo_from_mu(mu_u, mu_d, mu_s, H.T, params)
        e_Qs = electron_thermo(mu_e, H.T)

        if charge_neutrality == "local":
            charge_eq = Qs.n_C - e_Qs.n
            Y_e_Qs = e_Qs.n / Qs.n_B
        else:  # global
            charge_eq = mu_e - H.mu_e
            Y_e_Qs = H.Y_e

        saddle = ((Qs.mu_B - H.mu_B)
                  + Qs.Y_C * (Qs.mu_C - H.mu_C)
                  + Qs.Y_S * (Qs.mu_S - H.mu_S)
                  + Y_e_Qs * (mu_e - H.mu_e))

        return [Qs.Y_C - H.Y_C, Qs.Y_S - H.Y_S, charge_eq, saddle]

    h_guess = hadronic_guess(H)
    guess = initial_guess if initial_guess is not None else h_guess

    sol = robust_root(equations, guess, h_guess)
    if sol is None:
        return None

    mu_u, mu_d, mu_s, mu_e = sol.x
    return compute_alphabag_total_thermo_from_mu(
        mu_u, mu_d, mu_s, mu_e, H.T, params,
        include_photons=include_photons,
        include_gluons=include_gluons,
        include_thermal_neutrinos=include_thermal_neutrinos,
        mu_nu=H.mu_nu,
    )


# =============================================================================
# Saddlepoint solver (unpaired)
# =============================================================================
def solve_saddlepoint(H, params, charge_neutrality,
                      include_photons, include_gluons,
                      include_thermal_neutrinos, initial_guess=None):
    """Solve for Q* by minimizing W w.r.t. flavor composition.

    4 unknowns: [mu_u, mu_d, mu_s, mu_e]
    4 equations:
      1. mu_C_Qs + mu_e = mu_C_H + mu_e_H  (minimize wrt Y_C)
      2. mu_S_Qs = mu_S_H                   (minimize wrt Y_S)
      3. charge neutrality (local or global)
      4. saddlepoint on n_B

    Returns AlphaBagEOSResult or None.
    """
    def equations(x):
        mu_u, mu_d, mu_s, mu_e = x
        Qs = compute_alphabag_thermo_from_mu(mu_u, mu_d, mu_s, H.T, params)
        e_Qs = electron_thermo(mu_e, H.T)
        Y_e_Qs = e_Qs.n / Qs.n_B

        if charge_neutrality == "local":
            charge_eq = Qs.Y_C - Y_e_Qs
        else:  # global
            charge_eq = mu_e - H.mu_e

        saddle = ((Qs.mu_B - H.mu_B)
                  + Qs.Y_C * (Qs.mu_C - H.mu_C)
                  + Qs.Y_S * (Qs.mu_S - H.mu_S)
                  + Y_e_Qs * (mu_e - H.mu_e))

        return [
            Qs.mu_C + mu_e - (H.mu_C + H.mu_e),
            Qs.mu_S - H.mu_S,
            charge_eq,
            saddle,
        ]

    h_guess = hadronic_guess(H)
    guess = initial_guess if initial_guess is not None else h_guess

    sol = robust_root(equations, guess, h_guess)
    if sol is None:
        return None

    mu_u, mu_d, mu_s, mu_e = sol.x
    return compute_alphabag_total_thermo_from_mu(
        mu_u, mu_d, mu_s, mu_e, H.T, params,
        include_photons=include_photons,
        include_gluons=include_gluons,
        include_thermal_neutrinos=include_thermal_neutrinos,
        mu_nu=H.mu_nu,
    )


# =============================================================================
# Saddlepoint solver (CFL)
# =============================================================================
def solve_saddlepoint_cfl(H, params, Delta0, charge_neutrality,
                          include_photons, include_gluons,
                          include_thermal_neutrinos, initial_guess=None):
    """Solve for CFL Q* thermodynamics.

    4 unknowns: [mu_u, mu_d, mu_s, mu_e]
    4 equations:
      1. Y_C_Qs = 0                        (CFL constraint)
      2. Y_S_Qs = 1                        (CFL constraint)
      3. charge neutrality (local or global)
      4. saddlepoint on n_B

    Returns CFLEOSResult or None.
    """
    def equations(x):
        mu_u, mu_d, mu_s, mu_e = x
        Qs = compute_cfl_thermo_from_mu(mu_u, mu_d, mu_s, H.T, Delta0, params)
        e_Qs = electron_thermo(mu_e, H.T)
        Y_e_Qs = e_Qs.n / Qs.n_B if Qs.n_B > 0 else 0.0

        if charge_neutrality == "local":
            charge_eq = Qs.Y_C - Y_e_Qs
        else:  # global
            charge_eq = mu_e - H.mu_e

        saddle = ((Qs.mu_B - H.mu_B)
                  + Qs.Y_C * (Qs.mu_C - H.mu_C)
                  + Qs.Y_S * (Qs.mu_S - H.mu_S)
                  + Y_e_Qs * (mu_e - H.mu_e))

        return [Qs.Y_C, Qs.Y_S - 1.0, charge_eq, saddle]

    h_guess = hadronic_guess(H)
    guess = initial_guess if initial_guess is not None else h_guess

    sol = robust_root(equations, guess, h_guess)
    if sol is None:
        return None

    mu_u, mu_d, mu_s, mu_e = sol.x
    return compute_cfl_total_thermo_from_mu(
        mu_u, mu_d, mu_s, mu_e, H.T, Delta0, params,
        include_photons=include_photons,
        include_gluons=include_gluons,
        include_thermal_neutrinos=include_thermal_neutrinos,
        mu_nu=H.mu_nu,
    )


# =============================================================================
# Coulomb solver (unpaired, full minimization including R)
# =============================================================================
def solve_coulomb(H, params, sigma,
                  include_photons, include_gluons,
                  include_thermal_neutrinos, initial_guess=None):
    """Solve for Q* AND critical radius R* with Coulomb corrections.

    5 unknowns: [mu_u, mu_d, mu_s, mu_e, R]
    5 equations:
      1. mu_B_Qs = mu_B_H                                       (saddle point)
      2. mu_C_Qs + mu_e = mu_C_H + mu_e_H                      (minimize Y_C)
      3. mu_S_Qs = mu_S_H                                       (minimize Y_S)
      4. mu_e = mu_e_H + 8/5 alpha pi hc R^2 delta_n_C          (Coulomb GCN)
      5. P_Qs - P_H - 2sigma/R + coulomb_delta_P = 0            (stationarity)

    Returns (AlphaBagEOSResult, R_c) or None.
    """
    def equations(x):
        mu_u, mu_d, mu_s, mu_e, R = x
        if R <= 0:
            return [1e10] * 5

        Qs = compute_alphabag_total_thermo_from_mu(
            mu_u, mu_d, mu_s, mu_e, H.T, params,
            include_photons=include_photons,
            include_gluons=include_gluons,
            include_thermal_neutrinos=include_thermal_neutrinos,
            mu_nu=H.mu_nu,
        )
        delta_n_C = (Qs.Y_C - Qs.Y_e) * Qs.n_B

        return [
            Qs.mu_B - H.mu_B,
            Qs.mu_C + mu_e - (H.mu_C + H.mu_e),
            Qs.mu_S - H.mu_S,
            mu_e - coulomb_delta_mu_e(R, delta_n_C) - H.mu_e,
            Qs.P_total - H.P_total - 2.0 * sigma / R + coulomb_delta_P(R, delta_n_C),
        ]

    # Physics-based guess with R estimate
    h_guess_4 = hadronic_guess(H)
    mu_u_h, mu_d_h, mu_s_h = h_guess_4[0], h_guess_4[1], h_guess_4[2]
    Qs_guess = compute_alphabag_total_thermo_from_mu(
        mu_u_h, mu_d_h, mu_s_h, H.mu_e, H.T, params,
        include_photons=include_photons,
        include_gluons=include_gluons,
        include_thermal_neutrinos=include_thermal_neutrinos,
        mu_nu=H.mu_nu,
    )
    Delta_P_guess = Qs_guess.P_total - H.P_total
    R_guess = 2.0 * sigma / Delta_P_guess if Delta_P_guess > 0 else 1.0
    h_guess = np.array([mu_u_h, mu_d_h, mu_s_h, H.mu_e, R_guess])

    guess = initial_guess if initial_guess is not None else h_guess

    sol = robust_root(equations, guess, h_guess)
    if sol is None:
        return None

    mu_u, mu_d, mu_s, mu_e, R_c = sol.x

    if R_c <= 0:
        return None

    residual = np.sum(np.array(equations(sol.x))**2)
    if residual > 1e-10:
        return None

    result = compute_alphabag_total_thermo_from_mu(
        mu_u, mu_d, mu_s, mu_e, H.T, params,
        include_photons=include_photons,
        include_gluons=include_gluons,
        include_thermal_neutrinos=include_thermal_neutrinos,
        mu_nu=H.mu_nu,
    )
    return (result, R_c)


# =============================================================================
# Coulomb solver at fixed R (for computing W(R))
# =============================================================================
def solve_coulomb_at_R(R, H, params, sigma,
                       include_photons, include_gluons,
                       include_thermal_neutrinos, initial_guess=None):
    """Solve for Q* at given R with Coulomb corrections.

    4 unknowns: [mu_u, mu_d, mu_s, mu_e]
    4 equations:
      1. mu_B_Qs = mu_B_H
      2. mu_C_Qs + mu_e = mu_C_H + mu_e_H
      3. mu_S_Qs = mu_S_H
      4. mu_e = mu_e_H + coulomb_delta_mu_e(R, delta_n_C)

    Returns AlphaBagEOSResult or None.
    """
    def equations(x):
        mu_u, mu_d, mu_s, mu_e = x

        Qs = compute_alphabag_total_thermo_from_mu(
            mu_u, mu_d, mu_s, mu_e, H.T, params,
            include_photons=include_photons,
            include_gluons=include_gluons,
            include_thermal_neutrinos=include_thermal_neutrinos,
            mu_nu=H.mu_nu,
        )
        delta_n_C = (Qs.Y_C - Qs.Y_e) * Qs.n_B

        return [
            Qs.mu_B - H.mu_B,
            Qs.mu_C + mu_e - (H.mu_C + H.mu_e),
            Qs.mu_S - H.mu_S,
            mu_e - coulomb_delta_mu_e(R, delta_n_C) - H.mu_e,
        ]

    h_guess = hadronic_guess(H)
    guess = initial_guess if initial_guess is not None else h_guess

    sol = robust_root(equations, guess, h_guess)
    if sol is None:
        return None

    mu_u, mu_d, mu_s, mu_e = sol.x
    return compute_alphabag_total_thermo_from_mu(
        mu_u, mu_d, mu_s, mu_e, H.T, params,
        include_photons=include_photons,
        include_gluons=include_gluons,
        include_thermal_neutrinos=include_thermal_neutrinos,
        mu_nu=H.mu_nu,
    )


# =============================================================================
# CFL Coulomb solver (full minimization including R)
# =============================================================================
def solve_coulomb_cfl(H, params, Delta0, sigma,
                      include_photons, include_gluons,
                      include_thermal_neutrinos, initial_guess=None):
    """Solve for CFL Q* AND R* with Coulomb corrections.

    5 unknowns: [mu_u, mu_d, mu_s, mu_e, R]
    5 equations:
      1. Y_C_Qs = 0                                              (CFL)
      2. Y_S_Qs = 1                                              (CFL)
      3. mu_e = mu_e_H + coulomb_delta_mu_e(R, delta_n_C)        (Coulomb GCN)
      4. mu_B_Qs = mu_B_H                                        (saddle point)
      5. P_Qs - P_H - 2sigma/R + coulomb_delta_P = 0             (stationarity)

    Returns (CFLEOSResult, R_c) or None.
    """
    def equations(x):
        mu_u, mu_d, mu_s, mu_e, R = x
        if R <= 0:
            return [1e10] * 5

        Qs = compute_cfl_total_thermo_from_mu(
            mu_u, mu_d, mu_s, mu_e, H.T, Delta0, params,
            include_photons=include_photons,
            include_gluons=include_gluons,
            include_thermal_neutrinos=include_thermal_neutrinos,
            mu_nu=H.mu_nu,
        )
        delta_n_C = (Qs.Y_C - Qs.Y_e) * Qs.n_B

        return [
            Qs.Y_C,
            Qs.Y_S - 1.0,
            mu_e - coulomb_delta_mu_e(R, delta_n_C) - H.mu_e,
            Qs.mu_B - H.mu_B,
            Qs.P_total - H.P_total - 2.0 * sigma / R + coulomb_delta_P(R, delta_n_C),
        ]

    # Physics-based guess with R estimate
    h_guess_4 = hadronic_guess(H)
    mu_u_h, mu_d_h, mu_s_h = h_guess_4[0], h_guess_4[1], h_guess_4[2]
    Qs_guess = compute_cfl_total_thermo_from_mu(
        mu_u_h, mu_d_h, mu_s_h, H.mu_e, H.T, Delta0, params,
        include_photons=include_photons,
        include_gluons=include_gluons,
        include_thermal_neutrinos=include_thermal_neutrinos,
        mu_nu=H.mu_nu,
    )
    Delta_P_guess = Qs_guess.P_total - H.P_total
    R_guess = 2.0 * sigma / Delta_P_guess if Delta_P_guess > 0 else 1.0
    h_guess = np.array([mu_u_h, mu_d_h, mu_s_h, H.mu_e, R_guess])

    guess = initial_guess if initial_guess is not None else h_guess

    sol = robust_root(equations, guess, h_guess)
    if sol is None:
        return None

    mu_u, mu_d, mu_s, mu_e, R_c = sol.x

    if R_c <= 0:
        return None

    residual = np.sum(np.array(equations(sol.x))**2)
    if residual > 1e-10:
        return None

    result = compute_cfl_total_thermo_from_mu(
        mu_u, mu_d, mu_s, mu_e, H.T, Delta0, params,
        include_photons=include_photons,
        include_gluons=include_gluons,
        include_thermal_neutrinos=include_thermal_neutrinos,
        mu_nu=H.mu_nu,
    )
    return (result, R_c)


# =============================================================================
# CFL Coulomb solver at fixed R
# =============================================================================
def solve_coulomb_cfl_at_R(R, H, params, Delta0, sigma,
                           include_photons, include_gluons,
                           include_thermal_neutrinos, initial_guess=None):
    """Solve for CFL Q* at given R with Coulomb corrections.

    4 unknowns: [mu_u, mu_d, mu_s, mu_e]
    4 equations:
      1. Y_C_Qs = 0
      2. Y_S_Qs = 1
      3. mu_e = mu_e_H + coulomb_delta_mu_e(R, delta_n_C)
      4. mu_B_Qs = mu_B_H

    Returns CFLEOSResult or None.
    """
    def equations(x):
        mu_u, mu_d, mu_s, mu_e = x

        Qs = compute_cfl_total_thermo_from_mu(
            mu_u, mu_d, mu_s, mu_e, H.T, Delta0, params,
            include_photons=include_photons,
            include_gluons=include_gluons,
            include_thermal_neutrinos=include_thermal_neutrinos,
            mu_nu=H.mu_nu,
        )
        delta_n_C = (Qs.Y_C - Qs.Y_e) * Qs.n_B

        return [
            Qs.Y_C,
            Qs.Y_S - 1.0,
            mu_e - coulomb_delta_mu_e(R, delta_n_C) - H.mu_e,
            Qs.mu_B - H.mu_B,
        ]

    h_guess = hadronic_guess(H)
    guess = initial_guess if initial_guess is not None else h_guess

    sol = robust_root(equations, guess, h_guess)
    if sol is None:
        return None

    mu_u, mu_d, mu_s, mu_e = sol.x
    return compute_cfl_total_thermo_from_mu(
        mu_u, mu_d, mu_s, mu_e, H.T, Delta0, params,
        include_photons=include_photons,
        include_gluons=include_gluons,
        include_thermal_neutrinos=include_thermal_neutrinos,
        mu_nu=H.mu_nu,
    )


# =============================================================================
# Work of formation at given R (with Coulomb)
# =============================================================================
def work_of_formation_coulomb_at_R(R, H, params, sigma,
                                   include_photons=True, include_gluons=True,
                                   include_thermal_neutrinos=True,
                                   quark_phase='unpaired', Delta0=None,
                                   initial_guess=None):
    """Work of formation W(R) with Coulomb corrections.

    Solves the Coulomb-corrected equations at given R, then computes:
    W(R) = -4/3 pi R^3 (P_Qs - P_H) + 4 pi R^2 sigma + W_Coulomb(R)

    Returns float or None.
    """
    if quark_phase == 'cfl':
        result = solve_coulomb_cfl_at_R(
            R, H, params, Delta0, sigma,
            include_photons, include_gluons,
            include_thermal_neutrinos, initial_guess)
    else:
        result = solve_coulomb_at_R(
            R, H, params, sigma,
            include_photons, include_gluons,
            include_thermal_neutrinos, initial_guess)

    if result is None:
        return None

    delta_n_C = (result.Y_C - result.Y_e) * result.n_B

    W = (-4.0 / 3.0 * np.pi * R**3 * (result.P_total - H.P_total)
         + 4.0 * np.pi * R**2 * sigma
         + coulomb_W(R, delta_n_C))

    return W
