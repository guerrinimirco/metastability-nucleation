"""
Q* Computation: Saddlepoint Nucleation with Coulomb Corrections
===================================================================

Given initial hadronic conditions, find the quark droplet Q* thermodynamics
AND the critical radius R* by solving the Coulomb-modified saddlepoint equations.

Minimization case (GCN with Coulomb):
  Simultaneously solves for quark chemical potentials and droplet radius,
  with Coulomb corrections to the electron chemical potential and pressure balance.

  The Coulomb energy of a uniformly charged sphere with net charge
  density Δn_C = n_C_Q - n_e modifies:
    - Electron chemical potential: μ_e_H = μ_e - 8/5 α_em π ℏc R² Δn_C
    - Pressure balance: P_Qs = P_H + 2σ/R - 4/15 α_em π ℏc R² Δn_C²
    - Work of formation: W_Coulomb = -16/15 π² ℏc α_em Δn_C² R⁵

Screening case: TODO
"""

import os
import numpy as np
from types import SimpleNamespace
from scipy.optimize import root
from dataclasses import dataclass
from eos.alphabag.eos import compute_alphabag_total_thermo_from_mu, compute_cfl_total_thermo_from_mu
from nucleation.general.general_nucleation import (
    coulomb_delta_mu_e, coulomb_delta_P, coulomb_P, coulomb_W,
)


# =============================================================================
# Unpaired: solve with Coulomb (full minimization including R)
# =============================================================================
def solve_Qstar_coulomb(H, params, sigma,
                        include_photons, include_gluons,
                        include_thermal_neutrinos, initial_guess):
    """
    Solve for Q* thermodynamics AND critical radius R* with Coulomb corrections.

    5 unknowns: [mu_u, mu_d, mu_s, mu_e, R]
    5 equations:
      1. mu_B_Qs = mu_B_H                                      (saddle point on n_B)
      2. mu_C_Qs + mu_e = mu_C_H + mu_e_H                      (minimize wrt Y_C)
      3. mu_S_Qs = mu_S_H                                      (minimize wrt Y_S)
      4. mu_e_Qs = mu_e_H + 8/5 alpha_em pi hc R^2 (n_C_Qs - n_e_Qs)  (Coulomb GCN)
      5. P_Qs - 2 sigma/R + 4/15 alpha_em pi hc R^2 (n_C_Q - n_e)^2 = P_H 
                                                                (stationarity wrt R)

    Note that the pressure balance equation is derived from the minimization of 
    W = -4/3 pi R^3 [P_Qs - P_H - n_BQs Sum_i Y_i_Qs (mu_i_Qs-mu_i_H) ] + 2 sigma 4 pi R^2 + 16/15 pi^2 R^5 alpha_em n_B_Qs(Y_C_Qs - Y_e_Qs)^2
    with respect to R, and then applying che chemical equilibrium relations 1-4. 
    The term + 4/15 alpha_em pi hc R^2 (n_C_Q - n_e)^2 collects the contribution from the derivative of ECoul=16/15 pi^2 R^5 alpha_em n_B_Qs(Y_C_Qs - Y_e_Qs)^2
    together with the second term of Wbulk, namely +4/3 pi R^3 [n_BQs Sum_i Y_i_Qs (mu_i_Qs-mu_i_H)]

    Parameters
    ----------
    H : object
        Hadronic state with attributes:
        T, mu_B, mu_C, mu_S, mu_e, mu_nu, P_total.
    params : AlphaBagParams
    sigma : float
        Surface tension (MeV/fm^2).
    include_photons, include_gluons, include_thermal_neutrinos : bool
    initial_guess : array-like or None
        Guess for [mu_u, mu_d, mu_s, mu_e, R] (MeV, fm).

    Returns
    -------
    (AlphaBagEOSResult, R_c) or None
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
            mu_nu=H.mu_nu
        )

        delta_n_C = Qs.Y_C * Qs.n_B - Qs.n_e

        eq1 = Qs.mu_B - H.mu_B
        eq2 = Qs.mu_C + mu_e - (H.mu_C + H.mu_e)
        eq3 = Qs.mu_S - H.mu_S
        eq4 = mu_e - coulomb_delta_mu_e(R, delta_n_C) - H.mu_e
        eq5 = Qs.P_total - H.P_total - 2.0 * sigma / R + coulomb_delta_P(R, delta_n_C)

        return [eq1, eq2, eq3, eq4, eq5]


    # Physics-based guess
    mu_d_h = (H.mu_B - H.mu_C) / 3.0
    mu_u_h = (H.mu_B + 2.0 * H.mu_C) / 3.0
    mu_s_h = mu_d_h + H.mu_S

    # Estimate R from standard CNT: R_c ~ 2 sigma / (P_Qs - P_H)
    Qs_guess = compute_alphabag_total_thermo_from_mu(
        mu_u_h, mu_d_h, mu_s_h, H.mu_e, H.T, params,
        include_photons=include_photons,
        include_gluons=include_gluons,
        include_thermal_neutrinos=include_thermal_neutrinos,
        mu_nu=H.mu_nu
    )
    Delta_P_guess = Qs_guess.P_total - H.P_total
    R_guess = 2.0 * sigma / Delta_P_guess if Delta_P_guess > 0 else 1.0

    h_guess = np.array([mu_u_h, mu_d_h, mu_s_h, H.mu_e, R_guess])

    if initial_guess is None:
        initial_guess = h_guess

    sol = root(equations, initial_guess, method='hybr')
    if not sol.success:
        sol = root(equations, initial_guess, method='lm')

    if not sol.success and not np.allclose(initial_guess[:4], h_guess[:4]):
        sol = root(equations, h_guess, method='hybr')
        if not sol.success:
            sol = root(equations, h_guess, method='lm')

    if not sol.success:
        return None

    mu_u, mu_d, mu_s, mu_e, R_c = sol.x

    if R_c <= 0:
        return None

    # Verify solution quality
    residual = np.sum(np.array(equations(sol.x))**2)
    if residual > 1e-10:
        return None

    result = compute_alphabag_total_thermo_from_mu(
        mu_u, mu_d, mu_s, mu_e, H.T, params,
        include_photons=include_photons,
        include_gluons=include_gluons,
        include_thermal_neutrinos=include_thermal_neutrinos,
        mu_nu=H.mu_nu
    )

    return (result, R_c)


# =============================================================================
# Unpaired: solve at fixed R (for computing W(R))
# =============================================================================
def solve_Qstar_coulomb_at_R(R, H, params, sigma,
                              include_photons, include_gluons,
                              include_thermal_neutrinos, initial_guess):
    """
    Solve for Q* thermodynamics at given R with Coulomb corrections.

    4 unknowns: [mu_u, mu_d, mu_s, mu_e]
    4 equations:
      1. mu_B_Qs = mu_B_H
      2. mu_C_Qs + mu_e = mu_C_H + mu_e_H
      3. mu_S_Qs = mu_S_H
      4. mu_e_H = mu_e - 8/5 alpha_em pi hc R^2 (n_C_Q - n_e)

    Parameters
    ----------
    R : float
        Droplet radius (fm).
    H : object
        Hadronic state with attributes:
        T, mu_B, mu_C, mu_S, mu_e, mu_nu.
    params : AlphaBagParams
    sigma : float
        Surface tension (MeV/fm^2).
    initial_guess : array-like or None
        Guess for [mu_u, mu_d, mu_s, mu_e] (MeV).

    Returns
    -------
    AlphaBagEOSResult or None
    """

    def equations(x):
        mu_u, mu_d, mu_s, mu_e = x

        Qs = compute_alphabag_total_thermo_from_mu(
            mu_u, mu_d, mu_s, mu_e, H.T, params,
            include_photons=include_photons,
            include_gluons=include_gluons,
            include_thermal_neutrinos=include_thermal_neutrinos,
            mu_nu=H.mu_nu
        )

        delta_n_C = Qs.Y_C * Qs.n_B - Qs.n_e

        eq1 = Qs.mu_B - H.mu_B
        eq2 = Qs.mu_C + mu_e - (H.mu_C + H.mu_e)
        eq3 = Qs.mu_S - H.mu_S
        eq4 = mu_e - coulomb_delta_mu_e(R, delta_n_C) - H.mu_e

        return [eq1, eq2, eq3, eq4]


    if initial_guess is None:
        mu_d_h = (H.mu_B - H.mu_C) / 3.0
        mu_u_h = (H.mu_B + 2.0 * H.mu_C) / 3.0
        mu_s_h = mu_d_h + H.mu_S
        h_guess = np.array([mu_u_h, mu_d_h, mu_s_h, H.mu_e])
        initial_guess = h_guess

    sol = root(equations, initial_guess, method='hybr')
    if not sol.success:
        sol = root(equations, initial_guess, method='lm')

    if not sol.success and not np.allclose(initial_guess, h_guess):
        sol = root(equations, h_guess, method='hybr')
        if not sol.success:
            sol = root(equations, h_guess, method='lm')

    if not sol.success:
        return None

    mu_u, mu_d, mu_s, mu_e = sol.x

    return compute_alphabag_total_thermo_from_mu(
        mu_u, mu_d, mu_s, mu_e, H.T, params,
        include_photons=include_photons,
        include_gluons=include_gluons,
        include_thermal_neutrinos=include_thermal_neutrinos,
        mu_nu=H.mu_nu
    )


# =============================================================================
# CFL: solve with Coulomb (full minimization including R)
# =============================================================================
def solve_Qstar_cfl_coulomb(H, params, Delta0, sigma,
                             include_photons, include_gluons,
                             include_thermal_neutrinos, initial_guess):
    """
    Solve for CFL Q* thermodynamics AND critical radius R* with Coulomb.

    5 unknowns: [mu_u, mu_d, mu_s, mu_e, R]
    5 equations:
      1. Y_C_Qs = 0                                            (CFL constraint)
      2. Y_S_Qs = 1                                            (CFL constraint)
      3. mu_e_H = mu_e - 8/5 alpha_em pi hc R^2 (n_C_Q - n_e)  (Coulomb GCN)
      4. mu_B_Qs = mu_B_H                                      (saddle point on n_B)
      5. P_Qs = P_H + 2 sigma/R - 4/15 alpha_em pi hc R^2 (n_C_Q - n_e)^2
                                                                (stationarity wrt R)

    Parameters
    ----------
    H : object
        Hadronic state with attributes:
        T, mu_B, mu_C, mu_S, mu_e, mu_nu, P_total.
    params : AlphaBagParams
    Delta0 : float
        Zero-temperature CFL pairing gap (MeV).
    sigma : float
        Surface tension (MeV/fm^2).
    initial_guess : array-like or None
        Guess for [mu_u, mu_d, mu_s, mu_e, R].

    Returns
    -------
    (CFLEOSResult, R_c) or None
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
            mu_nu=H.mu_nu
        )

        delta_n_C = Qs.Y_C * Qs.n_B - Qs.n_e

        eq1 = Qs.Y_C                 # = 0
        eq2 = Qs.Y_S - 1.0           # = 0
        eq3 = mu_e - coulomb_delta_mu_e(R, delta_n_C) - H.mu_e
        eq4 = Qs.mu_B - H.mu_B
        eq5 = Qs.P_total - H.P_total - 2.0 * sigma / R + coulomb_delta_P(R, delta_n_C)

        return [eq1, eq2, eq3, eq4, eq5]


    # Physics-based guess
    mu_d_h = (H.mu_B - H.mu_C) / 3.0
    mu_u_h = (H.mu_B + 2.0 * H.mu_C) / 3.0
    mu_s_h = mu_d_h + H.mu_S

    # Estimate R from standard CNT: R_c ~ 2 sigma / (P_Qs - P_H)
    Qs_guess = compute_cfl_total_thermo_from_mu(
        mu_u_h, mu_d_h, mu_s_h, H.mu_e, H.T, Delta0, params,
        include_photons=include_photons,
        include_gluons=include_gluons,
        include_thermal_neutrinos=include_thermal_neutrinos,
        mu_nu=H.mu_nu
    )
    Delta_P_guess = Qs_guess.P_total - H.P_total
    R_guess = 2.0 * sigma / Delta_P_guess if Delta_P_guess > 0 else 1.0

    h_guess = np.array([mu_u_h, mu_d_h, mu_s_h, H.mu_e, R_guess])

    if initial_guess is None:
        initial_guess = h_guess

    sol = root(equations, initial_guess, method='hybr')
    if not sol.success:
        sol = root(equations, initial_guess, method='lm')

    if not sol.success and not np.allclose(initial_guess[:4], h_guess[:4]):
        sol = root(equations, h_guess, method='hybr')
        if not sol.success:
            sol = root(equations, h_guess, method='lm')

    if not sol.success:
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
        mu_nu=H.mu_nu
    )

    return (result, R_c)


# =============================================================================
# CFL: solve at fixed R
# =============================================================================
def solve_Qstar_cfl_coulomb_at_R(R, H, params, Delta0, sigma,
                                  include_photons, include_gluons,
                                  include_thermal_neutrinos, initial_guess):
    """
    Solve for CFL Q* thermodynamics at given R with Coulomb corrections.

    4 unknowns: [mu_u, mu_d, mu_s, mu_e]
    4 equations:
      1. Y_C_Qs = 0
      2. Y_S_Qs = 1
      3. mu_e_H = mu_e - 8/5 alpha_em pi hc R^2 (n_C_Q - n_e)
      4. mu_B_Qs = mu_B_H

    Parameters
    ----------
    R : float
        Droplet radius (fm).
    H : object
        Hadronic state.
    params : AlphaBagParams
    Delta0 : float
        CFL pairing gap (MeV).
    sigma : float
        Surface tension (MeV/fm^2).
    initial_guess : array-like or None

    Returns
    -------
    CFLEOSResult or None
    """

    def equations(x):
        mu_u, mu_d, mu_s, mu_e = x

        Qs = compute_cfl_total_thermo_from_mu(
            mu_u, mu_d, mu_s, mu_e, H.T, Delta0, params,
            include_photons=include_photons,
            include_gluons=include_gluons,
            include_thermal_neutrinos=include_thermal_neutrinos,
            mu_nu=H.mu_nu
        )

        delta_n_C = Qs.Y_C * Qs.n_B - Qs.n_e

        eq1 = Qs.Y_C
        eq2 = Qs.Y_S - 1.0
        eq3 = mu_e - coulomb_delta_mu_e(R, delta_n_C) - H.mu_e
        eq4 = Qs.mu_B - H.mu_B

        return [eq1, eq2, eq3, eq4]

    if initial_guess is None:
        mu_d_h = (H.mu_B - H.mu_C) / 3.0
        mu_u_h = (H.mu_B + 2.0 * H.mu_C) / 3.0
        mu_s_h = mu_d_h + H.mu_S
        h_guess = np.array([mu_u_h, mu_d_h, mu_s_h, H.mu_e])
        initial_guess = h_guess

    sol = root(equations, initial_guess, method='hybr')
    if not sol.success:
        sol = root(equations, initial_guess, method='lm')

    if not sol.success and not np.allclose(initial_guess, h_guess):
        sol = root(equations, h_guess, method='hybr')
        if not sol.success:
            sol = root(equations, h_guess, method='lm')

    if not sol.success:
        return None

    mu_u, mu_d, mu_s, mu_e = sol.x

    return compute_cfl_total_thermo_from_mu(
        mu_u, mu_d, mu_s, mu_e, H.T, Delta0, params,
        include_photons=include_photons,
        include_gluons=include_gluons,
        include_thermal_neutrinos=include_thermal_neutrinos,
        mu_nu=H.mu_nu
    )


# =============================================================================
# Work of formation at given R (with Coulomb)
# =============================================================================
def work_of_formation_coulomb_at_R(R, H, params, sigma,
                                    include_photons=True, include_gluons=True,
                                    include_thermal_neutrinos=True,
                                    quark_phase='unpaired', Delta0=None,
                                    initial_guess=None):
    """
    Work of formation W(R) with Coulomb corrections.

    Solves the Coulomb-corrected equations at given R, then computes:
    W(R) = -4/3 pi R^3 (P_Qs - P_H) + 4 pi R^2 sigma
           - 16/15 pi^2 hbar*c alpha_em (n_C_Q - n_e)^2 R^5

    Parameters
    ----------
    R : float
        Droplet radius (fm).
    H : object
        Hadronic state with P_total.
    params : AlphaBagParams
    sigma : float
        Surface tension (MeV/fm^2).
    quark_phase : str
        'unpaired' or 'cfl'.
    Delta0 : float or None
        CFL pairing gap (MeV). Required if quark_phase='cfl'.
    initial_guess : array-like or None

    Returns
    -------
    float or None
        W(R) in MeV, or None if solver fails.
    """
    if quark_phase == 'cfl':
        result = solve_Qstar_cfl_coulomb_at_R(
            R, H, params, Delta0, sigma,
            include_photons, include_gluons,
            include_thermal_neutrinos, initial_guess)
    else:
        result = solve_Qstar_coulomb_at_R(
            R, H, params, sigma,
            include_photons, include_gluons,
            include_thermal_neutrinos, initial_guess)

    if result is None:
        return None

    delta_n_C = result.Y_C * result.n_B - result.n_e

    W = (-4.0 / 3.0 * np.pi * R**3 * (result.P_total - H.P_total)
         + 4.0 * np.pi * R**2 * sigma
         + coulomb_W(R, delta_n_C))

    return W


# =============================================================================
# Grid computation: beta-equilibrium with Coulomb
# =============================================================================
def compute_Qstar_coulomb_table_betaeq(hadronic_table, params, sigma,
                                        include_photons=True, include_gluons=True,
                                        include_thermal_neutrinos=True,
                                        quark_phase='unpaired', Delta0=None,
                                        initial_guess=None, verbose=False,
                                        save_table=False, output_file=None):
    """
    Compute Q* + R_c table over a 2D grid (n_B_H, T) with Coulomb corrections.

    Parameters
    ----------
    hadronic_table : EOSTableData
        Table from load_eos_table(..., 'beta_eq').
        grids: 'n_B', 'T'
        data:  'Y_C', 'Y_S', 'mu_B', 'mu_C', 'mu_S', 'mu_e', 'P_total'
    params : AlphaBagParams
    sigma : float
        Surface tension (MeV/fm^2).
    quark_phase : str
        'unpaired' or 'cfl'.
    Delta0 : float or None
        CFL pairing gap. Required if quark_phase='cfl'.

    Returns
    -------
    QstarCoulombTableData
    """
    if quark_phase == 'cfl' and Delta0 is None:
        raise ValueError("Delta0 must be provided for quark_phase='cfl'")

    n_B_arr = hadronic_table.grids['n_B']
    T_arr = hadronic_table.grids['T']
    d = hadronic_table.data
    shape = (len(n_B_arr), len(T_arr))
    data = _init_coulomb_data(shape)
    row_start_guess = initial_guess

    for i_T in range(len(T_arr)):
        current_guess = row_start_guess
        row_converged = 0
        for i_nB in range(len(n_B_arr)):
            idx = (i_nB, i_T)

            Y_C_H = d['Y_C'][i_nB, i_T]
            H = SimpleNamespace(
                T=T_arr[i_T],
                Y_C=Y_C_H,
                Y_S=d['Y_S'][i_nB, i_T],
                Y_e=Y_C_H,
                Y_nu=0.0,
                mu_B=d['mu_B'][i_nB, i_T],
                mu_C=d['mu_C'][i_nB, i_T],
                mu_S=d['mu_S'][i_nB, i_T],
                mu_e=d['mu_e'][i_nB, i_T],
                mu_nu=0.0,
                P_total=d['P_total'][i_nB, i_T],
            )

            if quark_phase == 'cfl':
                sol = solve_Qstar_cfl_coulomb(
                    H, params, Delta0, sigma,
                    include_photons, include_gluons,
                    include_thermal_neutrinos, current_guess)
            else:
                sol = solve_Qstar_coulomb(
                    H, params, sigma,
                    include_photons, include_gluons,
                    include_thermal_neutrinos, current_guess)

            current_guess = _store_coulomb_result(data, idx, sol, current_guess)
            if sol is not None:
                row_converged += 1
                if i_nB == 0:
                    row_start_guess = current_guess

        if verbose:
            print(f"  [{i_T+1}/{len(T_arr)}] T={T_arr[i_T]:.1f} "
                  f"-> {row_converged}/{len(n_B_arr)} converged")

    hadronic_grids = {'n_B_H': n_B_arr, 'T': T_arr}
    result = QstarCoulombTableData(
        eq_type='betaeq', hadronic_grids=hadronic_grids, data=data)

    if save_table:
        if output_file is None:
            output_file = f"Qstar_coulomb_{result.eq_type}.dat"
        _export_table(result, params, sigma, output_file)

    return result


# =============================================================================
# Grid computation: trapped neutrinos with Coulomb
# =============================================================================
def compute_Qstar_coulomb_table_trapped(hadronic_table, params, sigma,
                                         include_photons=True, include_gluons=True,
                                         include_thermal_neutrinos=True,
                                         quark_phase='unpaired', Delta0=None,
                                         initial_guess=None, verbose=False,
                                         save_table=False, output_file=None):
    """
    Compute Q* + R_c table over a 3D grid (n_B_H, Y_L_H, T) with Coulomb.

    Parameters
    ----------
    hadronic_table : EOSTableData
        Table from load_eos_table(..., 'trapped_neutrinos').
        grids: 'n_B', 'Y_L', 'T'
        data:  'Y_C', 'Y_S', 'mu_B', 'mu_C', 'mu_S', 'mu_e', 'mu_nu', 'P_total'
    """
    if quark_phase == 'cfl' and Delta0 is None:
        raise ValueError("Delta0 must be provided for quark_phase='cfl'")

    n_B_arr = hadronic_table.grids['n_B']
    Y_L_arr = hadronic_table.grids['Y_L']
    T_arr = hadronic_table.grids['T']
    d = hadronic_table.data
    shape = (len(n_B_arr), len(Y_L_arr), len(T_arr))
    data = _init_coulomb_data(shape)
    row_start_guess = initial_guess

    for i_YL in range(len(Y_L_arr)):
        for i_T in range(len(T_arr)):
            current_guess = row_start_guess
            row_converged = 0
            for i_nB in range(len(n_B_arr)):
                idx = (i_nB, i_YL, i_T)

                Y_C_H = d['Y_C'][i_nB, i_YL, i_T]
                Y_L_H = Y_L_arr[i_YL]
                H = SimpleNamespace(
                    T=T_arr[i_T],
                    Y_C=Y_C_H,
                    Y_S=d['Y_S'][i_nB, i_YL, i_T],
                    Y_e=Y_C_H,
                    Y_nu=Y_L_H - Y_C_H,
                    mu_B=d['mu_B'][i_nB, i_YL, i_T],
                    mu_C=d['mu_C'][i_nB, i_YL, i_T],
                    mu_S=d['mu_S'][i_nB, i_YL, i_T],
                    mu_e=d['mu_e'][i_nB, i_YL, i_T],
                    mu_nu=d['mu_nu'][i_nB, i_YL, i_T],
                    P_total=d['P_total'][i_nB, i_YL, i_T],
                )

                if quark_phase == 'cfl':
                    sol = solve_Qstar_cfl_coulomb(
                        H, params, Delta0, sigma,
                        include_photons, include_gluons,
                        include_thermal_neutrinos, current_guess)
                else:
                    sol = solve_Qstar_coulomb(
                        H, params, sigma,
                        include_photons, include_gluons,
                        include_thermal_neutrinos, current_guess)

                current_guess = _store_coulomb_result(data, idx, sol, current_guess)
                if sol is not None:
                    row_converged += 1
                    if i_nB == 0:
                        row_start_guess = current_guess

            if verbose:
                step = i_YL * len(T_arr) + i_T + 1
                n_steps = len(Y_L_arr) * len(T_arr)
                print(f"  [{step}/{n_steps}] Y_L_H={Y_L_H:.2f}, T={T_arr[i_T]:.1f} "
                      f"-> {row_converged}/{len(n_B_arr)} converged")

    hadronic_grids = {'n_B_H': n_B_arr, 'Y_L_H': Y_L_arr, 'T': T_arr}
    result = QstarCoulombTableData(
        eq_type='trapped', hadronic_grids=hadronic_grids, data=data)

    if save_table:
        if output_file is None:
            output_file = f"Qstar_coulomb_{result.eq_type}.dat"
        _export_table(result, params, sigma, output_file)

    return result


# =============================================================================
# Helpers
# =============================================================================

@dataclass
class QstarCoulombTableData:
    """Q* saddlepoint + Coulomb nucleation table.

    Maps hadronic conditions to Q* thermodynamics AND critical radius R_c.
    """
    eq_type: str
    hadronic_grids: dict
    data: dict
    filepath: str = ""

    def __repr__(self):
        shapes = [f"{k}={len(v)}" for k, v in self.hadronic_grids.items()]
        return (f"QstarCoulombTableData(eq_type='{self.eq_type}', "
                f"grids=({', '.join(shapes)}))")


def _init_coulomb_data(shape):
    """Initialize output arrays including R_c and delta_n_C."""
    return {
        'n_B': np.full(shape, np.nan),
        'mu_B': np.full(shape, np.nan),
        'mu_C': np.full(shape, np.nan),
        'mu_S': np.full(shape, np.nan),
        'mu_u': np.full(shape, np.nan),
        'mu_d': np.full(shape, np.nan),
        'mu_s': np.full(shape, np.nan),
        'mu_e': np.full(shape, np.nan),
        'Y_C': np.full(shape, np.nan),
        'Y_S': np.full(shape, np.nan),
        'Y_u': np.full(shape, np.nan),
        'Y_d': np.full(shape, np.nan),
        'Y_s': np.full(shape, np.nan),
        'P_total': np.full(shape, np.nan),
        'e_total': np.full(shape, np.nan),
        's_total': np.full(shape, np.nan),
        'f_total': np.full(shape, np.nan),
        'R_c': np.full(shape, np.nan),
        'delta_n_C': np.full(shape, np.nan),
        'converged': np.zeros(shape, dtype=bool),
    }


def _store_coulomb_result(data, idx, sol, current_guess):
    """Store (result, R_c) tuple from Coulomb solver."""
    if sol is not None:
        Qs, R_c = sol
        data['n_B'][idx] = Qs.n_B
        data['mu_B'][idx] = Qs.mu_B
        data['mu_C'][idx] = Qs.mu_C
        data['mu_S'][idx] = Qs.mu_S
        data['mu_u'][idx] = Qs.mu_u
        data['mu_d'][idx] = Qs.mu_d
        data['mu_s'][idx] = Qs.mu_s
        data['mu_e'][idx] = Qs.mu_e
        data['Y_C'][idx] = Qs.Y_C
        data['Y_S'][idx] = Qs.Y_S
        data['Y_u'][idx] = Qs.Y_u
        data['Y_d'][idx] = Qs.Y_d
        data['Y_s'][idx] = Qs.Y_s
        data['P_total'][idx] = Qs.P_total
        data['e_total'][idx] = Qs.e_total
        data['s_total'][idx] = Qs.s_total
        data['f_total'][idx] = Qs.f_total
        data['R_c'][idx] = R_c
        data['delta_n_C'][idx] = Qs.Y_C * Qs.n_B - Qs.n_e
        data['converged'][idx] = True
        current_guess = np.array([Qs.mu_u, Qs.mu_d, Qs.mu_s, Qs.mu_e, R_c])
    return current_guess


# =============================================================================
# Export / Load
# =============================================================================
QSTAR_COULOMB_GRID_AXES = {
    'betaeq':  ['n_B_H', 'T'],
    'trapped': ['n_B_H', 'Y_L_H', 'T'],
}

_GRID_KEY_MAP = {'n_B_H': 'n_B_H', 'Y_L_H': 'Y_L_H', 'T': 'T'}


def _export_table(result, params, sigma, output_file):
    """Export Q* Coulomb results to a text file with input grid + output columns."""
    axes = QSTAR_COULOMB_GRID_AXES[result.eq_type]
    grid_arrays = [result.hadronic_grids[ax] for ax in axes]
    mesh = np.meshgrid(*grid_arrays, indexing='ij')
    input_cols = [m.ravel(order='F') for m in mesh]

    data_keys = ['n_B', 'mu_B', 'mu_C', 'mu_S', 'mu_u', 'mu_d', 'mu_s', 'mu_e',
                 'Y_C', 'Y_S', 'Y_u', 'Y_d', 'Y_s',
                 'P_total', 'e_total', 's_total', 'f_total',
                 'R_c', 'delta_n_C', 'converged']
    output_names = [k + '_Qs' for k in data_keys]
    output_cols = [result.data[k].ravel(order='F') for k in data_keys]

    all_names = list(axes) + output_names
    all_cols = np.column_stack(input_cols + output_cols)

    col_header = "  ".join(f"{name:>14s}" for name in all_names)
    meta = (
        f"# Q* saddlepoint + Coulomb nucleation table\n"
        f"# hadronic_eq_type: {result.eq_type}\n"
        f"# sigma: {sigma} MeV/fm^2\n"
        f"# quark_eos: {params.name}\n"
        f"# m_u={params.m_u} MeV  m_d={params.m_d} MeV  m_s={params.m_s} MeV"
        f"  alpha={params.alpha}  B4={params.B4} MeV\n"
        f"# {col_header}\n"
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(meta)
    with open(output_file, 'ab') as f:
        np.savetxt(f, all_cols, fmt='%16.8e')
    print(f"  Saved Q* Coulomb table ({all_cols.shape[0]} rows) -> {output_file}")


def load_Qstar_table(filepath):
    """
    Load a Q* Coulomb table from a .dat file saved by _export_table.

    Parameters
    ----------
    filepath : str
        Path to the .dat file.

    Returns
    -------
    QstarCoulombTableData
    """
    eq_type = None
    col_names = None
    with open(filepath) as f:
        for line in f:
            if not line.startswith('#'):
                break
            if 'hadronic_eq_type:' in line:
                eq_type = line.split('hadronic_eq_type:')[1].strip()
            col_names = line.lstrip('#').split()

    if eq_type is None or col_names is None:
        raise ValueError(f"Cannot parse header in {filepath}")

    axes = QSTAR_COULOMB_GRID_AXES[eq_type]
    n_axes = len(axes)

    raw = np.loadtxt(filepath)

    hadronic_grids = {}
    shape = []
    for i, ax in enumerate(axes):
        vals = np.unique(raw[:, i])
        hadronic_grids[_GRID_KEY_MAP[ax]] = vals
        shape.append(len(vals))
    shape = tuple(shape)

    data_col_names = col_names[n_axes:]
    data = {}
    for j, cname in enumerate(data_col_names):
        key = cname.replace('_Qs', '')
        col_flat = raw[:, n_axes + j]
        arr = col_flat.reshape(shape, order='F')
        if key == 'converged':
            arr = arr.astype(bool)
        data[key] = arr

    return QstarCoulombTableData(
        eq_type=eq_type,
        hadronic_grids=hadronic_grids,
        data=data,
        filepath=filepath,
    )
