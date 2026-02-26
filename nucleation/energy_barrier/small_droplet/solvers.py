"""
Q* Solvers
==========

Equation systems for finding the quark droplet Q* thermodynamics.

Three solver families:

    Frozen           : Flavor fractions frozen from hadronic phase
    Saddlepoint      : Minimization of W w.r.t. flavor composition (+ CFL variant)
    Minimize_Coulomb : As Saddlepoint, but with Coulomb energy in W (+ CFL variant)

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
from nucleation.energy_barrier.small_droplet.barrier import (
    coulomb_delta_mu_e, coulomb_delta_P, driving_force,
)


# =============================================================================
# Shared utilities
# =============================================================================
def hadronic_guess(H):
    """Initial guess [mu_u, mu_d, mu_s, mu_e] from hadronic state."""
    mu_d_h = (H.mu_B - H.mu_C) / 3.0
    mu_u_h = (H.mu_B + 2.0 * H.mu_C) / 3.0
    mu_s_h = mu_d_h + H.mu_S
    return np.array([mu_u_h, mu_d_h, mu_s_h, H.mu_e])


def robust_root(equations, guess, h_guess, atol=1e-8):
    """Solve nonlinear system with fallback strategy.

    Strategy:
    1. hybr with given guess
    2. lm with given guess
    3. hybr with hadronic guess (if different from given guess)
    4. lm with hadronic guess

    Returns scipy OptimizeResult or None.
    """
    sol = root(equations, guess, method='hybr')
    if np.max(np.abs(sol.fun)) < atol:
        return sol

    sol = root(equations, guess, method='lm')
    if np.max(np.abs(sol.fun)) < atol:
        return sol

    if not np.allclose(guess[:len(h_guess)], h_guess):
        sol = root(equations, h_guess, method='hybr')
        if np.max(np.abs(sol.fun)) < atol:
            return sol
        sol = root(equations, h_guess, method='lm')
        if np.max(np.abs(sol.fun)) < atol:
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
      1. Y_C_Qs = Y_C_H                         (frozen charge)
      2. Y_S_Qs = Y_S_H                         (frozen strangeness)
      3. charge neutrality                       (local or global)
      4. dW/dn_B = 0                             (saddlepoint)

    Returns AlphaBagEOSResult or None.
    """
    def equations(x):
        mu_u, mu_d, mu_s, mu_e = x
        Qs = compute_alphabag_thermo_from_mu(mu_u, mu_d, mu_s, H.T, params)
        e_Qs = electron_thermo(mu_e, H.T)

        if charge_neutrality == "local":
            charge_eq = Qs.n_C - e_Qs.n
            Y_e_Qs = e_Qs.n / Qs.n_B
        elif charge_neutrality == "global":
            charge_eq = mu_e - H.mu_e
            Y_e_Qs = H.Y_e
        else:
            raise ValueError("charge_neutrality must be 'local' or 'global'")

        saddle = ((Qs.mu_B - H.mu_B)
                  + Qs.Y_C * (Qs.mu_C - H.mu_C)
                  + Qs.Y_S * (Qs.mu_S - H.mu_S)
                  + Y_e_Qs * (mu_e - H.mu_e)) # note: given other eqs. it reduces to Qs.gibbs - H.gibbs (Qs.mu_B - H.mu_B if H in beta eq.)

        return [Qs.Y_C - H.Y_C,
                Qs.Y_S - H.Y_S,
                charge_eq / (Qs.n_B if charge_neutrality == "local" else H.mu_B),
                saddle / H.mu_B]

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
      1. mu_C_Qs + mu_e = mu_C_H + mu_e_H       (dW/dY_C = 0)
      2. mu_S_Qs = mu_S_H                        (dW/dY_S = 0)
      3. charge neutrality                        (local or global)
      4. dW/dn_B = 0                              (saddlepoint)

    Returns AlphaBagEOSResult or None.
    """
    def equations(x):
        mu_u, mu_d, mu_s, mu_e = x
        Qs = compute_alphabag_thermo_from_mu(mu_u, mu_d, mu_s, H.T, params)
        e_Qs = electron_thermo(mu_e, H.T)
        Y_e_Qs = e_Qs.n / Qs.n_B

        if charge_neutrality == "local":
            charge_eq = Qs.Y_C - Y_e_Qs
        elif charge_neutrality == "global":
            charge_eq = mu_e - H.mu_e
        else:
            raise ValueError("charge_neutrality must be 'local' or 'global'")

        saddle = ((Qs.mu_B - H.mu_B)
                  + Qs.Y_C * (Qs.mu_C - H.mu_C)
                  + Qs.Y_S * (Qs.mu_S - H.mu_S)
                  + Y_e_Qs * (mu_e - H.mu_e)) # note: given other eqs. it reduces to Qs.mu_B - H.mu_B

        return [
            (Qs.mu_C + mu_e - (H.mu_C + H.mu_e)) / H.mu_B,
            (Qs.mu_S - H.mu_S) / H.mu_B,
            charge_eq / (Qs.n_B if charge_neutrality == "local" else H.mu_B),
            saddle / H.mu_B,
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
    """Solve for CFL Q* by minimizing W w.r.t. flavor composition.

    4 unknowns: [mu_u, mu_d, mu_s, mu_e]
    4 equations:
      1. Y_C_Qs = 0                              (CFL constraint)
      2. Y_S_Qs = 1                              (CFL constraint)
      3. charge neutrality                        (local or global)
      4. dW/dn_B = 0                              (saddlepoint)

    Returns CFLEOSResult or None.
    """
    def equations(x):
        mu_u, mu_d, mu_s, mu_e = x
        Qs = compute_cfl_thermo_from_mu(mu_u, mu_d, mu_s, H.T, Delta0, params)
        e_Qs = electron_thermo(mu_e, H.T)
        Y_e_Qs = e_Qs.n / Qs.n_B

        if charge_neutrality == "local":
            charge_eq = Qs.Y_C - Y_e_Qs
        elif charge_neutrality == "global":
            charge_eq = mu_e - H.mu_e
        else:
            raise ValueError("charge_neutrality must be 'local' or 'global'")

        saddle = ((Qs.mu_B - H.mu_B)
                  + Qs.Y_C * (Qs.mu_C - H.mu_C)
                  + Qs.Y_S * (Qs.mu_S - H.mu_S)
                  + Y_e_Qs * (mu_e - H.mu_e))

        return [Qs.Y_C,
                Qs.Y_S - 1.0,
                charge_eq / (Qs.n_B if charge_neutrality == "local" else H.mu_B),
                saddle / H.mu_B]

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
def solve_saddlepoint_minimizecoulomb(H, params, sigma,
                  include_photons, include_gluons,
                  include_thermal_neutrinos, initial_guess=None):
    """Solve for Q* and R* with Coulomb corrections.

    5 unknowns: [mu_u, mu_d, mu_s, mu_e, R]
    5 equations:
      1. mu_B_Qs = mu_B_H                        (dW/dn_B = 0)
      2. mu_C_Qs + mu_e = mu_C_H + mu_e_H        (dW/dY_C = 0)
      3. mu_S_Qs = mu_S_H                         (dW/dY_S = 0)
      4. mu_e = mu_e_H + coulomb_delta_mu_e        (Coulomb GCN)
      5. P_Qs - P_H - 2sigma/R + dP_Coulomb = 0   (dW/dR = 0)

    Returns (AlphaBagEOSResult, R_c) or None.
    """

    # If GCN saddle point has not R_c (H stable), then there is no point in compute solve_saddlepoint_minimizecoulomb
    result_gcn = solve_saddlepoint(H, params, charge_neutrality='global', include_photons=include_photons, include_gluons=include_gluons,
                     include_thermal_neutrinos=include_thermal_neutrinos)

    if result_gcn is None or driving_force(result_gcn, H) >= 0:
        return None

    # if R_c_gcn exist, then proceed with the computation of solve_saddlepoint_minimizecoulombs
    
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
            (Qs.mu_B - H.mu_B) / H.mu_B,
            (Qs.mu_C + mu_e - (H.mu_C + H.mu_e)) / H.mu_B,
            (Qs.mu_S - H.mu_S) / H.mu_B,
            (mu_e - coulomb_delta_mu_e(R, delta_n_C) - H.mu_e) / H.mu_B,
            (Qs.P_total - H.P_total - 2.0 * sigma / R + coulomb_delta_P(R, delta_n_C)) / H.P_total,
        ]


    # compute R_c_gcn as guess for R_c_minimizecoulomb
    R_c_gcn = 2.0 * sigma / (result_gcn.P_total - H.P_total)
    mu_d_gcn = result_gcn.mu_d
    mu_u_gcn = result_gcn.mu_u
    mu_s_gcn = result_gcn.mu_s
    guess_gcn = np.array([mu_u_gcn, mu_d_gcn, mu_s_gcn, result_gcn.mu_e, R_c_gcn])
    h_guess = guess_gcn

    # First attempt
    guess = initial_guess if initial_guess is not None else guess_gcn
    sol = robust_root(equations, guess, h_guess)

    # If failed, refine R guess using LCN
    if sol is None:
        result_lcn = solve_saddlepoint(H, params, charge_neutrality='local', include_photons=include_photons, include_gluons=include_gluons,
                     include_thermal_neutrinos=include_thermal_neutrinos)
        if result_lcn is not None and result_lcn.P_total > H.P_total:
            R_c_lcn = 2.0 * sigma / (result_lcn.P_total - H.P_total)
            R_mid = (R_c_gcn + R_c_lcn) / 2.0
        else:
            R_mid = (R_c_gcn + 200.0) / 2.0

        guess_mid = np.array([mu_u_gcn, mu_d_gcn, mu_s_gcn, result_gcn.mu_e, R_mid])
        h_guess_mid = np.append(hadronic_guess(H), R_mid)
        sol = robust_root(equations, guess_mid, h_guess_mid)

    if sol is None:
        return None

    mu_u, mu_d, mu_s, mu_e, R_c = sol.x

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
def solve_saddlepoint_minimizecoulomb_at_R(R, H, params, sigma,
                       include_photons, include_gluons,
                       include_thermal_neutrinos, initial_guess=None):
    """Solve for Q* at fixed R with Coulomb corrections.

    4 unknowns: [mu_u, mu_d, mu_s, mu_e]
    4 equations:
      1. mu_B_Qs = mu_B_H                        (dW/dn_B = 0)
      2. mu_C_Qs + mu_e = mu_C_H + mu_e_H        (dW/dY_C = 0)
      3. mu_S_Qs = mu_S_H                         (dW/dY_S = 0)
      4. mu_e = mu_e_H + coulomb_delta_mu_e        (Coulomb GCN)

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
            (Qs.mu_B - H.mu_B) / H.mu_B,
            (Qs.mu_C + mu_e - (H.mu_C + H.mu_e)) / H.mu_B,
            (Qs.mu_S - H.mu_S) / H.mu_B,
            (mu_e - coulomb_delta_mu_e(R, delta_n_C) - H.mu_e) / H.mu_B,
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
def solve_saddlepoint_minimizecoulomb_cfl(H, params, Delta0, sigma,
                      include_photons, include_gluons,
                      include_thermal_neutrinos, initial_guess=None):
    """Solve for CFL Q* and R* with Coulomb corrections.

    5 unknowns: [mu_u, mu_d, mu_s, mu_e, R]
    5 equations:
      1. Y_C_Qs = 0                               (CFL constraint)
      2. Y_S_Qs = 1                                (CFL constraint)
      3. mu_e = mu_e_H + coulomb_delta_mu_e        (Coulomb GCN)
      4. mu_B_Qs = mu_B_H                          (dW/dn_B = 0)
      5. P_Qs - P_H - 2sigma/R + dP_Coulomb = 0   (dW/dR = 0)

    Returns (CFLEOSResult, R_c) or None.
    """

    # If GCN saddle point has not R_c (H stable), then there is no point in compute solve_saddlepoint_minimizecoulomb
    result_gcn = solve_saddlepoint_cfl(H, params, Delta0, charge_neutrality='global',
                     include_photons=include_photons, include_gluons=include_gluons,
                     include_thermal_neutrinos=include_thermal_neutrinos)

    if result_gcn is None or driving_force(result_gcn, H) >= 0:
        return None

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
            (mu_e - coulomb_delta_mu_e(R, delta_n_C) - H.mu_e) / H.mu_B,
            (Qs.mu_B - H.mu_B) / H.mu_B,
            (Qs.P_total - H.P_total - 2.0 * sigma / R + coulomb_delta_P(R, delta_n_C)) / H.P_total,
        ]

    # compute R_c_gcn as guess for R_c_minimizecoulomb
    R_c_gcn = 2.0 * sigma / (result_gcn.P_total - H.P_total)
    guess_gcn = np.array([result_gcn.mu_u, result_gcn.mu_d, result_gcn.mu_s,
                          result_gcn.mu_e, R_c_gcn])
    h_guess = guess_gcn

    # First attempt
    guess = initial_guess if initial_guess is not None else guess_gcn
    sol = robust_root(equations, guess, h_guess)

    # If failed, refine R guess using LCN
    if sol is None:
        result_lcn = solve_saddlepoint_cfl(H, params, Delta0, charge_neutrality='local',
                         include_photons=include_photons, include_gluons=include_gluons,
                         include_thermal_neutrinos=include_thermal_neutrinos)
        if result_lcn is not None and result_lcn.P_total > H.P_total:
            R_c_lcn = 2.0 * sigma / (result_lcn.P_total - H.P_total)
            R_mid = (R_c_gcn + R_c_lcn) / 2.0
        else:
            R_mid = (R_c_gcn + 200.0) / 2.0

        guess_mid = np.array([result_gcn.mu_u, result_gcn.mu_d, result_gcn.mu_s,
                              result_gcn.mu_e, R_mid])
        h_guess_mid = np.append(hadronic_guess(H), R_mid)
        sol = robust_root(equations, guess_mid, h_guess_mid)

    if sol is None:
        return None

    mu_u, mu_d, mu_s, mu_e, R_c = sol.x

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
def solve_saddlepoint_minimizecoulomb_cfl_at_R(R, H, params, Delta0, sigma,
                           include_photons, include_gluons,
                           include_thermal_neutrinos, initial_guess=None):
    """Solve for CFL Q* at fixed R with Coulomb corrections.

    4 unknowns: [mu_u, mu_d, mu_s, mu_e]
    4 equations:
      1. Y_C_Qs = 0                               (CFL constraint)
      2. Y_S_Qs = 1                                (CFL constraint)
      3. mu_e = mu_e_H + coulomb_delta_mu_e        (Coulomb GCN)
      4. mu_B_Qs = mu_B_H                          (dW/dn_B = 0)

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
            (mu_e - coulomb_delta_mu_e(R, delta_n_C) - H.mu_e) / H.mu_B,
            (Qs.mu_B - H.mu_B) / H.mu_B,
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
# Solver dispatch
# =============================================================================
def get_solver_Qs(flavor_mode, electric_charge_mode, params,
                    quark_phase='unpaired', Delta0=None, sigma=None,
                    include_photons=True, include_gluons=True,
                    include_thermal_neutrinos=True):
    """Return a solver callable with signature solver(H, initial_guess=None).

    Maps (flavor_mode, electric_charge_mode, quark_phase) to the appropriate
    solver function, binding all required parameters.

    Parameters
    ----------
    flavor_mode : str
        'frozen' or 'saddlepoint'.
    electric_charge_mode : str
        'lcn', 'gcn', 'gcn_coulomb', or 'coulomb_minimize'.
    params : AlphaBagParams
        Quark EOS parameters.
    quark_phase : str
        'unpaired' or 'cfl'. Default 'unpaired'.
    Delta0 : float or None
        CFL pairing gap (MeV). Required if quark_phase='cfl'.
    sigma : float or None
        Surface tension (MeV/fm^2). Required for 'coulomb_minimize'.
    include_photons, include_gluons, include_thermal_neutrinos : bool
        Default True.

    Returns
    -------
    solver : callable
    """
    valid_flavor = ('frozen', 'saddlepoint')
    valid_charge = ('lcn', 'gcn', 'gcn_coulomb', 'coulomb_minimize')
    if flavor_mode not in valid_flavor:
        raise ValueError(
            f"Invalid flavor_mode: '{flavor_mode}'. Valid: {list(valid_flavor)}")
    if electric_charge_mode not in valid_charge:
        raise ValueError(
            f"Invalid electric_charge_mode: '{electric_charge_mode}'. "
            f"Valid: {list(valid_charge)}")
    if flavor_mode == 'frozen':
        if electric_charge_mode not in ('lcn', 'gcn'):
            raise ValueError(
                f"frozen flavor_mode only supports 'lcn' or 'gcn', "
                f"got '{electric_charge_mode}'")
        if quark_phase == 'cfl':
            raise ValueError(
                "frozen flavor_mode does not support quark_phase='cfl'")
    if quark_phase == 'cfl' and Delta0 is None:
        raise ValueError("Delta0 must be provided when quark_phase='cfl'")
    if electric_charge_mode == 'coulomb_minimize' and sigma is None:
        raise ValueError(
            "sigma must be provided for electric_charge_mode='coulomb_minimize'")

    charge_neutrality = 'local' if electric_charge_mode == 'lcn' else 'global'
    common_kw = dict(
        include_photons=include_photons,
        include_gluons=include_gluons,
        include_thermal_neutrinos=include_thermal_neutrinos,
    )

    # --- frozen ---
    if flavor_mode == 'frozen':
        def solver(H, initial_guess=None):
            return solve_frozen(H, params, charge_neutrality, **common_kw,
                                initial_guess=initial_guess)
        return solver

    # --- saddlepoint (lcn, gcn, gcn_coulomb) ---
    if electric_charge_mode in ('lcn', 'gcn', 'gcn_coulomb'):
        if quark_phase == 'cfl':
            def solver(H, initial_guess=None):
                return solve_saddlepoint_cfl(H, params, Delta0, charge_neutrality,
                                             **common_kw, initial_guess=initial_guess)
        else:
            def solver(H, initial_guess=None):
                return solve_saddlepoint(H, params, charge_neutrality, **common_kw,
                                         initial_guess=initial_guess)
        return solver

    # --- coulomb_minimize ---
    if quark_phase == 'cfl':
        def solver(H, initial_guess=None):
            return solve_saddlepoint_minimizecoulomb_cfl(H, params, Delta0, sigma,
                                                         **common_kw, initial_guess=initial_guess)
    else:
        def solver(H, initial_guess=None):
            return solve_saddlepoint_minimizecoulomb(H, params, sigma, **common_kw,
                                                     initial_guess=initial_guess)
    return solver
