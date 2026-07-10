"""
Q* droplet composition solvers
===============================

Given a hadronic point H and a set of physical prescriptions, solve for the
thermodynamic state Q* of the critical quark droplet: the quark chemical
potentials (mu_u, mu_d, mu_s, mu_e) -- and, for the self-consistent Coulomb
mode, the critical radius R -- that make the droplet a stationary point of the
work of formation W.

Prescriptions
-------------
flavor_mode
    'frozen'      : droplet inherits the hadronic flavor fractions
                    (Y_C^Qs = Y_C^H, Y_S^Qs = Y_S^H). Only lcn / gcn, unpaired.
    'saddlepoint' : composition minimizes W -> strong chemical equilibrium
                    across the interface (mu_C+mu_e, mu_S, mu_B matched).
electric_charge_mode
    'lcn'              : local charge neutrality inside the droplet (no Coulomb).
    'gcn'              : global neutrality, mu_e^Qs = mu_e^H (no Coulomb).
    'gcn_coulomb'      : GCN composition; Coulomb added a posteriori downstream.
    'screening'        : GCN composition; Debye-screened Coulomb downstream. The
                         composition is IDENTICAL to gcn because the electrostatic
                         potential e*phi shifts mu_C and mu_e equally at the
                         interface and cancels in every equilibrium condition.
    'coulomb_minimize' : Coulomb inside the minimization; composition and R are
                         co-solved self-consistently (R feels the composition
                         through delta_mu_e(R), and vice versa).
quark_phase
    'unpaired' | 'cfl'   (CFL: Y_C^Qs = 0, Y_S^Qs = 1).

Public API
----------
    get_solver_Qs(flavor_mode, electric_charge_mode, params, ...) -> solver(H)
    solve_frozen / solve_saddlepoint / solve_saddlepoint_cfl
    solve_coulomb_minimize[_cfl] (5-eq, co-solves R)
    solve_coulomb_minimize[_cfl]_at_R (4-eq, fixed R -> for building W(R))

Internal helpers: hadronic_guess, robust_root.
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
from nucleation.barrier import (
    coulomb_delta_mu_e, coulomb_delta_P, driving_force,
)


# Cutoff (fm) on the GCN critical radius above which the full coulomb_minimize
# solve is treated as hopeless: E_Coul ~ R^5 delta_n_C^2, so for large R_c_gcn
# there is usually no dW/dR=0 root and the expensive fallback chain almost always
# fails. np.inf disables the heuristic (no points skipped).
R_GCN_SKIP_DEFAULT = np.inf


# =============================================================================
# Shared utilities  (internal)
# =============================================================================
def hadronic_guess(H):
    """Physics-based initial guess [mu_u, mu_d, mu_s, mu_e] from the H state.

    Inverts the strong-charge -> quark-flavor map at the hadronic chemical
    potentials, giving a starting point already close to chemical equilibrium.
    """
    mu_d_h = (H.mu_B - H.mu_C) / 3.0
    mu_u_h = (H.mu_B + 2.0 * H.mu_C) / 3.0
    mu_s_h = mu_d_h + H.mu_S
    return np.array([mu_u_h, mu_d_h, mu_s_h, H.mu_e])


def robust_root(equations, guess, h_guess, atol=1e-8):
    """Solve a nonlinear system with a 4-step fallback strategy.

    1) hybr from `guess`; 2) lm from `guess`; 3) hybr from the hadronic guess;
    4) lm from the hadronic guess. Returns the first OptimizeResult whose
    residual is below `atol`, else None.
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
# Frozen solver  (public)
# =============================================================================
def solve_frozen(H, params, charge_neutrality,
                 include_photons, include_gluons,
                 include_thermal_neutrinos, initial_guess=None):
    """Q* with frozen flavor fractions (droplet inherits the hadronic composition).

    4 unknowns [mu_u, mu_d, mu_s, mu_e], 4 equations:
      1. Y_C^Qs = Y_C^H          2. Y_S^Qs = Y_S^H
      3. charge neutrality (local or global)
      4. dW/dn_B = 0  (reduces to the Gibbs-per-baryon match given 1-3)
    Returns an AlphaBagEOSResult or None.
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
                  + Y_e_Qs * (mu_e - H.mu_e))

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
        include_photons=include_photons, include_gluons=include_gluons,
        include_thermal_neutrinos=include_thermal_neutrinos, mu_nu=H.mu_nu)


# =============================================================================
# Saddlepoint solver -- unpaired  (public)
# =============================================================================
def solve_saddlepoint(H, params, charge_neutrality,
                      include_photons, include_gluons,
                      include_thermal_neutrinos, initial_guess=None):
    """Q* by minimizing W wrt flavor composition (strong chemical equilibrium).

    4 unknowns [mu_u, mu_d, mu_s, mu_e], 4 equations:
      1. mu_C^Qs + mu_e = mu_C^H + mu_e^H   (dW/dY_C = 0)
      2. mu_S^Qs = mu_S^H                    (dW/dY_S = 0)
      3. charge neutrality (local or global)
      4. dW/dn_B = 0  (reduces to mu_B^Qs = mu_B^H given 1-3)
    Returns an AlphaBagEOSResult or None.
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
                  + Y_e_Qs * (mu_e - H.mu_e))

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
        include_photons=include_photons, include_gluons=include_gluons,
        include_thermal_neutrinos=include_thermal_neutrinos, mu_nu=H.mu_nu)


# =============================================================================
# Saddlepoint solver -- CFL  (public)
# =============================================================================
def solve_saddlepoint_cfl(H, params, Delta0, charge_neutrality,
                          include_photons, include_gluons,
                          include_thermal_neutrinos, initial_guess=None):
    """Q* CFL droplet by minimizing W wrt flavor composition.

    4 unknowns [mu_u, mu_d, mu_s, mu_e], 4 equations:
      1. Y_C^Qs = 0    2. Y_S^Qs = 1   (CFL flavor lock)
      3. charge neutrality (local or global)
      4. dW/dn_B = 0.  For CFL the saddle chem-sum reduces to
         mu_B^Qs + mu_S^Qs = mu_B^H + mu_S^H (Gibbs per baryon incl. strangeness).
    Returns a CFLEOSResult or None.
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
        include_photons=include_photons, include_gluons=include_gluons,
        include_thermal_neutrinos=include_thermal_neutrinos, mu_nu=H.mu_nu)


# =============================================================================
# coulomb_minimize -- unpaired, full (5-eq: composition + R co-solved)  (public)
# =============================================================================
def solve_coulomb_minimize(H, params, sigma,
                           include_photons, include_gluons,
                           include_thermal_neutrinos, initial_guess=None,
                           R_gcn_skip=R_GCN_SKIP_DEFAULT):
    """Q* AND R* with Coulomb inside the minimization (self-consistent).

    5 unknowns [mu_u, mu_d, mu_s, mu_e, R], 5 equations:
      1. mu_B^Qs = mu_B^H                         (dW/dn_B = 0)
      2. mu_C^Qs + mu_e = mu_C^H + mu_e^H         (dW/dY_C = 0)
      3. mu_S^Qs = mu_S^H                          (dW/dY_S = 0)
      4. mu_e = mu_e^H + coulomb_delta_mu_e(R)     (dW/dY_e = 0, GCN + Coulomb)
      5. P_Qs - P_H - 2 sigma/R + coulomb_delta_P(R) = 0   (dW/dR = 0)

    Returns (AlphaBagEOSResult, R_c) or None. `R_gcn_skip` (fm): above this
    GCN critical radius, make only a single quick attempt (no fallbacks).
    """
    result_gcn = solve_saddlepoint(
        H, params, charge_neutrality='global', include_photons=include_photons,
        include_gluons=include_gluons,
        include_thermal_neutrinos=include_thermal_neutrinos)
    if result_gcn is None or driving_force(result_gcn, H) >= 0:
        return None

    def equations(x):
        mu_u, mu_d, mu_s, mu_e, R = x
        if R <= 0:
            return [1e10] * 5
        Qs = compute_alphabag_total_thermo_from_mu(
            mu_u, mu_d, mu_s, mu_e, H.T, params,
            include_photons=include_photons, include_gluons=include_gluons,
            include_thermal_neutrinos=include_thermal_neutrinos, mu_nu=H.mu_nu)
        delta_n_C = (Qs.Y_C - Qs.Y_e) * Qs.n_B
        return [
            (Qs.mu_B - H.mu_B) / H.mu_B,
            (Qs.mu_C + mu_e - (H.mu_C + H.mu_e)) / H.mu_B,
            (Qs.mu_S - H.mu_S) / H.mu_B,
            (mu_e - coulomb_delta_mu_e(R, delta_n_C) - H.mu_e) / H.mu_B,
            (Qs.P_total - H.P_total - 2.0 * sigma / R
             + coulomb_delta_P(R, delta_n_C)) / H.P_total,
        ]

    R_c_gcn = 2.0 * sigma / (result_gcn.P_total - H.P_total)
    guess_gcn = np.array([result_gcn.mu_u, result_gcn.mu_d, result_gcn.mu_s,
                          result_gcn.mu_e, R_c_gcn])
    guess = initial_guess if initial_guess is not None else guess_gcn

    if R_c_gcn > R_gcn_skip:
        sol = root(equations, guess, method='hybr')
        if np.max(np.abs(sol.fun)) >= 1e-8:
            return None
    else:
        sol = robust_root(equations, guess, guess_gcn)
        if sol is None:
            result_lcn = solve_saddlepoint(
                H, params, charge_neutrality='local',
                include_photons=include_photons, include_gluons=include_gluons,
                include_thermal_neutrinos=include_thermal_neutrinos)
            if result_lcn is not None and driving_force(result_lcn, H) < 0:
                R_c_lcn = 2.0 * sigma / (result_lcn.P_total - H.P_total)
                R_mid = (R_c_gcn + R_c_lcn) / 2.0
                guess_mid = np.array([
                    (result_gcn.mu_u + result_lcn.mu_u) / 2.0,
                    (result_gcn.mu_d + result_lcn.mu_d) / 2.0,
                    (result_gcn.mu_s + result_lcn.mu_s) / 2.0,
                    (result_gcn.mu_e + result_lcn.mu_e) / 2.0, R_mid])
            else:
                R_mid = (R_c_gcn + 200.0) / 2.0
                guess_mid = np.array([result_gcn.mu_u, result_gcn.mu_d,
                                      result_gcn.mu_s, result_gcn.mu_e, R_mid])
            h_guess_mid = np.append(hadronic_guess(H), R_mid)
            sol = robust_root(equations, guess_mid, h_guess_mid)
        if sol is None:
            return None

    mu_u, mu_d, mu_s, mu_e, R_c = sol.x
    result = compute_alphabag_total_thermo_from_mu(
        mu_u, mu_d, mu_s, mu_e, H.T, params,
        include_photons=include_photons, include_gluons=include_gluons,
        include_thermal_neutrinos=include_thermal_neutrinos, mu_nu=H.mu_nu)
    return (result, R_c)


# =============================================================================
# coulomb_minimize -- unpaired, fixed R (4-eq: composition at given R)  (public)
# =============================================================================
def solve_coulomb_minimize_at_R(R, H, params, sigma,
                                include_photons, include_gluons,
                                include_thermal_neutrinos, initial_guess=None):
    """Q* at a FIXED R with Coulomb in the composition (for building W(R)).

    Same equations 1-4 as ``solve_coulomb_minimize`` but R held fixed (no
    dW/dR). Returns an AlphaBagEOSResult or None.
    """
    def equations(x):
        mu_u, mu_d, mu_s, mu_e = x
        Qs = compute_alphabag_total_thermo_from_mu(
            mu_u, mu_d, mu_s, mu_e, H.T, params,
            include_photons=include_photons, include_gluons=include_gluons,
            include_thermal_neutrinos=include_thermal_neutrinos, mu_nu=H.mu_nu)
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
        include_photons=include_photons, include_gluons=include_gluons,
        include_thermal_neutrinos=include_thermal_neutrinos, mu_nu=H.mu_nu)


# =============================================================================
# coulomb_minimize -- CFL, full (5-eq)  (public)   [CFL saddle fixed here]
# =============================================================================
def solve_coulomb_minimize_cfl(H, params, Delta0, sigma,
                               include_photons, include_gluons,
                               include_thermal_neutrinos, initial_guess=None,
                               R_gcn_skip=R_GCN_SKIP_DEFAULT):
    """CFL Q* AND R* with Coulomb inside the minimization (self-consistent).

    5 unknowns [mu_u, mu_d, mu_s, mu_e, R], 5 equations:
      1. Y_C^Qs = 0    2. Y_S^Qs = 1                       (CFL flavor lock)
      3. mu_e = mu_e^H + coulomb_delta_mu_e(R)              (dW/dY_e = 0)
      4. mu_B^Qs + mu_S^Qs = mu_B^H + mu_S^H               (dW/dn_B = 0)
      5. P_Qs - P_H - 2 sigma/R + coulomb_delta_P(R) = 0    (dW/dR = 0)

    Equation 4 is the CFL saddle condition: for Y_S = 1 the strangeness carries
    chemical work, so the Gibbs-per-baryon match INCLUDES mu_S (mu_B + mu_S),
    NOT mu_B alone. This makes R from the 5-eq solve coincide with argmax_R W(R).
    Returns (CFLEOSResult, R_c) or None.
    """
    result_gcn = solve_saddlepoint_cfl(
        H, params, Delta0, charge_neutrality='global',
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
            include_photons=include_photons, include_gluons=include_gluons,
            include_thermal_neutrinos=include_thermal_neutrinos, mu_nu=H.mu_nu)
        delta_n_C = (Qs.Y_C - Qs.Y_e) * Qs.n_B
        return [
            Qs.Y_C,
            Qs.Y_S - 1.0,
            (mu_e - coulomb_delta_mu_e(R, delta_n_C) - H.mu_e) / H.mu_B,
            ((Qs.mu_B + Qs.mu_S) - (H.mu_B + H.mu_S)) / H.mu_B,
            (Qs.P_total - H.P_total - 2.0 * sigma / R
             + coulomb_delta_P(R, delta_n_C)) / H.P_total,
        ]

    R_c_gcn = 2.0 * sigma / (result_gcn.P_total - H.P_total)
    guess_gcn = np.array([result_gcn.mu_u, result_gcn.mu_d, result_gcn.mu_s,
                          result_gcn.mu_e, R_c_gcn])
    guess = initial_guess if initial_guess is not None else guess_gcn

    if R_c_gcn > R_gcn_skip:
        sol = root(equations, guess, method='hybr')
        if np.max(np.abs(sol.fun)) >= 1e-8:
            return None
    else:
        sol = robust_root(equations, guess, guess_gcn)
        if sol is None:
            result_lcn = solve_saddlepoint_cfl(
                H, params, Delta0, charge_neutrality='local',
                include_photons=include_photons, include_gluons=include_gluons,
                include_thermal_neutrinos=include_thermal_neutrinos)
            if result_lcn is not None and result_lcn.P_total > H.P_total:
                R_c_lcn = 2.0 * sigma / (result_lcn.P_total - H.P_total)
                R_mid = (R_c_gcn + R_c_lcn) / 2.0
            else:
                R_mid = (R_c_gcn + 200.0) / 2.0
            guess_mid = np.array([result_gcn.mu_u, result_gcn.mu_d,
                                  result_gcn.mu_s, result_gcn.mu_e, R_mid])
            h_guess_mid = np.append(hadronic_guess(H), R_mid)
            sol = robust_root(equations, guess_mid, h_guess_mid)
        if sol is None:
            return None

    mu_u, mu_d, mu_s, mu_e, R_c = sol.x
    result = compute_cfl_total_thermo_from_mu(
        mu_u, mu_d, mu_s, mu_e, H.T, Delta0, params,
        include_photons=include_photons, include_gluons=include_gluons,
        include_thermal_neutrinos=include_thermal_neutrinos, mu_nu=H.mu_nu)
    return (result, R_c)


# =============================================================================
# coulomb_minimize -- CFL, fixed R (4-eq)  (public)   [CFL saddle fixed here]
# =============================================================================
def solve_coulomb_minimize_cfl_at_R(R, H, params, Delta0, sigma,
                                    include_photons, include_gluons,
                                    include_thermal_neutrinos,
                                    initial_guess=None):
    """CFL Q* at a FIXED R with Coulomb in the composition (for building W(R)).

    Equations 1-4 of ``solve_coulomb_minimize_cfl`` with R held fixed; eq 4 is
    the corrected CFL saddle mu_B^Qs + mu_S^Qs = mu_B^H + mu_S^H.
    Returns a CFLEOSResult or None.
    """
    def equations(x):
        mu_u, mu_d, mu_s, mu_e = x
        Qs = compute_cfl_total_thermo_from_mu(
            mu_u, mu_d, mu_s, mu_e, H.T, Delta0, params,
            include_photons=include_photons, include_gluons=include_gluons,
            include_thermal_neutrinos=include_thermal_neutrinos, mu_nu=H.mu_nu)
        delta_n_C = (Qs.Y_C - Qs.Y_e) * Qs.n_B
        return [
            Qs.Y_C,
            Qs.Y_S - 1.0,
            (mu_e - coulomb_delta_mu_e(R, delta_n_C) - H.mu_e) / H.mu_B,
            ((Qs.mu_B + Qs.mu_S) - (H.mu_B + H.mu_S)) / H.mu_B,
        ]

    h_guess = hadronic_guess(H)
    guess = initial_guess if initial_guess is not None else h_guess
    sol = robust_root(equations, guess, h_guess)
    if sol is None:
        return None
    mu_u, mu_d, mu_s, mu_e = sol.x
    return compute_cfl_total_thermo_from_mu(
        mu_u, mu_d, mu_s, mu_e, H.T, Delta0, params,
        include_photons=include_photons, include_gluons=include_gluons,
        include_thermal_neutrinos=include_thermal_neutrinos, mu_nu=H.mu_nu)


# =============================================================================
# Solver dispatch  (public)
# =============================================================================
VALID_FLAVOR = ('frozen', 'saddlepoint')
VALID_CHARGE = ('lcn', 'gcn', 'gcn_coulomb', 'coulomb_minimize', 'screening')


def get_solver_Qs(flavor_mode, electric_charge_mode, params,
                  quark_phase='unpaired', Delta0=None, sigma=None,
                  include_photons=True, include_gluons=True,
                  include_thermal_neutrinos=True,
                  R_gcn_skip=R_GCN_SKIP_DEFAULT):
    """Return a solver callable ``solver(H, initial_guess=None) -> Q* (or tuple)``.

    Maps (flavor_mode, electric_charge_mode, quark_phase) to the right solver,
    binding params/Delta0/sigma. For 'coulomb_minimize' the callable returns the
    (Q*, R_c) tuple; otherwise it returns the Q* EOS result. 'screening' uses the
    plain GCN composition (Coulomb enters only downstream in the barrier), so its
    solver is the gcn saddlepoint solver.
    """
    if flavor_mode not in VALID_FLAVOR:
        raise ValueError(
            f"Invalid flavor_mode: '{flavor_mode}'. Valid: {list(VALID_FLAVOR)}")
    if electric_charge_mode not in VALID_CHARGE:
        raise ValueError(
            f"Invalid electric_charge_mode: '{electric_charge_mode}'. "
            f"Valid: {list(VALID_CHARGE)}")
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

    # screening and gcn_coulomb use the plain GCN composition; lcn uses local.
    charge_neutrality = 'local' if electric_charge_mode == 'lcn' else 'global'
    common_kw = dict(
        include_photons=include_photons, include_gluons=include_gluons,
        include_thermal_neutrinos=include_thermal_neutrinos)

    if flavor_mode == 'frozen':
        def solver(H, initial_guess=None):
            return solve_frozen(H, params, charge_neutrality, **common_kw,
                                initial_guess=initial_guess)
        return solver

    if electric_charge_mode in ('lcn', 'gcn', 'gcn_coulomb', 'screening'):
        if quark_phase == 'cfl':
            def solver(H, initial_guess=None):
                return solve_saddlepoint_cfl(H, params, Delta0,
                                             charge_neutrality, **common_kw,
                                             initial_guess=initial_guess)
        else:
            def solver(H, initial_guess=None):
                return solve_saddlepoint(H, params, charge_neutrality,
                                         **common_kw,
                                         initial_guess=initial_guess)
        return solver

    # --- coulomb_minimize (co-solves R) ---
    if quark_phase == 'cfl':
        def solver(H, initial_guess=None):
            return solve_coulomb_minimize_cfl(
                H, params, Delta0, sigma, **common_kw,
                initial_guess=initial_guess, R_gcn_skip=R_gcn_skip)
    else:
        def solver(H, initial_guess=None):
            return solve_coulomb_minimize(
                H, params, sigma, **common_kw,
                initial_guess=initial_guess, R_gcn_skip=R_gcn_skip)
    return solver
