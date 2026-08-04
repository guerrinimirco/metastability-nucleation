"""
sigma_crit: the critical surface tension at a point
===================================================

The single-point nucleation core. Given one hadronic state (a PNS centre, or a
shell inside the star), find the surface tension sigma_crit at which the
nucleation time equals the target: below it a quark droplet forms and the star
converts, above it the hadronic star survives.

sigma_crit is the paper's observable because sigma is the least constrained
input in the problem: rather than assume a value and report yes/no, we report
the threshold value and let the reader place their own sigma against it.

Star-wide vs centre-only: nucleation can be fastest in a SHELL rather than at
the centre (density falls outward but so does temperature). `star_shell_states`
samples several shells and `sigma_target_pt` takes the most permissive; passing
shells=None reduces to the centre-only answer, which underestimates sigma_crit
by up to ~2x near the SQM corner.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from eos.alphabag.parameters import get_alphabag_custom
from eos.alphabag.thermodynamics_quarks import T_critical   # CFL gap melting temp (single source)
from eos.general.physics_constants import hc
from nucleation.critical import critical_droplet
from nucleation.rates import nucleation_rate, nucleation_time
from nucleation.analysis.config import NucConfig, StarMatch

# =============================================================================
#  Single-point nucleation
# =============================================================================
def crossover_radius(T, Delta0):
    """unpCFL coherence radius R_x(T) = hc/Delta(T), using the SAME CFL gap as the
    EoS: Delta(T)=Delta0*sqrt(1-(T/Tc)^2) with Tc=T_critical(Delta0) (=0.57*2^(1/3)
    *Delta0, incl. the CFL enhancement). inf at/above Tc (no pairing -> unpaired).
    Tc is taken from eos.T_critical so R_x can never drift from the gap in gap_cfl."""
    if Delta0 <= 0:
        return np.inf
    T_c = T_critical(Delta0)
    ratio = np.asarray(T / T_c)
    gap = np.where(ratio < 1, Delta0 * np.sqrt(np.maximum(0, 1 - ratio**2)), 0.0)
    return np.where(gap > 0, hc / gap, np.inf)


def hadronic_point(H_trapped: dict, nB, YL, T):
    """Trapped hadronic state (SimpleNamespace) at one (n_B, Y_L, T) point.

    Evaluates the trapped-neutrino interpolator dict at the point and packs
    every field the Q* solvers and ``driving_force`` need. Charge neutrality
    of the bulk hadronic phase fixes Y_e = Y_C; lepton-number conservation
    fixes Y_nu = Y_L - Y_C."""
    pt = (nB, YL, T)
    h = SimpleNamespace(
        n_B=float(nB), T=float(T),
        P_total=float(H_trapped['P'](*pt)),   e_total=float(H_trapped['eps'](*pt)),
        mu_B=float(H_trapped['mu_B'](*pt)),   mu_C=float(H_trapped['mu_C'](*pt)),
        mu_S=float(H_trapped['mu_S'](*pt)),   mu_e=float(H_trapped['mu_e'](*pt)),
        mu_nu=float(H_trapped['mu_nu'](*pt)),
        Y_C=float(H_trapped['Y_C'](*pt)),     Y_S=float(H_trapped['Y_S'](*pt)))
    h.Y_e = h.Y_C
    h.Y_nu = float(YL) - h.Y_C
    return h


def central_state(MT0, star: StarMatch):
    """(nBHc, T_c, H_pt) at the centre of the trapped star with Mb = Mb(MT0)."""
    Mb = float(star.Mb_of_MT0(MT0))
    nBHc = float(star.nBHc_of_Mb(Mb))
    T_c = float(star.H_iso_trapped['T'](nBHc, star.YLH, star.S))
    H_pt = hadronic_point(star.H_trapped, nBHc, star.YLH, T_c)
    return nBHc, T_c, H_pt


def star_shell_states(MT0, star: StarMatch, n_shells, nB_min=0.25):
    """Hadronic states [(H_pt, T)] at n_shells densities nB_min..nB_centre.

    The nucleation site is NOT always the centre: near the SQM-stability corner
    the driving force |Delta_f| peaks at ~2 n_sat and weakens toward the centre,
    so lower-density shells nucleate at higher sigma. sigma_crit must therefore
    demand tau > tau_target at EVERY shell up to the centre, not just at r=0.
    T on each shell follows the (Y_L, S) isentrope. Centre is always included
    (last element)."""
    nBHc, T_c, H_pt = central_state(MT0, star)
    shells = []
    for nB in np.linspace(nB_min, nBHc, int(n_shells)):
        T = float(star.H_iso_trapped['T'](nB, star.YLH, star.S))
        shells.append((hadronic_point(star.H_trapped, nB, star.YLH, T), T))
    return shells


def critical_droplet_pt(sigma, H_pt, T_c, flavor, charge, phase, cache,
                        params, Delta0, nuc: NucConfig):
    """Critical droplet at one hadronic point: (R_c [fm], W_c [MeV], Qs).

    Thin wrapper over ``nucleation.critical.critical_droplet`` -- the single
    barrier engine shared with the tables and the notebook W(R) plots, so the
    physics can never diverge. ``cache`` (a dict) memoizes the sigma-independent
    composition solves across the sigma scan.

    Returns
    -------
    (R_c, W_c, Qs) with the conventions
      * (nan, nan, None)  -- solver failed / no critical droplet found;
      * (nan, inf,  Qs)   -- hadronic phase STABLE (Delta_f >= 0): infinite
                             barrier, nucleation never happens;
      * finite values     -- genuine critical droplet; Qs is the droplet state
                             (for unpCFL: the phase selected at the peak).
    """
    Rx = float(crossover_radius(T_c, Delta0)) if phase == 'unpCFL' else None
    cd = critical_droplet(
        H_pt, sigma=sigma, params=params, flavor_mode=flavor,
        electric_charge_mode=charge, quark_phase=phase, Delta0=Delta0, Rx=Rx,
        cache=cache, include_photons=nuc.include_photons,
        include_gluons=nuc.include_gluons,
        include_thermal_neutrinos=nuc.include_thermal_neutrinos)
    return cd.R_c, cd.W_c, cd.Qs


def tau_pt(sigma, H_pt, T_c, flavor, charge, phase, cache, params, Delta0, nuc: NucConfig):
    """Central nucleation time tau [s] at one sigma, for one droplet phase.

    Thin wrapper over ``critical_droplet_pt``: converts (R_c, W_c, Qs) to a
    Langer rate and tau = 1/(Gamma V). Returns NaN on solver failure, +inf
    when the hadronic phase is stable (W_c = inf -> never nucleates)."""
    R_c, W_c, Qs = critical_droplet_pt(sigma, H_pt, T_c, flavor, charge, phase,
                                       cache, params, Delta0, nuc)
    if np.isposinf(W_c):
        return np.inf
    if Qs is None or not (np.isfinite(R_c) and R_c > 0
                          and np.isfinite(W_c) and W_c > 0):
        return np.nan
    Gamma = float(nucleation_rate(W_c, R_c, sigma, T_c, H_pt, Qs))
    return float(nucleation_time(Gamma, nuc.V)) if Gamma > 0 else np.inf


def sigma_target_pt(H_pt, T_c, flavor, charge, phase, params, Delta0,
                    nuc: NucConfig, shells=None):
    """sigma where tau(sigma) = nuc.tau_target -- at the star centre, or, when
    ``shells`` is given, star-wide.

    ``shells`` is a list of (H_pt, T) hadronic states along the star profile
    (see ``star_shell_states``). The scanned quantity is then
    tau_star(sigma) = min over shells of tau(sigma, shell), so the returned
    sigma_crit is the largest sigma at which ANY shell still nucleates -- i.e.
    the minimum sigma that guarantees tau > tau_target EVERYWHERE with
    n_B <= n_B_centre, per the star-wide no-nucleation definition. shells=None
    keeps the original centre-only behaviour.

    Robust to solver drop-outs. tau(sigma) can be non-monotonic and can be NaN
    over whole sigma sub-ranges where no critical droplet exists (e.g. the unpCFL
    forced-unpaired-below-Rx window). A naive brentq on [sig_lo, sig_hi] treats
    such a NaN edge as a tau=tau_target crossing and returns a spurious (tiny)
    sigma_crit. Instead we scan a coarse sigma grid and locate the crossing ONLY
    between adjacent CONVERGED (finite, positive tau) grid points, keeping the
    largest-sigma upward crossing (the ultimate high-sigma suppression edge), then
    brentq-refine on that sub-bracket.

    Returns
    -------
    float
      finite : sigma_crit (largest-sigma tau=tau_target crossing);
      +inf   : tau < tau_target for EVERY converged sigma up to sig_hi
               (nucleates for all tested sigma -> sigma_crit lies above sig_hi);
      NaN    : no converged nucleating point (never reaches tau_target from below,
               or no critical droplet anywhere in [sig_lo, sig_hi]).
    """
    lo = np.log10(nuc.tau_target)

    if shells is None:
        shells = [(H_pt, T_c)]
    caches = [{} for _ in shells]          # per-shell composition memoization
    last_hit = [len(shells) - 1]           # warm start: try last nucleating shell first

    def tau(s):
        # tau_star = min over shells; early exit once one shell nucleates (the
        # sign vs tau_target -- all the crossing logic needs -- is then fixed,
        # and near the crossing only the controlling shell is marginal anyway).
        best = np.nan
        order = [last_hit[0]] + [k for k in range(len(shells)) if k != last_hit[0]]
        for k in order:
            hp, tc = shells[k]
            t = tau_pt(s, hp, tc, flavor, charge, phase, caches[k],
                       params, Delta0, nuc)
            if np.isnan(t):
                continue                   # unknown shell: skip (NaN if all skip)
            if np.isnan(best) or t < best:
                best = t
            if t <= nuc.tau_target:
                last_hit[0] = k
                return t
        return best

    sig = np.linspace(nuc.sig_lo, nuc.sig_hi, int(nuc.n_sigma_scan))
    t = np.array([tau(s) for s in sig])
    with np.errstate(divide='ignore', invalid='ignore'):
        g = np.where(np.isfinite(t) & (t > 0), np.log10(t) - lo, np.nan)

    # sigma_crit = largest sigma that still nucleates (tau <= tau_target), then look
    # at what happens just above it. Using the LARGEST such sigma automatically steps
    # over any lower-sigma drop-out gaps where no critical droplet exists (so those
    # NaN edges are never mistaken for the threshold).
    nuc_idx = np.where(np.isfinite(g) & (g < 0))[0]
    if nuc_idx.size == 0:
        return np.nan                                # never nucleates in [sig_lo, sig_hi]
    i = int(nuc_idx.max())
    if i == len(sig) - 1:
        return np.inf                                # still nucleating at sig_hi
    gj = g[i + 1]
    if np.isfinite(gj) and gj >= 0:                  # converged upward crossing -> refine
        try:
            return float(brentq(lambda s: np.log10(tau(s)) - lo,
                                 sig[i], sig[i + 1], xtol=1e-3, rtol=1e-5))
        except (ValueError, FloatingPointError):
            return float(sig[i] + (sig[i + 1] - sig[i]) * g[i] / (g[i] - gj))
    # successor is NaN (no critical droplet) or tau=inf: nucleation stops between
    # sig[i] and sig[i+1]. But this is only a *physical* sigma_crit if tau was
    # genuinely approaching tau_target at sig[i]. In the re-hadronization boundary
    # band the only "nucleating" points are a barrierless island (W_c~=0, tau ~
    # 1e-70 s, i.e. spinodal, not thermal nucleation) that then vanishes: there is
    # no tau=tau_target threshold at all. Flag those as NaN rather than emit a fake
    # low sigma_crit painted as "easy to nucleate".
    # ponytail: -6 = tau within 1e6 of target; raise/lower if boundary cells shift.
    if g[i] < -6.0:
        return np.nan
    # Refine the drop-out edge with a fine local sub-scan.
    last = sig[i]
    for s in np.linspace(sig[i], sig[i + 1], 12)[1:]:
        ts = tau(s)
        if np.isfinite(ts) and ts > 0 and (np.log10(ts) - lo) < 0:
            last = float(s)
        else:
            break
    return float(last)

