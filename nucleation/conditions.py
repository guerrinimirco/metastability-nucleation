"""
Nucleation conditions
=====================

**Where does a quark droplet actually form?** Everything else in the package
computes how fast nucleation happens at a given state; this module inverts that
question and returns the CONDITION -- the temperature, or the density, at which
the nucleation time reaches a chosen target tau_target.

Two ways in, depending on what you have:

  a single point   ``nucleation_point(H, n_B, T, sigma, params=...)``
                   -> everything about that state at once: the hadronic EoS, the
                      droplet composition, R*, W*, W*/T, Gamma and tau.

  the locus        ``NucleationCondition`` / ``T_nuc`` / ``nB_nuc``
                   -> T(n_B, Y_L; tau_target) and its inverse n_B(T, Y_L),
                      as scalars, as arrays, or as a ready-to-plot curve.

A condition can be built three ways, all with the same methods afterwards:

    NucleationCondition.from_table(obs, tau_target)          # a computed grid
    NucleationCondition.from_interpolator(fn, grids, eq)     # a tau interpolator
    NucleationCondition.from_point_solver(H, T_grid, n_B_grid, ...)
                                                             # no table at all

The last one is the useful one for exploring parameter space: it answers "at
what T does this (alpha_s, B^1/4, Delta_0) nucleate?" without building a table
first. Thermal and quantum observables both work, because both expose ``.tau``.

Quick start::

    from nucleation import nucleation_point, NucleationCondition, T_nuc

    pt = nucleation_point(H, n_B_H=0.9, T=25.0, sigma=30.0, params=p, Y_L_H=0.25)
    pt.tau, pt.W_over_T, pt.nucleates(tau_target=1e-3)

    cond = NucleationCondition.from_table(obs, tau_target=1e-3)
    cond.T_of_nB(0.9, Y_L=0.25)          # the nucleation temperature
    n_B, T = cond.curve(Y_L=0.25)        # the whole locus, ready to plot

Implementation note: exactly two crossing finders are used anywhere in this
module -- ``_find_T_root`` (evaluates a callable/interpolator along T, with
secant warm-starts) and ``_find_crossing_root`` (works on precomputed table
values along either axis). Everything public delegates to one of them, so the
"first DOWNWARD crossing" rule (tau falling through tau_target) is defined once.
"""

import numpy as np
from types import SimpleNamespace
from dataclasses import dataclass, field
from scipy.optimize import root_scalar
from scipy.interpolate import RegularGridInterpolator, interp1d

from eos.alphabag.thermodynamics_quarks import T_critical
from eos.general.physics_constants import hc
from nucleation.tables import GRID_AXES
from nucleation.critical import critical_droplet
from nucleation.rates import nucleation_rate, nucleation_time

# Nucleation volume [fm^3]: a sphere of radius 100 m, the scale over which the
# central conditions of a proto-neutron star are roughly uniform. tau = 1/(V*Gamma).
V_NUCLEATION = 4.18879e51


@dataclass
class NucleationTemperatureResult:
    """Result of a nucleation-temperature computation over the grid."""
    hadronic_grids: dict
    T_nuc: np.ndarray
    tau_target: float
    converged: np.ndarray
    residual: np.ndarray
    eq_type: str


# =============================================================================
# Crossing finders  (internal)
# =============================================================================
def _find_T_root(f, T_arr, T_guesses=None):
    """First downward zero crossing of f(T) (f>0 low T, f<0 high T).

    Fast path: secant from warm-start guesses; fallback: grid sign-change scan +
    brentq. Returns (T_root, converged, residual=|f(T_root)|).
    """
    vals = np.array([f(T) for T in T_arr])
    valid = np.isfinite(vals) & (T_arr > 0)
    if np.sum(valid) < 2:
        return np.nan, False, np.nan
    T_valid = T_arr[valid]
    vals_valid = vals[valid]

    if T_guesses is not None:
        eps = (T_valid[-1] - T_valid[0]) * 1e-4

        def _try_secant(x0, x1):
            try:
                sol = root_scalar(f, x0=x0, x1=x1, method='secant')
                if sol.converged and T_valid[0] <= sol.root <= T_valid[-1]:
                    if f(sol.root - eps) > 0 and f(sol.root + eps) < 0:
                        return sol.root, True, abs(f(sol.root))
            except (ValueError, RuntimeError):
                pass
            return None

        for guess in T_guesses:
            result = _try_secant(guess, guess * 1.01)
            if result is not None:
                return result
        if len(T_guesses) >= 2:
            result = _try_secant(T_guesses[0], T_guesses[1])
            if result is not None:
                return result

    sign_changes = np.where(np.diff(np.sign(vals_valid)))[0]
    downward = [sc for sc in sign_changes
                if vals_valid[sc] > 0 and vals_valid[sc + 1] < 0]
    if len(downward) == 0:
        return np.nan, False, np.nan
    j = downward[0]
    T_lo, T_hi = float(T_valid[j]), float(T_valid[j + 1])
    try:
        sol = root_scalar(f, bracket=[T_lo, T_hi], method='brentq')
        if sol.converged:
            return sol.root, True, abs(f(sol.root))
    except (ValueError, RuntimeError):
        pass
    return np.nan, False, np.nan


def _find_crossing_root(x_valid, f_vals):
    """First downward zero-crossing of tabulated f(x) via local interp + brentq.

    Axis-agnostic: x may be n_B (fixed T) or T (fixed n_B).
    """
    sign_changes = np.where(np.diff(np.sign(f_vals)))[0]
    downward = [sc for sc in sign_changes
                if f_vals[sc] > 0 and f_vals[sc + 1] < 0]
    if len(downward) == 0:
        return np.nan, False, np.nan
    idx = downward[0]
    x_lo, x_hi = float(x_valid[idx]), float(x_valid[idx + 1])
    kind = 'cubic' if len(x_valid) >= 4 else 'linear'
    f_interp = interp1d(x_valid, f_vals, kind=kind)
    try:
        sol = root_scalar(f_interp, bracket=[x_lo, x_hi], method='brentq')
        if sol.converged:
            return sol.root, True, abs(f_interp(sol.root))
    except (ValueError, RuntimeError):
        pass
    return np.nan, False, np.nan


# =============================================================================
# Nucleation temperature from an interpolator  (public)
# =============================================================================
def compute_nucleation_temperature(log10_tau_fn, hadronic_grids, eq_type,
                                   tau_target=1.0, T_guess=None, verbose=False):
    """T where tau = tau_target over the grid, root-finding an interpolator along T.

    Works with any log10(tau) interpolator (thermal or quantum). Returns a
    ``NucleationTemperatureResult``.
    """
    n_B_arr = hadronic_grids['n_B_H']
    T_arr = hadronic_grids['T']
    n_nB = len(n_B_arr)
    log_tau_target = np.log10(tau_target)
    axes = GRID_AXES[eq_type]
    non_T_axes = [ax for ax in axes if ax not in ('n_B_H', 'T')]

    def _guess_list(prev_prev_T, prev_T, prev_prev_nB, prev_nB, nB):
        gl = []
        if (prev_prev_T is not None and prev_T is not None
                and prev_prev_nB is not None and prev_nB is not None):
            dnB = prev_nB - prev_prev_nB
            if abs(dnB) > 0:
                T_extrap = prev_T + (prev_T - prev_prev_T) * (nB - prev_nB) / dnB
                if T_extrap > 0:
                    gl.append(T_extrap)
        if prev_T is not None:
            gl.append(prev_T)
        return gl

    if len(non_T_axes) == 0:
        T_nuc = np.full(n_nB, np.nan)
        conv = np.zeros(n_nB, dtype=bool)
        resid = np.full(n_nB, np.nan)
        prev_T = T_guess
        prev_nB = prev_prev_T = prev_prev_nB = None
        for i in range(n_nB):
            nB = float(n_B_arr[i])

            def f(T, _nB=nB):
                return log10_tau_fn(_nB, T) - log_tau_target

            gl = _guess_list(prev_prev_T, prev_T, prev_prev_nB, prev_nB, nB)
            T_root, ok, res = _find_T_root(f, T_arr, gl or None)
            if ok:
                T_nuc[i] = T_root
                conv[i] = True
                resid[i] = res
                prev_prev_T, prev_prev_nB = prev_T, prev_nB
                prev_T, prev_nB = T_root, nB
        out_grids = {'n_B_H': n_B_arr}
    elif len(non_T_axes) == 1:
        ax_name = non_T_axes[0]
        ax_arr = hadronic_grids[ax_name]
        n_outer = len(ax_arr)
        T_nuc = np.full((n_nB, n_outer), np.nan)
        conv = np.zeros((n_nB, n_outer), dtype=bool)
        resid = np.full((n_nB, n_outer), np.nan)
        for k in range(n_outer):
            prev_T = T_guess
            prev_nB = prev_prev_T = prev_prev_nB = None
            outer_val = float(ax_arr[k])
            for i in range(n_nB):
                nB = float(n_B_arr[i])

                def f(T, _nB=nB, _ov=outer_val):
                    return log10_tau_fn(_nB, _ov, T) - log_tau_target

                gl = _guess_list(prev_prev_T, prev_T, prev_prev_nB, prev_nB, nB)
                T_root, ok, res = _find_T_root(f, T_arr, gl or None)
                if ok:
                    T_nuc[i, k] = T_root
                    conv[i, k] = True
                    resid[i, k] = res
                    prev_prev_T, prev_prev_nB = prev_T, prev_nB
                    prev_T, prev_nB = T_root, nB
        out_grids = {'n_B_H': n_B_arr, ax_name: ax_arr}
    else:
        raise ValueError(f"Unsupported eq_type: '{eq_type}'")

    return NucleationTemperatureResult(
        hadronic_grids=out_grids, T_nuc=T_nuc, tau_target=tau_target,
        converged=conv, residual=resid, eq_type=eq_type)


# =============================================================================
# Nucleation density curve from raw table values  (public)
# =============================================================================
def _clip_log_tau(tau):
    with np.errstate(divide='ignore', invalid='ignore'):
        log_tau = np.log10(tau)
    log_tau = np.clip(log_tau, -50, 100)
    return np.where(np.isnan(log_tau), 100.0, log_tau)


def _compute_nucleation_density_by_nB(nucleation_obs, tau_target=1.0, verbose=False):
    """Scan T per n_B (transpose of the default): robust when tau is non-monotonic in T."""
    eq_type = nucleation_obs.eq_type
    grids = nucleation_obs.hadronic_grids
    axes = GRID_AXES[eq_type]
    n_B_arr = grids['n_B_H']
    T_arr = grids['T']
    n_nB = len(n_B_arr)
    log_tau_target = np.log10(tau_target)
    log_tau_full = _clip_log_tau(nucleation_obs.tau)
    non_T_axes = [ax for ax in axes if ax not in ('n_B_H', 'T')]

    if len(non_T_axes) == 0:
        T_nuc = np.full(n_nB, np.nan)
        conv = np.zeros(n_nB, dtype=bool)
        resid = np.full(n_nB, np.nan)
        for i in range(n_nB):
            slice_ = log_tau_full[i, :]
            valid = nucleation_obs.converged[i, :] & np.isfinite(slice_)
            if np.sum(valid) < 2:
                continue
            T_root, ok, res = _find_crossing_root(
                T_arr[valid], slice_[valid] - log_tau_target)
            if ok:
                T_nuc[i], conv[i], resid[i] = T_root, True, res
        return SimpleNamespace(scan='n_B', n_B_arr=n_B_arr, T_nuc=T_nuc,
                               tau_target=tau_target, converged=conv,
                               residual=resid, eq_type=eq_type)
    elif len(non_T_axes) == 1:
        ax_name = non_T_axes[0]
        ax_arr = grids[ax_name]
        n_outer = len(ax_arr)
        T_nuc = np.full((n_nB, n_outer), np.nan)
        conv = np.zeros((n_nB, n_outer), dtype=bool)
        resid = np.full((n_nB, n_outer), np.nan)
        for i in range(n_nB):
            for m in range(n_outer):
                slice_ = log_tau_full[i, m, :]
                valid = nucleation_obs.converged[i, m, :] & np.isfinite(slice_)
                if np.sum(valid) < 2:
                    continue
                T_root, ok, res = _find_crossing_root(
                    T_arr[valid], slice_[valid] - log_tau_target)
                if ok:
                    T_nuc[i, m], conv[i, m], resid[i, m] = T_root, True, res
        return SimpleNamespace(scan='n_B', n_B_arr=n_B_arr, outer_name=ax_name,
                               outer_arr=ax_arr, T_nuc=T_nuc,
                               tau_target=tau_target, converged=conv,
                               residual=resid, eq_type=eq_type)
    raise ValueError(f"Unsupported eq_type: '{eq_type}'")


def compute_nucleation_density(nucleation_obs, tau_target=1.0, scan='T',
                               verbose=False):
    """Nucleation curve tau = tau_target in the (n_B_H, T) plane.

    scan='T' (default): loop T, root-find along n_B -> n_B_nuc(T). scan='n_B':
    loop n_B, root-find along T -> T_nuc(n_B) (use when tau is non-monotonic in
    T, e.g. unpCFL). Both take the first downward crossing.
    """
    if scan not in ('T', 'n_B'):
        raise ValueError(f"scan must be 'T' or 'n_B', got '{scan}'")
    if scan == 'n_B':
        return _compute_nucleation_density_by_nB(
            nucleation_obs, tau_target=tau_target, verbose=verbose)

    eq_type = nucleation_obs.eq_type
    grids = nucleation_obs.hadronic_grids
    axes = GRID_AXES[eq_type]
    n_B_arr = grids['n_B_H']
    T_arr = grids['T']
    n_T = len(T_arr)
    log_tau_target = np.log10(tau_target)
    log_tau_full = _clip_log_tau(nucleation_obs.tau)
    non_T_axes = [ax for ax in axes if ax not in ('n_B_H', 'T')]

    if len(non_T_axes) == 0:
        n_B_nuc = np.full(n_T, np.nan)
        conv = np.zeros(n_T, dtype=bool)
        resid = np.full(n_T, np.nan)
        for j in range(n_T):
            slice_ = log_tau_full[:, j]
            valid = nucleation_obs.converged[:, j] & np.isfinite(slice_)
            if np.sum(valid) < 2:
                continue
            nB_root, ok, res = _find_crossing_root(
                n_B_arr[valid], slice_[valid] - log_tau_target)
            if ok:
                n_B_nuc[j], conv[j], resid[j] = nB_root, True, res
        return SimpleNamespace(scan='T', T_arr=T_arr, n_B_nuc=n_B_nuc,
                               tau_target=tau_target, converged=conv,
                               residual=resid, eq_type=eq_type)
    elif len(non_T_axes) == 1:
        ax_name = non_T_axes[0]
        ax_arr = grids[ax_name]
        n_outer = len(ax_arr)
        n_B_nuc = np.full((n_T, n_outer), np.nan)
        conv = np.zeros((n_T, n_outer), dtype=bool)
        resid = np.full((n_T, n_outer), np.nan)
        for j in range(n_T):
            for m in range(n_outer):
                slice_ = log_tau_full[:, m, j]
                valid = nucleation_obs.converged[:, m, j] & np.isfinite(slice_)
                if np.sum(valid) < 2:
                    continue
                nB_root, ok, res = _find_crossing_root(
                    n_B_arr[valid], slice_[valid] - log_tau_target)
                if ok:
                    n_B_nuc[j, m], conv[j, m], resid[j, m] = nB_root, True, res
        return SimpleNamespace(scan='T', T_arr=T_arr, outer_name=ax_name,
                               outer_arr=ax_arr, n_B_nuc=n_B_nuc,
                               tau_target=tau_target, converged=conv,
                               residual=resid, eq_type=eq_type)
    raise ValueError(f"Unsupported eq_type: '{eq_type}'")


def nucleation_curve(result, iYL=None):
    """(n_B, T) arrays of the tau = tau_target curve from a scan='n_B' result."""
    nB = np.asarray(result.n_B_arr)
    T = np.asarray(result.T_nuc)
    if T.ndim == 2:
        if iYL is None:
            raise ValueError("iYL required for a 2D (trapped/fixed_yc) result")
        T = T[:, iYL]
        nB = nB[:, iYL] if nB.ndim == 2 else nB
    return nB, T


def build_nucleation_temperature_interpolator(result, kind='cubic'):
    """Smooth T_nuc interpolator from converged points only (bridges NaN gaps)."""
    n_B_arr = result.hadronic_grids['n_B_H']
    eq_type = result.eq_type
    if result.T_nuc.ndim == 1:
        mask = result.converged
        if np.sum(mask) < 2:
            return lambda n_B: np.nan
        f = interp1d(n_B_arr[mask], result.T_nuc[mask], kind=kind,
                     bounds_error=False, fill_value=np.nan)
        return lambda n_B: float(f(n_B))
    non_T_axes = [ax for ax in GRID_AXES[eq_type] if ax not in ('n_B_H', 'T')]
    ax_name = non_T_axes[0]
    ax_arr = result.hadronic_grids[ax_name]
    n_outer = len(ax_arr)
    slice_interps = []
    for k in range(n_outer):
        mask = result.converged[:, k]
        if np.sum(mask) < 2:
            slice_interps.append(None)
        else:
            slice_interps.append(interp1d(
                n_B_arr[mask], result.T_nuc[mask, k], kind=kind,
                bounds_error=False, fill_value=np.nan))
    T_filled = np.full_like(result.T_nuc, np.nan)
    for k, fi in enumerate(slice_interps):
        if fi is not None:
            T_filled[:, k] = fi(n_B_arr)
    rgi = RegularGridInterpolator(
        (n_B_arr, ax_arr), T_filled, method='linear',
        bounds_error=False, fill_value=np.nan)
    return lambda n_B, outer: float(rgi((n_B, outer)))


# =============================================================================
# Single-point physics  (public)
# =============================================================================
# These four are the per-point primitives the analysis layer used to own. They
# live here, in the core, because the tau = tau_target locus is built out of
# them and reaching up into `analysis` from the core would invert the layering.
# The core stays config-free: plain keyword arguments, no NucConfig.
# `nucleation.analysis` keeps thin NucConfig-taking adapters under the old
# names (critical_droplet_pt / tau_pt), so nuc_an.* callers are unaffected.

def crossover_radius(T, Delta0):
    """unpCFL coherence radius R_x(T) = hc/Delta(T) [fm].

    Uses the SAME CFL gap as the EoS: Delta(T) = Delta0*sqrt(1-(T/Tc)^2) with
    Tc = T_critical(Delta0) (= 0.57*2^(1/3)*Delta0, including the CFL
    enhancement). Returns inf at or above Tc -- no pairing, so the droplet is
    unpaired at every radius. Tc comes from eos.T_critical so R_x can never
    drift away from the gap actually used in the equation of state.
    """
    if Delta0 <= 0:
        return np.inf
    T_c = T_critical(Delta0)
    ratio = np.asarray(T / T_c)
    gap = np.where(ratio < 1, Delta0 * np.sqrt(np.maximum(0, 1 - ratio**2)), 0.0)
    return np.where(gap > 0, hc / gap, np.inf)


def hadronic_point(H_trapped, n_B, Y_L, T):
    """Trapped hadronic state at one (n_B, Y_L, T) point, as a namespace.

    Evaluates the trapped-neutrino interpolator dict and packs every field the
    Q* solvers and ``driving_force`` need. Charge neutrality of the bulk
    hadronic phase fixes Y_e = Y_C; lepton-number conservation fixes
    Y_nu = Y_L - Y_C.
    """
    pt = (n_B, Y_L, T)
    h = SimpleNamespace(
        n_B=float(n_B), T=float(T),
        P_total=float(H_trapped['P'](*pt)),   e_total=float(H_trapped['eps'](*pt)),
        mu_B=float(H_trapped['mu_B'](*pt)),   mu_C=float(H_trapped['mu_C'](*pt)),
        mu_S=float(H_trapped['mu_S'](*pt)),   mu_e=float(H_trapped['mu_e'](*pt)),
        mu_nu=float(H_trapped['mu_nu'](*pt)),
        Y_C=float(H_trapped['Y_C'](*pt)),     Y_S=float(H_trapped['Y_S'](*pt)))
    h.Y_e = h.Y_C
    h.Y_nu = float(Y_L) - h.Y_C
    return h


@dataclass
class NucleationPoint:
    """Everything about nucleation at ONE hadronic state.

    One call gives the ambient matter (``H``), the droplet that would form in it
    (``Qs``), the barrier it must cross (``R_star``, ``W_star``) and how long
    that takes (``Gamma``, ``tau``) -- so a single point can be interrogated
    without assembling a grid.

    Conventions inherited from the critical-droplet engine:
      * ``W_star = +inf``  -- the hadronic phase is STABLE here (Delta_f >= 0):
        there is no barrier to cross because there is nothing to cross to.
        ``tau`` is then +inf.
      * ``W_star = nan``   -- the solver failed; the point is unknown, not stable.
    """
    n_B_H: float
    Y_L_H: float
    T: float
    sigma: float
    H: object
    Qs: object
    R_star: float
    W_star: float
    Delta_f: float
    delta_n_C: float
    lambda_D: float
    Rx: float
    Gamma: float
    tau: float
    V: float
    quark_phase: str
    flavor_mode: str
    electric_charge_mode: str

    @property
    def W_over_T(self):
        """Barrier in units of the temperature -- the exponent that decides the
        thermal rate (Gamma ~ exp(-W*/T)). Values above ~100 mean no nucleation
        on any astrophysical timescale, whatever the prefactor does."""
        return self.W_star / self.T if self.T > 0 else np.inf

    @property
    def N_B_star(self):
        """Baryon number inside the critical droplet, (4/3)pi R*^3 n_B^Q*.

        The physical size of the fluctuation required: a few tens of baryons is
        plausible, 10^6 is not.
        """
        if self.Qs is None or not np.isfinite(self.R_star):
            return np.nan
        return (4.0 / 3.0) * np.pi * self.R_star**3 * float(self.Qs.n_B)

    def nucleates(self, tau_target):
        """True if a droplet forms within tau_target [s] at this state."""
        return bool(np.isfinite(self.tau) and self.tau <= tau_target)


def nucleation_point(H_trapped, n_B_H, T, sigma, *, params,
                     Y_L_H=None, quark_phase='unpaired', Delta0=None,
                     flavor_mode='saddlepoint', electric_charge_mode='gcn',
                     V=V_NUCLEATION, cache=None, Rx=None,
                     include_photons=True, include_gluons=True,
                     include_thermal_neutrinos=True):
    """EoS + nucleation state at ONE (n_B_H, [Y_L_H], T). See ``NucleationPoint``.

    Physics: solves the droplet composition Q* in equilibrium with the ambient
    hadronic matter, locates the barrier peak (R*, W*), and converts it to a
    Langer rate and a nucleation time tau = 1/(V*Gamma).

    Mechanics:
      H_trapped : the trapped-neutrino interpolator dict, OR an already-built
                  hadronic namespace (as returned by ``hadronic_point``), in
                  which case n_B_H / Y_L_H / T are only recorded, not re-evaluated.
      sigma     : surface tension [MeV/fm^2] -- the least-constrained input, and
                  the one ``sigma_crit`` solves for.
      quark_phase : 'unpaired', 'cfl', or 'unpCFL' (Rx defaults to the coherence
                  radius crossover_radius(T, Delta0)).
      V         : nucleation volume [fm^3]; only rescales tau, not the barrier.
      cache     : optional dict memoizing the sigma-independent composition
                  solves -- pass the same dict across a sigma scan.
    """
    if hasattr(H_trapped, 'mu_B'):          # already a hadronic namespace
        H = H_trapped
    else:
        if Y_L_H is None:
            raise ValueError(
                "Y_L_H is required when H_trapped is an interpolator dict")
        H = hadronic_point(H_trapped, n_B_H, Y_L_H, T)

    if quark_phase == 'unpCFL' and Rx is None:
        if Delta0 is None:
            raise ValueError("Delta0 required for quark_phase='unpCFL'")
        Rx = float(crossover_radius(T, Delta0))

    cd = critical_droplet(
        H, sigma=sigma, params=params, flavor_mode=flavor_mode,
        electric_charge_mode=electric_charge_mode, quark_phase=quark_phase,
        Delta0=Delta0, Rx=Rx, cache=cache,
        include_photons=include_photons, include_gluons=include_gluons,
        include_thermal_neutrinos=include_thermal_neutrinos)

    # tau conventions mirror the engine's: +inf where H is stable (no barrier to
    # cross), NaN where the solve failed (unknown, NOT stable).
    if np.isposinf(cd.W_c):
        Gamma, tau = 0.0, np.inf
    elif cd.Qs is None or not (np.isfinite(cd.R_c) and cd.R_c > 0
                               and np.isfinite(cd.W_c) and cd.W_c > 0):
        Gamma, tau = np.nan, np.nan
    else:
        Gamma = float(nucleation_rate(cd.W_c, cd.R_c, sigma, T, H, cd.Qs))
        tau = float(nucleation_time(Gamma, V)) if Gamma > 0 else np.inf

    return NucleationPoint(
        n_B_H=float(getattr(H, 'n_B', n_B_H)),
        Y_L_H=(np.nan if Y_L_H is None else float(Y_L_H)),
        T=float(T), sigma=float(sigma), H=H, Qs=cd.Qs,
        R_star=cd.R_c, W_star=cd.W_c, Delta_f=cd.Delta_f,
        delta_n_C=cd.delta_n_C, lambda_D=cd.lambda_D,
        Rx=(np.nan if Rx is None else float(Rx)),
        Gamma=Gamma, tau=tau, V=V, quark_phase=quark_phase,
        flavor_mode=flavor_mode, electric_charge_mode=electric_charge_mode)


def tau_at(H_trapped, n_B_H, T, sigma, **kw):
    """Nucleation time tau [s] at one point. Shorthand for
    ``nucleation_point(...).tau`` when the rest of the state is not needed."""
    return nucleation_point(H_trapped, n_B_H, T, sigma, **kw).tau


# =============================================================================
# The tau = tau_target locus  (public facade)
# =============================================================================
class NucleationCondition:
    """The locus tau(n_B_H, Y_L_H, T) = tau_target, however it was obtained.

    Below the locus the star is too cold (or too dilute) for a droplet to form
    within tau_target; above it, nucleation happens. The same three methods work
    no matter which constructor was used, so a figure does not care whether the
    numbers came from a saved table or a live solve:

        cond.T_of_nB(n_B, Y_L=None)      -> T_nuc   [MeV]     scalar or array
        cond.nB_of_T(T,   Y_L=None)      -> n_B_nuc [fm^-3]   scalar or array
        cond.curve(Y_L=None)             -> (n_B, T) arrays, ready to plot
        cond.nucleates(n_B, T, Y_L=None) -> bool/array (is T above T_nuc?)

    Construct with one of:
        from_table(obs, tau_target)                -- a computed grid (thermal
                                                      OR quantum: both have .tau)
        from_interpolator(fn, grids, eq_type, ...) -- any log10(tau) callable
        from_point_solver(H, T_grid, n_B_grid,...) -- solve on the fly, no table

    No new root finder is defined here: every constructor delegates to
    ``compute_nucleation_density`` / ``compute_nucleation_temperature``, which
    share the two crossing finders at the top of this module.
    """

    def __init__(self, tau_target, eq_type, n_B_arr, T_nuc, outer_arr=None,
                 outer_name=None, converged=None, _source=None):
        self.tau_target = float(tau_target)
        self.eq_type = eq_type
        self.n_B_arr = np.asarray(n_B_arr, dtype=float)
        self.T_nuc = np.asarray(T_nuc, dtype=float)
        self.outer_arr = None if outer_arr is None else np.asarray(outer_arr, float)
        self.outer_name = outer_name
        self.converged = (np.isfinite(self.T_nuc) if converged is None
                          else np.asarray(converged, dtype=bool))
        self._source = _source          # kept for nB_of_T, computed lazily
        self._nB_of_T = None

    # -- constructors ---------------------------------------------------------
    @classmethod
    def from_table(cls, obs, tau_target=1e-3, scan='n_B'):
        """From computed observables (``ThermalNucleationObservables`` or
        ``QuantumNucleationObservables`` -- both expose ``.tau``).

        scan='n_B' (the default) root-finds along T at each density. Prefer it:
        tau is NOT monotonic in T for unpCFL (the CFL gap melts as T rises), and
        scanning the other way silently picks the wrong branch there.
        """
        res = compute_nucleation_density(obs, tau_target=tau_target, scan=scan)
        return cls._from_scan_result(res, tau_target, source=obs)

    @classmethod
    def from_interpolator(cls, log10_tau_fn, hadronic_grids, eq_type,
                          tau_target=1e-3, T_guess=None):
        """From any log10(tau) interpolator, e.g. the 'log10_tau' entry of
        ``build_thermal_nucleation_interpolators`` or its quantum twin."""
        res = compute_nucleation_temperature(
            log10_tau_fn, hadronic_grids, eq_type,
            tau_target=tau_target, T_guess=T_guess)
        outer = [ax for ax in GRID_AXES[eq_type] if ax not in ('n_B_H', 'T')]
        return cls(
            tau_target=tau_target, eq_type=eq_type,
            n_B_arr=res.hadronic_grids['n_B_H'], T_nuc=res.T_nuc,
            outer_arr=(res.hadronic_grids[outer[0]] if outer else None),
            outer_name=(outer[0] if outer else None),
            converged=res.converged)

    @classmethod
    def from_point_solver(cls, H_trapped, T_grid, n_B_grid, *, sigma, params,
                          tau_target=1e-3, Y_L_H=None, eq_type='trapped_neutrinos',
                          verbose=False, **droplet_kw):
        """Solve the locus directly, with NO precomputed table.

        This is the constructor for exploring parameter space: it answers "at
        what T does THIS (alpha_s, B^1/4, Delta_0) nucleate?" in seconds, where
        building a Q* table first would take minutes.

        Walks n_B, root-finding tau(T) = tau_target along T at each density, and
        warm-starts every density from the previous solution -- the same trick
        ``compute_nucleation_temperature`` uses, and the reason this is not
        simply len(n_B_grid) independent scans.
        """
        T_grid = np.asarray(T_grid, dtype=float)
        n_B_grid = np.asarray(n_B_grid, dtype=float)
        log_target = np.log10(tau_target)
        cache = {}

        T_nuc = np.full(n_B_grid.size, np.nan)
        conv = np.zeros(n_B_grid.size, dtype=bool)
        prev_T = None
        for i, nB in enumerate(n_B_grid):
            def f(T, _nB=float(nB)):
                tau = tau_at(H_trapped, _nB, float(T), sigma, params=params,
                             Y_L_H=Y_L_H, cache=cache, **droplet_kw)
                # Clip exactly as the table path does, so a +inf ("never
                # nucleates") point is a large finite residual rather than a NaN
                # that would abort the bracket search.
                with np.errstate(divide='ignore', invalid='ignore'):
                    lt = np.log10(tau)
                lt = 100.0 if not np.isfinite(lt) else float(np.clip(lt, -50, 100))
                return lt - log_target
            root, ok, _ = _find_T_root(
                f, T_grid, [prev_T] if prev_T is not None else None)
            if ok:
                T_nuc[i], conv[i], prev_T = root, True, root
            if verbose:
                print(f"  n_B={float(nB):.4f} -> "
                      f"T_nuc={T_nuc[i]:.3f}" if ok else
                      f"  n_B={float(nB):.4f} -> no crossing")
        return cls(tau_target=tau_target, eq_type=eq_type, n_B_arr=n_B_grid,
                   T_nuc=T_nuc, converged=conv)

    @classmethod
    def _from_scan_result(cls, res, tau_target, source=None):
        """Wrap a compute_nucleation_density result (either scan direction)."""
        if getattr(res, 'scan', 'n_B') == 'n_B':
            return cls(tau_target=tau_target, eq_type=res.eq_type,
                       n_B_arr=res.n_B_arr, T_nuc=res.T_nuc,
                       outer_arr=getattr(res, 'outer_arr', None),
                       outer_name=getattr(res, 'outer_name', None),
                       converged=res.converged, _source=source)
        # scan='T' gives n_B(T); store it transposed so the object always
        # presents T_nuc(n_B) as its primary form.
        obj = cls(tau_target=tau_target, eq_type=res.eq_type,
                  n_B_arr=res.n_B_nuc, T_nuc=res.T_arr,
                  outer_arr=getattr(res, 'outer_arr', None),
                  outer_name=getattr(res, 'outer_name', None),
                  converged=res.converged, _source=source)
        return obj

    # -- queries --------------------------------------------------------------
    def _slice(self, Y_L=None):
        """(n_B, T_nuc) for one outer-axis value, dropping unconverged points."""
        T = self.T_nuc
        if T.ndim == 2:
            if Y_L is None:
                raise ValueError(
                    f"this condition has a {self.outer_name} axis; pass Y_L=")
            k = int(np.argmin(np.abs(self.outer_arr - Y_L)))
            T = T[:, k]
            ok = self.converged[:, k]
        else:
            ok = self.converged
        m = ok & np.isfinite(T)
        return self.n_B_arr[m], T[m]

    def curve(self, Y_L=None):
        """(n_B, T) arrays of the locus, ready to plot."""
        return self._slice(Y_L)

    def T_of_nB(self, n_B, Y_L=None):
        """Nucleation temperature [MeV] at one or many densities.

        NaN outside the range where a crossing was found -- that is a real
        answer ("no nucleation at any T on this grid"), not a failure.
        """
        x, y = self._slice(Y_L)
        if x.size < 2:
            return np.full(np.shape(n_B), np.nan) if np.ndim(n_B) else np.nan
        f = interp1d(x, y, kind='cubic' if x.size >= 4 else 'linear',
                     bounds_error=False, fill_value=np.nan)
        out = f(n_B)
        return float(out) if np.ndim(n_B) == 0 else np.asarray(out)

    def nB_of_T(self, T, Y_L=None):
        """Nucleation density [fm^-3] at one or many temperatures (the inverse).

        Computed by inverting the same locus, so it is guaranteed consistent
        with ``T_of_nB`` rather than being a second, independently scanned curve.
        """
        x, y = self._slice(Y_L)
        if x.size < 2:
            return np.full(np.shape(T), np.nan) if np.ndim(T) else np.nan
        order = np.argsort(y)                     # invert: T becomes the abscissa
        ys, xs = y[order], x[order]
        keep = np.concatenate([[True], np.diff(ys) > 0])   # strictly increasing
        ys, xs = ys[keep], xs[keep]
        if ys.size < 2:
            return np.full(np.shape(T), np.nan) if np.ndim(T) else np.nan
        f = interp1d(ys, xs, kind='cubic' if ys.size >= 4 else 'linear',
                     bounds_error=False, fill_value=np.nan)
        out = f(T)
        return float(out) if np.ndim(T) == 0 else np.asarray(out)

    def nucleates(self, n_B, T, Y_L=None):
        """True where (n_B, T) lies on the nucleating side of the locus."""
        T_thr = self.T_of_nB(n_B, Y_L)
        with np.errstate(invalid='ignore'):
            return np.asarray(T) >= T_thr

    def __repr__(self):
        n_ok = int(np.sum(self.converged))
        return (f"NucleationCondition(tau_target={self.tau_target:g} s, "
                f"eq_type='{self.eq_type}', {n_ok}/{self.converged.size} "
                f"points on the locus)")


# =============================================================================
# Shorthands: the literal T[n_B, Y_L, tau] / n_B[T, Y_L, tau] form
# =============================================================================
def _as_condition(source, tau_target, **kw):
    """Coerce whatever the caller has into a NucleationCondition."""
    if isinstance(source, NucleationCondition):
        return source
    if hasattr(source, 'tau') and hasattr(source, 'hadronic_grids'):
        return NucleationCondition.from_table(source, tau_target=tau_target)
    if callable(source):
        raise TypeError(
            "a bare interpolator needs its grid: use "
            "NucleationCondition.from_interpolator(fn, grids, eq_type, ...)")
    raise TypeError(f"cannot build a NucleationCondition from {type(source)!r}")


def T_nuc(source, n_B_H, Y_L_H=None, tau_target=1e-3, **kw):
    """Nucleation temperature T(n_B_H, Y_L_H; tau_target) [MeV].

    ``source`` is nucleation observables (thermal or quantum) or an existing
    ``NucleationCondition``. Scalars in, scalar out; arrays in, array out.
    Build the condition once and reuse it if you are calling this in a loop.
    """
    return _as_condition(source, tau_target, **kw).T_of_nB(n_B_H, Y_L_H)


def nB_nuc(source, T, Y_L_H=None, tau_target=1e-3, **kw):
    """Nucleation density n_B(T, Y_L_H; tau_target) [fm^-3]. See ``T_nuc``."""
    return _as_condition(source, tau_target, **kw).nB_of_T(T, Y_L_H)
