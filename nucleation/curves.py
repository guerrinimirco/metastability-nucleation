"""
Nucleation curves
==================

Locate the nucleation threshold tau(n_B, T) = tau_target in the (n_B_H, T)
plane, from grid-level nucleation observables or from any log10(tau)
interpolator.

Public API
----------
    compute_nucleation_temperature(log10_tau_fn, grids, eq_type, ...)
        -> T_nuc(n_B[, Y]) by root-finding an interpolator along T.
    compute_nucleation_density(nucleation_obs, ..., scan='T'|'n_B')
        -> the tau=tau_target curve from raw table values (robust to non-monotonic
           tau, e.g. unpCFL).
    nucleation_curve(result, iYL=None)
        -> (n_B, T) arrays ready to plot.
    build_nucleation_temperature_interpolator(result, kind='cubic')

Two crossing finders are used: ``_find_T_root`` (evaluates a callable/interpolator
along T, with secant warm-starts) and ``_find_crossing_root`` (works on precomputed
table values along either axis). Both take the FIRST DOWNWARD crossing (tau falling
through tau_target).
"""

import numpy as np
from types import SimpleNamespace
from dataclasses import dataclass
from scipy.optimize import root_scalar
from scipy.interpolate import RegularGridInterpolator, interp1d

from nucleation.tables import GRID_AXES


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
