"""
Droplet observables over the parameter plane
============================================

A sigma_crit scan answers "how much surface tension does this parameter set
tolerate". These answer the follow-up questions a referee asks about the
droplet itself, evaluated at each cell's OWN sigma_crit:

  * ``barrier_ratio_map``    -- is W*/T roughly invariant across the viable
                                region? (It largely is, which is the reason
                                sigma_crit is a useful single number at all.)
  * ``droplet_regime_grid``  -- which length scale sets the critical radius:
                                the unpaired peak, the CFL peak, or the pairing
                                coherence radius R_Delta where the barrier is
                                pinned at the unpaired->CFL kink.
  * ``sigma_crit_along_sequence`` -- sigma_crit as a function of position along
                                a stellar sequence, for the M_PNS panels.

All three are DATA PRODUCTION -- each cell costs several droplet solves -- so
they belong in Part II of the notebook and should be cached to .npz, not
recomputed every time a figure is redrawn.
"""
from __future__ import annotations

import numpy as np

from eos.alphabag.parameters import get_alphabag_custom
from nucleation.conditions import crossover_radius, hadronic_point
from nucleation.analysis.sigma_crit import (central_state, critical_droplet_pt,
                                            sigma_target_pt, tau_pt)

# Which radius the barrier peak coincides with.
REGIME = {'unpaired': 0, 'R_Delta': 1, 'cfl': 2, 'none': -1}


def _parallel(n_jobs):
    """joblib Parallel if available, else a serial stand-in with the same call
    shape -- so an optional dependency never changes the result, only the wall
    clock."""
    try:
        from joblib import Parallel, delayed
        return Parallel(n_jobs=n_jobs), delayed
    except ImportError:
        def _serial(tasks):
            return [f(*a, **k) for f, a, k in tasks]

        def _delayed(f):
            return lambda *a, **k: (f, a, k)
        return _serial, _delayed


def droplet_regime_grid(sig_crit, alpha_slices, B4_grid, Delta0_grid, *,
                        star, nuc, MT0, m_s, flavor='saddlepoint',
                        charge='coulomb_minimize', slices=None, n_jobs=-1):
    """Classify the unpCFL critical radius R* over the parameter plane.

    Physics: the unpCFL barrier is the composite of an unpaired core and a CFL
    mantle, so its peak sits at one of three radii -- the pure-unpaired R_unp,
    the pure-CFL R_CFL, or the pairing coherence radius
    R_Delta = R_x(T) = hc/Delta(T), when the peak is pinned at the switching
    kink. Which one it is says which physics is actually controlling the
    threshold in that part of the plane, and the sigma_crit map alone does not
    show it.

    Each cell is evaluated at its OWN sigma_crit, so this is a property of the
    threshold configuration rather than of an arbitrary reference sigma. The
    saved scan stores only sigma_crit, so all three radii are recomputed here.

    The star centre depends only on MT0, so it is computed once for the grid.
    `slices` selects which alpha indices to fill (default: all).

    Returns an int array of REGIME codes, -1 where there is no droplet.
    """
    parallel, delayed = _parallel(n_jobs)
    _, T_c, H_pt = central_state(float(MT0), star)
    base = (H_pt, T_c, flavor, charge)

    def code(alpha, B4, Delta0, sc):
        if not np.isfinite(sc):
            return REGIME['none']
        # Delta0 = 0 makes crossover_radius divide by zero; the +inf it returns
        # is the correct answer (no pairing -> unpaired at every radius).
        with np.errstate(divide='ignore', invalid='ignore'):
            p = get_alphabag_custom(alpha=alpha, B4=B4, m_s=m_s)
            R_star = critical_droplet_pt(sc, *base, 'unpCFL', {}, p, Delta0, nuc)[0]
            if not np.isfinite(R_star):
                return REGIME['none']
            R_cfl = critical_droplet_pt(sc, *base, 'cfl', {}, p, Delta0, nuc)[0]
            R_unp = critical_droplet_pt(sc, *base, 'unpaired', {}, p, Delta0, nuc)[0]
            R_x = float(crossover_radius(T_c, Delta0))
        cand = [(v, c) for v, c in ((R_unp, REGIME['unpaired']),
                                    (R_x, REGIME['R_Delta']),
                                    (R_cfl, REGIME['cfl'])) if np.isfinite(v)]
        if not cand:
            return REGIME['none']
        return min(cand, key=lambda t: abs(R_star - t[0]))[1]

    n_alpha, n_D0, n_B4 = sig_crit.shape
    out = np.full(sig_crit.shape, REGIME['none'], dtype=int)
    for ia in (range(n_alpha) if slices is None else slices):
        res = parallel(delayed(code)(alpha_slices[ia], B4_grid[j],
                                     Delta0_grid[i], sig_crit[ia, i, j])
                       for i in range(n_D0) for j in range(n_B4))
        out[ia] = np.array(res).reshape(n_D0, n_B4)
    return out


def barrier_ratio_map(sig_crit, alpha_slices, B4_grid, Delta0_grid, *,
                      shells, nuc, m_s, sigma=None, flavor='saddlepoint',
                      charge='coulomb_minimize', phase='unpCFL',
                      slices=None, n_jobs=-1, verbose=True):
    """W*/T at the DECIDING shell, over the parameter plane.

    Physics: the star-wide threshold is set by whichever shell nucleates fastest,
    not by the centre, so W* is read there. If W*/T comes out roughly constant
    across the viable region, then sigma_crit is capturing essentially all of the
    parameter dependence -- which is what justifies quoting it as a single number.

    Mechanics:
      shells : [(H_pt, T)] along the star (see ``star_shell_states``).
      sigma  : evaluate at this fixed surface tension; None (default) uses each
               cell's own sigma_crit, which is the threshold configuration.

    tau_pt returns +inf where the hadronic phase is stable and NaN on solver
    failure; both are excluded from the argmin, and a cell with no usable shell
    comes back NaN.

    Returns (W_over_T, T_deciding, shell_index), each with sig_crit's shape.
    """
    parallel, delayed = _parallel(n_jobs)

    def cell(alpha, B4, Delta0, sc):
        sig = sc if sigma is None else sigma
        if not (np.isfinite(sc) and np.isfinite(sig)):
            return np.nan, np.nan, -1
        with np.errstate(divide='ignore', invalid='ignore'):
            p = get_alphabag_custom(alpha=alpha, B4=B4, m_s=m_s)
            # One cache per shell, reused for the second call: re-solving the
            # winning shell for W* then costs only the barrier maximization,
            # not the composition solve.
            caches = [{} for _ in shells]
            t = np.array([tau_pt(sig, hp, T, flavor, charge, phase,
                                 caches[k], p, Delta0, nuc)
                          for k, (hp, T) in enumerate(shells)], dtype=float)
            ok = np.isfinite(t) & (t > 0)
            if not ok.any():
                return np.nan, np.nan, -1
            k = int(np.flatnonzero(ok)[np.argmin(t[ok])])
            hp, T = shells[k]
            W = critical_droplet_pt(sig, hp, T, flavor, charge, phase,
                                    caches[k], p, Delta0, nuc)[1]
        return (float(W) / T if np.isfinite(W) and T > 0 else np.nan), T, k

    n_alpha, n_D0, n_B4 = sig_crit.shape
    W_over_T = np.full(sig_crit.shape, np.nan)
    T_dec = np.full(sig_crit.shape, np.nan)
    k_dec = np.full(sig_crit.shape, -1, dtype=int)
    for ia in (range(n_alpha) if slices is None else slices):
        res = parallel(delayed(cell)(alpha_slices[ia], B4_grid[j],
                                     Delta0_grid[i], sig_crit[ia, i, j])
                       for i in range(n_D0) for j in range(n_B4))
        for col, arr in enumerate((W_over_T, T_dec, k_dec)):
            arr[ia] = np.array([r[col] for r in res]).reshape(n_D0, n_B4)
        if verbose:
            print(f"  alpha_s slice {ia}: "
                  f"{int(np.isfinite(W_over_T[ia]).sum())}/"
                  f"{int(np.isfinite(sig_crit[ia]).sum())} accessible cells "
                  f"returned a W*/T")
    return W_over_T, T_dec, k_dec


def sigma_crit_along_sequence(n_B_seq, T_seq, *, Y_L_H, H_trapped, params,
                              Delta0, nuc, shells=None, flavor='saddlepoint',
                              charge='coulomb_minimize', phase='unpCFL'):
    """sigma_crit at each (n_B, T) along a stellar sequence.

    Drives the sigma_crit-vs-M_PNS panel: as a proto-neutron star gets more
    massive its centre gets denser and hotter, so the surface tension it can
    tolerate changes along the sequence.

    `shells` selects the definition, and is NOT optional in spirit -- passing
    None gives the CENTRE-ONLY sigma_crit, which is a different quantity from
    the star-wide value in the saved scan grids (see analysis.outcomes).
    """
    n_B_seq = np.asarray(n_B_seq, dtype=float)
    T_seq = np.asarray(T_seq, dtype=float)
    out = np.full(n_B_seq.shape, np.nan)
    for i, (nB, T) in enumerate(zip(n_B_seq, T_seq)):
        if not (np.isfinite(nB) and np.isfinite(T)):
            continue
        H_pt = hadronic_point(H_trapped, nB, Y_L_H, T)
        out[i] = sigma_target_pt(H_pt, T, flavor, charge, phase, params,
                                 Delta0, nuc, shells=shells)
    return out
