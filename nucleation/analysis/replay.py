"""
Replay of accepted parameter cells
==================================

Once a scan has decided which (alpha_s, B^1/4, Delta_0) cells are viable, these
re-solve the cold EoS + TOV sequence for each one so figures can draw the
*bundle* of admissible stars (M-R, P(mu_B)) coloured by sigma_crit.

Separate from `scan` because this is presentation-time work: a scan decides
yes/no per cell, a replay reconstructs the curves behind the surviving cells.
"""
from __future__ import annotations

import numpy as np

from eos.general.state import EOSTable_for_TOV
from eos.astro.tov.solver import generate_ec_logspace, compute_tov_sequence
from nucleation.analysis.config import FilterConfig
from nucleation.analysis.filters import cfl_eos_at_params

try:
    import joblib  # noqa: F401  -- presence check; Parallel imported in-function
    _HAVE_JOBLIB = True
except Exception:
    _HAVE_JOBLIB = False

def replay_cfl(alpha, B4, Delta0, cfg: FilterConfig, e_c_vec=None):
    """Cold CFL EoS + TOV for one set -> dict(mu, P, R, M) or None on failure.

    Plot-oriented replay: the M-R curve only feeds a figure, so the TOV sweep
    uses a coarser central-density grid than the acceptance filter (40 points
    instead of cfg.e_c_vec_tov's 80), skips the baryonic-mass quadrature (M_b
    is never plotted) and slices the stable branch at argmax(M) directly
    (no find_mmax_precise refinement). ~2x faster per set than the filter path
    with a visually identical curve. Pass e_c_vec to override the grid.
    (Raw TOV cols without M_b: 0=e_c 1=n_c 2=P_c 3=R[km] 4=M[Msun].)"""
    P, e, mu, ok = cfl_eos_at_params(alpha, B4, Delta0, cfg)
    if ok.sum() < 20:
        return None
    n_ok, P_ok, e_ok, mu_ok = cfg.n_B_grid[ok], P[ok], e[ok], mu[ok]
    pos = P_ok > 0
    if pos.sum() < 10:
        return None
    if e_c_vec is None:
        e_c_vec = generate_ec_logspace(e_min=100, e_max=2000, n_points=40)
    try:
        tov = compute_tov_sequence(
            EOSTable_for_TOV(P=P_ok[pos], epsilon=e_ok[pos], nB=n_ok[pos]),
            e_c_vec=e_c_vec, add_crust_table='No',
            compute_baryonic_mass=False, compute_tidal=False, verbose=False,
            backend=cfg.tov_backend)
    except Exception:
        return None
    if tov.shape[0] < 3:
        return None
    i_max = int(np.argmax(tov[:, 4])) + 1          # stable branch: up to M_max
    return dict(mu=mu_ok, P=P_ok, R=tov[:i_max, 3], M=tov[:i_max, 4])


def replay_accepted(sig_crit, alpha_slices, B4_grid, Delta0_grid,
                    cfg: FilterConfig, max_curves=None, n_jobs=-1, verbose=True,
                    with_params=False):
    """Replay the cold CFL EoS + TOV of every ACCEPTED cell of a sigma_crit grid.

    Accepted = finite sigma_crit (CFL-pass AND nucleating). Each replay is an
    independent (EoS solve + TOV) -> joblib-parallel over cells. ``max_curves``
    evenly subsamples the accepted list first: for an overlay plot a few
    hundred curves are visually identical to several thousand at a fraction of
    the cost (None = replay all).

    ``with_params=True`` also returns the (alpha_s, B^1/4, Delta_0) each curve
    came from. Without it the caller has a bundle of curves and no way to ask
    WHICH parameter drives their spread -- and reconstructing the mapping
    outside would mean duplicating the accept ordering and the subsample stride,
    which is exactly the kind of silent drift a plot never reveals.

    Returns
    -------
    list of (curve_dict, sigma_crit), or of (curve_dict, sigma_crit,
    (alpha, B4, Delta0)) when ``with_params``. curve_dict comes from
    ``replay_cfl`` (keys mu, P, R, M); failed replays are dropped.
    """
    alpha_slices = np.atleast_1d(alpha_slices)
    accept = [(alpha_slices[ia], B4_grid[jx], Delta0_grid[i],
               float(sig_crit[ia, i, jx]))
              for ia in range(len(alpha_slices))
              for i in range(len(Delta0_grid))
              for jx in range(len(B4_grid))
              if np.isfinite(sig_crit[ia, i, jx])]
    n_total = len(accept)
    if max_curves and n_total > max_curves:
        accept = accept[::int(np.ceil(n_total / max_curves))]
    if verbose:
        print(f"replay: {len(accept)}/{n_total} accepted cells "
              f"(max_curves={max_curves}), "
              f"{'parallel' if (n_jobs != 1 and _HAVE_JOBLIB) else 'serial'}...",
              flush=True)
    if n_jobs != 1 and _HAVE_JOBLIB:
        from joblib import Parallel, delayed
        out = Parallel(n_jobs=n_jobs, verbose=(5 if verbose else 0))(
            delayed(replay_cfl)(a, b, d, cfg) for a, b, d, _ in accept)
    else:
        out = [replay_cfl(a, b, d, cfg) for a, b, d, _ in accept]
    if with_params:
        curves = [(c, sc, (a, b, d)) for c, (a, b, d, sc) in zip(out, accept)
                  if c is not None]
    else:
        curves = [(c, sc) for c, (_, _, _, sc) in zip(out, accept)
                  if c is not None]
    if verbose:
        print(f"replay: reconstructed {len(curves)}/{len(accept)} curves.", flush=True)
    return curves

