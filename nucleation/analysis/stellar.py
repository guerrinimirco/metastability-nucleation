"""
TOV sequence bookkeeping
========================

Everything about reading a solved TOV sequence: which column is which, where
the stable branch ends, how to interpolate along it, and how to find the
sequence matching a (Y_L, S) snapshot.

No nucleation physics here -- this is the stellar-structure side, kept separate
so a figure that only needs "M at this M_B" does not drag in the droplet engine.

Column convention
-----------------
``compute_tov_sequence`` returns one row per central density with columns
``TOV_COL``. Only the stable branch (up to the first maximum in gravitational
mass) describes real stars; everything past it is unstable to radial
oscillations, so ALWAYS slice with ``stable_branch`` before interpolating.
"""
from __future__ import annotations

import glob
import os

import numpy as np
from scipy.interpolate import interp1d

from eos.tov.solver import (EOSTable_for_TOV, generate_ec_logspace,
                            compute_tov_sequence, truncate_to_stable_branch)
from nucleation.analysis.filters import cfl_eos_at_params

# Column index -> meaning, for the arrays compute_tov_sequence returns.
# Named because `arr[:, 4]` appeared ~30 times in the notebook and a wrong
# integer there is a silent, plausible-looking wrong answer.
TOV_COL = {'e_c': 0, 'P_c': 1, 'n_Bc': 2, 'R': 3, 'M': 4, 'M_B': 5}


def stable_branch(seq):
    """Slice a TOV sequence to its stable branch (up to the first M_max).

    dM/de_c > 0 is the stability criterion; past the maximum the star is
    unstable to radial collapse, so those rows are not physical configurations.
    """
    seq = np.asarray(seq)
    return seq[:int(np.argmax(seq[:, TOV_COL['M']])) + 1]


def max_masses(seq):
    """(M_max, M_B_max) [M_sun] of the stable branch."""
    st = stable_branch(seq)
    return float(st[:, TOV_COL['M']].max()), float(st[:, TOV_COL['M_B']].max())


def branch_interp(seq, xcol, ycol, kind='cubic'):
    """Interpolate y(x) along a sequence's STABLE branch.

    `xcol` / `ycol` are keys of TOV_COL (or raw integers). Sorts and de-duplicates
    in x before interpolating: the cold CFL quark-star sequence can be
    non-monotone or repeat a baryonic mass, and scipy's cubic spline rejects
    duplicate abscissae outright. The notebook grew three near-copies of this
    function for exactly that reason -- this is the union of all three.

    Returns a callable; on fewer than 4 usable points it returns a
    NaN-producing stub rather than raising, so one degenerate parameter cell
    cannot abort a whole figure.
    """
    xcol = TOV_COL.get(xcol, xcol)
    ycol = TOV_COL.get(ycol, ycol)
    st = stable_branch(seq)
    x, y = st[:, xcol], st[:, ycol]
    order = np.argsort(x)
    x, y = x[order], y[order]
    keep = np.concatenate(([True], np.diff(x) > 1e-9))
    x, y = x[keep], y[keep]
    if len(x) < 4:
        if len(x) < 2:
            return lambda q: np.nan
        kind = 'linear'
    return interp1d(x, y, kind=kind, bounds_error=False, fill_value=np.nan)


def snapshot_key(Y_L, S):
    """Canonical dict key for a (Y_L, S) PNS snapshot.

    ONE rounding policy, used by the writer, the loader and every lookup. The
    notebook had three (exact tuple, round to 3 dp, nearest-neighbour search)
    and they agreed only because f'YL{Y_L:.2f}' happens to turn
    0.35000000000000003 back into 0.35.
    """
    return (round(float(Y_L), 3), round(float(S), 3))


def load_tov_trapped(dirpath, xsd_tag, pattern='tov_hadronic_trapped_2famphi_{tag}_YL*_S*.dat'):
    """{(Y_L, S): sequence} for every trapped TOV table in `dirpath`.

    Keys come from ``snapshot_key`` so lookups match regardless of how the
    filename was formatted.
    """
    out = {}
    for path in sorted(glob.glob(os.path.join(dirpath,
                                              pattern.format(tag=xsd_tag)))):
        stem = os.path.basename(path)[:-4]
        y_part = stem.split('_YL')[1]
        y_str, s_str = y_part.split('_S')
        out[snapshot_key(float(y_str), float(s_str))] = np.loadtxt(path)
    return out


def nearest_trapped_sequence(tov_trapped, Y_L, S):
    """The stored sequence closest to (Y_L, S) in the (Y_L, S) plane."""
    if not tov_trapped:
        raise ValueError("no trapped TOV sequences loaded")
    key = min(tov_trapped, key=lambda k: abs(k[0] - Y_L) + abs(k[1] - S))
    return tov_trapped[key]


def cold_quark_star_branch(pset, cfg, *, e_c_vec=None, backend='fast',
                           cache_path=None, verbose=False):
    """Cold (T=0) CFL quark star: EoS -> crust-less TOV -> stable branch.

    Physics: a strange quark star is self-bound, so there is no crust to match
    and the sequence starts at P=0 with finite density. The baryonic mass is
    computed because the PNS -> QS conversion conserves baryon number, not
    gravitational mass -- the remnant is read off at fixed M_B.

    Mechanics:
      pset      : dict with alpha, B4, Delta0, m_s.
      cfg       : FilterConfig supplying the n_B grid the EoS is solved on.
      e_c_vec   : central energy densities [MeV/fm^3]; a sensible default span
                  is used if omitted.
      backend   : 'fast' (numba) or 'scipy', forwarded to compute_tov_sequence.
      cache_path: optional .dat to read/write, so the same parameter set is not
                  re-solved by every figure that needs it. The notebook solved
                  this three separate times per set.

    Returns (stable_sequence, M_max).
    """
    if cache_path and os.path.exists(cache_path):
        seq = np.loadtxt(cache_path)
        return seq, float(seq[:, TOV_COL['M']].max())

    P, e, mu, ok = cfl_eos_at_params(pset['alpha'], pset['B4'],
                                     pset['Delta0'], cfg)
    pos = ok & (P > 0)
    if e_c_vec is None:
        e_c_vec = generate_ec_logspace(e_min=100, e_max=2000, n_points=80)
    tov = compute_tov_sequence(
        EOSTable_for_TOV(P=P[pos], epsilon=e[pos], nB=cfg.n_B_grid[pos]),
        e_c_vec=e_c_vec, add_crust_table='No',
        compute_baryonic_mass=True, compute_tidal=False,
        backend=backend, verbose=verbose)
    seq, M_max, _ = truncate_to_stable_branch(tov, verbose=verbose)

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        np.savetxt(cache_path, seq, fmt='%16.8e',
                   header="  ".join(f"{k:>14s}" for k in
                                    sorted(TOV_COL, key=TOV_COL.get)))
    return seq, float(M_max)
