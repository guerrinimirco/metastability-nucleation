"""
Absolute-stability boundaries at P = 0
======================================

Where in the (alpha_s, B^1/4) plane is strange quark matter absolutely stable?

The Witten hypothesis needs two things at once, and they pull in opposite
directions:

  * 3-flavour (uds) matter MUST be bound: e/n_B < 930 MeV at P = 0, or a strange
    star could not hold itself together;
  * 2-flavour (ud) matter must NOT be, or ordinary nuclei would decay into it and
    there would be no ordinary matter to observe.

Both boundaries are curves B^1/4(alpha_s) obtained by root-finding e/n_B = 930
MeV, and both are drawn on the parameter-plane figures as the region within
which a strange star is admissible at all.

Note e/n_B and mu_B coincide at P = 0 (from e = -P + mu_B n_B), so "the Witten
energy per baryon" and "mu_B at zero pressure" are the same number -- the
notebook computed it twice under two names.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy.optimize import brentq

from nucleation.analysis.filters import (unpaired_eos_at_params, zero_crossing,
                                         ud_eps_per_nB)

# The neutron mass minus a nominal binding: below this, matter is more bound
# than the most bound nucleus and is therefore absolutely stable.
E_PER_BARYON_STABLE = 930.0     # MeV


def energy_per_baryon_at_P0(alpha, B4, cfg, m_s=None):
    """e/n_B [MeV] at P = 0 for UNPAIRED 3-flavour beta-equilibrated SQM.

    This is the Witten energy per baryon. Equal to mu_B at P = 0, since
    e = -P + mu_B n_B.

    Returns NaN when the EoS has no P = 0 crossing on ``cfg.n_B_grid`` -- that
    is a real answer (this parameter set has no self-bound surface), not a
    failure, so callers should treat NaN as "not stable" rather than retry.

    ``m_s`` overrides the strange-quark mass in ``cfg`` for m_s scans.
    """
    if m_s is not None:
        cfg = dataclasses.replace(cfg, m_s=m_s)
    P, e, _mu, ok = unpaired_eos_at_params(alpha, B4, cfg)
    n, P, e = cfg.n_B_grid[ok], P[ok], e[ok]
    if n.size < 5:
        return np.nan
    cr = zero_crossing(n, P)
    if cr is None:
        return np.nan
    j, fr = cr
    n_P0 = n[j] + fr * (n[j + 1] - n[j])
    e_P0 = e[j] + fr * (e[j + 1] - e[j])
    return float(e_P0 / n_P0) if n_P0 > 0 else np.nan


def B4_at_energy(scalar_fn, alpha, lo, hi, target=E_PER_BARYON_STABLE, xtol=0.05):
    """B^1/4 where ``scalar_fn(alpha, B4) == target``, or NaN if unbracketed.

    e/n_B rises monotonically with the bag constant (more vacuum pressure costs
    energy), so there is one clean root and a bracket test is enough -- no need
    to scan for multiple crossings.
    """
    def f(b):
        return scalar_fn(alpha, b) - target

    f_lo, f_hi = f(lo), f(hi)
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0:
        return np.nan
    return brentq(f, lo, hi, xtol=xtol)


def stability_curve(scalar_fn, alpha_grid, B4_lo, B4_hi,
                    target=E_PER_BARYON_STABLE):
    """B^1/4(alpha_s) along which ``scalar_fn`` equals `target`.

    NaN wherever no root brackets in [B4_lo, B4_hi]; pass the result through
    ``resample_curve`` before plotting on a finer alpha grid.
    """
    return np.array([B4_at_energy(scalar_fn, float(a), B4_lo, B4_hi, target)
                     for a in np.asarray(alpha_grid, dtype=float)])


def resample_curve(curve, src_alpha, target_alpha, fallback):
    """Put a coarse stability curve onto a finer alpha grid.

    Root-finding the boundary is expensive, so it is evaluated on a coarse alpha
    grid and interpolated for drawing. An all-NaN curve (no crossing anywhere in
    range) collapses to a constant `fallback`, which keeps the shaded region
    well defined instead of vanishing.
    """
    target_alpha = np.asarray(target_alpha, dtype=float)
    if curve is None:
        return np.full_like(target_alpha, fallback, dtype=float)
    curve = np.asarray(curve, dtype=float)
    m = np.isfinite(curve)
    if m.sum() < 2:
        return np.full_like(target_alpha, fallback, dtype=float)
    return np.interp(target_alpha, np.asarray(src_alpha, dtype=float)[m], curve[m])


def two_flavour_energy_at_P0(alpha, B4, cfg):
    """e/n_B [MeV] at P = 0 for 2-flavour (ud + e) matter.

    The other half of the Witten window: this one must stay ABOVE 930 MeV.
    Thin alias over ``filters.ud_eps_per_nB`` so both boundaries are reached
    through the same vocabulary; the implementation is not duplicated.
    """
    return ud_eps_per_nB(alpha, B4, cfg)
