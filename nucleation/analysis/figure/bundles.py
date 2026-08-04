"""
Drawing a bundle of many replayed curves
========================================

A parameter scan leaves hundreds to thousands of admissible (alpha_s, B^1/4,
Delta_0) cells, each with its own M-R and P(mu_B) curve. Over-plotting them all
is an ink blot; plotting one "representative" curve throws away the spread that
is the actual result.

So: bin the curves by sigma_crit, and draw each bin as a median line with a
p16-p84 envelope. The reader sees both the typical star and how much the
prediction moves across the viable parameter space, with colour still carrying
sigma_crit.
"""
from __future__ import annotations

import numpy as np


def resample_profiles(curves, xkey, ykey, xgrid, stable=False):
    """y(xgrid) for every curve in the bundle, NaN outside each curve's own span.

    A percentile across curves only means something if they are all evaluated at
    the same abscissae, so every curve is first interpolated onto one common
    grid. NaN outside its span (rather than an edge value) is what stops a short
    curve from silently propping up the envelope where it does not reach.

    `stable=True` clips each curve to its stable branch (up to M_max) first --
    that is what makes R(M) single-valued, and therefore interpolable, on the
    M-R plane.

    `curves` is a sequence of (curve_dict, sigma_crit) pairs, as returned by
    ``analysis.replay.replay_accepted``.
    """
    out = []
    for c, _ in curves:
        x = np.asarray(c[xkey], dtype=float)
        y = np.asarray(c[ykey], dtype=float)
        if stable:
            k = int(np.argmax(np.asarray(c['M']))) + 1
            x, y = x[:k], y[:k]
        order = np.argsort(x)                 # np.interp requires increasing xp
        out.append(np.interp(xgrid, x[order], y[order],
                             left=np.nan, right=np.nan))
    return np.asarray(out)


def band(ax, profiles, xgrid, colour, label=None, *, swap=False, min_n=3,
         alpha=0.30, lw=1.7, zorder=2):
    """Median + p16-p84 envelope of a bundle of resampled profiles.

    Grid columns backed by fewer than `min_n` curves are dropped: at the
    high-mass end only a few members of a bin still exist, and a percentile over
    one or two curves is noise drawn with the authority of a band.

    `swap=True` puts the independent variable on the y-axis -- the M-R panel
    interpolates R at fixed M, but plots R horizontally.
    """
    profiles = np.asarray(profiles)
    ok = np.isfinite(profiles).sum(axis=0) >= min_n
    if not ok.any():
        return
    lo, md, hi = (np.nanpercentile(profiles[:, ok], p, axis=0)
                  for p in (16, 50, 84))
    g = np.asarray(xgrid)[ok]
    if swap:
        ax.fill_betweenx(g, lo, hi, color=colour, alpha=alpha, lw=0, zorder=zorder)
        ax.plot(md, g, color=colour, lw=lw, zorder=zorder + 1, label=label)
    else:
        ax.fill_between(g, lo, hi, color=colour, alpha=alpha, lw=0, zorder=zorder)
        ax.plot(g, md, color=colour, lw=lw, zorder=zorder + 1, label=label)


def quantile_bins(curves, n_bins, cmap, norm):
    """Split a bundle into equal-COUNT sigma_crit bins.

    Equal-count rather than equal-width: sigma_crit is far from uniform over the
    parameter plane, so equal-width bins would leave some holding a handful of
    cells and others holding most of them -- and a band drawn over three curves
    next to one drawn over three hundred invites the wrong comparison.

    Each bin's colour is taken at its MEDIAN sigma_crit on the same `norm` as
    the heatmap, so a band's colour means the same sigma_crit everywhere in the
    figure.

    Returns [(lo, hi, members, colour), ...].
    """
    s = np.array([sc for _, sc in curves], dtype=float)
    edges = np.quantile(s, np.linspace(0, 1, n_bins + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)      # make the top edge inclusive
    out = []
    for k in range(n_bins):
        m = (s >= edges[k]) & (s < edges[k + 1])
        if m.any():
            members = [curves[i] for i in np.flatnonzero(m)]
            out.append((float(edges[k]), float(edges[k + 1]), members,
                        cmap(norm(np.median(s[m])))))
    return out


def summary_points(curves, radius_def='R_1.4'):
    """(M_max, R) summary point per curve, for the scatter panel.

    `radius_def` picks WHICH radius, and they answer different questions:
      'R_Mmax' radius of the maximum-mass star -- the compactness endpoint
      'R_1.4'  radius of the 1.4 M_sun star    -- what NICER-type measurements
               actually pin, so this is the default
      'R_max'  largest radius anywhere on the curve

    R is NaN when the branch never reaches 1.4 M_sun (for 'R_1.4').
    """
    out = []
    for c, _ in curves:
        M = np.asarray(c['M'], dtype=float)
        R = np.asarray(c['R'], dtype=float)
        k = int(np.argmax(M))
        if radius_def == 'R_Mmax':
            r = R[k]
        elif radius_def == 'R_max':
            r = np.nanmax(R)
        elif radius_def == 'R_1.4':
            r = np.interp(1.4, M[:k + 1], R[:k + 1], left=np.nan, right=np.nan)
        else:
            raise ValueError(f"unknown radius_def: {radius_def!r}")
        out.append((float(M[k]), float(r)))
    return np.asarray(out)
