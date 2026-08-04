"""
Annotation primitives for stellar-sequence panels
=================================================

Marking WHICH star on a curve, not the curve itself. A sequence plotted against
baryonic mass or central density is unreadable without saying where the
1.4 M_sun star sits and where the branch ends, so these three helpers appear on
almost every panel of the paper.

Kept out of the notebook because the marker geometry (sizes, white outlines,
z-order, label offsets in points) has to agree across figures for the reader to
learn it once; kept out of `eos` because it knows about this paper's TOV column
convention.
"""
from __future__ import annotations

import numpy as np

from nucleation.analysis.stellar import TOV_COL, stable_branch

# Gravitational masses [M_sun] marked on every sequence. ONE source of truth --
# the notebook carried two copies of this tuple under different names and they
# could silently drift apart between figures.
MASS_MARKS_DEFAULT = (1.0, 1.2, 1.4, 1.6)

# Marker geometry, shared so the reader learns it once:
#   dot  = a star of a given gravitational mass
#   star = the branch tip, i.e. M_max
_DOT_KW = dict(marker='o', ms=5.5, mec='white', mew=0.7, zorder=7)
_TIP_KW = dict(marker='*', ms=13, mec='white', mew=0.6, zorder=7)
_LABEL_FS = 8.5


def mass_marks(ax, seq, yvals, color, *, masses=MASS_MARKS_DEFAULT,
               xcol='M_B', dot_off=(0, 7, 'center', 'bottom'),
               star_off=(6, 0, 'left', 'center'), label=True, dot_arrow=False):
    """Mark a sequence at the given gravitational masses, plus its M_max tip.

    Physics: panels whose x-axis is baryonic mass (or central density) cannot be
    read directly, because the observable label on a star is its GRAVITATIONAL
    mass. This interpolates each requested M onto the curve and marks it there.
    M_grav is monotone along the stable branch, so np.interp keys off it safely.

    Mechanics:
      seq       : TOV sequence array (columns per TOV_COL).
      yvals     : the y quantity already evaluated on the same rows as `seq`.
      xcol      : which column is the x-axis ('M_B' by default, 'n_Bc' also used).
      dot_off / star_off : (dx_pt, dy_pt, ha, va) nudges for the labels, in
                  POINTS, so they stay put when the axis limits change.
      label     : False draws markers only (for a panel that is already busy).
      dot_arrow : draw a leader arrow from label to marker, for crowded panels.
    """
    xcol = TOV_COL.get(xcol, xcol)
    seq = np.asarray(seq)
    Mg, Xv = seq[:, TOV_COL['M']], seq[:, xcol]
    yvals = np.asarray(yvals)
    m = np.isfinite(Mg) & np.isfinite(Xv) & np.isfinite(yvals)
    Mg, Xv, yv = Mg[m], Xv[m], yvals[m]
    if Mg.size < 2:
        return

    dx, dy, dha, dva = dot_off
    for mt in masses:
        if Mg.min() <= mt <= Mg.max():
            xb, yb = np.interp(mt, Mg, Xv), np.interp(mt, Mg, yv)
            ax.plot(xb, yb, color=color, **_DOT_KW)
            if label:
                ax.annotate(f"{mt:g}", (xb, yb), textcoords='offset points',
                            xytext=(dx, dy), ha=dha, va=dva,
                            fontsize=_LABEL_FS, color=color, zorder=8,
                            arrowprops=(dict(arrowstyle='->', color=color,
                                             lw=0.7, shrinkA=1, shrinkB=2)
                                        if dot_arrow else None))

    ax.plot(Xv[-1], yv[-1], color=color, **_TIP_KW)          # the M_max tip
    if label:
        sx, sy, sha, sva = star_off
        ax.annotate(r"$M_{\max}$", (Xv[-1], yv[-1]),
                    textcoords='offset points', xytext=(sx, sy),
                    ha=sha, va=sva, fontsize=_LABEL_FS, color=color, zorder=8,
                    arrowprops=(dict(arrowstyle='->', color=color, lw=0.7,
                                     shrinkA=1, shrinkB=2)
                                if dot_arrow else None))


def label_with_arrows(ax, points, label, color, dy=5.0, fontsize=_LABEL_FS):
    """One label with a leader arrow to EACH of several markers.

    Used where the same physical star appears twice in a panel -- once at fixed
    gravitational mass and once baryon-number-matched -- and labelling both
    would repeat the text. Every arrow starts from the SAME anchor, `dy` above
    the highest marker, and the text is drawn separately so re-rendering never
    shifts an arrow tail. Non-finite points are skipped.
    """
    pts = [(x, y) for x, y in points if np.isfinite(x) and np.isfinite(y)]
    if not pts:
        return
    xt = float(np.mean([p[0] for p in pts]))
    yt = max(p[1] for p in pts) + dy
    ax.text(xt, yt, label, ha='center', va='bottom', fontsize=fontsize,
            color=color, zorder=8)
    arrow = dict(arrowstyle='->', color=color, lw=0.6, shrinkA=2, shrinkB=3,
                 mutation_scale=6)                   # small heads, common tail
    for p in pts:
        ax.annotate('', xy=p, xytext=(xt, yt), textcoords='data',
                    arrowprops=arrow)


def isentrope_mass_markers(ax, tov_seq, T_of_nB, color, *, n_sat,
                           masses=MASS_MARKS_DEFAULT, fontsize=_LABEL_FS):
    """Mark where given gravitational masses sit on a (n_B, T) isentrope.

    Ties the nucleation-condition plane back to actual stars: a T_nuc(n_B) curve
    means little until you can see that a 1.4 M_sun proto-neutron star sits at
    THIS density and THIS temperature on it.

    `T_of_nB` is the isentrope temperature as a function of central density
    (an interpolator); densities are drawn in units of `n_sat`.
    """
    seq = stable_branch(np.asarray(tov_seq))
    Mg, nBc = seq[:, TOV_COL['M']], seq[:, TOV_COL['n_Bc']]
    m = np.isfinite(Mg) & np.isfinite(nBc)
    Mg, nBc = Mg[m], nBc[m]
    if Mg.size < 2:
        return
    for mt in masses:
        if not (Mg.min() <= mt <= Mg.max()):
            continue
        nb = float(np.interp(mt, Mg, nBc))
        T = float(T_of_nB(nb))
        if not np.isfinite(T):
            continue
        ax.plot(nb / n_sat, T, color=color, **_DOT_KW)
        ax.annotate(f"{mt:g}", (nb / n_sat, T), textcoords='offset points',
                    xytext=(0, 7), ha='center', va='bottom',
                    fontsize=fontsize, color=color, zorder=8)
