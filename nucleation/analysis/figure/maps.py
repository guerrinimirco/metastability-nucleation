"""
Parameter-plane map layers
==========================

The layers that make up a (B^1/4, Delta_0) or (B^1/4, alpha_s) figure: the
sigma_crit heatmap itself, iso-value contours over it, the outlines of the
regions each acceptance filter excludes, and the diverging map used for
differences between two scans.

Why these live here rather than in the notebook: the rejection outlines had
THREE implementations in the notebook (the paper figure, the unpaired-matter
variant and the scan's quick-look plot), with different colours, different
labels and a median-vs-mean centroid. A reader comparing two panels was
comparing two conventions. There is now one.

Colour conventions, fixed once:
  * sigma_crit / any positive scalar field -> viridis (perceptually uniform)
  * a signed DIFFERENCE                    -> RdBu_r, symmetric about zero, so
                                              white always means "no change"
  * no data / not viable                   -> 0.85 grey, never white, because on
                                              a diverging map white is a VALUE
  * contours over viridis                  -> white (iso-sigma) and light grey
                                              (iso-M_max); warm hues clash with
                                              the vermillion rejection edge
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from eos.general.figure_style import OKAB
from nucleation.analysis.config import REASON_CODE

# Which excluded region is drawn how.
#   (reason code, colour, label, label rotation [deg], left-panel-only)
# `left_only` marks a zone whose label would collide in every panel but the
# first of a multi-panel row -- the outline is still drawn, only the text is
# suppressed.
REASON_STYLE = [
    (REASON_CODE['mmax'],        OKAB['vermillion'],
     r'$M_{\rm QS}^{\rm max}<2M_\odot$', 45, False),
    (REASON_CODE['witten'],      OKAB['green'],
     'SQM not abs.\nstable', 0, False),
    (REASON_CODE['twoflavor'],   OKAB['grey'],
     '2 flav.\nstability', 90, True),
    (REASON_CODE['rehadr'],      OKAB['blue'],
     're-hadr.', 0, False),
    (REASON_CODE['rehad_quasi'], OKAB['purple'],
     'quasi-\nre-hadr.', 0, False),
]

# Critical-radius regime of the droplet at the barrier peak, codes 0/1/2.
REGIME_STYLE = [
    (0, OKAB['orange'], r'$R_*=R^{\rm unp}_*$'),
    (1, OKAB['purple'], r'$R_*=R_{\Delta}$'),
    (2, OKAB['sky'],    r'$R_*=R^{\rm CFL}_*$'),
]

ISO_SIGMA_COLOUR = 'white'        # iso-sigma_crit over viridis
ISO_MASS_COLOUR = 'lightgray'     # iso-M_max over viridis
NO_DATA_GREY = '0.85'
_LABEL_FS = 8.5
_LABEL_FS_EMPH = 9.5              # the M_max edge, one size up


def _zone_centroid(X, Y, mask):
    """Median (x, y) of the cells a boolean mask selects.

    Accepts X/Y either as the 1-D axis vectors or as the 2-D meshes, because
    matplotlib's contour/pcolormesh take both and a caller should not have to
    remember which form this expects. Median rather than mean: these zones are
    L-shaped or crescent-shaped often enough that their mean lands outside the
    region entirely, putting the label on empty plane.
    """
    rows, cols = np.where(mask)
    X, Y = np.asarray(X), np.asarray(Y)
    xs = X[rows, cols] if X.ndim == 2 else X[cols]
    ys = Y[rows, cols] if Y.ndim == 2 else Y[rows]
    return float(np.median(xs)), float(np.median(ys))


def sigma_map(ax, X, Y, field, *, cmap='viridis', vmin=None, vmax=None,
              bad=NO_DATA_GREY, shading='nearest'):
    """Heatmap of a positive scalar over the parameter plane.

    Non-viable cells are masked and drawn `bad` grey rather than left white --
    white is a data value on any diverging companion panel, and a reader should
    never have to guess whether a pale cell means "small" or "not computed".
    Returns the mesh, for a colorbar.
    """
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(bad)
    return ax.pcolormesh(X, Y, np.ma.masked_invalid(field), cmap=cm,
                         vmin=vmin, vmax=vmax, shading=shading)


def diverging_map(ax, X, Y, D, vlim, *, cmap='RdBu_r', bad=NO_DATA_GREY,
                  shading='nearest'):
    """Signed-difference panel, symmetric about zero.

    Symmetry is not cosmetic: it puts 0 on the colormap's neutral midpoint, so
    white reliably means "no change". Missing cells are grey for the same
    reason -- on a diverging map, white is taken.
    """
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(bad)
    return ax.pcolormesh(X, Y, np.ma.masked_invalid(D), cmap=cm,
                         vmin=-vlim, vmax=+vlim, shading=shading)


def symmetric_vlim(fields, pct=99.0, override=None):
    """Shared colour limit for a set of difference maps.

    The `pct`-th percentile of |Delta| rather than the max, so a handful of
    outlier cells cannot flatten every other panel into featureless white.
    Pass `override` to pin it by hand across a figure series.
    """
    if override is not None:
        return float(override)
    if not fields:
        return 1.0
    v = np.concatenate([np.abs(np.asarray(f)[np.isfinite(f)]).ravel()
                        for f in fields])
    v = v[v > 0]
    return float(np.percentile(v, pct)) if v.size else 1.0


def iso_lines(ax, X, Y, field, levels, colour, fmt=None, *, fontsize=_LABEL_FS,
              linewidths=1.0, linestyles='-', inline=True, zorder=5, **kw):
    """Labelled iso-value contours over a heatmap.

    Used for iso-sigma_crit (white) and iso-M_max (light grey): the heatmap
    shows the field, the contours let a reader read a number off it.
    """
    cs = ax.contour(X, Y, np.ma.masked_invalid(field), levels=levels,
                    colors=[colour], linewidths=linewidths,
                    linestyles=linestyles, zorder=zorder, **kw)
    if fmt is not None:
        ax.clabel(cs, inline=inline, fontsize=fontsize, fmt=fmt)
    return cs


def reject_outlines(ax, X, Y, reason, *, spec=REASON_STYLE, labels=True,
                    panel_index=0, annotate_offsets=None, zorder=4):
    """Outline every excluded region, in its own colour, with an inline label.

    Outlines and no fill: the regions overlap in places, and filling them would
    hide the sigma_crit map underneath, which is the actual result.

    Physics: `reason` holds the integer code of the FIRST filter each cell
    failed (REASON_CODE). The codes are nested by how cheap the filter is, so
    contouring this field at half-integer levels draws each filter's pass edge.

    Mechanics:
      panel_index      : index within a multi-panel row; zones marked left-only
                         in `spec` are labelled only at index 0.
      annotate_offsets : {reason_code: (dx_pt, dy_pt)} to pull a label out of a
                         thin zone with a leader arrow back to it. Defaults to
                         doing this for 're-hadr.', whose band is too narrow to
                         hold its own text.
    """
    reason = np.asarray(reason)
    if annotate_offsets is None:
        annotate_offsets = {REASON_CODE['rehadr']: (42, 0)}

    for code, colour, label, rotation, left_only in spec:
        m = reason == code
        if not m.any():
            continue
        ax.contour(X, Y, m.astype(float), levels=[0.5], colors=[colour],
                   linewidths=1.8, zorder=zorder)
        if not labels or (left_only and panel_index > 0):
            continue
        # Median, not mean: these zones are L-shaped or crescent-shaped often
        # enough that their mean lands outside the region entirely.
        bc, dc = _zone_centroid(X, Y, m)
        fs = _LABEL_FS_EMPH if code == REASON_CODE['mmax'] else _LABEL_FS
        if code in annotate_offsets:
            ax.annotate(label, xy=(bc, dc), xycoords='data',
                        xytext=annotate_offsets[code], textcoords='offset points',
                        color=colour, fontsize=fs, fontweight='bold',
                        ha='left', va='center', zorder=8,
                        arrowprops=dict(arrowstyle='->', color=colour, lw=0.8,
                                        mutation_scale=6, shrinkB=2))
        else:
            ax.text(bc, dc, label, color=colour, fontsize=fs, fontweight='bold',
                    ha='center', va='center', rotation=rotation, zorder=7)


def regime_outlines(ax, X, Y, regime, *, spec=REGIME_STYLE, labels=True,
                    zorder=4):
    """Outline the critical-radius regime zones (unpaired / gap / CFL).

    Which length scale sets R_* -- the unpaired barrier peak, the CFL coherence
    radius R_Delta, or the CFL peak -- changes across the plane, and the
    sigma_crit map alone does not show where.
    """
    regime = np.asarray(regime)
    for code, colour, label in spec:
        m = regime == code
        if not m.any():
            continue
        ax.contour(X, Y, m.astype(float), levels=[0.5], colors=[colour],
                   linewidths=1.2, linestyles='--', zorder=zorder)
        if labels:
            bc, dc = _zone_centroid(X, Y, m)
            ax.text(bc, dc, label, color=colour, fontsize=_LABEL_FS,
                    fontweight='bold', ha='center', va='center', zorder=7)
