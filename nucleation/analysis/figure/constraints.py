"""Observational M-R constraint overlays (NICER / HESS posteriors + mass bands).

The 68% / 95% credible contours are precomputed offline (by
``eos/plot/compute_contours.py`` in the sibling ``eos`` project) and stored as
CSV files, so plotting never rebuilds KDEs. ``add_observational_constraints``
draws them behind the model curves on any M-R axes.
"""
import csv
from pathlib import Path

import numpy as np

from .style import STANDARD_COLORS

# 2D M-R posteriors: CSV basename -> (full label, colour, label-anchor).
# Soft fills (no border) so they read as background; anchor = (dx, dy, ha, va)
# offsets the inline label from the blob centroid [data units] -- tune to taste.
# anchor = (dx, dy, ha, va): dx/dy shift the label from the blob centroid in
# DATA units (dx in km, dy in M_sun); ha/va are its text alignment. dy>0 + 'bottom'
# = above the blob, dy<0 + 'top' = below it. MR_LABEL_FS sets the font size.
MR_LABEL_FS = 10          # match the J0952-0607 mass-band label
MR_CONSTRAINTS = {
    "J0030": ("PSR J0030+0451", STANDARD_COLORS['Orange'],  (0.0,  -0.1, 'center',  'center')),     
    "J0740": ("PSR J0740+6620", STANDARD_COLORS['Blue'],    (0.0,  0.0, 'center', 'center')),
    "J0614": ("PSR J0614-3329", STANDARD_COLORS['Green'],   (-0.3, 0.0, 'center', 'center')),     
    "HESS":  ("HESS J1731-347", STANDARD_COLORS['Magenta'], (0.0, -0.2, 'center', 'center')),
}
# 1D mass-only measurements -> horizontal bands. Only J0952-0607 is shown as a
# band (J0740 is already an M-R contour). Gold fill + darker brown line/label.
MASS_LABELS = {"J0952-0607": "PSR J0952-0607"}
MASS_COLORS = {"J0952-0607": STANDARD_COLORS['Yellow']}
MASS_TEXT = {"J0952-0607": STANDARD_COLORS['Brown']}


def _smooth_closed(x, y, frac=0.04):
    """Low-pass a *closed* contour so a jagged posterior boundary reads as a
    smooth blob (like published M-R figures). Periodic moving average: wrap-pad
    the loop, box-filter x(t) and y(t), unwrap. ``frac`` = window as a fraction
    of the point count (bigger = smoother, but shrinks convex bulges more)."""
    n = len(x)
    if n < 8:
        return x, y                                   # too few points to smooth
    w = max(3, int(n * frac) | 1)                     # odd window
    k = np.ones(w) / w
    xp, yp = np.r_[x[-w:], x, x[:w]], np.r_[y[-w:], y, y[:w]]   # periodic pad
    return np.convolve(xp, k, 'same')[w:-w], np.convolve(yp, k, 'same')[w:-w]


def add_observational_constraints(ax, contour_dir, show_mass_bands=True,
                                  inline_labels=False):
    """Overlay precomputed observational constraints *behind* the model curves.

    - 2D M-R posteriors: 95% + 68% credible regions as soft pastel fills (no
      border). With inline_labels=False a boundary line carries a legend entry;
      with inline_labels=True the full source name is written near the blob.
    - 1D mass-only measurements: horizontal ``axhspan`` bands (68% + 95%),
      annotated at the left edge.
    Everything is drawn at low zorder (0-1) so the model curves stay on top.

    Parameters
    ----------
    ax : matplotlib Axes
    contour_dir : str or Path
        Directory holding the ``<key>_68.csv`` / ``<key>_95.csv`` contour
        files and ``mass_bounds.csv``. Missing files are silently skipped.
    """
    contour_dir = Path(contour_dir)
    for key, (label, colour, anchor) in MR_CONSTRAINTS.items():
        f95, f68 = contour_dir / f"{key}_95.csv", contour_dir / f"{key}_68.csv"
        if not f95.exists():
            continue
        R95, M95 = _smooth_closed(*np.loadtxt(f95, delimiter=",", skiprows=1).T)
        ax.fill(R95, M95, color=colour, alpha=0.20, lw=0, zorder=0)      # 2 sigma light
        if f68.exists():
            R68, M68 = _smooth_closed(*np.loadtxt(f68, delimiter=",", skiprows=1).T)
            ax.fill(R68, M68, color=colour, alpha=0.40, lw=0, zorder=1)  # 1 sigma darker
            Rc, Mc = R68, M68
        else:
            Rc, Mc = R95, M95
        if inline_labels:
            if anchor is not None:                    # label once per source
                dx, dy, ha, va = anchor
                ax.text(Rc.mean() + dx, Mc.mean() + dy, label, color=colour,
                        fontsize=MR_LABEL_FS, fontweight='bold', ha=ha, va=va, zorder=3,
                        clip_on=True)
        else:                                         # legend path: draw boundary
            ax.plot(Rc, Mc, color=colour, lw=1.2, zorder=1, label=label)

    if show_mass_bands and (contour_dir / "mass_bounds.csv").exists():
        with open(contour_dir / "mass_bounds.csv") as fh:
            rows = list(csv.DictReader(fh))
        for name, disp in MASS_LABELS.items():
            by_lvl = {r["level"]: r for r in rows if r["name"] == name}
            colour = MASS_COLORS.get(name, "gray")
            txt_c = MASS_TEXT.get(name, colour)       # darker tone for line/label
            for lvl, a in (("95", 0.10), ("68", 0.18)):   # 2 sigma lighter
                r = by_lvl[lvl]
                ax.axhspan(float(r["lower"]), float(r["upper"]),
                           color=colour, alpha=a, lw=0, zorder=0)
            # x in axes fraction (blended transform) -> pinned to the left edge,
            # independent of the current xlim, so it never escapes the panel.
            ax.text(0.02, float(by_lvl["68"]["upper"]), " " + disp,
                    transform=ax.get_yaxis_transform(), ha="left", va="bottom",
                    fontsize=10, fontweight='bold', color=txt_c, zorder=3,
                    clip_on=True)
