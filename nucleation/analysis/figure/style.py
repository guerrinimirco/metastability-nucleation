"""Shared figure styling for the nucleation analysis plots.

One place for the publication rcParams, the panel-letter helper and the
project colour palette, so every notebook figure looks the same and a style
tweak edits a single file.
"""
import matplotlib as mpl

# Project palette (muted tones that survive both screen and print).
STANDARD_COLORS = {
    'Red':     (0.8,  0.25, 0.33),
    'Green':   (0.24, 0.6,  0.44),
    'Blue':    (0.24, 0.6,  0.8),
    'Gray':    (0.4,  0.4,  0.4),
    'Cyan':    (0.1,  0.6,  0.6),
    'Magenta': (0.6,  0.24, 0.6),
    'Yellow':  (0.95, 0.75, 0.1),
    'Brown':   (0.6,  0.4,  0.2),
    'Orange':  (0.9,  0.4,  0.0),
    'Pink':    (0.9,  0.4,  0.6),
    'Purple':  (0.5,  0.35, 0.65),
}


def set_paper_style():
    """Publication rcParams: CMU Serif + Computer-Modern math (matches a LaTeX
    two-column layout), inward ticks on all four sides, frame-less legends,
    300 dpi tight saves."""
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['CMU Serif', 'STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',                 # CM math to match CMU Serif
        'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 13,
        'xtick.labelsize': 13, 'ytick.labelsize': 13,
        'legend.fontsize': 12, 'legend.frameon': False,
        'lines.linewidth': 1.8, 'axes.linewidth': 0.9,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.minor.visible': True, 'ytick.minor.visible': True,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'axes.unicode_minus': False,              # CMU Serif lacks U+2212 minus
    })


def panel_label(ax, lab, corner='upper'):
    """Put an (a)/(b)/... tag in a panel corner ('upper' or 'lower' left)."""
    y, va = (0.945, 'top') if corner == 'upper' else (0.055, 'bottom')
    ax.text(0.045, y, lab, transform=ax.transAxes,
            fontweight='bold', va=va, ha='left')
