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


# PRD / REVTeX two-column geometry [inches]. A figure MUST be built at the
# width it will occupy on the page and included at 1:1 (no \includegraphics
# width= rescaling, which would shrink the fonts). Then the pt sizes below land
# as-is on paper. Single column = \columnwidth, full width = \textwidth span.
PRD_COL_W  = 3.375
PRD_FULL_W = 7.0


def set_paper_style():
    """Publication rcParams: CMU Serif + Computer-Modern math (matches a LaTeX
    two-column layout), inward ticks on all four sides, frame-less legends,
    300 dpi tight saves. All text is 10 pt = PRD body-text size, so labels,
    ticks and legends match the surrounding paper when the figure is placed at
    its true width (see PRD_COL_W / PRD_FULL_W)."""
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['CMU Serif', 'STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',                 # CM math to match CMU Serif
        'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 10,
        'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 10, 'legend.frameon': False,
        'lines.linewidth': 1.8, 'axes.linewidth': 0.9,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.minor.visible': True, 'ytick.minor.visible': True,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'axes.unicode_minus': False,              # CMU Serif lacks U+2212 minus
    })


def panel_label(ax, lab, corner='upper left'):
    """Put an (a)/(b)/... tag in a panel corner. `corner` is a vertical word
    ('upper'/'lower') optionally followed by a horizontal one ('left'/'right');
    horizontal defaults to 'left'. e.g. 'upper', 'lower right', 'upper right'."""
    words = corner.split()
    y, va = (0.945, 'top') if words[0] == 'upper' else (0.055, 'bottom')
    right = len(words) > 1 and words[1] == 'right'
    x, ha = (0.955, 'right') if right else (0.045, 'left')
    ax.text(x, y, lab, transform=ax.transAxes,
            fontweight='bold', va=va, ha=ha)
