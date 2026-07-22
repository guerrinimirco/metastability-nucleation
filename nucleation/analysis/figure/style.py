"""Shared figure styling for the nucleation analysis plots.

One place for the publication rcParams, the panel-letter helper and the
project colour palette, so every notebook figure looks the same and a style
tweak edits a single file.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

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
# Height cap for a full-width four-panel (2×2) figure: keep it within ~half the
# PRD text height (~9 in) so two figures fit per page. Panels go landscape at
# this height; fonts stay 10 pt (PRD body). 1×2 figures use half of this.
PRD_2X2_H  = 4.5
# Mid-size square figure: wider than one column but kept under half the PRD text
# height (~9 in) so two stack on a page. Used by paper_grid(mode='centered').
PRD_CENTERED_W = 4.75


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


def paper_grid(layout='2x2', mode='single', placeholder=True, square=True):
    """Build a square-panelled PRD figure with the publication style applied.

    Physics: each panel is forced to an exact 1:1 box aspect so, e.g., an M-R
    plane or a phase diagram isn't visually stretched. The figure is built at the
    true page width it will occupy (no LaTeX rescaling), so the 10 pt fonts from
    set_paper_style() land as PRD body text.

    Mechanics:
      layout : '2x2' (four panels) or '1x2' (a two-panel row = the 2×2 top row).
      mode   : sets the figure width W -
                 'single'   -> W = PRD_COL_W     (3.375", one column)
                 'centered' -> W = PRD_CENTERED_W (4.75", mid-size)
                 'double'   -> W = PRD_FULL_W     (7.0", two columns)
               2×2 is (W, W); 1×2 is (W, W/2) so its panels match a 2×2 top row.
      square : True (default) forces an exact 1:1 box on every panel — crisp
               squares, but because the figsize is fixed the slack dimension
               leaves a centring margin (harmless: savefig(bbox_inches='tight')
               crops it). False drops the box constraint so panels stretch to
               fill the figure (quasi-square, zero centring whitespace) — use it
               when the on-screen/inline gap bothers you and near-square is fine.

    Panels do NOT share axes: every one carries its own x/y axis. With
    placeholder=True (default) each panel gets dummy x/y labels and a title so
    the bare layout reads as a template; pass placeholder=False for a real
    figure that sets its own labels (and tags panels with panel_label instead of
    a title). Returns (fig, axes) with axes a 2-D ndarray; unpack as
    ((axA, axB), (axC, axD)) or (axA, axB).
    """
    widths = {'single': PRD_COL_W, 'centered': PRD_CENTERED_W, 'double': PRD_FULL_W}
    if mode not in widths:
        raise ValueError(f"mode must be one of {sorted(widths)}, got {mode!r}")
    if layout not in ('2x2', '1x2'):
        raise ValueError(f"layout must be '2x2' or '1x2', got {layout!r}")

    set_paper_style()
    w = widths[mode]
    nrows = 2 if layout == '2x2' else 1
    figsize = (w, w) if layout == '2x2' else (w, w / 2)   # square panels either way

    # squeeze=False -> axes always 2-D, so callers unpack uniformly.
    # sharex/sharey default False: independent axes per the request.
    fig, axes = plt.subplots(nrows, 2, figsize=figsize, layout='constrained',
                             squeeze=False)
    # Tighten the constrained-layout padding so panels sit closer together and to
    # the figure edge (less whitespace); edit here to rescale for all figures.
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)
    for ax in axes.flat:
        if square:
            ax.set_box_aspect(1)    # exact square box, independent of label room
        if placeholder:
            # Dummy labels/title so the empty template looks complete - real
            # figures pass placeholder=False and set their own.
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$y$')
            ax.set_title('panel')
    return fig, axes


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


if __name__ == '__main__':
    # ponytail: smallest check that the sizing/squareness contract holds.
    mpl.use('Agg')
    _W = {'single': 3.375, 'centered': 4.75, 'double': 7.0}
    for _lay in ('2x2', '1x2'):
        for _m, _w in _W.items():
            _fig, _ax = paper_grid(_lay, _m)
            _fw, _fh = _fig.get_size_inches()
            assert abs(_fw - _w) < 1e-9
            assert abs(_fh - (_w if _lay == '2x2' else _w / 2)) < 1e-9
            assert all(abs(a.get_box_aspect() - 1) < 1e-9 for a in _ax.flat)
            plt.close(_fig)
    print('paper_grid self-check ok')
