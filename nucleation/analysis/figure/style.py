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


def set_paper_style(fontsize=10, labelsize=None, legendsize=None, rc=None):
    """Publication rcParams: CMU Serif + Computer-Modern math (matches a LaTeX
    two-column layout), inward ticks on all four sides, frame-less legends,
    300 dpi tight saves. All text defaults to 10 pt = PRD body-text size, so
    labels, ticks and legends match the surrounding paper when the figure is
    placed at its true width (see PRD_COL_W / PRD_FULL_W).

      fontsize  : base text size in pt (title + anything not overridden below).
                  Keep 10 for a figure placed at true column width; only raise it
                  if the figure will be down-scaled in \\includegraphics.
      labelsize : size of the axis names AND the x/y tick numbers. Defaults to
                  `fontsize`; set separately to shrink/grow axis text alone.
      legendsize: size of the legend text. Defaults to `fontsize`; set separately
                  (journals often run legends a touch smaller than axis labels).
      rc        : optional dict of extra rcParams to override on top, e.g.
                  rc={'axes.linewidth': 0.5}. Escape hatch for one-off tweaks.
    """
    labelsize = fontsize if labelsize is None else labelsize
    legendsize = fontsize if legendsize is None else legendsize
    base = {
        'font.family': 'serif',
        'font.serif': ['CMU Serif', 'STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',                 # CM math to match CMU Serif
        'font.size': fontsize, 'axes.titlesize': fontsize,
        'axes.labelsize': labelsize,              # axis names
        'xtick.labelsize': labelsize, 'ytick.labelsize': labelsize,  # tick numbers
        'legend.fontsize': legendsize, 'legend.frameon': False,
        # Shorter legend handle lines + tighter text gap so keys read compact.
        'legend.handlelength': 1.4, 'legend.handletextpad': 0.5,
        # Thin box + ticks: spines 0.7, ticks 0.6/0.5 (lighter than mpl default).
        'lines.linewidth': 1.8, 'axes.linewidth': 0.7,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.minor.width': 0.5, 'ytick.minor.width': 0.5,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.minor.visible': True, 'ytick.minor.visible': True,
        'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'axes.unicode_minus': False,              # CMU Serif lacks U+2212 minus
    }
    if rc:
        base.update(rc)
    mpl.rcParams.update(base)


def paper_grid(layout='2x2', mode='single', placeholder=True, square=True,
               aspect=1.2, fontsize=10, labelsize=None, legendsize=None, rc=None):
    """Build a PRD panel grid with the publication style applied.

    Physics: each panel is forced to a fixed box aspect (W/H = `aspect`) so a
    plane you must read geometrically — an M-R plane, a phase diagram — isn't
    silently stretched. Line-vs-parameter panels read better landscape (aspect
    ~1.2, the default); plane plots want `aspect=1.0` (square). The figure is
    built at the true page width it will occupy (no LaTeX rescaling), so the
    `fontsize`-pt fonts from set_paper_style() land as PRD body text.

    Mechanics:
      layout   : '2x2' (four panels) or '1x2' (a two-panel row = the 2×2 top row).
      mode     : sets the figure width W -
                   'single'   -> W = PRD_COL_W     (3.375", one column)
                   'centered' -> W = PRD_CENTERED_W (4.75", mid-size)
                   'double'   -> W = PRD_FULL_W     (7.0", two columns)
                 The height is derived from `aspect` so panels come out at that
                 W/H with minimal slack: 2×2 is (W, W/aspect); 1×2 is
                 (W, (W/2)/aspect) so its panels match a 2×2 top row.
      aspect   : panel width/height. 1.2 (default) = mild landscape; 1.0 = square.
      fontsize : base text size in pt (default 10 = PRD body; title + fallback).
      labelsize: axis-name + tick-number size (default = fontsize).
      legendsize: legend text size (default = fontsize).
      rc       : optional dict of extra rcParams (e.g. line/axes widths),
                 forwarded to set_paper_style for per-figure tweaks.
                 All four are forwarded to set_paper_style.
      square   : True (default) pins every panel to the exact `aspect` box —
                 because the figsize is fixed the slack dimension leaves a
                 centring margin (harmless: savefig(bbox_inches='tight') crops
                 it). False drops the box constraint so panels stretch to fill
                 the figure (zero centring whitespace) — use it when the inline
                 gap bothers you and near-`aspect` is fine.

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
    if aspect <= 0:
        raise ValueError(f"aspect must be > 0, got {aspect!r}")

    set_paper_style(fontsize=fontsize, labelsize=labelsize,
                    legendsize=legendsize, rc=rc)
    w = widths[mode]
    nrows = 2 if layout == '2x2' else 1
    # Height follows the panel aspect: one panel is w/2 wide, so w/2/aspect tall;
    # 2×2 stacks two of those (≈ w/aspect), 1×2 is a single row.
    fig_h = w / aspect if layout == '2x2' else (w / 2) / aspect
    figsize = (w, fig_h)

    # squeeze=False -> axes always 2-D, so callers unpack uniformly.
    # sharex/sharey default False: independent axes per the request.
    fig, axes = plt.subplots(nrows, 2, figsize=figsize, layout='constrained',
                             squeeze=False)
    # Tighten the constrained-layout padding so panels sit closer together and to
    # the figure edge (less whitespace); edit here to rescale for all figures.
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)
    for ax in axes.flat:
        if square:
            ax.set_box_aspect(1 / aspect)    # fixed W/H box, independent of labels
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
    for _asp in (1.0, 1.2):
        for _lay in ('2x2', '1x2'):
            for _m, _w in _W.items():
                _fig, _ax = paper_grid(_lay, _m, aspect=_asp, fontsize=11,
                                       labelsize=9, legendsize=8)
                _fw, _fh = _fig.get_size_inches()
                _exp_h = _w / _asp if _lay == '2x2' else (_w / 2) / _asp
                assert abs(_fw - _w) < 1e-9
                assert abs(_fh - _exp_h) < 1e-9
                assert all(abs(a.get_box_aspect() - 1 / _asp) < 1e-9 for a in _ax.flat)
                assert abs(mpl.rcParams['font.size'] - 11) < 1e-9
                assert abs(mpl.rcParams['axes.labelsize'] - 9) < 1e-9
                assert abs(mpl.rcParams['legend.fontsize'] - 8) < 1e-9
                plt.close(_fig)
    print('paper_grid self-check ok')
