"""Figure styling and shared plot helpers for the nucleation analysis.

This is the single import surface the paper notebook uses. The *generic*
publication styling and the observational M-R overlays are NOT defined here:
they live in the sibling ``eos`` project (``eos.general.figure_style`` and
``eos.general.observational_constraints``) so that every project built on
``eos`` draws the same way, and they are simply re-exported below.

What IS nucleation-specific -- and therefore defined in this subpackage -- are
the drawing primitives that know about this paper's quantities: droplet-phase
line styles, the (B^1/4, Delta_0) parameter-plane layers, the CFL rejection
reason codes and the many-curve replay bundles.

Usage::

    from nucleation.analysis.figure import paper_grid, panel_label
    from nucleation.analysis.figure import add_observational_constraints
    fig, ((axA, axB), (axC, axD)) = paper_grid('2x2', mode='double',
                                               placeholder=False)
    add_observational_constraints(axA)      # packaged contours, no path needed
"""
from eos.general.figure_style import (
    set_paper_style, paper_grid, panel_label,
    STANDARD_COLORS, OKAB, OKAB_CAT, PHASE_LW, PHASE_ALPHA,
    PRD_COL_W, PRD_FULL_W, PRD_CENTERED_W,
)
from eos.general.observational_constraints import (
    add_observational_constraints, MR_CONSTRAINTS, DEFAULT_CONTOUR_DIR,
)

__all__ = [
    # --- publication style (from eos.general.figure_style) ---
    "set_paper_style", "paper_grid", "panel_label",
    "STANDARD_COLORS", "OKAB", "OKAB_CAT", "PHASE_LW", "PHASE_ALPHA",
    "PRD_COL_W", "PRD_FULL_W", "PRD_CENTERED_W",
    # --- observational overlays (from eos.general.observational_constraints) ---
    "add_observational_constraints", "MR_CONSTRAINTS", "DEFAULT_CONTOUR_DIR",
]
