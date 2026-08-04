"""
Grid tables & I/O
=================

Grid-level drivers that sweep the critical-droplet physics over a hadronic
(n_B_H, [Y_L_H | Y_C_H], T) grid, plus the .dat readers/writers. Everything here
is "the same single-point physics, many times, warm-started", built on
``nucleation.composition`` (Q* solves), ``nucleation.barrier`` (R_c / W_c),
``nucleation.critical`` (unpCFL peak logic) and ``nucleation.rates``.

Layout (bottom to top -- each imports only from the ones above it):

    grid     -- axes, shapes, per-point storage. No physics.
    qstar    -- the Q* droplet-composition table + its .dat I/O.
    thermal  -- Langer activation OVER the barrier (finite T).
    quantum  -- WKB tunnelling THROUGH the barrier (T -> 0).

`thermal` and `quantum` are siblings on purpose: they consume the same Q* table
and quote the same (R_c, W_c), differing only in the escape mechanism, so their
outputs are directly comparable point by point.

All three table kinds share ONE on-disk format: axis columns first, then the
data columns, ravelled in Fortran order, under a `# key: value` header.

This module re-exports the full public surface, so
``from nucleation.tables import X`` keeps working for every X it ever exposed.

Grid W_c convention: NaN where a point did not converge / has no critical
droplet (this is what round-trips through the .dat files). The engine's +inf
"H-stable" convention lives in ``nucleation.critical`` / sigma_crit only.
"""

from nucleation.tables.grid import (
    GRID_AXES, QstarTableData, expand_grid,
    # Private names kept importable: tables.py exposed them and other modules
    # (critical.py, analysis/) import them by these exact names.
    _EQ_TYPE_ALIASES, _BASE_DATA_KEYS, _COULOMB_EXTRA_KEYS,
    _THERMAL_DATA_KEYS, _QUANTUM_DATA_KEYS, _COULOMB_MODES,
    _grid_layout, _lambda_D_grid, _init_data, _store_result,
    _compute_and_store_Rc, _build_H_from_table,
)
from nucleation.tables.qstar import (
    compute_Qstar_table, build_Qstar_interpolators,
    load_Qstar_table, export_table, _compute_Qs_at_R,
)
from nucleation.tables.thermal import (
    ThermalNucleationObservables, compute_thermal_nucleation_observables,
    build_thermal_nucleation_interpolators,
    export_thermal_nucleation_table, load_thermal_nucleation_table,
    _compute_thermal_nucleation_unpCFL,
)
from nucleation.tables.quantum import (
    QuantumNucleationObservables, compute_quantum_nucleation_observables,
    build_quantum_nucleation_interpolators,
    export_quantum_nucleation_table, load_quantum_nucleation_table,
)

__all__ = [
    # grid
    "GRID_AXES", "QstarTableData", "expand_grid",
    # Q*
    "compute_Qstar_table", "build_Qstar_interpolators",
    "load_Qstar_table", "export_table",
    # thermal
    "ThermalNucleationObservables", "compute_thermal_nucleation_observables",
    "build_thermal_nucleation_interpolators",
    "export_thermal_nucleation_table", "load_thermal_nucleation_table",
    # quantum
    "QuantumNucleationObservables", "compute_quantum_nucleation_observables",
    "build_quantum_nucleation_interpolators",
    "export_quantum_nucleation_table", "load_quantum_nucleation_table",
]
