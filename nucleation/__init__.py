"""
Nucleation package for the hadron-to-quark phase transition.

Public API:
    compute_nucleation_observables  — thermal nucleation over the full grid
    compute_energy_barrier          — W(R) at a single grid point
    compute_nucleation_temperature  — find T where tau = tau_target
    compute_quantum_nucleation      — quantum (WKB) tunneling over the full grid
    QstarTableData                  — unified Q* table container
    load_Qstar_table                — load Q* table from .dat file
    build_Qstar_interpolators       — build interpolators from Q* table
    compute_Qstar_table             — generic grid computation
    export_table                    — export Q* table to .dat file
    build_solver_fn                 — build solver callable from mode strings
"""

from nucleation.observables import (
    compute_nucleation_observables,
    compute_energy_barrier,
    compute_nucleation_temperature,
    compute_quantum_nucleation,
    NucleationObservables,
    EnergyBarrierResult,
    NucleationTemperatureResult,
    QuantumNucleationGrid,
)
from nucleation.table import (
    QstarTableData,
    load_Qstar_table,
    build_Qstar_interpolators,
    compute_Qstar_table,
    export_table,
)
from nucleation.solvers import build_solver_fn
from nucleation.quantum import QuantumNucleationResult
