"""
Nucleation package for the hadron-to-quark phase transition.

Public API:
    compute_nucleation_observables  — main computation
    compute_energy_barrier          — W(R) at a single grid point
    compute_nucleation_temperature  — find T where tau = tau_target
    QstarTableData                  — unified Q* table container
    load_Qstar_table                — load Q* table from .dat file
    build_Qstar_interpolators       — build interpolators from Q* table
    compute_Qstar_table             — generic grid computation
    export_table                    — export Q* table to .dat file
"""

from nucleation.observables import (
    compute_nucleation_observables,
    compute_energy_barrier,
    compute_nucleation_temperature,
    NucleationObservables,
    EnergyBarrierResult,
    NucleationTemperatureResult,
)
from nucleation.table import (
    QstarTableData,
    load_Qstar_table,
    build_Qstar_interpolators,
    compute_Qstar_table,
    export_table,
)
