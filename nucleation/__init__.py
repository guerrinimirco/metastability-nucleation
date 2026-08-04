"""Nucleation of quark-matter droplets in metastable hadronic matter.

Module layout (physics flows top to bottom):

    barrier      -- W(R) components (bulk, surface, Coulomb, screening)
    composition  -- Q* droplet composition solvers (flavor/charge/phase)
    critical     -- the single critical-droplet engine + W(R) profile builder
    rates        -- Langer thermal + relativistic-WKB quantum rates
    tables       -- grid drivers over (n_B_H, [Y], T) + .dat readers/writers
    conditions   -- WHERE nucleation happens: the tau = tau_target locus,
                    plus the single-point API (nucleation_point)
    analysis     -- sigma_crit parameter scans + figures (see nucleation.analysis)

The names below are the public API; import them directly, e.g.
``from nucleation import compute_Qstar_table, compute_energy_barrier``.
"""

# --- barrier components ------------------------------------------------------
from nucleation.barrier import (
    driving_force, work_of_formation,
    bulk_W, surface_W, coulomb_W, coulomb_P,
    coulomb_screened_W, coulomb_screened_P,
    critical_radius_noCoulomb, critical_work_noCoulomb,
    critical_radius_coulomb, critical_radius_coulomb_screened,
    f_screening, df_screening, electron_susceptibility, debye_length,
    switching_step, switching_tanh, get_switching_function,
)

# --- composition solvers -----------------------------------------------------
from nucleation.composition import (
    get_solver_Qs,
    solve_frozen, solve_saddlepoint, solve_saddlepoint_cfl,
    solve_coulomb_minimize, solve_coulomb_minimize_cfl,
    solve_coulomb_minimize_at_R, solve_coulomb_minimize_cfl_at_R,
    R_GCN_SKIP_DEFAULT,
)

# --- critical-droplet engine + barrier profile -------------------------------
from nucleation.critical import (
    CriticalDroplet, critical_droplet,
    EnergyBarrierResult, compute_energy_barrier,
)

# --- rates -------------------------------------------------------------------
from nucleation.rates import (
    nucleation_rate, nucleation_time,
    effective_inertia, quantum_nucleation_time, QuantumNucleationResult,
)

# --- tables + I/O ------------------------------------------------------------
from nucleation.tables import (
    GRID_AXES, QstarTableData,
    compute_Qstar_table, build_Qstar_interpolators,
    load_Qstar_table, export_table,
    ThermalNucleationObservables, compute_thermal_nucleation_observables,
    build_thermal_nucleation_interpolators,
    export_thermal_nucleation_table, load_thermal_nucleation_table,
    QuantumNucleationObservables, compute_quantum_nucleation_observables,
    build_quantum_nucleation_interpolators,
    export_quantum_nucleation_table, load_quantum_nucleation_table,
)

# --- nucleation conditions (tau = tau_target loci + the single-point API) -----
from nucleation.conditions import (
    NucleationTemperatureResult,
    compute_nucleation_temperature, compute_nucleation_density,
    nucleation_curve, build_nucleation_temperature_interpolator,
    NucleationPoint, nucleation_point, tau_at,
    NucleationCondition, T_nuc, nB_nuc,
    crossover_radius, hadronic_point, V_NUCLEATION,
)

__all__ = [
    # barrier
    "driving_force", "work_of_formation", "bulk_W", "surface_W", "coulomb_W",
    "coulomb_P", "coulomb_screened_W", "coulomb_screened_P",
    "critical_radius_noCoulomb", "critical_work_noCoulomb",
    "critical_radius_coulomb", "critical_radius_coulomb_screened",
    "f_screening", "df_screening", "electron_susceptibility", "debye_length",
    "switching_step", "switching_tanh", "get_switching_function",
    # composition
    "get_solver_Qs", "solve_frozen", "solve_saddlepoint", "solve_saddlepoint_cfl",
    "solve_coulomb_minimize", "solve_coulomb_minimize_cfl",
    "solve_coulomb_minimize_at_R", "solve_coulomb_minimize_cfl_at_R",
    "R_GCN_SKIP_DEFAULT",
    # critical
    "CriticalDroplet", "critical_droplet", "EnergyBarrierResult",
    "compute_energy_barrier",
    # rates
    "nucleation_rate", "nucleation_time", "effective_inertia",
    "quantum_nucleation_time", "QuantumNucleationResult",
    # tables
    "GRID_AXES", "QstarTableData", "compute_Qstar_table",
    "build_Qstar_interpolators", "load_Qstar_table", "export_table",
    "ThermalNucleationObservables", "compute_thermal_nucleation_observables",
    "build_thermal_nucleation_interpolators",
    "export_thermal_nucleation_table", "load_thermal_nucleation_table",
    "QuantumNucleationObservables", "compute_quantum_nucleation_observables",
    "build_quantum_nucleation_interpolators",
    "export_quantum_nucleation_table", "load_quantum_nucleation_table",
    # conditions
    "NucleationTemperatureResult", "compute_nucleation_temperature",
    "compute_nucleation_density", "nucleation_curve",
    "build_nucleation_temperature_interpolator",
    "NucleationPoint", "nucleation_point", "tau_at",
    "NucleationCondition", "T_nuc", "nB_nuc",
    "crossover_radius", "hadronic_point", "V_NUCLEATION",
]
