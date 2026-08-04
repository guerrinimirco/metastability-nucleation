"""The public import surface after the rewrite (old paths are gone)."""


def test_root_public_api():
    from nucleation import (
        compute_Qstar_table, load_Qstar_table, build_Qstar_interpolators,
        compute_thermal_nucleation_observables,
        build_thermal_nucleation_interpolators, load_thermal_nucleation_table,
        compute_nucleation_density, nucleation_curve, export_table,
        QstarTableData, compute_energy_barrier, get_solver_Qs,
        critical_droplet, CriticalDroplet, work_of_formation, driving_force,
        nucleation_rate, nucleation_time, quantum_nucleation_time,
        compute_quantum_nucleation_observables, f_screening,
        electron_susceptibility,
    )
    assert callable(critical_droplet)


def test_per_module_imports():
    from nucleation.barrier import work_of_formation, coulomb_screened_W
    from nucleation.composition import (
        get_solver_Qs, solve_coulomb_minimize_cfl)
    from nucleation.critical import critical_droplet, compute_energy_barrier
    from nucleation.rates import nucleation_rate, quantum_nucleation_time
    from nucleation.tables import compute_Qstar_table, GRID_AXES
    from nucleation.conditions import compute_nucleation_density
    assert 'trapped_neutrinos' in GRID_AXES


def test_analysis_surface():
    import nucleation.analysis as nuc_an
    for name in ('FilterConfig', 'NucConfig', 'critical_droplet_pt', 'tau_pt',
                 'sigma_target_pt', 'run_sigma_crit_scan', 'crossover_radius',
                 'REASON_CODE', '_HAVE_JOBLIB'):
        assert hasattr(nuc_an, name), name


def test_old_paths_removed():
    import importlib
    import pytest
    for mod in ('nucleation.energy_barrier', 'nucleation.general_nucleation',
                'nucleation.MRE'):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(mod)
