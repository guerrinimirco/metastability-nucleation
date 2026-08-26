"""The public import surface after the rewrite (old paths are gone)."""
import pytest


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


def test_no_undefined_names():
    """Static sweep for undefined names across the package.

    Why: splitting tables.py and analysis/sigma_crit.py into packages moved
    functions away from the imports they relied on, and two names
    (`ud_eps_per_nB`, and the alphaBag custom-parameter helper) ended up used
    but not imported.
    Both sat inside joblib-parallel branches that no test reached, so the suite
    stayed green and the failure only appeared during a full notebook run.

    Import-time checks cannot catch this -- the module imports fine and only
    raises NameError when that specific line executes. A static pass can.
    """
    import pathlib
    import subprocess
    import sys

    pkg = pathlib.Path(__file__).resolve().parent.parent
    try:
        proc = subprocess.run([sys.executable, '-m', 'pyflakes', str(pkg)],
                              capture_output=True, text=True)
    except FileNotFoundError:                      # pragma: no cover
        pytest.skip("pyflakes not installed")
    if proc.returncode != 0 and not proc.stdout:
        pytest.skip(f"pyflakes unavailable: {proc.stderr.strip()[:200]}")

    undefined = [ln for ln in proc.stdout.splitlines() if 'undefined name' in ln]
    assert not undefined, "undefined names:\n  " + "\n  ".join(undefined)
