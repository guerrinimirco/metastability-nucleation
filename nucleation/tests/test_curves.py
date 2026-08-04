"""Nucleation-curve finders on a computed thermal grid."""
import numpy as np
import pytest

from nucleation.tables import compute_thermal_nucleation_observables
from nucleation.curves import compute_nucleation_density, nucleation_curve


@pytest.fixture(scope='module')
def thermal_obs(params, H_table):
    """A real thermal grid: tau(n_B, Y_L, T) over the committed fixture table.

    Uses the whole fixture (936 points, ~20 s) rather than an index slice --
    the old index slice silently depended on the shape of the production table.
    """
    return compute_thermal_nucleation_observables(
        H_table, sigma=30.0, params=params, flavor_mode='saddlepoint',
        electric_charge_mode='gcn', quark_phase='unpaired')


def test_density_scans_produce_curve(thermal_obs):
    obs = thermal_obs
    r_T = compute_nucleation_density(obs, tau_target=1e-3, scan='T')
    r_n = compute_nucleation_density(obs, tau_target=1e-3, scan='n_B')
    assert np.isfinite(r_n.T_nuc).sum() > 0
    # nucleation_curve slices the n_B scan into plottable arrays
    nB, T = nucleation_curve(r_n, iYL=0)
    assert nB.shape == T.shape
    assert np.isfinite(T).sum() > 0
