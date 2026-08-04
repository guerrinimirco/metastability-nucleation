"""Regenerate the committed hadronic test fixture.

    python -m nucleation.tests.make_fixture

Writes ``data/eos_hadronic_trapped_fixture.dat`` (~250 KB, ~5 s). You only need
to run this if the SFHo EoS or its parametrization changes -- the file is
committed, so a normal checkout just uses it.

Why the grid looks irregular
----------------------------
The golden values in ``golden/regression.json`` were captured against the FULL
production table (n_B: 300 nodes, Y_L: 7, T: 52 -- 37 MB). We can ship neither
that nor a naive subsample: ``eos.sfho.build_interpolators`` is MULTILINEAR, so
the value at an off-node test point depends on its two BRACKETING nodes per
axis. Drop a bracket and the golden stops reproducing.

But multilinear interpolation touches ONLY those two nodes, and
``RegularGridInterpolator`` does not require uniform spacing. So a grid built
from

  (a) the exact bracketing node pairs of every golden test point, plus
  (b) a coarse sweep, so tau(n_B, T) still has a real downward crossing for the
      nucleation-curve tests,

reproduces the goldens BIT-FOR-BIT at ~2% of the points. Verified: perturbing a
golden by 1 part in 1e4 fails the suite (tolerance 1.5e-7).

Adding the coarse-sweep nodes is safe because every bracket pair is ADJACENT in
the production grid -- no extra node can land strictly between a pair and change
which two nodes an interpolation uses.

If you add a test that evaluates at a NEW (n_B, Y_L, T), add it to GOLDEN_PTS
below and regenerate, or the interpolator will bracket it with the wrong nodes.
"""
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'data', 'eos_hadronic_trapped_fixture.dat')

# --- production grids, verbatim from the notebook (Part I) -------------------
# These must match the notebook exactly: the goldens were captured on them.
N_SAT = 0.1583
N_B_FULL = np.linspace(0.1, 12, 300) * N_SAT      # fm^-3
T_FULL = np.concatenate([[0.01, 0.1], np.arange(2, 101., 2)])   # MeV
Y_L_FULL = np.arange(0.1, 0.401, 0.05)

# --- every (n_B, Y_L, T) the suite evaluates the interpolators at ------------
# Keep in sync with the build_H(...) call sites in test_composition.py,
# test_critical.py, test_rates.py and test_sigma_crit.py.
GOLDEN_PTS = [
    (0.8, 0.25, 20.0),    # test_composition, test_critical, test_rates
    (1.1, 0.10, 30.0),    # test_composition
    (0.3, 0.25, 20.0),    # test_critical, test_sigma_crit
    (0.9, 0.25, 25.0),    # test_critical, test_composition, test_sigma_crit
]

# Coarse sweep for the nucleation-curve tests. Exact values are irrelevant there
# (they only assert that a root EXISTS), so a subsample is fine.
COARSE_NB_IDX = range(20, 221, 20)
COARSE_T_IDX = [2, 6, 10, 14, 18, 22, 28, 34, 40, 46]     # T = 2,10,...,90 MeV
N_Y_L = 4                                                  # 0.10,0.15,0.20,0.25

# --- SFHo parametrization, verbatim from the notebook (Part I) ---------------
PARAM_KW = dict(U_Lambda_N=-28.0, U_Sigma_N=+30.0, U_Xi_N=-18.0,
                x_sigma_delta=1.15, x_omega_delta=1.0, x_rho_delta=1.0,
                name="2fam_phi_xsd115")


def bracket_indices(full, value):
    """Indices of the two nodes bracketing `value`, or [j] if it IS a node."""
    j = int(np.searchsorted(full, value))
    if j < len(full) and np.isclose(full[j], value):
        return [j]
    if j == 0 or j >= len(full):
        raise ValueError(f"{value} lies outside the production grid")
    return [j - 1, j]


def fixture_grids():
    """The (n_B, Y_L, T) node arrays the fixture is built on."""
    nb_idx, t_idx = set(), set()
    for nB, YL, T in GOLDEN_PTS:
        nb_idx.update(bracket_indices(N_B_FULL, nB))
        t_idx.update(bracket_indices(T_FULL, T))
        if not np.isclose(Y_L_FULL[:N_Y_L], YL).any():
            raise ValueError(f"Y_L={YL} is not among the fixture's Y_L nodes")
    nb_idx.update(COARSE_NB_IDX)
    t_idx.update(COARSE_T_IDX)
    return (N_B_FULL[np.array(sorted(nb_idx))],
            Y_L_FULL[:N_Y_L],
            T_FULL[np.array(sorted(t_idx))])


def main():
    from eos.sfho.parameters import create_custom_parametrization
    from eos.sfho.compute_tables import TableSettings, compute_table

    n_B, Y_L, T = fixture_grids()
    print(f"grid: n_B={len(n_B)}  Y_L={len(Y_L)}  T={len(T)}"
          f"  -> {len(n_B) * len(Y_L) * len(T)} points")
    print(f"  n_B {n_B.min():.4f} .. {n_B.max():.4f} fm^-3")
    print(f"  T   {T.min()} .. {T.max()} MeV")

    t0 = time.time()
    compute_table(TableSettings(
        parametrization='2fam_phi',
        custom_params=create_custom_parametrization(**PARAM_KW),
        particle_content='nucleons_hyperons_deltas',
        n_B_values=n_B, Y_L_values=Y_L, T_values=T,
        equilibrium='trapped_neutrinos',
        include_photons=True, include_pseudoscalar_mesons=True,
        include_thermal_neutrinos=True,
        print_results=False, print_errors=True, print_timing=False,
        save_to_file=True, output_filename=OUT))
    print(f"wrote {OUT} in {time.time() - t0:.0f}s")
    print("now run: pytest nucleation -q   (goldens must still pass)")


if __name__ == '__main__':
    main()
