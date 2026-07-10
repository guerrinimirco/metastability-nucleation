# Stale data artifacts after the CFL coulomb_minimize fix

The 2026-07 `nucleation` rewrite fixed the CFL saddle condition in the
`coulomb_minimize` charge mode: the baryon-density equation was
`mu_B^Q = mu_B^H`; it is now the correct `mu_B^Q + mu_S^Q = mu_B^H + mu_S^H`
(Gibbs energy per baryon *including strangeness*, since a CFL droplet has
Y_S = 1). This changes the critical radius for CFL droplets (e.g. 2.55 -> 2.89 fm
at one test point; ~0.53 -> 0.60 fm at another). Every saved artifact whose
values flow through a CFL `coulomb_minimize` solve is therefore stale and must be
regenerated before use.

## STALE — regenerate (CFL / unpCFL × coulomb_minimize)

- `output/tables_Qstar/Qstar_*_coulomb_minimize_cfl_*.dat`            (12 files)
- `output/tables_nucleation/nucleation_*_coulomb_minimize_cfl_*.dat`  (12 files)
- `output/tables_nucleation/nucleation_*_coulomb_minimize_unpCFL*.dat` (12 files)
  (unpCFL builds its CFL core from the same solver -> stale)
- `output/mc_cfl/sigma_crit_grid_xsd115_MT01.40_saddlepoint-coulomb_minimize-cfl.npz`
- `output/mc_cfl/sigma_crit_grid_xsd115_MT01.40_saddlepoint-coulomb_minimize-unpCFL.npz`

## VALID — unchanged (no CFL coulomb_minimize path)

- All `lcn` / `gcn` / `gcn_coulomb` Q* and thermal tables (any phase).
- All `coulomb_minimize` **unpaired** Q* and thermal tables (the unpaired saddle
  `mu_B^Q = mu_B^H` was already correct).
- All hadronic (`tables_Hphase`), quark-EoS (`tables_Qphase`) and TOV
  (`tables_tov`) tables.

## How to regenerate

Re-run the notebook cells that build the affected tables:
- Section that calls `compute_Qstar_table(... electric_charge_mode='coulomb_minimize',
  quark_phase='cfl' ...)` and saves to `tables_Qstar/`.
- Section that calls `compute_thermal_nucleation_observables(...)` for CFL / unpCFL
  coulomb_minimize and saves to `tables_nucleation/`.
- The `run_sigma_crit_scan(...)` cells for `coulomb_minimize` `cfl` / `unpCFL`,
  which overwrite the `mc_cfl/*.npz` grids.

New `screening` charge-mode tables (if generated) are additional, not replacements.
