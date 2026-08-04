# Reproducing the paper

Everything comes from one notebook, `notebooks/2fam_PNS_nucleation.py`. This page
is the operational companion: what to run, how long it takes, how much disk it
needs, and what to do when something is missing.

---

## 1. Before you start

```bash
pip install -e ../eos          # must come first
pip install -e .
pytest nucleation -q           # 71 passed, 0 skipped
```

If the test suite does not come back green, stop here — every number below
depends on it. In particular a **skipped** test is not a pass: the suite is
designed to run with no external data, so a skip means something is misconfigured.

The notebook is a [jupytext](https://jupytext.readthedocs.io/) `py:percent` file.
Run it either way:

```bash
python notebooks/2fam_PNS_nucleation.py     # as a script, start to finish
jupytext --sync notebooks/2fam_PNS_nucleation.py   # then open the .ipynb
```

The `.py` is the source of truth and the only version in git. The `.ipynb` is
generated, carries megabytes of cell output, and is gitignored.

---

## 2. The two run modes

One switch in Part I decides everything:

```python
REDUCED_GRID = True      # or False
```

|  | `True` (smoke) | `False` (production) |
| --- | --- | --- |
| Writes to | `output/smoke/` | `output/paper/` |
| n_B grid | 60 points | 300 |
| T grid | 15 | 52 |
| σ values | 3 | 6 |
| Scan plane | 1 × 11 × 11 | 4 × 51 × 51 |
| Shells for σ_crit | 2 | 6 |
| Part II wall clock | **~10 min** | **hours** |
| Numbers are | indicative only | the published values |

**The physics is identical in both modes.** Only the sampling changes. Smoke
figures are jagged and the numbers are not paper values, but every code path is
exercised — which is the point: it proves the notebook runs end-to-end before you
commit hours to it.

The two modes write to **separate directories**, so a smoke run can never
overwrite paper data or be mistaken for it. Part I prints a loud banner in smoke
mode.

### Recommended order

1. `REDUCED_GRID = True`, run the whole notebook. ~15 min total.
2. Look at the figures. Fix anything broken.
3. `REDUCED_GRID = False`, run **Part II only**. Hours; do it once.
4. Re-run Parts III–V on the paper data. Minutes.

Steps 3 and 4 are separate because Part II is the only expensive one and Parts
III–V are the loop you actually iterate in.

---

## 3. What each Part costs

Wall clock measured on an Apple M-series laptop, 10 cores, `joblib` and `numba`
available.

| Part | Section | Smoke | Production | Writes |
| --- | --- | --- | --- | --- |
| I | setup | instant | instant | — |
| II.1 | hadronic EoS × 4 | ~2 min | ~40 min | `tables/hadronic_eos/` |
| II.2 | hadronic TOV | ~30 s | ~10 min | `tables/hadronic_tov/` |
| II.3 | quark EoS | ~30 s | ~5 min | `tables/quark_eos/` |
| II.5 | quark TOV | ~10 s | ~1 min | `tables/quark_tov/` |
| II.6 | Q\* tables | ~3 min | ~1–2 h | `tables/qstar/` |
| II.7 | nucleation observables | ~3 min | ~1–2 h | `tables/nucleation/` |
| II.8 | σ_crit scan | ~1 min | ~2–4 h | `tables/sigma_crit/` |
| III | load everything | seconds | seconds | — |
| IV | paper figures + table | ~1 min | ~2 min | `figures/paper/` |
| V | supplementary | ~2 min | ~20 min | `figures/supplementary/` |

**Part II is resumable.** Every section skips work whose output file already
exists, so an interrupted run picks up where it stopped. To force a
recomputation, delete the file.

### Disk

| Directory | Smoke | Production | In git? |
| --- | --- | --- | --- |
| `tables/hadronic_eos/` | 20 MB | ~37 MB | ✗ |
| `tables/quark_eos/` | 15 MB | ~64 MB | ✗ |
| `tables/qstar/` | 50 MB | **~9 GB** | ✗ |
| `tables/nucleation/` | 35 MB | **~5 GB** | ✗ |
| `tables/hadronic_tov/`, `quark_tov/` | < 1 MB | < 1 MB | ✓ |
| `tables/sigma_crit/` | < 1 MB | ~2 MB | ✓ |
| `figures/`, `figure_data/` | ~5 MB | ~8 MB | ✓ |

Budget **~15 GB** for a production run. The Q\* and nucleation tables dominate
and are the two that stay out of git — everything a reader needs to redraw the
figures is committed.

---

## 4. When something is missing

This is the failure you are most likely to hit, so it fails loudly rather than
silently.

**`KeyError: no nucleation table for ...`**
Part III could not find a table Part IV asked for. The message lists what Part II
does tabulate. Usually you changed `SIGMA_LIST`, `CHARGE_MODES` or
`quark_param_sets` and did not re-run Part II.

**`AssertionError: figures request sigma [...] but SIGMA_LIST is [...]`**
Part I's consistency check, firing *before* anything expensive runs. A figure
wants a surface tension that will never be tabulated. Either add it to
`SIGMA_LIST` or fix `FIG_SIGMAS`. (Without this check the figure would simply
drop a curve without saying so — which is exactly what the previous version of
this notebook did.)

**`RuntimeError: no sigma_crit scan for MT0=...`**
Part II.8 has not run for that stellar mass. Check `MT0_GRID`.

**"sensitivity map needs >= 2 scanned M_T0 values"**
Not an error. Section V.2 needs two scans to difference and only found one; it
names the setting to change. Set `MT0_GRID = [1.17, 1.4, 1.6]` and re-run II.8.

**A section prints "too few accepted cells"**
On the smoke grid the parameter scan is 11 × 11, so few cells survive the
acceptance filters and the curve-bundle figures have nothing to draw. Expected;
it resolves on the production grid.

---

## 5. Things worth checking on the production grid

Three quantities are sensitive to resolution in ways the smoke grid cannot show:

1. **Hadronic M_max.** The smoke n_B grid (60 points) under-resolves the EoS and
   gives M_max ≈ 1.7 M⊙. The production grid should recover the correct, higher
   value. If it does not, the hadronic tables are wrong, not the nucleation code.

2. **`N_SHELLS` and `n_sigma_scan` are NOT safely reducible.** `N_SHELLS = 6` is
   converged to < 0.5 % against 12; the smoke value of 2 is not. And
   `sigma_target_pt` picks *which* bracket brentq then refines off a 40-point
   coarse scan, so shrinking `n_sigma_scan` changes the answer rather than its
   precision. Neither is reduced in production; do not reduce them to save time.

3. **The W\*/T invariance (§V.3).** On the smoke grid the spread comes out
   implausibly tight because only two shells and one α_s slice are sampled.
   Re-check the spread on the production grid before quoting it as evidence of
   invariance.

---

## 6. Regenerating the test fixture

Only needed if the SFHo EoS or its parametrization changes:

```bash
python -m nucleation.tests.make_fixture
pytest nucleation -q          # the goldens must still pass
```

The fixture is a deliberately irregular grid — it carries the exact bracketing
nodes of every point the goldens are evaluated at, because the interpolators are
multilinear and dropping a bracket changes the answer. `make_fixture.py` explains
the construction. If you add a test that evaluates at a *new* (n_B, Y_L, T), add
it to `GOLDEN_PTS` there and regenerate.

If a golden fails after regeneration, find out **why** before touching a
tolerance. The goldens are a tripwire, not a target.
