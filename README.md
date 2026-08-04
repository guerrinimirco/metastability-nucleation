# nucleation

**Quark-droplet nucleation in proto-neutron stars.**

Code and data behind *"Coexistence of strange quark stars and neutron stars:
metastability and nucleation in proto-neutron stars"*.

If strange quark matter is the true ground state of matter, why is any neutron
star still a neutron star? Because converting one requires **nucleating** a quark
droplet, and a droplet must pay surface energy before it repays bulk energy. The
barrier that results is crossed at a rate $\propto e^{-W_*/T}$, so the answer
hinges on the quark–hadron **surface tension** $\sigma$ — a quantity nobody has
measured.

Rather than assume a value, this code inverts the question and computes the
threshold $\sigma_{\rm crit}$ at which a proto-neutron star nucleates within its
lifetime. A star converts if $\sigma < \sigma_{\rm crit}$; you can place your own
preferred $\sigma$ against the maps.

> **Citation.** If you use this code, please cite the paper.
> A BibTeX entry will be added here on publication.

---

## Install

Requires Python ≥ 3.9 and the sibling [`eos`](https://github.com/guerrinimirco/eos)
package, which supplies the equations of state, the TOV solver and the shared
publication figure style.

```bash
git clone https://github.com/guerrinimirco/eos.git
git clone https://github.com/guerrinimirco/metastability-nucleation.git nucleation

pip install -e ./eos          # must come first: nucleation depends on it
pip install -e ./nucleation
```

Editable installs are recommended: this is research code you will want to read
and modify, and the notebook imports both packages directly.

**Optional, both worth having:**

| Package | Without it |
| --- | --- |
| `joblib` | parameter scans run serially (~10× slower) |
| `numba`  | use `tov_backend='scipy'`; correct, ~100× slower |

Check the install:

```bash
pytest nucleation -q          # 71 tests, no external data needed
```

---

## Repository map

The package is split by **what a piece of code is allowed to know about**.

```
nucleation/                CORE. Knows nothing about stars, about grids of
                           quark parameters, or about matplotlib.
  barrier.py               W(R): bulk, surface, Coulomb, Debye screening
  composition.py           what is inside the droplet (the Q* solvers)
  critical.py              find the saddle: R*, W*, and the W(R) profile
  rates.py                 Langer thermal activation + relativistic-WKB tunnelling
  conditions.py            WHERE nucleation happens -- the tau = tau_target
                           locus, plus the single-point API
  tables/                  the same physics over a grid, plus .dat I/O
    grid.py qstar.py thermal.py quantum.py

nucleation/analysis/       PAPER LAYER. This study's specific choices.
  config.py                FilterConfig / NucConfig / StarMatch
  filters.py               which quark parameters describe a viable star at all
  sigma_crit.py            the threshold surface tension at a point
  scan.py                  sweep the parameter plane (data production)
  replay.py                re-solve accepted cells into M-R / P(mu_B) bundles
  stellar.py               TOV sequence bookkeeping
  droplet.py               droplet observables + parameter-plane maps
  outcomes.py              the PNS-evolution NS / QS / BH decision engine
  stability.py             absolute-stability boundaries at P = 0
  figure/                  publication style (from eos) + drawing primitives

notebooks/                 the paper notebook (the .py jupytext mirror is the
                           source of truth -- reviewable diffs)
docs/                      physics notes and the reproduction guide
output/                    generated; see below
```

The dependency rule runs one way only: `analysis` builds on the core, the core
never imports `analysis`, and **`eos` never imports `nucleation`**.

---

## Quick start

### The nucleation time at one point

```python
from eos.sfho.compute_tables import load_eos_table, build_interpolators
from eos.alphabag.parameters import get_alphabag_custom
from nucleation import nucleation_point

H = build_interpolators(load_eos_table('...trapped....dat', 'trapped_neutrinos'))
params = get_alphabag_custom(alpha=0.157, B4=145.0, m_s=100.0)

pt = nucleation_point(H, n_B_H=0.9, T=25.0, sigma=30.0,
                      params=params, Y_L_H=0.25,
                      quark_phase='unpCFL', Delta0=80.0,
                      electric_charge_mode='coulomb_minimize')

pt.R_star        # critical radius [fm]
pt.W_star        # barrier [MeV]
pt.W_over_T      # the exponent that decides the rate
pt.N_B_star      # baryons inside the critical droplet
pt.tau           # nucleation time [s]
pt.nucleates(tau_target=1e-3)
```

### The nucleation condition: where does it happen?

```python
from nucleation import NucleationCondition, T_nuc

# From a computed grid (thermal OR quantum -- both expose .tau)
cond = NucleationCondition.from_table(obs, tau_target=1e-3)
cond.T_of_nB(0.9, Y_L=0.25)        # the nucleation temperature [MeV]
cond.nB_of_T(30.0, Y_L=0.25)       # the inverse [fm^-3]
n_B, T = cond.curve(Y_L=0.25)      # the whole locus, ready to plot

# Or with no table at all -- solves on the fly. This is the one for exploring
# parameter space: "at what T does THIS (alpha_s, B^1/4, Delta_0) nucleate?"
cond = NucleationCondition.from_point_solver(
    H, T_grid=np.linspace(5, 80, 30), n_B_grid=np.linspace(0.6, 1.2, 20),
    sigma=80.0, params=params, tau_target=1e-3, Y_L_H=0.25)
```

### The critical surface tension

```python
from nucleation.analysis import (NucConfig, make_star_match,
                                 star_shell_states, sigma_target_pt)

nuc = NucConfig(tau_target=1e-3)
shells = star_shell_states(MT0=1.4, star=star, n_shells=6)   # star-wide
sigma_crit = sigma_target_pt(H_pt, T_c, 'saddlepoint', 'coulomb_minimize',
                             'unpCFL', params, Delta0=80.0, nuc=nuc,
                             shells=shells)
```

> **Star-wide vs centre-only.** Nucleation is not always fastest at the centre:
> near the strange-matter-stability corner the driving force peaks around
> 2 n_sat and *weakens* inward, so an off-centre shell nucleates first. The
> centre-only value can underestimate $\sigma_{\rm crit}$ by up to a factor ~2
> there. Pass `shells=` explicitly and say which definition you used.

---

## Reproducing the paper

Everything comes from one notebook, `notebooks/2fam_PNS_nucleation.py`, in five
parts:

| Part | Does | Cost |
| --- | --- | --- |
| **I**   | imports, parameters, grids | instant |
| **II**  | generate every table and scan | **the expensive one** |
| **III** | load what II produced | seconds |
| **IV**  | the paper figures + outcomes table | seconds |
| **V**   | supplementary material | minutes |

Part II is the only slow part, and once it has run you never run it again:
**Parts I + III is the figure-editing loop.**

### Smoke run first (~10 minutes)

```python
REDUCED_GRID = True     # Part I
```

Coarse grids, writing to `output/smoke/`. The **physics is identical** — only the
sampling is coarse, so figures are jagged and the numbers are not paper values.
Its job is to prove every cell runs end-to-end before you commit hours.

### Production run (hours)

```python
REDUCED_GRID = False
```

Writes to `output/paper/`. The two modes use **separate directories**, so a smoke
run can never overwrite or be mistaken for paper data.

Run the notebook as a script (`python 2fam_PNS_nucleation.py`) or open the paired
`.ipynb` via `jupytext --sync`.

### What is in the repository, and what is not

| Path | Committed? |
| --- | --- |
| `output/paper/figures/` | ✅ the paper figures |
| `output/paper/figure_data/` | ✅ the CSVs behind them |
| `output/paper/tables/sigma_crit/` | ✅ the scan grids (`.npz`) |
| `output/paper/tables/{hadronic,quark}_tov/` | ✅ stellar sequences |
| `output/paper/tables/{qstar,nucleation,*_eos}/` | ❌ many GB — regenerate from Part II |
| `output/smoke/` | ❌ never committed |

So a reader can redraw every figure from a fresh clone without recomputing
anything; only the multi-GB intermediates need Part II.

---

## Figure map

| Figure | Notebook | File |
| --- | --- | --- |
| 1 — nucleation barrier | IV.1 | `paper_fig1_barrier` |
| 2 — stellar sequences | IV.2 | `paper_fig2_stellar_sequences` |
| 3 — nucleation at the PNS centre | IV.3 | `paper_fig3_centre_vs_Mpns` |
| 4 — nucleation conditions $T_{\rm nuc}(n_B)$ | IV.4 | `paper_fig4_Tnuc` |
| 5 — $\sigma_{\rm crit}$ parameter plane | IV.5 | `paper_fig5_sigmacrit_map` |
| Appendix — method dependence | IV.6 | `paper_appendix_methods` |
| Outcomes table | IV.7 | `table_outcomes_*.csv` / `.tex` |
| S1 — viable region as stars | V.1 | `supp_viable_region_stars` |
| S2 — $\sigma_{\rm crit}$ sensitivity | V.2 | `supp_dsigma_MT0` |
| S3 — $W_*/T$ invariance | V.3 | `supp_WoverT_map` |
| S4 — unpaired matter | V.4 | `supp_unpaired_sigmacrit` |

Every figure writes its underlying arrays to `output/*/figure_data/` as CSV.

---

## Testing

```bash
pytest nucleation -q
```

71 tests, no external data: the hadronic fixture
(`nucleation/tests/data/`, 247 KB) is committed and regenerable with
`python -m nucleation.tests.make_fixture`.

`golden/regression.json` pins the engine's output at known points. Treat it as a
**tripwire, not a target** — if a golden fails, find out why before touching the
tolerance.

---

## Documentation

* [`docs/nucleation_physics.md`](docs/nucleation_physics.md) — the formalism,
  equation by equation, each tied to the routine implementing it: classical
  nucleation theory, droplet composition, the self-consistent Coulomb treatment,
  the unpCFL piecewise barrier, Debye screening, and relativistic-WKB tunnelling.
* [`docs/reproducing.md`](docs/reproducing.md) — run order, wall-clock and disk
  budget, and what to do when a table is missing.

---

## Licence

MIT. See [`LICENSE`](LICENSE).
