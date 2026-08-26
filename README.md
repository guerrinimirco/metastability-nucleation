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

Requires Python ≥ 3.11 and the sibling [`eos`](https://github.com/guerrinimirco/eos)
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
pytest test -q                # 72 tests, no external data needed
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
  quark.py                 the eos.alphabag interface: custom parameter sets
                           and the TOTAL droplet thermodynamics (quarks + bag
                           + electrons + photons + gluons + thermal neutrinos)
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
test/                      the suite, with its committed fixture and goldens
docs/                      physics notes and the reproduction guide
output/                    generated; see below
```

The dependency rule runs one way only: `analysis` builds on the core, the core
never imports `analysis`, and **`eos` never imports `nucleation`**.

---

## Quick start

All three examples are copy-paste runnable **from a fresh clone**, from the
repository root, with no table generation: they read the committed test fixture
`test/data/eos_hadronic_trapped_fixture.dat` (247 KB, an 18 x 4 x 13 trapped-SFHo
grid). The output below each is what they actually printed, on the stack the
suite is run with: CPython 3.14.2 with NumPy 2.3.5 and SciPy 1.17.0. Together
they take about ten seconds.

For real work, swap the fixture for a production table out of Part II of the
notebook — the fixture is deliberately coarse and exists to make the goldens
reproducible without shipping 37 MB.

### 1. The nucleation time at one point

```python
from eos.sfho.table import load_eos_table, build_interpolators
from nucleation.quark import custom_params
from nucleation import nucleation_point

H = build_interpolators(load_eos_table(
    'test/data/eos_hadronic_trapped_fixture.dat', 'trapped_neutrinos'))
params = custom_params(alpha=0.157, B4=145.0, m_s=100.0)

pt = nucleation_point(H, n_B_H=0.9, T=25.0, sigma=30.0,
                      params=params, Y_L_H=0.25)

print(f"R_star  = {pt.R_star:8.4f} fm")
print(f"W_star  = {pt.W_star:8.4f} MeV")
print(f"W_over_T= {pt.W_over_T:8.4f}")
print(f"N_B_star= {pt.N_B_star:8.4f}")
print(f"tau     = {pt.tau:.6e} s")
print(f"nucleates(tau_target=1e-3) = {pt.nucleates(tau_target=1e-3)}")
```

```
R_star  =   0.9885 fm
W_star  = 122.7818 MeV
W_over_T=   4.9113
N_B_star=   4.2014
tau     = 5.430414e-73 s
nucleates(tau_target=1e-3) = True
```

`R_star` is the critical radius, `W_star` the barrier, `W_over_T` the exponent
that decides the rate and `N_B_star` the baryons inside the critical droplet.
At sigma = 30 MeV/fm^2 the barrier is low enough that nucleation is effectively
instantaneous — which is the point of asking for sigma_crit instead (example 3).

Pass `quark_phase='unpCFL'` with `Delta0=` for a paired droplet, and
`electric_charge_mode='coulomb_minimize'` for the self-consistent Coulomb
treatment; the defaults are the unpaired phase under global charge neutrality.

### 2. The nucleation condition: where does it happen?

```python
import numpy as np
from eos.sfho.table import load_eos_table
from nucleation import NucleationCondition
from nucleation.tables import compute_thermal_nucleation_observables

H_table = load_eos_table(
    'test/data/eos_hadronic_trapped_fixture.dat', 'trapped_neutrinos')
obs = compute_thermal_nucleation_observables(
    H_table, sigma=80.0, params=params, flavor_mode='saddlepoint',
    electric_charge_mode='gcn', quark_phase='unpaired')

cond = NucleationCondition.from_table(obs, tau_target=1e-3)
print(cond)

n_B, T = cond.curve(Y_L=0.25)
print(f"locus at Y_Le = 0.25: {n_B.size} points")
for x, y in list(zip(n_B, T))[:4]:
    print(f"  n_B = {x:.4f} fm^-3   T_nuc = {y:8.4f} MeV")
print(f"T_of_nB(0.9, Y_L=0.25) = {float(cond.T_of_nB(0.9, Y_L=0.25)):.4f} MeV")
```

```
NucleationCondition(tau_target=0.001 s, eq_type='trapped_neutrinos', 72/72 points on the locus)
locus at Y_Le = 0.25: 18 points
  n_B = 0.1418 fm^-3   T_nuc =  34.5921 MeV
  n_B = 0.2678 fm^-3   T_nuc =  24.7061 MeV
  n_B = 0.2993 fm^-3   T_nuc =  22.8319 MeV
  n_B = 0.3056 fm^-3   T_nuc =  22.4899 MeV
T_of_nB(0.9, Y_L=0.25) = 15.6626 MeV
```

`cond.nB_of_T(T, Y_L=...)` is the inverse — one locus, not two curves, so the
two round-trip. `NucleationCondition.from_point_solver(H, T_grid, n_B_grid, ...)`
builds the same object with no table at all, solving on the fly; that is the one
for exploring parameter space ("at what T does THIS
(alpha_s, B^1/4, Delta_0) nucleate?"), at the cost of a solve per grid point.

### 3. The critical surface tension

```python
from nucleation.analysis import NucConfig, hadronic_point, sigma_target_pt

nuc = NucConfig(tau_target=1e-3)
H_pt = hadronic_point(H, n_B=0.9, Y_L=0.25, T=25.0)
sigma_crit = sigma_target_pt(H_pt, 25.0, 'saddlepoint', 'coulomb_minimize',
                             'unpCFL', params, 80.0, nuc)
print(f"sigma_crit (centre-only) = {sigma_crit:.4f} MeV/fm^2")
```

```
sigma_crit (centre-only) = 98.9248 MeV/fm^2
```

The star converts if the true sigma is below this. `sigma_target_pt` returns
`+inf` when the point nucleates for every sigma tested up to `nuc.sig_hi`, and
`NaN` when no converged nucleating point exists at all — read the value before
trusting it.

> **Star-wide vs centre-only.** Nucleation is not always fastest at the centre:
> near the strange-matter-stability corner the driving force peaks around
> 2 n_sat and *weakens* inward, so an off-centre shell nucleates first. The
> centre-only value above can underestimate $\sigma_{\rm crit}$ by up to a
> factor ~2 there. Pass `shells=star_shell_states(MT0, star, n_shells)` for the
> star-wide value — that needs a TOV sequence, so it is a notebook-scale call,
> not a fresh-clone one — and always say which definition you used.

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

The seven figures the paper includes, in `output/paper/figures/paper/`:

| Figure | Notebook | File stem |
| --- | --- | --- |
| 1 — nucleation barrier and critical quantities | IV.1 | `paper_fig1_barrier_{tag}_{set}` |
| 2 — stellar sequences | IV.2 | `paper_fig2_stellar_sequences_{tag}` |
| 3 — nucleation at the PNS centre vs $M_{\rm PNS}$ | IV.3 | `paper_fig3_Rstar_Wc_tau_sigmacrit_{tag}_{set}` |
| 4 — nucleation conditions $T_{\rm nuc}(n_B^H)$ | IV.4 | `paper_fig4_Tnuc_{tag}` |
| 5 — $\sigma_{\rm crit}$ parameter plane | IV.5 | `paper_fig5_sigcrit_map_isolines_{tag}` |
| A — electric-charge prescriptions | IV.6 | `paper_appA_charge_prescriptions_{tag}` |
| B — frozen vs saddle-point flavour | IV.7 | `paper_appB_frozen_vs_saddlepoint_{tag}` |
| Outcomes table | IV.8 | `table_outcomes_{tag}.csv` / `.tex` |

Supporting material, in `output/paper/figures/supplementary/`:

| Figure | Notebook | File stem |
| --- | --- | --- |
| viable region as stars ($M$–$R$, $P(\mu_B)$) | V.1 | `paper_fig5a_MR_PmuB_sigmacrit_{tag}` |
| $\Delta\sigma_{\rm crit}$: $M_{T0}$ and phase | V.2 | `dsigma_MT0_{tag}_{phase}`, `dsigma_phase_{tag}_MT0{m}` |
| $W_*/T$ over the accessible plane | V.3 | `WoverT_map_{tag}_MT0{m}` |
| unpaired $\sigma_{\rm crit}$, three $m_s$ | V.4 | `sigcrit_unpaired_ms_panels_{tag}` |

`{tag}` is the run tag (`xsd115`) and `{set}` the quark parameter set
(`B4145_D80_a0.16_ms100`). Every figure writes its underlying arrays to
`output/*/figure_data/` as CSV.

**These cells reproduce the published figures byte-for-byte in style** —
colours, sizes, limits, legends and panel anchors are the ones the paper
carries. The knob block at the top of each cell is what to edit; the drawing
code below it is not decoration but the figure contract.

---

## Testing

```bash
pytest test -q
```

72 tests, no external data and nothing under `output/`: the hadronic fixture
(`test/data/`, 247 KB) is committed and regenerable with
`python test/make_fixture.py`. The suite runs in a few seconds.

Regenerating the fixture is **not** a routine step. Against the current `eos` it
no longer reproduces the committed file bit-for-bit — the arithmetic moved by
~1 ulp when `eos` adopted its shared conserved-charge basis — so a regeneration
silently moves the input the goldens were captured against. Run it only when the
SFHo EoS or its parametrization genuinely changes, and re-capture the goldens
deliberately when you do.

`test/golden/regression.json` pins the engine's output at known points. Treat it
as a **tripwire, not a target** — if a golden fails, find out why before touching
the tolerance.

**Two goldens currently fail, and both are known**: `test_regression_solver_cases`
and `test_energy_barrier_matches_golden`. Neither is a physics change. Both
compare round-off — ten quantities the CFL flavour lock forces to zero, compared
*relatively*, and a `W(R)` curve reaching -1.5e+06 MeV compared *absolutely* at
1e-9 — and both moved by ~1 ulp when `eos` adopted its shared conserved-charge
basis. Every physically nonzero quantity still matches. Fixing them means
re-deciding what the golden asserts, which is open work; the tolerances have
**not** been loosened in the meantime.

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

**TODO — not yet chosen.** Pick one before making the repository public;
without a licence file, others have no legal right to use or build on this code.
MIT or BSD-3-Clause are the usual choices for research code of this kind.
