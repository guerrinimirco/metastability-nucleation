# Nucleation of quark droplets in hadronic matter — physics and formalism

This note documents the physics and mathematics behind the `nucleation` code,
with emphasis on the pieces you will describe in the paper: the **saddle-point
composition** of the critical droplet, the **self-consistent Coulomb
minimization** (`coulomb_minimize`), and the **unpaired→CFL (unpCFL) piecewise
barrier**. Every equation below is tied to the routine that implements it.

Conventions: energies in MeV, densities in fm$^{-3}$, radii in fm. `hc` $=\hbar c$
carries the MeV·fm unit; `alpha_EM` $=\alpha\simeq1/137$. The hadronic (parent)
phase is labelled $H$, the quark droplet (daughter) phase $Q^\ast$.

---

## 1. Classical nucleation theory in the thin-wall limit

A spherical quark droplet of radius $R$ nucleating inside metastable hadronic
matter costs a **work of formation**

$$
W(R) \;=\; \underbrace{\tfrac{4}{3}\pi R^3\,\Delta f}_{\text{bulk}}
\;+\; \underbrace{4\pi R^2\,\sigma}_{\text{surface}}
\;+\; \underbrace{E_{\rm Coul}(R)}_{\text{electrostatic}} ,
\tag{1}
$$

built from three competing pieces (`barrier.bulk_W`, `barrier.surface_W`,
`barrier.coulomb_W`, assembled in `barrier.work_of_formation`). The surface
tension $\sigma$ is an external parameter; the bulk term drives the transition;
the Coulomb term (§4) penalizes a charged droplet.

**Critical droplet.** Because $W_{\rm bulk}\sim -R^3$ (when nucleation is
favoured) and $W_{\rm surface}\sim +R^2$, $W(R)$ rises to a maximum at the
**critical radius** $R_c$ and then falls. The maximum

$$
\left.\frac{dW}{dR}\right|_{R_c}=0,\qquad W_c \equiv W(R_c)
\tag{2}
$$

is the **nucleation barrier**. $R_c$ is the saddle of the free-energy landscape:
sub-critical droplets ($R<R_c$) shrink, super-critical ones grow. The nucleation
rate is $\propto \exp(-W_c/T)$, so $W_c$ (equivalently the value of $\sigma$ that
makes $W_c$ reach a chosen threshold, `sigma_crit`) is the physical observable.

### 1.1 Bulk driving force

The bulk term per unit volume is the free-energy density difference between the
droplet interior at its own chemical state and the surrounding hadronic matter
(`barrier.driving_force`):

$$
\Delta f \;=\; -\big(P_{Q^\ast}-P_H\big)
\;+\; n_B^{Q^\ast}\!\!\sum_{i\in\{B,C,S,e,\nu\}}\!\! Y_i^{Q^\ast}\big(\mu_i^{Q^\ast}-\mu_i^{H}\big),
\qquad Y_B\equiv 1 .
\tag{3}
$$

Here $B,C,S$ are baryon number, electric charge, and strangeness; $e,\nu$ are
leptons. $Y_i = n_i/n_B$ are per-baryon fractions. The hadronic phase is
**metastable** — droplet growth is thermodynamically favoured — precisely when
$\Delta f < 0$. When $\Delta f \ge 0$ there is no barrier maximum: the droplet
always shrinks and the code returns $R_c=\text{NaN}$, $W_c=+\infty$ (the
"H stable" convention in `critical.CriticalDroplet`).

### 1.2 Analytic barrier without Coulomb

If the droplet composition is $R$-independent and uncharged ($E_{\rm Coul}=0$,
the `lcn`/`gcn` modes), (2) has a closed form (`barrier.critical_radius_noCoulomb`,
`barrier.critical_work_noCoulomb`):

$$
R_c = -\frac{2\sigma}{\Delta f},
\qquad
W_c = \frac{16\pi\,\sigma^3}{3\,\Delta f^{2}} .
\tag{4}
$$

Both are NaN where $\Delta f\ge 0$. These are the CNT textbook results and serve
as the fast path and as the seed for every numerical solve below.

---

## 2. Droplet composition: why it is a minimization

The interior state $Q^\ast$ is **not** a free choice: for each candidate radius
the droplet settles into the composition that makes $W$ stationary. Writing the
work as a function of the intensive interior variables and extremizing subject to
the conserved charges is what fixes $\mu_u,\mu_d,\mu_s,\mu_e$ (and, with Coulomb,
$R$). The `nucleation.composition` module is exactly this family of stationarity
solvers. There are two orthogonal prescriptions.

**Flavor prescription** (`flavor_mode`):

- `frozen` — the droplet inherits the hadronic flavor fractions,
  $Y_C^{Q^\ast}=Y_C^H$, $Y_S^{Q^\ast}=Y_S^H$. No flavor-changing (weak)
  equilibration across the interface; only the baryon-number balance is imposed.
  This is the fast, "just-nucleated" composition.
- `saddlepoint` — the composition itself minimizes $W$, i.e. strong/weak
  chemical equilibrium is reached across the interface. **This is the physically
  complete critical droplet and the focus of the paper.**

**Electric-charge prescription** (`electric_charge_mode`): how neutrality and
the Coulomb energy are treated — `lcn`, `gcn`, `gcn_coulomb`, `screening`,
`coulomb_minimize` (§3–4).

---

## 3. The saddle-point conditions

For a droplet whose interior is uniform, minimizing $W$ (1) with respect to the
composition, at fixed conserved charges, produces one stationarity equation per
independent flavor direction. With unknowns $x=(\mu_u,\mu_d,\mu_s,\mu_e)$ the
`saddlepoint` solver (`composition.solve_saddlepoint`) imposes four equations.

**(i) Charge stationarity** $\partial W/\partial Y_C = 0$ — matching the
electrochemical potential of charge across the interface:

$$
\mu_C^{Q^\ast} + \mu_e \;=\; \mu_C^{H} + \mu_e^{H}.
\tag{5}
$$

**(ii) Strangeness stationarity** $\partial W/\partial Y_S = 0$:

$$
\mu_S^{Q^\ast} \;=\; \mu_S^{H}.
\tag{6}
$$

**(iii) Charge neutrality.** Two variants:

$$
\text{local (lcn): } Y_C^{Q^\ast}=Y_e^{Q^\ast},
\qquad
\text{global (gcn): } \mu_e^{Q^\ast}=\mu_e^{H}.
\tag{7}
$$

Local neutrality forces the droplet interior itself to be neutral (no net charge,
no Coulomb). Global neutrality lets the droplet carry net charge, neutralized by
a background electron sea in equilibrium with the hadronic phase — this is the
charge that the Coulomb term in §4 acts on.

**(iv) Mechanical/Gibbs balance** $\partial W/\partial n_B = 0$. The full
stationarity of the chemical sum,

$$
\big(\mu_B^{Q^\ast}-\mu_B^H\big)
+ Y_C^{Q^\ast}\big(\mu_C^{Q^\ast}-\mu_C^H\big)
+ Y_S^{Q^\ast}\big(\mu_S^{Q^\ast}-\mu_S^H\big)
+ Y_e^{Q^\ast}\big(\mu_e-\mu_e^H\big) = 0,
\tag{8}
$$

which, **once (5)–(7) hold**, collapses to the Gibbs-per-baryon match

$$
\boxed{\;\mu_B^{Q^\ast} = \mu_B^{H}\;}
\tag{9}
$$

i.e. equal Gibbs free energy per baryon — the usual coexistence condition. The
code keeps the full form (8) as equation 4 of the system because it is what
remains exactly zero at the solution and it is numerically robust; (9) is its
reduced content. All four residuals are normalized by $\mu_B^H$ (or $n_B$) so the
root finder sees dimensionless, $O(1)$ equations.

### 3.1 CFL droplet

In the color-flavor-locked phase the pairing enforces equal quark flavors, so the
flavor prescription is replaced by the **CFL lock**
(`composition.solve_saddlepoint_cfl`):

$$
Y_C^{Q^\ast}=0,\qquad Y_S^{Q^\ast}=1 .
\tag{10}
$$

A CFL droplet is automatically charge-neutral in bulk. Crucially, because
$Y_S=1$ the strangeness carries chemical work, and the Gibbs-per-baryon balance
(iv) **includes strangeness**:

$$
\boxed{\;\mu_B^{Q^\ast}+\mu_S^{Q^\ast} = \mu_B^{H}+\mu_S^{H}\;}
\tag{11}
$$

**not** $\mu_B$ alone. This is the CFL saddle condition; getting it right is what
makes the radius from the self-consistent solve (§4) coincide with
$\arg\max_R W(R)$. (This correction is recorded in the repo history as the
`coulomb_minimize` CFL fix.)

### 3.2 Numerical strategy

Every composition system is solved by `composition.robust_root`, a four-step
fallback: `hybr` then `lm` from the supplied guess, then both again from the
physics-based `hadronic_guess` (which inverts the strong-charge→quark-flavor map
at the hadronic $\mu$'s, landing already near equilibrium). A solve is accepted
only if the max residual is below $10^{-8}$; otherwise `None` propagates upward
and the point is flagged as a solver failure ($R_c=W_c=\text{NaN}$), kept
distinct from the physical "H stable" ($W_c=+\infty$) outcome.

---

## 4. Coulomb energy and the self-consistent `coulomb_minimize`

### 4.1 Electrostatic energy of a charged droplet

A globally-neutral droplet carries a net charge density

$$
\delta n_C \;=\; \big(Y_C^{Q^\ast}-Y_e^{Q^\ast}\big)\,n_B^{Q^\ast}.
\tag{12}
$$

Treated as a uniformly charged sphere, its electrostatic self-energy is
$E=\tfrac35 Q^2/R$ with $Q=e\,\delta n_C\,\tfrac43\pi R^3$ and $e^2=\alpha\,\hbar c$,
giving (`barrier.coulomb_W`):

$$
E_{\rm Coul}(R) \;=\; \frac{16}{15}\,\pi^2\,\alpha\,\hbar c\;\delta n_C^{2}\,R^{5}.
\tag{13}
$$

The $R^5$ growth is what makes Coulomb decisive at large droplets: it can
suppress or even remove the barrier maximum.

### 4.2 Two ways to add Coulomb — a posteriori vs. self-consistent

There is a genuine physics fork here, and the code keeps them strictly separate
(see the "two Coulomb pressures" note at the bottom of `barrier.py`):

**(A) A-posteriori** (`gcn_coulomb`, `screening`). Solve the composition **without**
Coulomb (plain GCN saddle point, §3), then add $E_{\rm Coul}$ to $W(R)$ and locate
the maximum. The composition is $R$-independent, so $R_c$ follows from a scalar
root of $dW/dR=0$:

$$
\Delta f + \frac{2\sigma}{R} - P_{\rm Coul}(R) = 0,
\qquad
P_{\rm Coul} = -\frac{dE_{\rm Coul}}{dV}
= -\frac{4}{3}\,\pi\,\alpha\,\hbar c\,R^{2}\,\delta n_C^{2}
\tag{14}
$$

(`barrier.coulomb_P`, `barrier.critical_radius_coulomb`). This $P_{\rm Coul}$ is
the *thermodynamic* Coulomb pressure $-dE_{\rm Coul}/dV$.

**(B) Self-consistent** (`coulomb_minimize`). $E_{\rm Coul}$ is placed **inside**
the minimization. Now the composition feels the electrostatics: minimizing $W$
including (13) with respect to $Y_e$ no longer gives $\mu_e^{Q^\ast}=\mu_e^H$ but

$$
\mu_e^{Q^\ast} = \mu_e^{H} + \delta\mu_e(R),
\qquad
\delta\mu_e(R) = \frac{8}{5}\,\pi\,\alpha\,\hbar c\,R^{2}\,\delta n_C
\tag{15}
$$

(`barrier.coulomb_delta_mu_e`). Because the composition and $R$ are now coupled
through $\delta\mu_e(R)$ — and $\delta n_C$ in turn depends on the composition —
they must be **co-solved**. The pressure balance (2) picks up an extra electron
chemical-work term and becomes

$$
P_{Q^\ast} - P_H - \frac{2\sigma}{R} + \delta P(R) = 0,
\qquad
\delta P(R) = \frac{4}{15}\,\pi\,\alpha\,\hbar c\,R^{2}\,\delta n_C^{2}
\tag{16}
$$

(`barrier.coulomb_delta_P`). Note $\delta P \ne -dE_{\rm Coul}/dV$; the two are
related by

$$
\delta P = P_{\rm Coul} + \frac{8}{5}\,\pi\,\alpha\,\hbar c\,R^2\,\delta n_C^2
= P_{\rm Coul} + \delta\mu_e\cdot(\text{electron chem-sum}),
$$

the difference being exactly the electron contribution that the self-consistent
minimization introduces. **`screening` must never be mixed with the $\delta P$
machinery** — it is architecturally an a-posteriori mode.

### 4.3 The 5-equation self-consistent system

`coulomb_minimize` (unpaired: `composition.solve_coulomb_minimize`) solves for the
five unknowns $x=(\mu_u,\mu_d,\mu_s,\mu_e,R)$ simultaneously:

$$
\begin{aligned}
&\text{1. } \mu_B^{Q^\ast} = \mu_B^{H} && (\partial W/\partial n_B=0)\\
&\text{2. } \mu_C^{Q^\ast} + \mu_e = \mu_C^{H}+\mu_e^{H} && (\partial W/\partial Y_C=0)\\
&\text{3. } \mu_S^{Q^\ast} = \mu_S^{H} && (\partial W/\partial Y_S=0)\\
&\text{4. } \mu_e = \mu_e^{H} + \delta\mu_e(R) && (\partial W/\partial Y_e=0,\ \text{GCN}+\text{Coulomb})\\
&\text{5. } P_{Q^\ast} - P_H - \tfrac{2\sigma}{R} + \delta P(R) = 0 && (\partial W/\partial R=0)
\end{aligned}
\tag{17}
$$

Equation 4 replaces the plain GCN condition $\mu_e=\mu_e^H$ by (15); equation 5 is
the radius stationarity (16). The **CFL** version
(`composition.solve_coulomb_minimize_cfl`) swaps equations 1–3 for the CFL lock
(10) plus the strangeness-inclusive Gibbs match (11):

$$
Y_C^{Q^\ast}=0,\quad Y_S^{Q^\ast}=1,\quad
\mu_e = \mu_e^H + \delta\mu_e(R),\quad
\mu_B^{Q^\ast}+\mu_S^{Q^\ast}=\mu_B^H+\mu_S^H,\quad
(16).
$$

**Seeding and fallbacks.** The solve is bootstrapped from the plain GCN saddle
point: its composition plus $R_c^{\rm gcn}=2\sigma/(P_{Q^\ast}^{\rm gcn}-P_H)$ form
the initial guess. If the point is not favoured ($\Delta f\ge 0$ at the GCN
solution) the routine returns `None` immediately. If the primary solve fails, it
retries from a midpoint between the GCN and LCN critical radii, then from the
hadronic guess. Because $E_{\rm Coul}\sim R^5\delta n_C^2$, very large
$R_c^{\rm gcn}$ usually has no $dW/dR=0$ root; the `R_gcn_skip` cutoff (default
$\infty$, i.e. disabled) can short-circuit those hopeless points to a single quick
attempt.

**Building the profile.** To plot the barrier shape, `solve_coulomb_minimize_at_R`
freezes $R$ and solves only equations 1–4 (composition at fixed radius); sweeping
$R$ with warm-starts (`critical.compute_Qs_along_R`) traces $W(R)$, whose maximum
must coincide with the $R$ returned by the full 5-equation solve — a built-in
consistency check.

---

## 5. The unpCFL piecewise barrier

The physical picture: a nucleating quark droplet need not be a single phase. A
small droplet is **unpaired** quark matter; only once it is large enough to
support color-flavor-locking (radius beyond a coherence/crossover scale $R_x$)
does the interior become **CFL**. The `unpCFL` mode builds one barrier out of the
two phases joined at $R_x$.

### 5.1 Switching function

A blend $S(R)\in[0,1]$ interpolates unpaired ($S=0$) to CFL ($S=1$)
(`barrier.get_switching_function`):

$$
\text{step: } S(R)=\Theta(R-R_x),
\qquad
\text{tanh: } S(R)=\tfrac12\!\left[1+\tanh\!\frac{R-R_x}{\delta}\right].
\tag{18}
$$

The local bulk driving force and charge density are blended,

$$
\Delta f(R) = (1-S)\,\Delta f_{\rm unp} + S\,\Delta f_{\rm cfl},
\qquad
\delta n_C(R) = (1-S)\,\delta n_{C,\rm unp} + S\,\delta n_{C,\rm cfl},
\tag{19}
$$

and $W(R)$ is rebuilt from (1). For the step blend the barrier is genuinely
**piecewise**: the unpaired $W(R)$ on $[0,R_x]$ and the CFL (or, more precisely,
the more stable) branch on $(R_x,\infty)$, with a kink at $R_x$.

### 5.2 Global peak of a kinked barrier

The subtlety (handled in `critical._find_Rc_Wc_step`): the critical point is the
**global maximum** of the piecewise curve, and you cannot find it by comparing
radii alone. Two candidate peaks are formed and compared **by height**:

- **Below-region candidate** — the unpaired barrier on $[0,R_x]$. If the unpaired
  critical radius $R_c^{\rm unp}$ lies inside $[0,R_x]$ the peak sits there;
  otherwise the unpaired branch is still rising at $R_x$ and the candidate is
  pinned at the kink, $R=R_x$.
- **Above-region candidate** — the more stable phase (smaller CNT radius) on
  $(R_x,\infty)$, valid only if its critical radius actually exceeds $R_x$.

$$
W_c = \max\big(W_{\rm below},\,W_{\rm above}\big),
\qquad
R_c,\,S \text{ from the winning branch.}
\tag{20}
$$

Comparing heights is essential: if both critical radii exceed $R_x$ but the
upper-branch peak is *lower* than the value at the kink, the true maximum is
pinned at $R_x$ (unpaired), which a radius-only comparison would miss.

### 5.3 Existence and NaN-safety

The CFL core is required; the unpaired mantle may legitimately have no critical
droplet (then it falls back to the CFL upper branch). The blend
(`critical._blend_phase`) is NaN-safe: at $S=0$ it returns the unpaired value
exactly even if the CFL value is NaN, and vice versa at $S=1$, so a failed phase
never poisons the selected one via $0\times\text{NaN}$. A **no-barrier guard**
enforces the convention of §1.1: if $\Delta f\ge 0$ for *both* phases the barrier
rises without bound above $R_x$, there is no critical droplet, and $W_c=+\infty$.

The smooth (`tanh`) case has no closed piecewise form, so
`critical._find_Rc_Wc_tanh` maximizes $W(R)$ numerically per point
(`minimize(-W)`), seeded by the step-formula $R_c$.

### 5.4 unpCFL with `coulomb_minimize`

When `unpCFL` is combined with `coulomb_minimize`, the composition of *each* phase
is $R$-dependent, so the barrier cannot be assembled from two fixed compositions.
Instead (`critical.compute_energy_barrier`, unpCFL + coulomb_minimize branch) both
phases are re-solved at every $R$ via the fixed-$R$ solvers of §4.3, warm-started
along the sweep, and blended by $S(R)$ before $W(R)$ is formed. This is the most
expensive path and the reason the fixed-$R$ solvers exist.

---

## 6. Map from physics to code

| Physics | Function |
|---|---|
| $W(R)$ and components, eq. (1) | `barrier.work_of_formation`, `bulk_W`, `surface_W`, `coulomb_W` |
| Driving force $\Delta f$, eq. (3) | `barrier.driving_force` |
| Analytic $R_c,W_c$ (no Coulomb), eq. (4) | `barrier.critical_radius_noCoulomb`, `critical_work_noCoulomb` |
| Saddle-point composition, eqs. (5)–(9) | `composition.solve_saddlepoint` |
| CFL saddle, eqs. (10)–(11) | `composition.solve_saddlepoint_cfl` |
| Frozen composition | `composition.solve_frozen` |
| Coulomb energy / shifts, eqs. (13)(15)(16) | `barrier.coulomb_W`, `coulomb_delta_mu_e`, `coulomb_delta_P` |
| A-posteriori Coulomb $R_c$, eq. (14) | `barrier.coulomb_P`, `critical_radius_coulomb` |
| Self-consistent 5-eq system, eq. (17) | `composition.solve_coulomb_minimize[_cfl]` |
| Fixed-$R$ composition (build $W(R)$) | `composition.solve_coulomb_minimize[_cfl]_at_R` |
| Debye screening $f(R/\lambda_D)$ | `barrier.f_screening`, `coulomb_screened_W`, `critical_radius_coulomb_screened` |
| unpCFL switching $S(R)$, eq. (18) | `barrier.get_switching_function` |
| unpCFL global peak, eq. (20) | `critical._find_Rc_Wc_step` / `_find_Rc_Wc_tanh` |
| Single public entry point | `critical.critical_droplet` |
| Langer thermal rate, eq. (21) | `rates.nucleation_rate`, `nucleation_time` |
| Effective inertia $M(R)$, eq. (22) | `rates.effective_inertia` |
| Bohr–Sommerfeld $E_0$, eq. (25) | `rates.ground_state_energy` |
| WKB exponent $A(E_0)$, eq. (24) | `rates.tunneling_action` |
| Quantum time $\tau_{\rm qt}$, eq. (26) | `rates.quantum_nucleation_time` |
| Quantum grid driver | `tables.compute_quantum_nucleation_observables` |
| Single-point everything, eq. (27) | `conditions.nucleation_point` |
| $\tau=\tau_{\rm target}$ locus, eq. (28) | `conditions.NucleationCondition` |
| Star shells, eq. (29) | `analysis.star_shell_states` |
| $\sigma_{\rm crit}$, eq. (30) | `analysis.sigma_target_pt` |
| Acceptance filters, §8.2 | `analysis.passes_cfl_filters` |
| NS / QS / BH decision, §8.3 | `analysis.outcome_row` |

---

## 7. Quantum nucleation (relativistic WKB)

At high temperature the droplet escapes by thermal activation **over** the
barrier, at rate $\propto e^{-W_c/T}$ (§1). As $T\to0$ that channel shuts off and
the transition proceeds by **tunnelling through** the barrier instead. The
quantum rate then depends on the whole shape of $W(R)$, not on its peak alone —
which is why $\tau_{\rm qt}$ needs a callable $W(R)$ where the thermal rate needs
only $(R_c,W_c)$.

### 7.1 The droplet as a one-dimensional quantum degree of freedom

Treat the radius $R$ as a collective coordinate. Growing the droplet displaces
the surrounding hadronic matter, and that irrotational flow gives $R$ an
effective inertia (`rates.effective_inertia`):

$$
M(R) \;=\; 4\pi\rho_H\Big(1-\frac{n_B^{Q^\ast}}{n_B^{H}}\Big)^{2} R^{3},
\tag{22}
$$

with $\rho_H\simeq m_n n_B^H$ unless a custom $\rho_H(n_B,T)$ is supplied. The
factor $(1-n_B^{Q^\ast}/n_B^H)^2$ is the physical content: if the two phases had
equal density nothing would have to move, and the droplet would carry no inertia.

The motion is treated **relativistically**, so the actions carry the
$\sqrt{(2M+E-W)(\dots)}$ form rather than the Schrödinger $\sqrt{2M(\dots)}$;
$2M$ plays the role of the rest energy.

### 7.2 Actions

With $\hbar c$ restoring units, the bound sub-barrier motion has the
Bohr–Sommerfeld action (`rates.oscillation_action`)

$$
I(E) \;=\; \frac{2}{\hbar c}\int_{0}^{R_{\rm in}}
\sqrt{\big[2M(R)+E-W(R)\big]\big[E-W(R)\big]}\;dR ,
\tag{23}
$$

and the classically forbidden region gives the tunnelling exponent
(`rates.tunneling_action`)

$$
A(E) \;=\; \frac{2}{\hbar c}\int_{R_{\rm in}}^{R_{\rm out}}
\sqrt{\big[2M(R)+E-W(R)\big]\big[W(R)-E\big]}\;dR ,
\tag{24}
$$

where $R_{\rm in}<R_c<R_{\rm out}$ are the turning points $W(R)=E$.

### 7.3 Ground state and the nucleation time

The droplet does not tunnel from rest but from the lowest quantized level of the
bound motion. With $E_{\rm min}=\max_R[W-2M]$ the lower edge of the
positive-energy band and $m_0=\lfloor I(E_{\rm min})/2\pi+1/4\rfloor$ the number
of levels it holds, $E_0$ solves (`rates.ground_state_energy`)

$$
I(E_0) \;=\; 2\pi\big(m_0+\tfrac34\big).
\tag{25}
$$

The attempt frequency is $\nu_0=(dI/dE)^{-1}$ evaluated at $E_0$
(`rates.oscillation_frequency`), and with $N_c$ independent nucleation centres

$$
\tau_{\rm qt} \;=\; \big[N_c\,\nu_0\,e^{-A(E_0)}\big]^{-1}.
\tag{26}
$$

### 7.4 Two search-window subtleties (both are real, both bit)

Equations (23)–(25) are one-dimensional root-finds, and **both of their search
windows must be bounded by the barrier peak $R_c$**. This is not a numerical
nicety: it is what makes them well posed once the Coulomb term is on.

Without Coulomb, $W(R)\sim-R^3$ falls monotonically past $R_c$. With it, the
electrostatic energy grows as $+R^5$ (eq. 13) and eventually beats the bulk gain,
so $W(R)$ **turns back up** at large $R$. Consequently:

1. $E_{\rm min}=\max_R[W-2M]$ must be searched on $R<R_c$ only. Searched over a
   wide range it locks onto the outer Coulomb wall and returns
   $E_{\rm min}\sim10^{9}$ MeV instead of $\sim10$ MeV. The bound is physical:
   the band edge belongs to the bound oscillation, which lives entirely inside
   $R_{\rm in}<R_c$.
2. $R_{\rm out}$ is the **first** crossing of $W=E$ beyond $R_c$, found by
   walking outward until $W<E$. Bracketing $[R_c,R_{\rm max}]$ directly fails
   whenever $W$ has already climbed back above $E$ at $R_{\rm max}$: both
   endpoints then lie above $E$ and the bracket is rejected, even though a
   turning point plainly exists.

Beyond the outer minimum of $W$ the droplet is supercritical and the thin-wall
collective-coordinate picture no longer applies, so nothing physical is lost by
confining the search.

### 7.5 Thermal or quantum?

The two channels are computed by parallel drivers over the same grid —
`compute_thermal_nucleation_observables` and
`compute_quantum_nucleation_observables` — sharing the same $Q^\ast$ table, the
same $\Delta f$ and the *same* $R_c$ and $W_c$, so they can be compared point by
point. The physical nucleation time is the shorter of the two; their crossover
defines the temperature below which the transition is quantum-dominated. Both
observables expose `.tau`, so either can be fed to the $\tau=\tau_{\rm target}$
locus finders of `nucleation.curves` without special-casing.

---

### Appendix — Debye screening (the `screening` mode)

The `screening` mode is an a-posteriori Coulomb (family A of §4.2) in which the
uniform-sphere self-energy (13) is multiplied by a Debye form factor
$f(x)$, $x=R/\lambda_D$ (`barrier.coulomb_screened_W`):

$$
f(x)=\frac{5}{2x^5}\Big[x^3-3(x+1)(x\cosh x-\sinh x)e^{-x}\Big],
\quad f(0)=1,\quad f\!\to\!\frac{5}{2x^2}\ (x\!\to\!\infty).
$$

The Debye length comes from the electron susceptibility
$\chi_e=\partial n_e/\partial\mu_e|_T$,
$\lambda_D=1/\sqrt{4\pi\alpha\,\hbar c\,\chi_e}$
(`barrier.electron_susceptibility`, `debye_length`). Because $f(x)$ suffers a
three-fold catastrophic cancellation for the physically relevant $R\ll\lambda_D$,
the code uses an exact Maclaurin series below $x=0.5$ and the closed form above.
The composition is identical to plain GCN: the electrostatic potential shifts
$\mu_C$ and $\mu_e$ equally at the interface and cancels in every equilibrium
condition.


---

## 8. From a droplet to a star

Sections 1-7 answer "does a droplet form here". A star is not a point, and the
observable is not a nucleation time -- it is which kind of compact object is
left. Three further steps close that gap.

### 8.1 The critical surface tension $\sigma_{\rm crit}$

$\sigma$ is the least constrained input in the problem: it is not measured, and
model estimates span an order of magnitude. Quoting $\tau$ at an assumed
$\sigma$ therefore buries the dominant uncertainty in an assumption.

Instead we inverta the question. Since $W_*\propto\sigma^3$ (eq. 4) the barrier
rises steeply with $\sigma$, so $\tau(\sigma)$ is monotonic and there is a
unique threshold

$$
\tau\big(\sigma_{\rm crit}\big) \;=\; \tau_{\rm target},
\tag{30}
$$

with the star converting iff $\sigma<\sigma_{\rm crit}$
(`analysis.sigma_target_pt`). The root-find is a coarse scan over
$[\sigma_{\rm lo},\sigma_{\rm hi}]$ followed by `brentq` on the bracketing
pair; the coarse scan is what selects *which* crossing is refined, so its
resolution is part of the physics, not a tolerance.

Two conventions carry through the code:

* $\sigma_{\rm crit}=+\infty$ -- the hadronic phase is unstable at every
  $\sigma$, so nucleation always happens;
* $\sigma_{\rm crit}=$ NaN -- the solve failed. NaN is **not** "does not
  nucleate"; it is "unknown", and is excluded rather than counted as a negative.

### 8.2 Star-wide, not central

The obvious place to evaluate eq. (30) is the stellar centre, and that is wrong
often enough to matter. Near the corner of parameter space where strange quark
matter is only just absolutely stable, the driving force $|\Delta f|$ peaks
around $2\,n_{\rm sat}$ and *weakens* toward the centre, because the hadronic
EoS stiffens faster than the quark one. An off-centre shell then nucleates first.

$\sigma_{\rm crit}$ is therefore defined star-wide,

$$
\sigma_{\rm crit}^{\rm star} \;=\; \max_{\text{shells } i}\;
\sigma_{\rm crit}\big(n_B^{(i)}, T^{(i)}\big),
\tag{29}
$$

over `N_SHELLS` shells from $n_{B,\min}$ to the centre, each at the temperature
its $(Y_L,S)$ isentrope assigns (`analysis.star_shell_states`). Six shells are
converged to better than 0.5 % against twelve. The centre-only value can
underestimate the star-wide one by up to a factor $\sim2$ in that corner, so the
two are **not interchangeable** and every figure and table states which it uses.

### 8.3 Which parameters describe a real star

Before nucleation is even asked about, a quark parameter set must survive four
filters (`analysis.passes_cfl_filters`), applied cheap-to-expensive so a rejected
cell never pays for a TOV solve:

1. **Witten window, bound side** -- 3-flavour matter must satisfy
   $e/n_B<930$ MeV at $P=0$, or a strange star could not hold itself together.
2. **Witten window, unbound side** -- 2-flavour ($ud$) matter must *not*, or
   ordinary nuclei would decay and there would be no ordinary matter to observe.
3. **No re-hadronization** -- the quark pressure must stay above the hadronic one
   past their crossing in $\mu_B$, both in bulk and at droplet level; otherwise
   a converted star would convert back.
4. **Maximum mass** -- the cold sequence must reach the heaviest observed pulsar.

Each cell records the *first* filter it failed, as a nested integer code, so
contouring that field at half-integer levels draws every filter's pass edge in
one pass.

### 8.4 The outcome: NS, QS or BH

A proto-neutron star is a sequence of objects, not one. Between the two snapshots
$t_0$ (lepton-rich, $Y_L=0.35$, $S=1.5$) and $t_{T_{\max}}$ (deleptonized,
$Y_L=0.25$, $S=2.0$) its **baryon number is conserved** but its maximum
gravitational mass is not: trapped neutrinos stiffen the EoS, so $M_{\max}$
*falls* as they escape.

Asking eq. (30) at each snapshot then gives three outcomes
(`analysis.outcome_row`):

* **QS** -- $\sigma<\sigma_{\rm crit}$ at either snapshot. The remnant is read
  off the cold quark-star sequence at **fixed $M_B$**, and the gravitational-mass
  deficit is released:
  $E_{\rm conv}=(M_{\rm PNS}-M_{\rm QS})c^2$, of order $10^{53}$ erg.
* **BH** -- born above the $t_0$ maximum mass (prompt), or surviving $t_0$ only
  for deleptonization to drop $M_{\max}$ below its $M_B$ (delayed).
* **NS** -- neither: it cools into an ordinary neutron star of the same baryon
  number.

---

## 9. Nucleation conditions: the $\tau=\tau_{\rm target}$ locus

Rather than a rate at a point, what a figure usually wants is the *boundary*: the
curve in the $(n_B,T)$ plane along which

$$
\tau\big(n_B^H,\,Y_L^H,\,T\big) \;=\; \tau_{\rm target}.
\tag{28}
$$

Above it a droplet forms in time; below it the hadronic star survives. Overlaying
the PNS isentrope on this locus turns the microphysics into a statement about
stellar mass: the crossing is the mass at which conversion becomes possible.

`conditions.NucleationCondition` builds the locus three ways -- from a computed
grid, from any $\log_{10}\tau$ interpolator, or by solving on the fly with no
table at all -- and exposes the same $T(n_B)$, $n_B(T)$ and `curve()` afterwards.
Thermal and quantum observables both work, because both expose `.tau`.

**Scan direction is a physics choice, not a preference.** Root-finding along $T$
at fixed $n_B$ (`scan='n_B'`) is required for the unpCFL phase: the CFL gap melts
as $T$ rises, so $\tau$ is **non-monotonic** in $T$ and the other scan direction
silently locks onto the wrong branch. Both directions take the first *downward*
crossing.
