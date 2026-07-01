r"""Multiple Reflection Expansion (MRE) surface & curvature tension for quark droplets.

Finite-size thermodynamics of a *spherical* quark-matter droplet of radius ``R``
embedded in hadronic matter, at zero and finite temperature.  The MRE corrects
the single-particle density of states (DOS) of the quark Fermi gas with a surface
(``1/R``) and a curvature (``1/R^2``) term; integrating the corrected DOS gives
the surface tension ``sigma`` (MeV/fm^2) and curvature energy ``gamma`` (MeV/fm).

Physics (Balian-Bloch 1970; Madsen 1994; Berger-Jaffe; Lugones-Grunfeld):

    rho_MRE(k,m,R) = 1 + (6 pi^2 / x) f_S(k,m) + (12 pi^2 / x^2) f_C(k,m),
    x = k R / hbar c                                   (dimensionless! -- see below)

    f_S(k,m) = -(1/8pi) [1 - (2/pi) arctan(k/m)]       (Berger-Jaffe, < 0)
    f_C(k,m) = (1/12pi^2) [1 - (3k/2m)(pi/2 - arctan(k/m))]   (Madsen ansatz)

The grand potential of the drop is  Omega = -P V + sigma S + gamma C  with
V=(4/3)pi R^3, S=4 pi R^2, C=8 pi R, so S/V=3/R, C/V=6/R^2.

Two book-keeping caveats (NOT bugs):

* **Units (the factor that bites).**  ``rho_MRE`` contains ``k R`` which must be
  dimensionless; with ``k`` in MeV and ``R`` in fm we use ``x = k R / hbar c``.
  Dropping ``hbar c`` misplaces the infrared cutoff by ~200x.  The Lambda_IR
  validation (Section "validate") catches this immediately.

* **Two prescriptions** (agree as R->inf, differ at finite R):
    - ``'B'`` (bag convention, default): the MRE weight multiplies the ``k^4/E``
      *pressure* integrand -> matter part only, convergent, small sigma.  This is
      the natural match for the alpha-Bag EOS (a bag model).
    - ``'A'`` (NJL convention): the MRE weight multiplies the per-mode free energy
      ``L(k)`` -> carries the (large) Dirac-sea/vacuum term if ``include_vac``.

For the alpha-Bag EOS only the massive ideal Fermi-gas term feeds sigma, gamma;
the bag constant B, the alpha_s correction and the CFL ``Delta^2 mu^2`` terms are
bulk (volume) terms and enter sigma, gamma *only indirectly* through the
equilibrium chemical potentials.  Since ``f_S(k, m=0) = 0`` identically, massless
u,d quarks carry **zero** surface tension -- with ``m_u=m_d=0`` the surface
tension is entirely an ``m_s``-driven quantity.

References (verified against the literature):
* J. Madsen, PRD 50, 3328 (1994) -- curvature ansatz.
* M. S. Berger, R. L. Jaffe, PRC 35, 213 (1987) -- surface term.
* G. Lugones, A. G. Grunfeld, PRC 88, 045803 (2013), arXiv:1308.1452 (NJL).
* G. Lugones, A. G. Grunfeld et al., arXiv:1811.09954 (hot, magnetized).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq
from scipy.integrate import quad

# hbar c and pi: reuse the eos package constants when importable, else fall back
try:                                              # drops straight into the repo env
    from eos.general.physics_constants import hc as HBARC, PI as _PI
except Exception:                                 # standalone (tests, no eos)
    HBARC = 197.3269804                           # MeV * fm
    _PI = np.pi

PI = float(_PI)
PI2 = PI * PI
G_QUARK = 6.0                                     # 3 colours x 2 spins


# =============================================================================
#  Section 1 -- MRE density of states and infrared cutoff
# =============================================================================
def f_S(k, m):
    """Berger-Jaffe surface shape function (dimensionless, <= 0).

    f_S -> 0 as k->inf;  f_S -> -1/(8 pi) as k->0.  Identically 0 for m=0
    (massless quarks carry no surface tension)."""
    if m == 0:
        return np.zeros_like(np.asarray(k, dtype=float))
    return -(1.0 / (8 * PI)) * (1.0 - (2.0 / PI) * np.arctan(k / m))


def f_C(k, m):
    """Madsen curvature shape function (dimensionless).

    f_C -> -1/(24 pi^2) as m->0 (massless MIT-bag limit);  -> +1/(12 pi^2) as
    m->inf.  Uses arctan(m/k) = pi/2 - arctan(k/m) for a stable small-m limit."""
    if m == 0:
        return np.full_like(np.asarray(k, dtype=float), -1.0 / (24 * PI2))
    # (3k/2m)(pi/2 - arctan(k/m)); pi/2 - arctan(k/m) = arctan(m/k)
    return (1.0 / (12 * PI2)) * (1.0 - (3 * k / (2 * m)) * np.arctan2(m, k))


def rho_MRE(k, m, R):
    """MRE-corrected density of states, normalised to the bulk value 1.

    R in fm, k,m in MeV.  x = kR/hc is the dimensionless combination (Caveat 1).
    R = inf (bulk) -> 1."""
    if not np.isfinite(R):
        return np.ones_like(np.asarray(k, dtype=float))
    x = k * R / HBARC
    return 1.0 + (6 * PI2 / x) * f_S(k, m) + (12 * PI2 / x**2) * f_C(k, m)


def Lambda_IR(m, R, kmax=None, n=20000):
    """Infrared cutoff: the *largest* momentum root of rho_MRE(k)=0 [MeV].

    rho_MRE goes negative at small k (unphysical); we integrate above Lambda_IR.
    Depends only on (m, R), not on T or mu.  R=inf -> 0 (no cutoff in the bulk).

    The roots sit where x = kR/hc is O(1), so the largest root scales as ~1/R; the
    search ceiling must follow it (a fixed ceiling misses the root at small R and
    returns a spurious value, making sigma(R)/gamma(R) discontinuous)."""
    if not np.isfinite(R):
        return 0.0
    if kmax is None:
        kmax = max(600.0, 4.0 * HBARC / R)        # scale ceiling with 1/R
    ks = np.linspace(1e-3, kmax, n)
    s = np.sign(rho_MRE(ks, m, R))
    idx = np.where(np.diff(s) != 0)[0]
    if idx.size == 0:
        return 0.0
    i = idx[-1]                                   # bracket the largest root
    return float(brentq(rho_MRE, ks[i], ks[i + 1], args=(m, R)))


# =============================================================================
#  Section 2 -- single-particle building blocks
# =============================================================================
def E(k, m):
    """Relativistic dispersion sqrt(k^2 + m^2) [MeV]."""
    return np.sqrt(k * k + m * m)


def fpm(k, m, mu, T):
    """(f+, f-): Fermi-Dirac occupations of particles / antiparticles.

    T<=0 -> step function f+ = theta(k_F - k), f- = 0 (k_F = sqrt(mu^2 - m^2))."""
    if T <= 0:
        kF = np.sqrt(max(mu * mu - m * m, 0.0)) if mu > m else 0.0
        return (1.0 if k < kF else 0.0), 0.0
    e = E(k, m)
    fp = 1.0 / (1.0 + np.exp(np.clip((e - mu) / T, -500, 500)))
    fm = 1.0 / (1.0 + np.exp(np.clip((e + mu) / T, -500, 500)))
    return fp, fm


def L_thermal(k, m, mu, T):
    """Matter part of the per-mode free energy [MeV] (prescription A).

    T<=0 -> (mu - E) theta(mu - E).  Uses logaddexp for overflow safety."""
    if T <= 0:
        return max(mu - E(k, m), 0.0)
    e = E(k, m)
    return T * (np.logaddexp(0.0, -(e - mu) / T) + np.logaddexp(0.0, -(e + mu) / T))


def _kupper(m, mu, T):
    """Upper integration limit: Fermi momentum at T=0, else mu + 12T (thermal tail)."""
    if T <= 0:
        return np.sqrt(max(mu * mu - m * m, 0.0))
    return max(mu + 12 * T, 12 * T)


# =============================================================================
#  Section 3 -- per-flavour surface tension & curvature energy
# =============================================================================
def quark_sigma_gamma(mu, T, m, R, g=G_QUARK, prescription='B', include_vac=False,
                      Luv=None):
    """Surface tension sigma [MeV/fm^2] and curvature energy gamma [MeV/fm] of one
    quark flavour at chemical potential ``mu``, temperature ``T``, droplet radius
    ``R``.  Returns ``(sigma, gamma, Lambda_IR)``.

    prescription 'B' (bag, default): MRE weight on the k^4/E pressure integrand
        sigma = -(g/3) int (f+ + f-) f_S k^3/E dk ,  gamma = -(g/3) int (f+ + f-) f_C k^2/E dk
    prescription 'A' (NJL): MRE weight on the per-mode free energy L(k)
        sigma = -g int k f_S L dk ,  gamma = -g int f_C L dk
        (+ optional Dirac-sea/vacuum term up to the UV cutoff Luv if include_vac).
    """
    LIR = Lambda_IR(m, R)
    kup = _kupper(m, mu, T)
    if kup <= LIR:                                # droplet so small everything is cut off
        return 0.0, 0.0, LIR

    if prescription == 'B':
        def iS(k):
            fp, fm = fpm(k, m, mu, T)
            return (fp + fm) * f_S(k, m) * k**3 / E(k, m)

        def iC(k):
            fp, fm = fpm(k, m, mu, T)
            return (fp + fm) * f_C(k, m) * k**2 / E(k, m)

        sig = -(g / 3.0) * quad(iS, LIR, kup, limit=400)[0]
        gam = -(g / 3.0) * quad(iC, LIR, kup, limit=400)[0]

    elif prescription == 'A':
        sig = -g * quad(lambda k: k * f_S(k, m) * L_thermal(k, m, mu, T),
                        LIR, kup, limit=400)[0]
        gam = -g * quad(lambda k: f_C(k, m) * L_thermal(k, m, mu, T),
                        LIR, kup, limit=400)[0]
        if include_vac and Luv is not None:       # NJL Dirac sea (vacuum) term
            sig += -g * quad(lambda k: k * f_S(k, m) * E(k, m), LIR, Luv, limit=400)[0]
            gam += -g * quad(lambda k: f_C(k, m) * E(k, m), LIR, Luv, limit=400)[0]
    else:
        raise ValueError(f"prescription must be 'A' or 'B', got {prescription!r}")

    # Section 6 unit conversion: [MeV^3] -> MeV/fm^2, [MeV^2] -> MeV/fm
    return sig / HBARC**2, gam / HBARC, LIR


def quark_n_components(mu, T, m, g=G_QUARK):
    """Finite-size pieces of one flavour's number density.

    Returns ``(nV, nS, nC)`` with full density n = nV + (3/R) nS + (6/R^2) nC.
    nV [fm^-3], nS [fm^-2], nC [fm^-1] after the hc conversions below.  nS<0, nC<0
    in practice (antiparticles subtract: weight f+ - f-)."""
    LIR = 0.0                                     # density components evaluated without IR cut
    kup = _kupper(m, mu, T)
    d = lambda k: (fpm(k, m, mu, T)[0] - fpm(k, m, mu, T)[1])
    if kup <= 0:
        return 0.0, 0.0, 0.0
    nV = g / (2 * PI2) * quad(lambda k: d(k) * k * k, LIR, kup, limit=400)[0]
    nS = g * quad(lambda k: d(k) * f_S(k, m) * k, LIR, kup, limit=400)[0]
    nC = g * quad(lambda k: d(k) * f_C(k, m), LIR, kup, limit=400)[0]
    return nV / HBARC**3, nS / HBARC**2, nC / HBARC   # -> fm^-3, fm^-2, fm^-1


# =============================================================================
#  Section 4 -- alpha-Bag droplet wrappers (sum over u, d, s)
# =============================================================================
@dataclass
class MREResult:
    """Total surface tension & curvature energy of a 3-flavour quark droplet."""
    sigma: float                                  # MeV/fm^2 (sum over flavours)
    gamma: float                                  # MeV/fm
    T: float
    R: float
    prescription: str
    sigma_flavor: dict = field(default_factory=dict)
    gamma_flavor: dict = field(default_factory=dict)
    Lambda_IR_flavor: dict = field(default_factory=dict)
    nS_flavor: dict = field(default_factory=dict)
    nC_flavor: dict = field(default_factory=dict)
    mu: dict = field(default_factory=dict)

    def density_correction(self, flavor):
        """Finite-size correction (3/R) nS + (6/R^2) nC [fm^-3] to add on the bulk nV."""
        return (3.0 / self.R) * self.nS_flavor[flavor] + \
               (6.0 / self.R**2) * self.nC_flavor[flavor]


def _from_mu(mu_u, mu_d, mu_s, T, R, masses, prescription='B', include_vac=False,
             Luv=None, with_densities=True):
    """Core: sum per-flavour sigma, gamma (and density pieces) at given mu's."""
    mus = {'u': mu_u, 'd': mu_d, 's': mu_s}
    res = MREResult(sigma=0.0, gamma=0.0, T=T, R=R, prescription=prescription, mu=mus)
    for fl in ('u', 'd', 's'):
        s, g_, lir = quark_sigma_gamma(mus[fl], T, masses[fl], R,
                                       prescription=prescription,
                                       include_vac=include_vac, Luv=Luv)
        res.sigma += s
        res.gamma += g_
        res.sigma_flavor[fl] = s
        res.gamma_flavor[fl] = g_
        res.Lambda_IR_flavor[fl] = lir
        if with_densities:
            _, nS, nC = quark_n_components(mus[fl], T, masses[fl])
            res.nS_flavor[fl] = nS
            res.nC_flavor[fl] = nC
    return res


def _masses(params):
    """Quark mass dict from an AlphaBagParams (or any object with m_u/m_d/m_s)."""
    return {'u': float(params.m_u), 'd': float(params.m_d), 's': float(params.m_s)}


def mre_unpaired_from_mu(mu_u, mu_d, mu_s, T, R, params, prescription='B',
                         include_vac=False, Luv=None):
    """sigma, gamma of an *unpaired* alpha-Bag droplet from given quark mu's."""
    return _from_mu(mu_u, mu_d, mu_s, T, R, _masses(params),
                    prescription=prescription, include_vac=include_vac, Luv=Luv)


def mre_cfl_from_mu(mu_u, mu_d, mu_s, T, R, params, prescription='B'):
    """sigma, gamma of a *CFL* droplet from the flavour-locked quark mu's.

    The CFL gap enters sigma, gamma only through these (locked) mu's; the
    Delta^2 mu^2 condensate term is a bulk term and does not appear here."""
    return _from_mu(mu_u, mu_d, mu_s, T, R, _masses(params), prescription=prescription)


def mre_unpaired_beta_eq(n_B, T, R, params, prescription='B', **solve_kw):
    """Solve the unpaired alpha-Bag beta-equilibrium at (n_B, T), then sigma, gamma."""
    from eos.alphabag.eos import solve_alphabag_beta_eq
    r = solve_alphabag_beta_eq(n_B, T, params, **solve_kw)
    return mre_unpaired_from_mu(r.mu_u, r.mu_d, r.mu_s, T, R, params,
                                prescription=prescription)


def mre_cfl_beta_eq(n_B, T, R, Delta0, params, prescription='B', **solve_kw):
    """Solve the CFL alpha-Bag EOS at (n_B, T, Delta0), then sigma, gamma."""
    from eos.alphabag.eos import solve_cfl
    r = solve_cfl(n_B, T, Delta0, params, **solve_kw)
    return mre_cfl_from_mu(r.mu_u, r.mu_d, r.mu_s, T, R, params, prescription=prescription)


# =============================================================================
#  Section 5 -- validation harness (gate: Lambda_IR reference table)
# =============================================================================
def validate(verbose=True):
    """Reproduce the Lambda_IR reference table and the f_S/f_C limits.

    Reference (m = 6.5 MeV): R=3 -> 52.53, R=5 -> 33.54, R=10 -> 18.88 MeV.
    A ~200x error here means the hbar c in x = kR/hc was dropped (Caveat 1)."""
    ref = {3: 52.53, 5: 33.54, 10: 18.88}
    TOL = 5e-3                                     # ~0.2-0.3% agreement is expected (see note)
    ok = True
    for R, target in ref.items():
        got = Lambda_IR(6.5, R)
        rel = abs(got - target) / target
        ok &= rel < TOL
        if verbose:
            print(f"  Lambda_IR(m=6.5, R={R:2d}) = {got:7.3f} MeV  "
                  f"(ref {target}, rel {rel:.2e})")
    assert Lambda_IR(6.5, np.inf) == 0.0, "bulk Lambda_IR must vanish"
    # shape-function limits
    assert abs(float(f_S(1e9, 150.0))) < 1e-6, "f_S -> 0 as k->inf"
    assert abs(float(f_S(1e-6, 150.0)) + 1 / (8 * PI)) < 1e-3, "f_S -> -1/8pi as k->0"
    assert float(f_S(123.0, 0.0)) == 0.0, "massless f_S must be 0"
    assert abs(float(f_C(50.0, 0.0)) + 1 / (24 * PI2)) < 1e-12, "f_C(m=0) = -1/24pi^2"
    if verbose:
        print("  f_S/f_C limits OK; massless f_S = 0 OK.")
    assert ok, "Lambda_IR table mismatch -- check the kR/hc units (Caveat 1)"
    return ok


if __name__ == '__main__':
    print("MRE validation (Lambda_IR table + shape limits):")
    validate()

    # worked points: an alpha-Bag s-quark droplet, T=0 and finite T
    try:
        from eos.alphabag.parameters import get_alphabag_default
        p = get_alphabag_default()                # m_u=m_d=0, m_s=150
    except Exception:
        @dataclass
        class _P:
            m_u: float = 0.0; m_d: float = 0.0; m_s: float = 150.0
        p = _P()

    print("\nSingle s-quark droplet (mu=450, R=5):")
    for T in (0.0, 30.0):
        sB, gB, lir = quark_sigma_gamma(450.0, T, 150.0, 5.0, prescription='B')
        sA, gA, _ = quark_sigma_gamma(450.0, T, 150.0, 5.0, prescription='A')
        print(f"  T={T:4.0f}: sigma_B={sB:7.3f}  sigma_A={sA:7.3f} MeV/fm^2 | "
              f"gamma_B={gB:7.3f} MeV/fm | Lambda_IR={lir:5.2f} MeV")
    print(f"  sigma_u(m=0) = {quark_sigma_gamma(450.0, 0.0, 0.0, 5.0)[0]:.3e} "
          f"MeV/fm^2 (must be 0)")

    print("\n3-flavour unpaired droplet from mu's (mu_u=400, mu_d=mu_s=450, R=8):")
    for T in (0.0, 30.0):
        r = mre_unpaired_from_mu(400.0, 450.0, 450.0, T, 8.0, p)
        print(f"  T={T:4.0f}: sigma={r.sigma:7.3f} MeV/fm^2  gamma={r.gamma:7.3f} MeV/fm "
              f"(s carries {r.sigma_flavor['s']:.3f})")

    # bulk limit: Lambda_IR -> 0 and sigma -> its finite BULK value (~tens MeV/fm^2);
    # it is the contribution sigma*S/V = sigma*3/R to Omega/V that vanishes as R->inf.
    big = quark_sigma_gamma(450.0, 0.0, 150.0, 1e6, prescription='B')
    print(f"\nBulk check R=1e6: sigma={big[0]:.2f} MeV/fm^2 (finite bulk value), "
          f"Lambda_IR={big[2]:.2e} (-> 0)")
