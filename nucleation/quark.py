"""
The alphaBag interface: custom parameter sets and the total droplet thermodynamics
==================================================================================

`eos.alphabag` exposes the quark sector alone -- `thermo_from_mu` and
`cfl_thermo_from_mu` return the quarks plus the bag, with no leptons, no
photons and no gluons in them, because those carry no conserved charge that
the phase's own equilibrium is written in. A nucleating droplet needs the
TOTAL: the pressure that pushes against surface tension and the energy density
that enters the work of formation are the pressure and energy density of
everything inside the droplet.

`eos` assembles that total behind `eos_point`, which also owns the solve. This
package cannot use that door: the Q* composition is fixed by a 4- or 5-equation
saddlepoint of the work of formation, not by an equilibrium mode, and that
solver is the reason this package exists (see `nucleation.composition`). So the
assembly is done here, from the public pieces:

    eos.alphabag.thermo_from_mu / cfl_thermo_from_mu   quarks + bag
    eos.alphabag.gluon_thermo                          gluons
    eos.general.thermodynamics_leptons.electron_thermo electrons
    eos.general.thermodynamics_leptons.photon_thermo   photons
    eos.general.thermodynamics_leptons.neutrino_thermo neutrinos

The sums are

    P_total = P_quark + P_e + P_nu + P_gamma + P_g + N_th P_nu(mu=0)
    e_total = e_quark + e_e + e_nu + e_gamma + e_g + N_th e_nu(mu=0)
    s_total = s_quark + s_e + s_nu + s_gamma + s_g + N_th s_nu(mu=0)
    f_total = e_total - T s_total

every term entering P, eps and s with the same sign -- the bag is already
inside the quark block, and nothing here is a rearrangement term. The trapped
electron neutrino is present only when mu_nu != 0. `N_th` is the number of
thermal (mu = 0) flavours: 3 when nothing is trapped, 2 when the electron
family is, so the trapped flavour is never counted twice.

Units are the repository's throughout: n in fm^-3, mu and T in MeV, P and eps
in MeV/fm^3, s in fm^-3.

Public API
----------
    custom_params(alpha, B4, m_s)                -> eos.alphabag.Parameters
    total_thermo_from_mu(mu_u, mu_d, mu_s, mu_e, T, params, ...)  -> DropletThermo
    cfl_total_thermo_from_mu(..., Delta0, ...)                    -> DropletThermo
"""
from dataclasses import dataclass, replace

from eos.alphabag.parameters import Parameters
from eos.alphabag.thermodynamics import (
    cfl_thermo_from_mu, gluon_thermo, thermo_from_mu,
)
from eos.general.thermodynamics_leptons import (
    electron_thermo, neutrino_thermo, photon_thermo,
)


def custom_params(alpha, B4, m_s):
    """An alphaBag parameter set with the three axes this study scans.

    `alpha` is alpha_s, `B4` is B^(1/4) in MeV and `m_s` the strange current
    mass in MeV. Everything else keeps the shipped defaults. The set is
    *custom* rather than published, which is why it is built here by
    `dataclasses.replace` and not asked of `eos` -- `Parameters.named()` is
    reserved for parameter sets that appear in the literature.
    """
    return replace(Parameters.default(), name="alphabag_custom",
                   alpha=alpha, B4=B4, m_s=m_s)


@dataclass
class DropletThermo:
    """Everything inside a quark droplet at given chemical potentials.

    The quark block's own fields (densities, conserved charges and their
    potentials) alongside the totals over quarks, leptons, photons and gluons.
    Both the unpaired and the CFL phase come back in this one shape; in the
    CFL phase the flavour lock fixes Y_C = 0 and Y_S = 1 rather than solving
    for them.
    """
    # State
    n_B: float = 0.0        # baryon density (fm^-3)
    n_C: float = 0.0        # non-leptonic charge density (fm^-3)
    T: float = 0.0          # temperature (MeV)
    # Non-leptonic fractions, per baryon
    Y_C: float = 0.0        # charge fraction (quarks only -- leptons excluded)
    Y_S: float = 0.0        # strangeness fraction, S = +1 per s quark
    Y_u: float = 0.0        # up quark fraction
    Y_d: float = 0.0        # down quark fraction
    Y_s: float = 0.0        # strange quark fraction
    # Lepton fractions, per baryon
    Y_e: float = 0.0        # electron fraction
    Y_nu: float = 0.0       # trapped electron-neutrino fraction
    # Chemical potentials (MeV)
    mu_u: float = 0.0
    mu_d: float = 0.0
    mu_s: float = 0.0
    mu_e: float = 0.0
    mu_nu: float = 0.0      # trapped electron neutrino, 0 when free-streaming
    mu_B: float = 0.0       # mu_u + 2 mu_d
    mu_C: float = 0.0       # mu_u - mu_d
    mu_S: float = 0.0       # mu_s - mu_d
    # Totals over every sector present
    P_total: float = 0.0    # pressure (MeV/fm^3)
    e_total: float = 0.0    # energy density (MeV/fm^3)
    s_total: float = 0.0    # entropy density (fm^-3)
    f_total: float = 0.0    # free energy density f = eps - T s (MeV/fm^3)


def _assemble(quark, mu_e, mu_nu, T, alpha,
              include_photons, include_gluons, include_thermal_neutrinos):
    """Add the chargeless and leptonic sectors to a quark block.

    `quark` is an `eos.alphabag` MatterThermo or CFLThermo: the two carry the
    same names for everything read here, so the paired and unpaired phases
    share this assembly.
    """
    thermo_e = electron_thermo(mu_e, T)

    P_total = quark.P + thermo_e.P
    e_total = quark.e + thermo_e.e
    s_total = quark.s + thermo_e.s

    # The trapped electron neutrino, when there is one.
    n_nu = 0.0
    if mu_nu != 0.0:
        thermo_nu = neutrino_thermo(mu_nu, T)
        n_nu = thermo_nu.n
        P_total += thermo_nu.P
        e_total += thermo_nu.e
        s_total += thermo_nu.s

    if include_photons:
        thermo_gamma = photon_thermo(T)
        P_total += thermo_gamma.P
        e_total += thermo_gamma.e
        s_total += thermo_gamma.s

    if include_gluons:
        thermo_g = gluon_thermo(T, alpha)
        P_total += thermo_g.P
        e_total += thermo_g.e
        s_total += thermo_g.s

    # The flavours held at mu = 0. Two when the electron family is trapped
    # above, three when it is not, so no flavour is counted twice.
    if include_thermal_neutrinos:
        thermo_nu_th = neutrino_thermo(0.0, T)
        n_thermal_flavors = 2.0 if mu_nu != 0.0 else 3.0
        P_total += n_thermal_flavors * thermo_nu_th.P
        e_total += n_thermal_flavors * thermo_nu_th.e
        s_total += n_thermal_flavors * thermo_nu_th.s

    n_B = quark.n_B
    return DropletThermo(
        n_B=n_B, n_C=quark.n_C, T=T,
        Y_C=quark.Y_C, Y_S=quark.Y_S,
        Y_u=quark.n_u / n_B if n_B > 0 else 0.0,
        Y_d=quark.n_d / n_B if n_B > 0 else 0.0,
        Y_s=quark.n_s / n_B if n_B > 0 else 0.0,
        Y_e=thermo_e.n / n_B if n_B > 0 else 0.0,
        Y_nu=n_nu / n_B if n_B > 0 else 0.0,
        mu_u=quark.mu_u, mu_d=quark.mu_d, mu_s=quark.mu_s,
        mu_e=mu_e, mu_nu=mu_nu,
        mu_B=quark.mu_B, mu_C=quark.mu_C, mu_S=quark.mu_S,
        P_total=P_total, e_total=e_total, s_total=s_total,
        f_total=e_total - T * s_total,
    )


def total_thermo_from_mu(mu_u, mu_d, mu_s, mu_e, T, params,
                         include_photons=True, include_gluons=True,
                         include_thermal_neutrinos=True, mu_nu=0.0):
    """An unpaired quark droplet: quarks + bag + electrons + the flagged gases.

    Args:
        mu_u, mu_d, mu_s: quark chemical potentials (MeV)
        mu_e: electron chemical potential (MeV)
        T: temperature (MeV)
        params: an `eos.alphabag.Parameters`
        include_photons, include_gluons, include_thermal_neutrinos: sectors
        mu_nu: trapped electron-neutrino potential (MeV); 0 means
               free-streaming, and then all three flavours are thermal

    Returns:
        DropletThermo
    """
    quark = thermo_from_mu(mu_u, mu_d, mu_s, T, params)
    return _assemble(quark, mu_e, mu_nu, T, params.alpha,
                     include_photons, include_gluons,
                     include_thermal_neutrinos)


def cfl_total_thermo_from_mu(mu_u, mu_d, mu_s, mu_e, T, Delta0, params,
                             include_photons=True, include_gluons=True,
                             include_thermal_neutrinos=True, mu_nu=0.0):
    """A colour-flavour locked droplet, same assembly as the unpaired one.

    The electrons are kept: the phase is neutral in isolation, but a droplet
    under GLOBAL charge neutrality sits at the hadronic mu_e and carries the
    electron gas that goes with it.

    Args:
        mu_u, mu_d, mu_s: quark chemical potentials (MeV)
        mu_e: electron chemical potential (MeV)
        T: temperature (MeV)
        Delta0: zero-temperature pairing gap (MeV)
        params: an `eos.alphabag.Parameters`
        include_photons, include_gluons, include_thermal_neutrinos: sectors
        mu_nu: trapped electron-neutrino potential (MeV)

    Returns:
        DropletThermo
    """
    quark = cfl_thermo_from_mu(mu_u, mu_d, mu_s, T, Delta0, params)
    return _assemble(quark, mu_e, mu_nu, T, params.alpha,
                     include_photons, include_gluons,
                     include_thermal_neutrinos)
