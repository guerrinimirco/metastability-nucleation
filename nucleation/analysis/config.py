"""
Configuration objects for the sigma_crit analysis
=================================================

The notebook used to carry ~40 loose globals into every helper. They are
collected here into three small frozen-ish config objects so a scan's setup is
one explicit argument rather than whatever happened to be in the namespace:

    FilterConfig -- the CFL / unpaired acceptance filters (Witten bound,
                    no-rehadronization, TOV M_max window)
    NucConfig    -- the nucleation setup (tau target, sigma bracket, brentq)
    StarMatch    -- the M_T0 -> central-conditions map for one PNS snapshot

Build a StarMatch with ``make_star_match`` and the cold P_H(mu_B) comparator
with ``build_PH_of_muB``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.interpolate import interp1d

from eos.astro.tov.solver import generate_ec_logspace

# Filter-reason -> integer code. The filters short-circuit cheap->expensive, so
# codes are *nested*: contouring this field at 1.5 / 2.5 gives the Witten / (weak)
# re-hadronization pass edges, and 4.5 (== cfl_ok edge) the full-acceptance edge.
# The no-rehadronization filter is the bulk ΔP(mu_B)=P_quark-P_H test at T=0: a
# cell must pass no_rehad (else 'rehadr', "re-hadr.") AND no_rehad_strong (else
# 'rehad_quasi', "quasi-re-hadr."). merge_rehad_labels folds quasi into 'rehadr'.
REASON_CODE = {'solve': 0, 'witten': 1, 'rehadr': 2, 'rehad_quasi': 3,
               'mmax': 4, 'OK': 5,
               'twoflavor': 6}  # 2-flavor (ud) matter is bound -> nuclei unstable


# =============================================================================
#  Config objects
# =============================================================================
@dataclass
class FilterConfig:
    """Everything the CFL filters need beyond (alpha, B4, Delta0).

    P_H_of_muB / mu_B_H_sorted / P_H_sorted come from ``build_PH_of_muB``."""
    P_H_of_muB: Callable
    mu_B_H_sorted: np.ndarray
    P_H_sorted: np.ndarray
    m_s: float = 100.0
    n_B_grid: np.ndarray = field(default_factory=lambda: np.linspace(0.05, 2.5, 250))
    e_c_vec_tov: np.ndarray = field(
        default_factory=lambda: generate_ec_logspace(e_min=100, e_max=2000, n_points=80))
    M_max_window: tuple = (2.0, np.inf)
    e_over_nB_max: float = 930.0          # 3-flavor SQM must be bound: e/nB|P=0 < this
    T_eos: float = 0.0
    # 2-flavor (ud+e, charge-neutral, beta-eq) matter must NOT be bound, else ordinary
    # nuclei would decay to it -> require e/nB|P=0 > e_over_nB_2flavor (Delta0-independent).
    check_2flavor: bool = True
    e_over_nB_2flavor: float = 930.0
    # TOV backend for the M_max filter and the M-R replay: 'scipy' (trusted
    # reference) or 'fast' (numba, ~100x). The scan/replay already parallelise
    # over grid cells with joblib, so the fast solver is called serially here
    # (compute_tov_sequence is single-threaded) to avoid numba-threads x joblib oversubscription.
    tov_backend: str = 'scipy'
    # --- no-rehadronization filter (bulk ΔP at T=0) -----------------------------
    # The rehadronization filter works on the bulk pressure difference
    # ΔP(mu_B) = P_quark(mu_B) - P_H(mu_B) at T=0, over the mu_B overlap (uses the
    # P_H_of_muB comparator above -- no extra solves). A cell must pass no_rehad
    # (ΔP stays >0 past its crossing -> else 'rehadr') AND no_rehad_strong (ΔP
    # monotonically increasing in mu_B -> else 'rehad_quasi'). Both are acceptance
    # criteria. See _rehad_reason / rehad_flags.
    merge_rehad_labels: bool = False      # fold 'rehad_quasi' into 'rehadr' (one zone)


@dataclass
class NucConfig:
    """Nucleation / root-find setup for sigma_crit."""
    sig_lo: float = 1.0
    sig_hi: float = 300.0
    n_sigma_scan: int = 40          # coarse sigma grid for the robust sigma_crit scan
    tau_target: float = 1e-3        # s
    V: float = 4.18879e51           # fm^3 (sphere R = 100 m)
    include_photons: bool = True
    include_gluons: bool = True
    include_thermal_neutrinos: bool = True


@dataclass
class StarMatch:
    """Maps a cold-beta-eq gravitational mass MT0 to trapped central conditions."""
    H_iso_trapped: dict          # interpolator dict with key 'T'
    H_trapped: dict              # interpolator dict: P, eps, mu_*, Y_C, Y_S
    YLH: float
    S: float
    Mb_of_MT0: Callable          # from cold beta-eq TOV
    nBHc_of_Mb: Callable         # from trapped/isentropic TOV at (YLH, S)


def build_PH_of_muB(H_betaeq: dict, n_B_grid: np.ndarray, T: float):
    """Cold P_H(mu_B) comparator for the no-rehadronization filter.

    Returns (P_H_of_muB, mu_B_H_sorted, P_H_sorted)."""
    mu = H_betaeq['mu_B'](n_B_grid, T)
    P = H_betaeq['P'](n_B_grid, T)
    ok = np.isfinite(mu) & np.isfinite(P)
    order = np.argsort(mu[ok])
    mu_s, P_s = mu[ok][order], P[ok][order]
    return interp1d(mu_s, P_s, kind='linear', bounds_error=False, fill_value=np.nan), mu_s, P_s


def make_star_match(H: dict, YLH: float, S: float,
                    tov_betaeq_path: str, tov_trapped_path: str) -> StarMatch:
    """Build a StarMatch from the H interpolators + the two TOV tables.

    Columns: cold beta-eq TOV 4=M,5=Mb; trapped TOV 2=n_Bc,4=M,5=Mb."""
    tov0 = np.loadtxt(tov_betaeq_path)
    M0, Mb0 = tov0[:, 4], tov0[:, 5]
    i0 = int(np.argmax(M0)) + 1
    Mb_of_MT0 = interp1d(M0[:i0], Mb0[:i0], kind='cubic', bounds_error=True)
    tovh = np.loadtxt(tov_trapped_path)
    nBc, Mh, Mbh = tovh[:, 2], tovh[:, 4], tovh[:, 5]
    ih = int(np.argmax(Mh)) + 1
    nBHc_of_Mb = interp1d(Mbh[:ih], nBc[:ih], kind='cubic', bounds_error=True)
    return StarMatch(H_iso_trapped=H['iso_trapped'], H_trapped=H['trapped'],
                     YLH=YLH, S=S, Mb_of_MT0=Mb_of_MT0, nBHc_of_Mb=nBHc_of_Mb)

