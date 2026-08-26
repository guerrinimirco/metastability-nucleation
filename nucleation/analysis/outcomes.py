"""
PNS evolution outcomes: neutron star, quark star, or black hole
===============================================================

The paper's decision engine, and the one piece of NEW physics relative to the
rest of the package: everything else computes *whether a droplet forms*; this
follows one proto-neutron star through its evolution and says what is left at
the end.

The argument
------------
A PNS is born hot and lepton-rich (t_0), then deleptonizes and heats before
cooling (t_Tmax). Its BARYON number is conserved throughout, but its maximum
gravitational mass is not: the trapped-neutrino EoS is stiffer, so M_max FALLS
during deleptonization. Meanwhile the nucleation threshold sigma_crit changes
too, because both the central density and the temperature move.

So a fixed-M_B star is asked the same question twice, at t_0 and again at
t_Tmax, and there are three ways out:

  QS -- sigma < sigma_crit at either snapshot: a quark droplet nucleates and the
        star converts. The remnant mass follows from BARYON conservation, not
        mass conservation, and the difference (M_PNS - M_QS)c^2 is released.
  BH -- either the star was born above the t_0 maximum mass (prompt collapse),
        or it survives t_0 but deleptonization drops M_max below its M_B.
  NS -- it never nucleates and never exceeds M_max: an ordinary cold neutron
        star of the same baryon number.

sigma_crit definition: star-wide vs centre-only
-----------------------------------------------
``shells`` is a REQUIRED argument of every entry point here. sigma_crit computed
at the centre only, and sigma_crit maximised over shells through the star, are
DIFFERENT QUANTITIES -- the centre-only value underestimates by up to ~2x near
the SQM corner, because nucleation can be fastest in a shell rather than at r=0.
The notebook defaulted one way here and the other way in the saved scan grids,
so a table and a map placed side by side were not comparing the same thing.
Making it explicit is the fix; the column names record which was used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nucleation.conditions import hadronic_point
from nucleation.analysis.sigma_crit import sigma_target_pt, star_shell_states
from nucleation.analysis.stellar import branch_interp, max_masses

# Solar rest energy [erg]: converts a gravitational-mass deficit into the energy
# released by the conversion.
M_SUN_C2_ERG = 1.7827e54


@dataclass
class Snapshot:
    """One moment in the PNS evolution: its (Y_L, S) and its TOV sequence."""
    Y_L: float
    S: float
    tov: np.ndarray
    label: str = ""

    @property
    def M_max(self):
        return max_masses(self.tov)[0]

    @property
    def M_B_max(self):
        return max_masses(self.tov)[1]


@dataclass
class EvolutionTrack:
    """The two PNS snapshots plus the cold sequence they end on.

    Bundles what an outcome decision needs so the row function takes a track
    rather than reaching into a dozen notebook globals.
    """
    t0: Snapshot
    t_max: Snapshot
    cold: np.ndarray
    H_trapped: dict
    H_iso_trapped: dict

    @classmethod
    def build(cls, H_trapped, H_iso_trapped, tov_cold, tov_t0, tov_tmax,
              pns_t0, pns_tmax):
        """From the interpolator dicts and three TOV sequences.

        `pns_t0` / `pns_tmax` are dicts with 'YLH' and 'S'.
        """
        return cls(
            t0=Snapshot(pns_t0['YLH'], pns_t0['S'], np.asarray(tov_t0), 't_0'),
            t_max=Snapshot(pns_tmax['YLH'], pns_tmax['S'], np.asarray(tov_tmax),
                           't_Tmax'),
            cold=np.asarray(tov_cold),
            H_trapped=H_trapped, H_iso_trapped=H_iso_trapped)


def nucleates(sigma_crit, sigma):
    """Does a droplet form at this surface tension?

    sigma < sigma_crit means the barrier is low enough. sigma_crit = +inf is the
    engine's "always nucleates" convention (the hadronic phase is unstable at
    every sigma); NaN means the solve failed and is treated as "no".
    """
    return bool(np.isposinf(sigma_crit)
                or (np.isfinite(sigma_crit) and sigma < sigma_crit))


def sigma_crit_at_snapshot(track, snap, n_B_c, params, Delta0, nuc, *,
                           shells, flavor='saddlepoint',
                           charge='coulomb_minimize', phase='unpCFL',
                           star=None, MT0=None, n_shell_min=0.25):
    """sigma_crit [MeV/fm^2] at one snapshot's central density.

    `shells` is REQUIRED and selects the definition:
      None      -> centre-only sigma_crit
      int n     -> star-wide: the maximum over n shells from n_shell_min to the
                   centre (needs `star` and `MT0`), matching the saved scan grids

    Returns NaN if the central state is not defined (e.g. prompt collapse).
    """
    if not np.isfinite(n_B_c):
        return np.nan
    T_c = float(track.H_iso_trapped['T'](n_B_c, snap.Y_L, snap.S))
    if not np.isfinite(T_c):
        return np.nan
    H_pt = hadronic_point(track.H_trapped, n_B_c, snap.Y_L, T_c)

    shell_states = None
    if shells is not None:
        if star is None or MT0 is None:
            raise ValueError(
                "star-wide sigma_crit (shells=int) needs star= and MT0=; "
                "pass shells=None for the centre-only definition")
        shell_states = star_shell_states(MT0, star, shells, nB_min=n_shell_min)

    return sigma_target_pt(H_pt, T_c, flavor, charge, phase, params, Delta0,
                           nuc, shells=shell_states)


def outcome_row(track, pset, params, Delta0, sigma, M_pns_t0, *, nuc, shells,
                M_QS_of_Mb, n_sat, flavor='saddlepoint',
                charge='coulomb_minimize', phase='unpCFL', label=None,
                star_t0=None, star_tmax=None, MT0=None):
    """Follow ONE fixed-baryon-mass PNS from t_0 to t_Tmax; decide NS / QS / BH.

    Returns (row, sigma_crit_t0, sigma_crit_tmax). The two sigma_crit values come
    back alongside the verdict because they ARE the decision -- a figure that
    plots sigma against sigma_crit shows the reader why the outcome came out the
    way it did, which the verdict alone does not.

    sigma_crit is NaN for a snapshot never reached (prompt BH, or the star had
    already converted at t_0).

    `M_QS_of_Mb` is a callable M_B -> M for the cold quark star (see
    ``stellar.cold_quark_star_branch`` + ``stellar.branch_interp``); conversion
    conserves baryon number, so the remnant is read off at fixed M_B.
    """
    sc0 = sc_T = np.nan
    tag = label if label is not None else str(pset)
    Mb = float(branch_interp(track.t0.tov, 'M', 'M_B')(M_pns_t0))

    def _row(**kw):
        base = dict(params=tag, sigma=sigma, M_t0=M_pns_t0, Mb_t0=Mb,
                    nBc_t0=np.nan, YSc_t0=np.nan, M_tT=np.nan, nBc_tT=np.nan,
                    YSc_tT=np.nan, type='BH', M_rem=np.nan, E_conv_e53=np.nan)
        base.update(kw)
        return base

    # --- prompt BH: born above the t_0 maximum mass -> no stable configuration
    if (M_pns_t0 > track.t0.M_max + 1e-6) or not np.isfinite(Mb):
        return _row(), sc0, sc_T

    # --- t_0 central state, then the nucleation decision --------------------
    nBc0 = float(branch_interp(track.t0.tov, 'M', 'n_Bc')(M_pns_t0))
    Tc0 = float(track.H_iso_trapped['T'](nBc0, track.t0.Y_L, track.t0.S))
    YS0 = float(track.H_trapped['Y_S'](nBc0, track.t0.Y_L, Tc0))
    sc0 = sigma_crit_at_snapshot(track, track.t0, nBc0, params, Delta0, nuc,
                                 shells=shells, flavor=flavor, charge=charge,
                                 phase=phase, star=star_t0, MT0=MT0)

    if nucleates(sc0, sigma):
        # Converts already at t_0; the t_Tmax snapshot is never reached.
        typ, M_conv = 'QS', M_pns_t0
        M_pns_T = nBc_T = YS_T = np.nan
    elif Mb > track.t_max.M_B_max + 1e-6:
        # Survives t_0, but deleptonization drops M_max below its M_B -> collapse.
        typ, M_conv = 'BH', np.nan
        M_pns_T = nBc_T = YS_T = np.nan
    else:
        # --- t_Tmax at FIXED BARYON MASS: ask again -------------------------
        M_pns_T = float(branch_interp(track.t_max.tov, 'M_B', 'M')(Mb))
        nBc_T = float(branch_interp(track.t_max.tov, 'M_B', 'n_Bc')(Mb))
        Tc_T = (float(track.H_iso_trapped['T'](nBc_T, track.t_max.Y_L,
                                               track.t_max.S))
                if np.isfinite(nBc_T) else np.nan)
        YS_T = (float(track.H_trapped['Y_S'](nBc_T, track.t_max.Y_L, Tc_T))
                if np.isfinite(nBc_T) else np.nan)
        sc_T = sigma_crit_at_snapshot(track, track.t_max, nBc_T, params, Delta0,
                                      nuc, shells=shells, flavor=flavor,
                                      charge=charge, phase=phase,
                                      star=star_tmax, MT0=MT0)
        typ, M_conv = (('QS', M_pns_T) if nucleates(sc_T, sigma)
                       else ('NS', np.nan))

    # --- remnant mass and conversion energy ---------------------------------
    if typ == 'QS':
        M_qs = float(M_QS_of_Mb(Mb))
        M_rem = M_qs
        # Energy released = the gravitational-mass deficit at fixed baryon number.
        E_conv = ((M_conv - M_qs) * M_SUN_C2_ERG / 1e53
                  if np.isfinite(M_qs) else np.nan)
    elif typ == 'NS':
        M_rem = float(branch_interp(track.cold, 'M_B', 'M')(Mb))
        E_conv = np.nan
    else:
        M_rem = E_conv = np.nan

    return _row(nBc_t0=nBc0 / n_sat, YSc_t0=YS0, M_tT=M_pns_T,
                nBc_tT=(nBc_T / n_sat if np.isfinite(nBc_T) else np.nan),
                YSc_tT=YS_T, type=typ, M_rem=M_rem, E_conv_e53=E_conv), sc0, sc_T


def outcomes_table(track, psets, sigmas, M_pns_list, *, nuc, shells,
                   M_QS_of_Mb_factory, n_sat, params_factory, label_of=None,
                   **row_kw):
    """The full outcomes table as a pandas DataFrame (rows = set x sigma x M).

    `params_factory(pset) -> alpha-Bag params` and
    `M_QS_of_Mb_factory(pset) -> callable` are passed in so this module does not
    have to know how a quark parameter set is spelled or cached.

    The sigma_crit column name records which DEFINITION was used, because
    star-wide and centre-only values are not interchangeable (see module
    docstring).
    """
    import pandas as pd

    suffix = 'centre' if shells is None else 'star'
    rows = []
    for pset in psets:
        params = params_factory(pset)
        M_QS = M_QS_of_Mb_factory(pset)
        label = label_of(pset) if label_of else None
        for sigma in sigmas:
            for M0 in M_pns_list:
                row, sc0, scT = outcome_row(
                    track, pset, params, pset['Delta0'], sigma, M0,
                    nuc=nuc, shells=shells, M_QS_of_Mb=M_QS, n_sat=n_sat,
                    label=label, **row_kw)
                row[f'sigma_crit_{suffix}_t0'] = sc0
                row[f'sigma_crit_{suffix}_tT'] = scT
                rows.append(row)
    return pd.DataFrame(rows)


def to_latex_tabular(df, float_fmt='%.3g', na_rep='--'):
    """Minimal LaTeX tabular body for `df`.

    Hand-rolled because ``DataFrame.to_latex`` pulls in jinja2, which is not
    otherwise a dependency of this project.
    """
    cols = list(df.columns)
    lines = [r'\begin{tabular}{' + 'l' * len(cols) + '}', r'\hline',
             ' & '.join(str(c).replace('_', r'\_') for c in cols) + r' \\',
             r'\hline']
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float) and not np.isfinite(v):
                cells.append(na_rep)
            elif isinstance(v, float):
                cells.append(float_fmt % v)
            else:
                cells.append(str(v).replace('_', r'\_'))
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\hline', r'\end{tabular}']
    return '\n'.join(lines)


def pick_cell_by_Mmax(sig_crit, M_max, alpha_slices, B4_grid, Delta0_grid,
                      target, tol, m_s):
    """The LOWEST-sigma_crit viable cell whose M_max matches `target`.

    Used to anchor the outcomes table to an observed maximum mass: rather than
    quoting a hand-picked parameter set, pick the cell the data allows that is
    hardest to nucleate, so the reported outcome is the conservative one.

    Returns (pset, M_max, sigma_crit), or (None, nan, nan) if nothing matches.
    """
    sig_crit = np.asarray(sig_crit)
    M_max = np.asarray(M_max)
    ok = np.isfinite(sig_crit) & np.isfinite(M_max) & (np.abs(M_max - target) <= tol)
    if not ok.any():
        return None, np.nan, np.nan
    idx = np.argmin(np.where(ok, sig_crit, np.inf))
    i, j, k = np.unravel_index(idx, sig_crit.shape)
    pset = dict(alpha=float(alpha_slices[i]), B4=float(B4_grid[j]),
                Delta0=float(Delta0_grid[k]), m_s=float(m_s))
    return pset, float(M_max[i, j, k]), float(sig_crit[i, j, k])
