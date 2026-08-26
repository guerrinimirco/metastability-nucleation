"""
Parameter-plane scans
=====================

Sweep the acceptance filters and sigma_crit over a (alpha_s, B^1/4, Delta_0)
grid, cache the (expensive, sigma-independent) filter pass so repeated scans at
different M_T0 or droplet phase reuse it, and save the result as .npz.

This is Part-II work: DATA PRODUCTION. `plot_sigma_crit_grid` lives here as a
quick look at a scan you have just run -- the paper figures are composed from
the primitives in `nucleation.analysis.figure.maps` instead, so that the
published layers have exactly one implementation.
"""
from __future__ import annotations

import time

import numpy as np

from nucleation.quark import custom_params
from nucleation.analysis.config import FilterConfig, NucConfig, StarMatch, REASON_CODE
from nucleation.analysis.filters import (passes_cfl_filters,
                                         passes_unpaired_filters,
                                         ud_eps_per_nB)
from nucleation.analysis.sigma_crit import (
    central_state, star_shell_states, sigma_target_pt,
)

try:
    import joblib  # noqa: F401  -- presence check; Parallel imported in-function
    _HAVE_JOBLIB = True
except Exception:
    _HAVE_JOBLIB = False

# =============================================================================
#  Grid scan (filter cache + optional joblib)
# =============================================================================
_filter_cache = {}


def _rehad_key(cfg: FilterConfig):
    # Scalar signature of the no-rehadronization filter for cache invalidation.
    # ponytail: the ΔP(mu_B) profile itself is fixed by (alpha,B4,Delta0)+P_H, all
    # already in the key; only the quasi-vs-rehadr labelling toggle is extra.
    return (bool(cfg.merge_rehad_labels),)


def _grid_key(alpha_slices, B4_grid, Delta0_grid, cfg: FilterConfig):
    # Include the acceptance constants so the cache invalidates when they change.
    return (tuple(np.round(np.atleast_1d(alpha_slices), 6)),
            tuple(np.round(np.asarray(B4_grid), 6)),
            tuple(np.round(np.asarray(Delta0_grid), 6)),
            float(cfg.m_s), tuple(cfg.M_max_window), float(cfg.e_over_nB_max),
            bool(cfg.check_2flavor), float(cfg.e_over_nB_2flavor),
            _rehad_key(cfg))


def scan_cfl_filters(alpha_slices, B4_grid, Delta0_grid, cfg: FilterConfig,
                     reuse=True, verbose=True, n_jobs=-1):
    """(cfl_ok, M_max, reason) stacks of shape (NA, ND, NB) over the grid.

    The expensive part (CFL EoS + TOV per cell); INDEPENDENT of MT0 and of the
    nucleation method, so cached on the grid + acceptance constants. Cells are
    independent -> joblib-parallel (n_jobs=-1 all cores, 1 serial; serial if no
    joblib)."""
    alpha_slices = np.atleast_1d(np.asarray(alpha_slices, float))
    key = _grid_key(alpha_slices, B4_grid, Delta0_grid, cfg)
    if reuse and key in _filter_cache:
        if verbose:
            print("CFL filter: reusing cached grid result.", flush=True)
        return _filter_cache[key]
    NA, ND, NB = len(alpha_slices), len(Delta0_grid), len(B4_grid)
    cfl = np.zeros((NA, ND, NB), dtype=bool)
    mm = np.full((NA, ND, NB), np.nan)
    rs = np.zeros((NA, ND, NB), dtype=np.int8)
    idx = [(ia, i, jx) for ia in range(NA) for i in range(ND) for jx in range(NB)]
    use_par = (n_jobs != 1) and _HAVE_JOBLIB
    if verbose:
        print(f"CFL filter scan: {NA}x{ND}x{NB}={len(idx)} cells, "
              f"{('parallel n_jobs=%d' % n_jobs) if use_par else 'serial'}...", flush=True)
    t0 = time.perf_counter()
    if use_par:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=n_jobs, verbose=(10 if verbose else 0))(
            delayed(passes_cfl_filters)(alpha_slices[ia], B4_grid[jx], Delta0_grid[i], cfg)
            for (ia, i, jx) in idx)
    else:
        res = [passes_cfl_filters(alpha_slices[ia], B4_grid[jx], Delta0_grid[i], cfg)
               for (ia, i, jx) in idx]
    for (ia, i, jx), (ok, M_max, reason) in zip(idx, res):
        cfl[ia, i, jx] = ok
        mm[ia, i, jx] = M_max
        rs[ia, i, jx] = REASON_CODE[reason]
    n_2f = 0
    if cfg.check_2flavor:
        # 2-flavor (ud) stability: reject (alpha,B4) where ud matter is bound
        # (e/nB|P=0 <= threshold) -> ordinary nuclei would decay. Delta0-independent,
        # so one solve per (alpha,B4) column, broadcast across Delta0.
        cols = [(ia, jx) for ia in range(NA) for jx in range(NB)]
        if use_par:
            from joblib import Parallel, delayed
            ud = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(ud_eps_per_nB)(alpha_slices[ia], B4_grid[jx], cfg)
                for (ia, jx) in cols)
        else:
            ud = [ud_eps_per_nB(alpha_slices[ia], B4_grid[jx], cfg) for (ia, jx) in cols]
        for (ia, jx), v in zip(cols, ud):
            if np.isfinite(v) and v <= cfg.e_over_nB_2flavor:
                bad = rs[ia, :, jx] == REASON_CODE['OK']   # only override passing cells
                cfl[ia, bad, jx] = False
                rs[ia, bad, jx] = REASON_CODE['twoflavor']
                n_2f += int(bad.sum())
    if verbose:
        npass = int(cfl.sum())
        c = {n: int((rs == k).sum()) for n, k in REASON_CODE.items()}  # reason histogram
        Mok = mm[cfl]
        mr = (f"M_max in [{Mok.min():.2f}, {Mok.max():.2f}] M_sun"
              if Mok.size else "no surviving cells")
        print(f"  CFL filter done in {time.perf_counter()-t0:.1f}s: "
              f"pass {npass}/{len(idx)} ({100*npass/max(len(idx),1):.0f}%), {mr}\n"
              f"    rejected -> no CFL solve: {c['solve']}, "
              f"Witten (unstable / e/nB>{cfg.e_over_nB_max:g}): {c['witten']}, "
              f"re-hadr: {c['rehadr']}, quasi-re-hadr: {c['rehad_quasi']}, "
              f"M_max outside {cfg.M_max_window}: {c['mmax']}, "
              f"2-flavor bound (e/nB<={cfg.e_over_nB_2flavor:g}): {c['twoflavor']}",
              flush=True)
    _filter_cache[key] = (cfl, mm, rs)
    return cfl, mm, rs


def scan_unpaired_filters(alpha_slices, B4_grid, cfg: FilterConfig,
                          reuse=True, verbose=True, n_jobs=-1):
    """(ok, M_max, reason) stacks over the (alpha_s, B4) grid for UNPAIRED matter.

    Companion to ``scan_cfl_filters`` for the gap-free unpaired channel. Returns
    the same 3-D ``(NA, ND, NB)`` layout with a SINGLETON Delta_0 axis (ND=1), so
    the result drops straight into ``compute_sigma_crit`` / ``plot_sigma_crit_grid``
    (which expect an (alpha, Delta_0, B4) stack) without special-casing. Per cell
    it runs ``passes_unpaired_filters`` (no-rehadronization + unpaired-EoS M_max);
    joblib-parallel over cells, cached on the grid + acceptance constants."""
    alpha_slices = np.atleast_1d(np.asarray(alpha_slices, float))
    key = ('unpaired', tuple(np.round(alpha_slices, 6)),
           tuple(np.round(np.asarray(B4_grid), 6)),
           float(cfg.m_s), tuple(cfg.M_max_window), float(cfg.e_over_nB_max),
           bool(cfg.check_2flavor), float(cfg.e_over_nB_2flavor),
           _rehad_key(cfg))
    if reuse and key in _filter_cache:
        if verbose:
            print("Unpaired filter: reusing cached grid result.", flush=True)
        return _filter_cache[key]
    NA, NB = len(alpha_slices), len(B4_grid)
    idx = [(ia, jx) for ia in range(NA) for jx in range(NB)]
    use_par = (n_jobs != 1) and _HAVE_JOBLIB
    if verbose:
        print(f"Unpaired filter scan: {NA}x{NB}={len(idx)} cells, "
              f"{('parallel n_jobs=%d' % n_jobs) if use_par else 'serial'}...", flush=True)
    t0 = time.perf_counter()
    if use_par:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=n_jobs, verbose=(10 if verbose else 0))(
            delayed(passes_unpaired_filters)(alpha_slices[ia], B4_grid[jx], cfg)
            for (ia, jx) in idx)
    else:
        res = [passes_unpaired_filters(alpha_slices[ia], B4_grid[jx], cfg)
               for (ia, jx) in idx]
    cfl = np.zeros((NA, 1, NB), dtype=bool)
    mm = np.full((NA, 1, NB), np.nan)
    rs = np.zeros((NA, 1, NB), dtype=np.int8)
    for (ia, jx), (ok, M_max, reason) in zip(idx, res):
        cfl[ia, 0, jx] = ok
        mm[ia, 0, jx] = M_max
        rs[ia, 0, jx] = REASON_CODE[reason]
    if verbose:
        npass = int(cfl.sum())
        c = {n: int((rs == k).sum()) for n, k in REASON_CODE.items()}
        print(f"  Unpaired filter done in {time.perf_counter()-t0:.1f}s: "
              f"pass {npass}/{len(idx)} ({100*npass/max(len(idx),1):.0f}%); "
              f"rejected -> no solve: {c['solve']}, re-hadr: {c['rehadr']}, "
              f"quasi-re-hadr: {c['rehad_quasi']}, "
              f"M_max outside {cfg.M_max_window}: {c['mmax']}", flush=True)
    _filter_cache[key] = (cfl, mm, rs)
    return cfl, mm, rs


def compute_sigma_crit(cfl_ok, MT0, flavor, charge, phase, alpha_slices, B4_grid,
                       Delta0_grid, star: StarMatch, nuc: NucConfig, m_s=100.0,
                       n_jobs=-1, verbose=True, n_shells=1, nB_shell_min=0.25):
    """sigma_crit stack at one MT0 / method, only on CFL-pass cells. H_pt (plain
    floats) is computed once and shared -> joblib-parallel over CFL-pass cells.

    n_shells=1 (default) evaluates at the star CENTRE only. n_shells>1 makes
    sigma_crit STAR-WIDE: tau > tau_target is demanded at n_shells densities
    from nB_shell_min up to the centre (see star_shell_states) -- the centre is
    not always the easiest nucleation site."""
    alpha_slices = np.atleast_1d(np.asarray(alpha_slices, float))
    NA, ND, NB = cfl_ok.shape
    nBHc, T_c, H_pt = central_state(MT0, star)
    shells = (star_shell_states(MT0, star, n_shells, nB_shell_min)
              if n_shells > 1 else None)
    sig = np.full((NA, ND, NB), np.nan)
    cells = [(ia, i, jx) for ia in range(NA) for i in range(ND) for jx in range(NB)
             if cfl_ok[ia, i, jx]]
    use_par = (n_jobs != 1) and _HAVE_JOBLIB
    if verbose:
        print(f"  sigma_crit MT0={MT0:.2f} {flavor}/{charge}/{phase}: {len(cells)} "
              f"CFL-pass cells (nBHc={nBHc:.4f}, T_c={T_c:.2f} MeV, "
              f"{'star-wide %d shells' % n_shells if shells else 'centre only'}), "
              f"{'parallel' if use_par else 'serial'}", flush=True)
    if not cells:
        return sig
    pars = [custom_params(alpha=alpha_slices[ia], B4=B4_grid[jx], m_s=m_s)
            for (ia, i, jx) in cells]
    D0s = [Delta0_grid[i] for (ia, i, jx) in cells]
    t0 = time.perf_counter()
    if use_par:
        from joblib import Parallel, delayed
        vals = Parallel(n_jobs=n_jobs, verbose=(10 if verbose else 0))(
            delayed(sigma_target_pt)(H_pt, T_c, flavor, charge, phase, pp, d0,
                                     nuc, shells=shells)
            for pp, d0 in zip(pars, D0s))
    else:
        vals = [sigma_target_pt(H_pt, T_c, flavor, charge, phase, pp, d0, nuc,
                                shells=shells)
                for pp, d0 in zip(pars, D0s)]
    for (ia, i, jx), v in zip(cells, vals):
        sig[ia, i, jx] = v
    if verbose:
        fin = sig[np.isfinite(sig)]
        nnuc, nbrk = fin.size, len(cells) - fin.size
        rng = (f"sigma_crit in [{fin.min():.1f}, {fin.max():.1f}] MeV/fm^2"
               if nnuc else "none nucleate in bracket")
        print(f"    -> nucleating (finite sigma_crit): {nnuc}/{len(cells)} CFL-pass "
              f"cells in {time.perf_counter()-t0:.1f}s; {nbrk} never reach tau_target "
              f"in sigma=[{nuc.sig_lo:g}, {nuc.sig_hi:g}]. {rng}", flush=True)
    return sig


def run_sigma_crit_scan(MT0_grid_thr, flavor, charge, phase, alpha_slices,
                        B4_grid, Delta0_grid, cfg: FilterConfig, nuc: NucConfig,
                        star: StarMatch, n_jobs=-1, reuse_filter=True,
                        save_path_fmt=None, xsd_tag='', extra_save=None,
                        n_shells=1, nB_shell_min=0.25):
    """CFL filters (cached) + sigma_crit per MT0. Returns
    {MT0: dict(sig_crit, cfl_ok, M_max, reason)}. If save_path_fmt is given it is
    formatted with (xsd=xsd_tag, MT0=MT0, flavor, charge, phase) and an .npz saved.
    n_shells>1 -> STAR-WIDE sigma_crit (see compute_sigma_crit)."""
    alpha_slices = np.atleast_1d(np.asarray(alpha_slices, float))
    cfl, mm, rs = scan_cfl_filters(alpha_slices, B4_grid, Delta0_grid, cfg,
                                   reuse=reuse_filter, n_jobs=n_jobs)
    out = {}
    for MT0 in np.atleast_1d(MT0_grid_thr):
        MT0 = float(MT0)
        sig = compute_sigma_crit(cfl, MT0, flavor, charge, phase, alpha_slices,
                                 B4_grid, Delta0_grid, star, nuc, m_s=cfg.m_s,
                                 n_jobs=n_jobs, n_shells=n_shells,
                                 nB_shell_min=nB_shell_min)
        if save_path_fmt:
            npz = save_path_fmt.format(xsd=xsd_tag, MT0=MT0, flavor=flavor,
                                       charge=charge, phase=phase)
            payload = dict(B4_grid=B4_grid, Delta0_grid=Delta0_grid,
                           alpha_slices=alpha_slices, sig_crit=sig, cfl_ok=cfl,
                           M_max=mm, reason=rs, MT0=MT0, m_s=cfg.m_s,
                           mass_window=np.array(cfg.M_max_window, dtype=float),
                           flavor=flavor, charge=charge, phase=phase,
                           tau=nuc.tau_target)
            if extra_save:
                payload.update(extra_save)
            np.savez(npz, **payload)
            print(f"  saved -> {npz}", flush=True)
        out[MT0] = dict(sig_crit=sig, cfl_ok=cfl, M_max=mm, reason=rs)
    return out


# =============================================================================
#  Plotting
# =============================================================================
def plot_sigma_crit_grid(B4_grid, Delta0_grid, alpha_slices, sig_crit, cfl_ok,
                         M_max, reason, scan_label, mass_window=(2.0, np.inf),
                         title_extra='', show_split_grey=True, show_cfl_boundary=True,
                         show_filter_lines=True, show_mmax_lines=True,
                         show_sigcrit_lines=False, show_param_sets=True,
                         param_sets=None, mc_csv=None, alpha_tol=0.04):
    """sigma_crit heatmaps over (B4, Delta0), one panel per alpha, toggleable
    overlays. Pure function of the scan arrays -> use it to re-plot a reloaded npz."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    try:
        import pandas as pd
    except Exception:
        pd = None

    alpha_slices = np.atleast_1d(alpha_slices)
    NA = len(alpha_slices)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad((0, 0, 0, 0) if show_split_grey else 'lightgray')
    finite = sig_crit[np.isfinite(sig_crit)]
    vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)

    mc = None
    if show_param_sets and mc_csv and pd is not None:
        try:
            mc = pd.read_csv(mc_csv)
        except Exception:
            mc = None

    fig, axes = plt.subplots(1, NA, figsize=(4.6 * NA, 4.4), squeeze=False, sharey=True)
    for ia, alpha in enumerate(alpha_slices):
        ax = axes[0, ia]
        sa, ca, ma, ra = sig_crit[ia], cfl_ok[ia], M_max[ia], reason[ia].astype(float)
        if show_split_grey:
            # categorical background for the non-finite sigma_crit cells:
            #   0 grey  = not CFL-viable
            #   1 orange= CFL-viable but never nucleates in [sig_lo, sig_hi]  (NaN)
            #   2 teal  = CFL-viable and nucleates for ALL tested sigma        (+inf,
            #             i.e. sigma_crit lies above sig_hi -> most permissive)
            cat = np.where(np.isfinite(sa), np.nan,
                           np.where(~ca, 0.0, np.where(np.isposinf(sa), 2.0, 1.0)))
            ax.pcolormesh(B4_grid, Delta0_grid, np.ma.masked_invalid(cat),
                          cmap=ListedColormap(['0.85', '#f4a261', '#2a9d8f']),
                          vmin=0, vmax=2, shading='nearest')
        pcm = ax.pcolormesh(B4_grid, Delta0_grid, np.ma.masked_invalid(sa),
                            cmap=cmap, vmin=vmin, vmax=vmax, shading='nearest')
        if show_cfl_boundary and ca.any() and (~ca).any():
            ax.contour(B4_grid, Delta0_grid, ca.astype(float), levels=[0.5],
                       colors='k', linewidths=1.1, linestyles='--')
        if show_filter_lines:
            # Witten / no-rehadronization pass edges, labelled inline (no legend).
            for lev, name in [(1.5, 'Witten'), (2.5, 'rehadronization')]:
                if (ra > lev).any() and (ra < lev).any():
                    cf = ax.contour(B4_grid, Delta0_grid, ra, levels=[lev],
                                    colors='tab:blue', linewidths=1.1)
                    ax.clabel(cf, fmt={lev: name}, fontsize=8, inline=True)
        if show_mmax_lines and np.isfinite(ma).any():
            win = sorted(lv for lv in mass_window if np.isfinite(lv))
            wskip = {round(w, 1) for w in win}
            # white iso-M_max guides every 0.2 Msun, from 1.8 up to whatever appears
            # in the panel, skipping the red acceptance-window level(s).
            wl = [round(x, 1) for x in np.arange(1.8, 4.0001, 0.2)
                  if round(x, 1) not in wskip
                  and np.nanmin(ma) <= round(x, 1) <= np.nanmax(ma)]
            if wl:
                cs = ax.contour(B4_grid, Delta0_grid, ma, levels=wl,
                                colors='white', linewidths=0.9, alpha=0.9)
                ax.clabel(cs, fmt=r'%.1f $M_\odot$', fontsize=7, inline=True)
            if win:
                csw = ax.contour(B4_grid, Delta0_grid, ma, levels=win,
                                 colors='red', linewidths=1.6)
                ax.clabel(csw, fmt=r'%.2f $M_\odot$', fontsize=7, inline=True)
        if show_sigcrit_lines and np.isfinite(sa).any():
            cc = ax.contour(B4_grid, Delta0_grid, sa, levels=6,
                            colors='k', linewidths=0.5, alpha=0.5)
            ax.clabel(cc, fmt='%.0f', fontsize=6, inline=True)
        if show_param_sets:
            if mc is not None and {'alpha', 'B4', 'Delta0'} <= set(mc.columns):
                sel = np.abs(mc['alpha'].to_numpy() - alpha) < alpha_tol
                ax.scatter(mc['B4'].to_numpy()[sel], mc['Delta0'].to_numpy()[sel],
                           s=8, c='k', marker='.', alpha=0.5)
            for p in (param_sets or []):
                ja = int(np.argmin(np.abs(alpha_slices - p['alpha'])))
                if ja == ia:
                    ax.scatter([p['B4']], [p['Delta0']], s=110, marker='*',
                               c='yellow', edgecolors='k', zorder=5)
        # In-region text labels (replace the legend): centroid of each region's cells.
        for mask, txt, col in [(ra == REASON_CODE['mmax'], r'$M_{\max}<2$', '0.2'),
                               (ra == REASON_CODE['twoflavor'], '2-flavor\nbound', 'darkred'),
                               (ra == REASON_CODE['rehadr'], 're-hadr.', 'navy'),
                               (ra == REASON_CODE['rehad_quasi'], 'quasi-\nre-hadr.', 'purple'),
                               (ca & np.isnan(sa), 'no nucleation', 'saddlebrown'),
                               (ca & np.isposinf(sa), r'$\sigma_{\rm c}>\sigma_{\rm hi}$', 'teal')]:
            if mask.any():
                r_, c_ = np.where(mask)
                ax.text(float(B4_grid[c_].mean()), float(Delta0_grid[r_].mean()),
                        txt, color=col, fontsize=8, fontweight='bold',
                        ha='center', va='center')
        ax.set_title(rf"$\alpha_s$={alpha:.2f}")
        ax.set_xlabel(r'$B^{1/4}$ [MeV]')
        ax.set_xlim(np.min(B4_grid), np.max(B4_grid))
        ax.set_ylim(np.min(Delta0_grid), np.max(Delta0_grid))
        if ia == 0:
            ax.set_ylabel(r'$\Delta_0$ [MeV]')

    fig.colorbar(pcm, ax=axes.ravel().tolist(), label=r'$\sigma_{\rm crit}$ [MeV/fm$^2$]')
    fig.suptitle(rf"Acceptable region — {scan_label}{title_extra}", y=1.04)
    plt.show()
    return fig
