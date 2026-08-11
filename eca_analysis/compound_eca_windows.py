"""Compound-event ECA over sliding 30-year windows, with significance band.

Reproduces the multi-panel regional figures (fixed-threshold and
thermodynamic-threshold) from the original rpy2/CoinCalc analysis, using the
pure-Python engine in :mod:`eca_analysis`.

For each region and each overlapping 30-year window (1980-2010 ... 2050-2080)
this computes, on the JJAS extreme-rain series:

* ``k_trig``   -- observed trigger-coincidence count (self-ECA): the number of
                  extreme days that are followed, within the ECA window, by
                  another extreme day. This is the red point in the figures.
* ``band_lo`` / ``band_hi`` -- the 2.5 / 97.5 % envelope of the binomial null
                  distribution of that count (blue shaded band).
* ``significant`` -- 1 if ``k_trig > band_hi`` else 0 (the ``results_*`` dicts).

Two reproduction modes (``year_blocked``):
* ``True``  (default) -- JJAS days are grouped by year within each window;
  coincidences and the null never bridge a Sep->next-Jun boundary. The null is
  an exact Poisson-binomial over the ~30 JJAS blocks. Consistent with the rest
  of eca_analysis.
* ``False`` -- JJAS months concatenated across the window (Sep abuts next Jun),
  a single binomial null. Faithful to the original figures; numbers run higher
  because cross-year coincidences are retained.

Threshold regimes:
* fixed -- a constant threshold (default 20 mm/hr).
* thermodynamic -- for each window, the value matching the *baseline* window's
  20-mm percentile in that window's own distribution (a Clausius-Clapeyron
  style shift). As in the original, the thermodynamic figure draws its band
  from the shifted-threshold series but keeps the red points on the fixed
  threshold, so the two are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from .eca import (
    SEASON_JJAS, eca_blocked, eca_ts, flag_wet_events, jjas_year_blocks,
    null_band, trigger_null_pmf,
)

__all__ = [
    "WindowConfig", "sliding_windows", "get_percentile",
    "value_from_percentile", "analyse_region", "run_window_analysis",
    "plot_region_panels", "plot_region_4panel",
]

DEFAULT_PERIODS = ("1980-2010", "1990-2020", "2000-2030", "2010-2040",
                   "2020-2050", "2030-2060", "2040-2070", "2050-2080")


@dataclass
class WindowConfig:
    data_dir: str
    out_dir: str
    ensembles: Sequence[str] = ("2868",)
    regions: Sequence[str] = ()                # (code, display_name) handled below
    region_names: Optional[dict] = None        # code -> pretty name for titles
    fixed_threshold: float = 20.0
    delT: int = 4
    tau: int = 1
    len_wet: int = 1
    season: Sequence[int] = SEASON_JJAS
    periods: Sequence[str] = DEFAULT_PERIODS
    window_width: int = 30
    window_step: int = 10
    baseline_period: str = "1980-2010"
    year_blocked: bool = True                  # observed counts never bridge
    #                                            Sep -> next-Jun when True
    null_model: str = "pooled"                 # "pooled": Eq (1) binomial over
    #                                            the whole window (paper method);
    #                                            "blocked": exact per-season sum
    thermo_points: str = "thermo"              # "thermo" (this 4-panel script)
    #                                            or "fixed" (map/multi-panel code)
    band_lo: float = 0.025
    band_hi: float = 0.975
    ensemble_prefix: str = "p110"
    detector: Callable[[np.ndarray, int, float], np.ndarray] = field(
        default=flag_wet_events)


# --------------------------------------------------------------------------- #
# windows & thresholds
# --------------------------------------------------------------------------- #
def sliding_windows(periods=DEFAULT_PERIODS):
    """Parse period labels 'YYYY-YYYY' into (label, start_year, end_year)."""
    out = []
    for lbl in periods:
        a, b = lbl.split("-")
        out.append((lbl, int(a), int(b)))
    return out


def get_percentile(value, data):
    """Percentile rank of ``value`` within ``data`` (flattened)."""
    from scipy import stats
    return stats.percentileofscore(np.asarray(data).ravel(), value, kind="rank")


def value_from_percentile(percentile, data):
    """Value at ``percentile`` of ``data`` (flattened)."""
    return float(np.percentile(np.asarray(data).ravel(), percentile))


# --------------------------------------------------------------------------- #
# core per-window ECA + null band
# --------------------------------------------------------------------------- #
def _window_masks(years, months, season, windows):
    """For each window, the boolean mask of in-season days whose year is in
    [start, end]. Returns list of (label, mask)."""
    years = np.asarray(years)
    months = np.asarray(months)
    in_season = np.isin(months, list(season))
    out = []
    for lbl, y0, y1 in windows:
        out.append((lbl, in_season & (years >= y0) & (years <= y1)))
    return out


def _observed_and_band(wet, years, months, season, delT, tau,
                       year_blocked, band_lo, band_hi, null_model="pooled"):
    """Observed k_trig and (band_lo, band_hi) for one window's wet series.

    ``wet`` is the 0/1 series restricted to the window's in-season days;
    ``years``/``months`` are aligned to it.

    The OBSERVED count is always computed with the requested blocking (with
    ``year_blocked=True`` no coincidence bridges a Sep -> next-Jun boundary).
    ``null_model`` controls the null distribution of that count:

    * ``"pooled"``  -- Equation (1) of the paper: one binomial over the pooled
      window (total in-season length T, total events N). Equivalent to the
      per-season model at the window-average event density; exact when event
      counts are homogeneous across seasons. Note the mild construction
      mismatch: this null permits cross-boundary coincidences that the
      blocked observation cannot contain (boundary clipping affects
      ~(delT+tau)/season_length of trigger windows, a few percent).
    * ``"blocked"`` -- exact sum of per-season binomials (Poisson-binomial by
      convolution); reduces to Eq. (1) for a single season.
    """
    if year_blocked:
        blocks = jjas_year_blocks(years, months, season)
        res = eca_blocked(wet, wet, blocks, delT=delT, tau=tau, sigtest=None)
        block_stats = [(len(b), r.n_a, r.n_b)
                       for b, r in zip(blocks, res.per_block)]
    else:
        res = eca_ts(wet, wet, delT=delT, tau=tau, sigtest=None)
        block_stats = [(len(wet), res.n_a, res.n_b)]
    if null_model == "pooled":
        t_tot = sum(t for t, _, _ in block_stats)
        na_tot = sum(a for _, a, _ in block_stats)
        nb_tot = sum(b for _, _, b in block_stats)
        block_stats = [(t_tot, na_tot, nb_tot)]
    elif null_model != "blocked":
        raise ValueError(f"unknown null_model {null_model!r}")
    pmf = trigger_null_pmf(block_stats, delT=delT, tau=tau)
    lo, hi = null_band(pmf, lo=band_lo, hi=band_hi)
    return int(res.k_trigg), lo, hi


def analyse_region(region_data, cfg: WindowConfig, region_code: str):
    """Return a tidy DataFrame (one row per period) for one region cube.

    ``region_data`` = (data, yyyymmdd, years, months) as from the loader.
    Produces both fixed- and thermodynamic-threshold results in long form
    with a ``regime`` column.
    """
    data, yyyymmdd, years, months = region_data
    windows = sliding_windows(cfg.periods)
    win_masks = _window_masks(years, months, cfg.season, windows)

    # per-window in-season slices (kept aligned for blocking)
    slices = {}
    for lbl, mask in win_masks:
        slices[lbl] = (np.asarray(data)[mask], np.asarray(years)[mask],
                       np.asarray(months)[mask])

    # ---- thermodynamic thresholds: percentile-match the baseline 20mm ----
    base_data = slices[cfg.baseline_period][0]
    base_pct = get_percentile(cfg.fixed_threshold, base_data)
    thermo_thr = {lbl: value_from_percentile(base_pct, slices[lbl][0])
                  for lbl, _ in win_masks}

    rows = []
    for lbl, _ in win_masks:
        d, yy, mm = slices[lbl]

        # FIXED regime: detect at fixed threshold, band from same series
        wet_fixed = cfg.detector(d, cfg.len_wet, cfg.fixed_threshold)
        k_fix, lo_fix, hi_fix = _observed_and_band(
            wet_fixed, yy, mm, cfg.season, cfg.delT, cfg.tau,
            cfg.year_blocked, cfg.band_lo, cfg.band_hi,
            null_model=cfg.null_model)
        rows.append({
            "region": region_code, "period": lbl, "regime": "fixed",
            "threshold": cfg.fixed_threshold, "k_trig": k_fix,
            "band_lo": lo_fix, "band_hi": hi_fix,
            "significant": int(k_fix > hi_fix),
        })

        # THERMODYNAMIC regime: band from shifted-threshold series.
        # Red point (k_trig) source depends on cfg.thermo_points:
        #   "fixed"  -> keep the fixed-threshold count (map/multi-panel code)
        #   "thermo" -> recount on the thermodynamic series (this 4-panel code)
        wet_thermo = cfg.detector(d, cfg.len_wet, thermo_thr[lbl])
        k_thermo, lo_th, hi_th = _observed_and_band(
            wet_thermo, yy, mm, cfg.season, cfg.delT, cfg.tau,
            cfg.year_blocked, cfg.band_lo, cfg.band_hi,
            null_model=cfg.null_model)
        k_red = k_fix if cfg.thermo_points == "fixed" else k_thermo
        rows.append({
            "region": region_code, "period": lbl, "regime": "thermodynamic",
            "threshold": thermo_thr[lbl], "k_trig": k_red,
            "k_trig_fixed": k_fix, "k_trig_thermo": k_thermo,
            "band_lo": lo_th, "band_hi": hi_th,
            "significant": int(k_red > hi_th),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def plot_region_panels(df, cfg: WindowConfig, ensemble, regime, out_path):
    """Multi-panel figure: one panel per region, band + red k_trig points."""
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = df[df.regime == regime]
    regions = list(dict.fromkeys(sub.region))       # preserve order
    names = cfg.region_names or {}
    n = len(regions)
    ncols = 4
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 4))
    axes = np.atleast_1d(axes).flatten()
    periods = list(cfg.periods)
    x = range(len(periods))

    for i, rc in enumerate(regions):
        g = sub[sub.region == rc].set_index("period").reindex(periods)
        ax = axes[i]
        ax.plot(x, g.band_lo, color="royalblue", lw=0.7)
        ax.plot(x, g.band_hi, color="royalblue", lw=0.7)
        ax.fill_between(x, g.band_lo, g.band_hi, color="cornflowerblue", alpha=0.5)
        ax.scatter(x, g.k_trig, label="wet-[...]-wet", color="#E63946",
                   edgecolor="black", s=60)
        ax.set_xticks(list(x))
        ax.set_xticklabels(periods, rotation=45)
        ax.set_title(names.get(rc, rc), fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
    for j in range(len(regions), len(axes)):
        fig.delaxes(axes[j])

    mode = "year-blocked" if cfg.year_blocked else "concatenated"
    fig.suptitle(
        f"Compound Wet Events across UK regions, {cfg.fixed_threshold}mm/hr "
        f"(Ensemble {ensemble}) - {regime}, {mode}",
        fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def _load_region_series(cfg: WindowConfig, ensemble, region):
    """Load one region cube -> (data, yyyymmdd, years, months). Mirrors the
    pipeline loader; isolated for mocking/testing."""
    import iris
    import os
    path = os.path.join(cfg.data_dir, region,
                        f"{cfg.ensemble_prefix}{ensemble}_{region}.nc")
    cube = iris.load(path)[0]
    t = cube.coord("time")
    dates = t.units.num2date(t.points)
    yyyymmdd = np.array([f"{d.year:04d}{d.month:02d}{d.day:02d}" for d in dates])
    years = np.array([d.year for d in dates])
    months = np.array([d.month for d in dates])
    return np.asarray(cube.data), yyyymmdd, years, months


def run_window_analysis(cfg: WindowConfig):
    """Run the sliding-window compound-ECA for every ensemble/region.

    Writes per ensemble: ``compound_eca_windows_<regime>_<ens>.png`` and a
    combined ``compound_eca_windows.csv`` (all ensembles/regions/periods/both
    regimes, with k_trig, band, significant). Returns the tidy DataFrame.
    """
    import os
    os.makedirs(cfg.out_dir, exist_ok=True)
    all_frames = []
    for ensemble in cfg.ensembles:
        ens_frames = []
        for region in cfg.regions:
            try:
                rd = _load_region_series(cfg, ensemble, region)
            except Exception as e:   # missing member/region -> skip, warn
                print(f"  skip {region} ens {ensemble}: {e}")
                continue
            df = analyse_region(rd, cfg, region)
            df.insert(0, "ensemble", ensemble)
            ens_frames.append(df)
        if not ens_frames:
            continue
        ens_df = pd.concat(ens_frames, ignore_index=True)
        all_frames.append(ens_df)
        for regime in ("fixed", "thermodynamic"):
            out_png = os.path.join(
                cfg.out_dir,
                f"compound_eca_windows_{regime}_{ensemble}.png")
            plot_region_panels(ens_df, cfg, ensemble, regime, out_png)
            print(f"  wrote {out_png}")

    result = (pd.concat(all_frames, ignore_index=True)
              if all_frames else pd.DataFrame())
    result.to_csv(os.path.join(cfg.out_dir, "compound_eca_windows.csv"),
                  index=False)
    return result


def plot_region_4panel(df, cfg: WindowConfig, region_code, ensemble, out_path,
                       region_display=None):
    """Reproduce the 4-panel single-region figure:

    a) fixed-threshold band + observed points for ``ensemble``
    b) thermodynamic-threshold band + observed points for ``ensemble``
    c) fixed-threshold significance heatmap (all ensembles x periods)
    d) thermodynamic-threshold significance heatmap

    ``df`` must be the multi-ensemble tidy frame from run_window_analysis /
    analyse_region for this region (all ensembles present for c/d).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    name = region_display or region_code
    periods = list(cfg.periods)
    x = range(len(periods))
    reg = df[df.region == region_code]
    ensembles = sorted(reg.ensemble.unique())

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # ---- a) fixed, single ensemble ----
    def _panel_ab(ax, regime, tag, thr_label):
        g = (reg[(reg.ensemble == ensemble) & (reg.regime == regime)]
             .set_index("period").reindex(periods))
        ax.plot(x, g.band_lo, color="royalblue", lw=0.7)
        ax.plot(x, g.band_hi, color="royalblue", lw=0.7)
        ax.fill_between(x, g.band_lo, g.band_hi, color="cornflowerblue",
                        alpha=0.5, label="2.5-97.5 percentile")
        ax.scatter(x, g.k_trig, label="Observed coincidences",
                   color="#E63946", edgecolor="black", s=80)
        ax.set_xticks(list(x)); ax.set_xticklabels(periods, rotation=45)
        ax.set_ylabel("Number of coincidences", fontsize=16)
        ax.set_title(f"{tag} {thr_label}\n{name}, p110{ensemble}", fontsize=16)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.legend(loc="upper left", fontsize=9)

    _panel_ab(axes[0, 0], "fixed",
              "a) Fixed threshold (20 mm/hr)", "")
    _panel_ab(axes[0, 1], "thermodynamic",
              "b) Thermodynamic threshold", "")

    # ---- c/d) significance heatmaps ----
    def _panel_cd(ax, regime, tag):
        grid = np.zeros((len(ensembles), len(periods)))
        for i, ens in enumerate(ensembles):
            g = (reg[(reg.ensemble == ens) & (reg.regime == regime)]
                 .set_index("period").reindex(periods))
            grid[i, :] = g.significant.fillna(0).values
        ax.imshow(grid, cmap="Blues", aspect="auto", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_xticks(list(x)); ax.set_xticklabels(periods, rotation=45, ha="right")
        ax.set_yticks(range(len(ensembles))); ax.set_yticklabels(ensembles)
        ax.set_ylabel("Ensemble member", fontsize=16)
        ax.set_title(f"{tag}\n{name}", fontsize=16)
        ax.set_xticks(np.arange(len(periods)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(ensembles)) - 0.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.3)

    _panel_cd(axes[1, 0], "fixed",
              "c) Fixed threshold - Ensemble exceedances")
    _panel_cd(axes[1, 1], "thermodynamic",
              "d) Thermodynamic threshold - Ensemble exceedances")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out_path