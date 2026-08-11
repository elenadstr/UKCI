"""JJAS compound-precipitation-event pipeline.

Pure-Python replacement for the rpy2/CoinCalc driver script:

1. R/rpy2/CoinCalc removed; ECA is :mod:`eca_analysis.eca` (validated against
   the CoinCalc R source and cross-checked against the PyPI ``event-analysis``
   package).
2. **JJAS only**: analysis is restricted to June--September.
3. **Year-blocked**: ECA and compound classification operate within
   (year, JJAS) blocks, so nothing bridges September -> next June.
   "no events" days are JJAS days only.
4. **Corrected compound definition** (maintainer-confirmed): a wet event is
   compound iff another wet event lies within the ECA window
   (lags tau..tau+delT) before or after it, in the same block -- i.e. the
   union of the self-ECA precursor and trigger indices. The legacy
   meteo_window dedup + forward-expansion is removed from the pipeline: it
   misclassified the first event of every episode as single, and used a
   linkage window inconsistent with the ECA window (e.g. an event 6 days
   after an episode is NOT compound when tau+delT < 6). Legacy functions
   remain in :mod:`eca_analysis.compound` for comparison runs.
5. Cubes are loaded once per region (the original loaded each cube twice).
6. The summary table no longer references undefined globals (``p_wet`` /
   ``p_dry`` -- a latent NameError in the original); event probabilities are
   computed from JJAS day counts, and compound episode/day counts are added.

Everything the UI/advanced users may want to toggle is in :class:`ECAConfig`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .eca import (
    SEASON_JJAS,
    eca_blocked,
    flag_wet_events,
    jjas_year_blocks,
)
from .compound import classify_days, compound_episodes
from .episode_stats import episode_stats, national_series_from_dates

DEFAULT_REGIONS = (
    "East_Midlands", "East_Scotland", "East_of_England", "North_East_England",
    "North_Scotland", "North_West_England", "Northern_Ireland",
    "South_East_England", "South_West_England", "Wales", "West_Midlands",
    "West_Scotland", "Yorkshire_and_Humber",
)


@dataclass
class ECAConfig:
    data_dir: str
    out_dir: str
    ensembles: Sequence[str] = ("2868",)
    regions: Sequence[str] = DEFAULT_REGIONS
    wet_thresholds: Sequence[float] = (20, 24, 28)
    delT_values: Sequence[int] = (1, 2, 3, 4)
    tau: int = 1
    len_wet: int = 1
    season: Sequence[int] = SEASON_JJAS       # months kept; blocks are per year
    ensemble_prefix: str = "p110"
    sigtest: str = "poisson"                  # or "shuffle.surrogate"
    null_model: str = "pooled"                # "pooled": Eq (1) on window totals
    #                                           (paper method); "blocked": exact
    #                                           per-season Poisson-binomial
    shuffle_reps: int = 1000
    # pluggable event detector: (data, len_wet, threshold) -> binary array
    detector: Callable[[np.ndarray, int, float], np.ndarray] = field(
        default=flag_wet_events
    )
    seed: Optional[int] = None


def _load_region_series(cfg: ECAConfig, ensemble: str, region: str):
    """Load one region cube; return (precip data, yyyymmdd labels, years,
    months). Kept in one place so the iris dependency is isolated and the
    360-day calendar is never coerced to a Gregorian DatetimeIndex."""
    import iris  # local import: core ECA stays iris-free

    path = os.path.join(
        cfg.data_dir, region, f"{cfg.ensemble_prefix}{ensemble}_{region}.nc"
    )
    cube = iris.load(path)[0]
    tcoord = cube.coord("time")
    dates = tcoord.units.num2date(tcoord.points)  # calendar-aware (cftime)
    yyyymmdd = np.array([f"{d.year:04d}{d.month:02d}{d.day:02d}" for d in dates])
    years = np.array([d.year for d in dates])
    months = np.array([d.month for d in dates])
    return np.asarray(cube.data), yyyymmdd, years, months


def run(cfg: ECAConfig) -> pd.DataFrame:
    """Run the full sweep. Writes, per (ensemble, threshold, delT):

    * ``{prefix}{ensemble}_events_{thr}mmh_delT_{d}_JJAS.csv`` -- single /
      compound / no-events yyyymmdd columns (same layout as before, JJAS days
      only, dates pooled across regions exactly as in the original), and
    * one aggregate ``eca_summary_JJAS.csv`` with counts, rates and p-values
      per (ensemble, region, threshold, delT) -- the statistics the original
      computed but never saved.

    Returns the summary DataFrame.
    """
    os.makedirs(os.path.join(cfg.out_dir, "events_yyyymmdd"), exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    summary_rows = []
    regional_frames = []
    national_frames = []

    # cache region data per ensemble (original re-loaded every cube twice
    # per (threshold, delT) combination)
    for ensemble in cfg.ensembles:
        region_data = {r: _load_region_series(cfg, ensemble, r) for r in cfg.regions}

        for wet_threshold in cfg.wet_thresholds:
            for delT in cfg.delT_values:
                compound_dates, wet_dates, all_dates = [], [], []
                date_to_regions: dict = {}   # for the national 'regions' column

                for region, (data, yyyymmdd, years, months) in region_data.items():
                    blocks = jjas_year_blocks(years, months, cfg.season)
                    if not blocks:
                        raise ValueError(f"No {cfg.season} days in {region}")
                    season_mask = np.zeros(len(data), dtype=bool)
                    for blk in blocks:
                        season_mask[blk] = True
                    all_dates.extend(yyyymmdd[season_mask])

                    # detect events *within* blocks so len_wet windows cannot
                    # straddle a season boundary
                    wet = np.zeros(len(data), dtype=int)
                    for blk in blocks:
                        wet[blk] = cfg.detector(data[blk], cfg.len_wet, wet_threshold)
                    wet_dates.extend(yyyymmdd[wet == 1])
                    for d in yyyymmdd[wet == 1]:
                        date_to_regions.setdefault(d, set()).add(region)

                    res = eca_blocked(
                        wet, wet, blocks, delT=delT, tau=cfg.tau,
                        sigtest=cfg.sigtest, reps=cfg.shuffle_reps, rng=rng,
                        null_model=cfg.null_model,
                    )

                    # compound = events with a neighbour inside the ECA
                    # window before OR after (union of precursor & trigger
                    # indices); reuses the ECA result, no extra pass
                    compound_pos = np.union1d(res.prec_indices, res.trigg_indices)
                    compound_dates.extend(yyyymmdd[compound_pos])
                    episodes = compound_episodes(wet, blocks, delT, cfg.tau)

                    # per-episode stats: start / end / duration / length
                    ep_df = episode_stats(
                        wet, yyyymmdd, blocks, delT=delT, tau=cfg.tau,
                        years=years,
                    )
                    if len(ep_df):
                        ep_df.insert(0, "delT", delT)
                        ep_df.insert(0, "wet_threshold", wet_threshold)
                        ep_df.insert(0, "region", region)
                        ep_df.insert(0, "ensemble", ensemble)
                        regional_frames.append(ep_df)

                    n_days = int(season_mask.sum())
                    summary_rows.append({
                        "ensemble": ensemble, "region": region,
                        "wet_threshold": wet_threshold, "delT": delT,
                        "tau": cfg.tau, "sigtest": cfg.sigtest,
                        "n_jjas_days": n_days, "n_events": res.n_a,
                        "p_wet": res.n_a / n_days,
                        "k_precursor": res.k_prec, "k_trigger": res.k_trigg,
                        "precursor_rate": res.rate_prec,
                        "trigger_rate": res.rate_trigg,
                        "pvalue_precursor": res.pvalue_prec,
                        "pvalue_trigger": res.pvalue_trigg,
                        "n_compound_days": len(compound_pos),
                        "n_single_days": res.n_a - len(compound_pos),
                        "n_compound_episodes": len(episodes),
                    })

                single, compound, no_events = classify_days(
                    np.array(all_dates), np.array(wet_dates), np.array(compound_dates)
                )

                # -------- UK-wide (national) episodes --------
                # A day is nationally wet if ANY region is wet; linkage then
                # runs on this pooled daily series, so an event that is in one
                # region one day and another region the next counts once.
                nat_wet, nat_axis, nat_blocks, nat_years = \
                    national_series_from_dates(all_dates, wet_dates)
                nat_df = episode_stats(
                    nat_wet, nat_axis, nat_blocks, delT=delT, tau=cfg.tau,
                    years=nat_years,
                )
                if len(nat_df):
                    # which regions contributed to each national episode
                    nat_df["regions"] = nat_df["rain_days"].apply(
                        lambda s: ";".join(sorted(
                            set().union(*(date_to_regions.get(d, set())
                                          for d in s.split(";")))))
                    )
                    nat_df.insert(0, "delT", delT)
                    nat_df.insert(0, "wet_threshold", wet_threshold)
                    nat_df.insert(0, "ensemble", ensemble)
                    national_frames.append(nat_df)

                df = pd.concat(
                    [pd.Series(single, name="single"),
                     pd.Series(compound, name="compound"),
                     pd.Series(no_events, name="no events")],
                    axis=1,
                )
                fname = (f"{cfg.ensemble_prefix}{ensemble}_events_"
                         f"{wet_threshold:g}mmh_delT_{delT}_JJAS.csv")
                df.to_csv(
                    os.path.join(cfg.out_dir, "events_yyyymmdd", fname), index=False
                )
                print(f"Completed: ensemble={ensemble}, "
                      f"threshold={wet_threshold}, delT={delT}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(cfg.out_dir, "eca_summary_JJAS.csv"), index=False)

    _REGIONAL_COLS = ["ensemble", "region", "wet_threshold", "delT", "year",
                      "event_type", "start_index", "end_index",
                      "start_yyyymmdd", "end_yyyymmdd", "duration", "length",
                      "rain_days"]
    _NATIONAL_COLS = ["ensemble", "wet_threshold", "delT", "year",
                      "event_type", "start_index", "end_index",
                      "start_yyyymmdd", "end_yyyymmdd", "duration", "length",
                      "rain_days", "regions"]
    regional = (pd.concat(regional_frames, ignore_index=True)
                if regional_frames else pd.DataFrame(columns=_REGIONAL_COLS))
    regional.to_csv(
        os.path.join(cfg.out_dir, "eca_episodes_regional_JJAS.csv"), index=False)

    national = (pd.concat(national_frames, ignore_index=True)
                if national_frames else pd.DataFrame(columns=_NATIONAL_COLS))
    national.to_csv(
        os.path.join(cfg.out_dir, "eca_episodes_national_JJAS.csv"), index=False)
    return summary


def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        description="JJAS year-blocked compound-event ECA pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", required=True,
                   help="root containing <region>/<prefix><ensemble>_<region>.nc")
    p.add_argument("--out-dir", required=True,
                   help="output root; CSVs go to <out-dir>/events_yyyymmdd/")
    p.add_argument("--ensembles", nargs="+",
                   default=["0000", "1113", "1554", "1649", "1843", "1935",
                            "2123", "2242", "2305", "2335", "2491", "2868"],
                   help="ensemble member IDs")
    p.add_argument("--regions", nargs="+", default=list(DEFAULT_REGIONS))
    p.add_argument("--thresholds", nargs="+", type=float, default=[20, 24, 28],
                   help="wet-day thresholds (mm/h)")
    p.add_argument("--delT", nargs="+", type=int, default=[1, 2, 3, 4])
    p.add_argument("--tau", type=int, default=1)
    p.add_argument("--len-wet", type=int, default=1)
    p.add_argument("--season", nargs="+", type=int, default=list(SEASON_JJAS),
                   help="months to keep (blocks are per calendar year)")
    p.add_argument("--ensemble-prefix", default="p110")
    p.add_argument("--sigtest", choices=["poisson", "shuffle.surrogate"],
                   default="poisson")
    p.add_argument("--null-model", choices=["pooled", "blocked"],
                   default="pooled",
                   help="null for p-values: 'pooled' = Eq (1) binomial on "
                        "window totals (paper method); 'blocked' = exact "
                        "per-season sum")
    p.add_argument("--shuffle-reps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed (only affects shuffle.surrogate)")
    return p


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    cfg = ECAConfig(
        data_dir=args.data_dir, out_dir=args.out_dir,
        ensembles=tuple(args.ensembles), regions=tuple(args.regions),
        wet_thresholds=tuple(args.thresholds), delT_values=tuple(args.delT),
        tau=args.tau, len_wet=args.len_wet, season=tuple(args.season),
        ensemble_prefix=args.ensemble_prefix, sigtest=args.sigtest,
        null_model=args.null_model,
        shuffle_reps=args.shuffle_reps, seed=args.seed,
    )
    return run(cfg)


if __name__ == "__main__":
    main()