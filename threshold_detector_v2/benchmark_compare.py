#!/usr/bin/env python3
"""
Benchmark v1 threshold_detector against threshold_detector_v2.

Runs repeatable synthetic workloads for core hot-path functions and prints a
speed comparison table plus output-consistency checks.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure sibling packages `threshold_detector` and `threshold_detector_v2`
# are importable when this script is run as a file path.
_HERE = Path(__file__).resolve()
_ECA_ROOT = _HERE.parent.parent
if str(_ECA_ROOT) not in sys.path:
    sys.path.insert(0, str(_ECA_ROOT))

import threshold_detector.detector as det_v1
import threshold_detector.plotting as plot_v1
import threshold_detector_v2.detector as det_v2
import threshold_detector_v2.plotting as plot_v2


@dataclass
class BenchResult:
    name: str
    v1_mean_s: float
    v2_mean_s: float
    speedup: float
    details: str = ""


def timed_many(fn, repeat):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(statistics.mean(times))


def assert_equal_array(a, b, name):
    if not np.array_equal(np.asarray(a), np.asarray(b)):
        raise AssertionError(f"{name}: arrays differ")


def assert_equal_df(df1, df2, sort_cols, numeric_cols=None, name="df"):
    left = df1.sort_values(sort_cols).reset_index(drop=True).copy()
    right = df2.sort_values(sort_cols).reset_index(drop=True).copy()

    if list(left.columns) != list(right.columns):
        raise AssertionError(f"{name}: columns differ: {left.columns} vs {right.columns}")

    numeric_cols = numeric_cols or []
    for c in left.columns:
        if c in numeric_cols:
            lv = left[c].to_numpy(dtype=float)
            rv = right[c].to_numpy(dtype=float)
            both_nan = np.isnan(lv) & np.isnan(rv)
            ok = both_nan | np.isclose(lv, rv, rtol=0, atol=1e-12)
            if not np.all(ok):
                idx = int(np.flatnonzero(~ok)[0])
                raise AssertionError(f"{name}: numeric mismatch in {c} at row {idx}: {lv[idx]} vs {rv[idx]}")
        else:
            if not left[c].equals(right[c]):
                raise AssertionError(f"{name}: mismatch in column {c}")


def bench_flag_extreme_events(rng, n_steps, repeat):
    x = rng.normal(size=n_steps)

    def run_v1():
        det_v1.flag_extreme_events(x, threshold=0.25, N=15, min_days=11,
                                   direction="above", flag="last")

    def run_v2():
        det_v2.flag_extreme_events(x, threshold=0.25, N=15, min_days=11,
                                   direction="above", flag="last")

    out1 = det_v1.flag_extreme_events(x, threshold=0.25, N=15, min_days=11,
                                      direction="above", flag="last")
    out2 = det_v2.flag_extreme_events(x, threshold=0.25, N=15, min_days=11,
                                      direction="above", flag="last")
    assert_equal_array(out1, out2, "flag_extreme_events")

    t1 = timed_many(run_v1, repeat)
    t2 = timed_many(run_v2, repeat)
    return BenchResult("flag_extreme_events", t1, t2, (t1 / t2) if t2 > 0 else math.inf)


def bench_make_season_blocks(n_years, repeat):
    # 360-day calendar: 30 days/month for simplicity
    years = np.repeat(np.arange(1980, 1980 + n_years), 360)
    months = np.tile(np.repeat(np.arange(1, 13), 30), n_years)

    def run_v1():
        det_v1.make_season_blocks(years=years, months=months,
                                  season_start=10, season_length=5)

    def run_v2():
        det_v2.make_season_blocks(years=years, months=months,
                                  season_start=10, season_length=5)

    b1 = det_v1.make_season_blocks(years=years, months=months,
                                   season_start=10, season_length=5)
    b2 = det_v2.make_season_blocks(years=years, months=months,
                                   season_start=10, season_length=5)
    if len(b1) != len(b2):
        raise AssertionError("make_season_blocks: different number of blocks")
    for i, (x, y) in enumerate(zip(b1, b2)):
        if not np.array_equal(x, y):
            raise AssertionError(f"make_season_blocks: block {i} differs")

    t1 = timed_many(run_v1, repeat)
    t2 = timed_many(run_v2, repeat)
    return BenchResult("make_season_blocks", t1, t2, (t1 / t2) if t2 > 0 else math.inf)


def _synthetic_binary_pair(rng, n_steps):
    # sparse-ish event series
    b1 = (rng.random(n_steps) < 0.05).astype(int)
    b2 = (rng.random(n_steps) < 0.04).astype(int)
    return b1, b2


def _synthetic_years_months(n_steps):
    years = np.repeat(np.arange(1980, 1980 + (n_steps // 360) + 1), 360)[:n_steps]
    months = np.tile(np.repeat(np.arange(1, 13), 30), (n_steps // 360) + 1)[:n_steps]
    return years, months


def bench_detect_compound_events_bivariate(rng, n_steps, repeat):
    b1, b2 = _synthetic_binary_pair(rng, n_steps)
    years, months = _synthetic_years_months(n_steps)

    def run_v1():
        det_v1.detect_compound_events_bivariate(
            b1, b2, delT=4, tau=1, min_duration_1=1, min_duration_2=1,
            years=years, months=months, season_start=6, season_length=4
        )

    def run_v2():
        det_v2.detect_compound_events_bivariate(
            b1, b2, delT=4, tau=1, min_duration_1=1, min_duration_2=1,
            years=years, months=months, season_start=6, season_length=4
        )

    out1 = det_v1.detect_compound_events_bivariate(
        b1, b2, delT=4, tau=1, min_duration_1=1, min_duration_2=1,
        years=years, months=months, season_start=6, season_length=4
    )
    out2 = det_v2.detect_compound_events_bivariate(
        b1, b2, delT=4, tau=1, min_duration_1=1, min_duration_2=1,
        years=years, months=months, season_start=6, season_length=4
    )
    assert_equal_df(
        out1, out2,
        sort_cols=["start_idx", "end_idx", "length", "n_extreme_cases"],
        numeric_cols=["start_idx", "end_idx", "length", "n_extreme_cases",
                      "n_extreme_cases_1", "n_extreme_cases_2"],
        name="detect_compound_events_bivariate",
    )

    t1 = timed_many(run_v1, repeat)
    t2 = timed_many(run_v2, repeat)
    return BenchResult("detect_compound_events_bivariate", t1, t2, (t1 / t2) if t2 > 0 else math.inf)


def _synthetic_events_df(rng, n_rows=250_000, n_ens=28, start_year=1981, end_year=2080):
    ens = np.array([f"{i:02d}" for i in range(n_ens)])
    years = np.arange(start_year, end_year + 1)

    df = pd.DataFrame(
        {
            "ensemble": rng.choice(ens, size=n_rows),
            "year": rng.choice(years, size=n_rows),
            "month_number": rng.integers(1, 13, size=n_rows),
            "length": rng.integers(1, 20, size=n_rows),
            "n_extreme_cases": rng.integers(1, 20, size=n_rows),
            "n_extreme_cases_1": rng.integers(0, 15, size=n_rows),
            "n_extreme_cases_2": rng.integers(0, 15, size=n_rows),
        }
    )
    df["n_coincident_cases"] = np.minimum(df["n_extreme_cases_1"], df["n_extreme_cases_2"])
    return df


def bench_get_annual_stats(df, repeat):
    years = list(range(int(df["year"].min()), int(df["year"].max()) + 1))

    def run_v1():
        plot_v1.get_annual_stats(
            df,
            year_col="year",
            ensemble_col="ensemble",
            duration_col="n_extreme_cases",
            length_col="length",
            years=years,
            months_col="month_number",
            season_start=6,
            season_length=4,
        )

    def run_v2():
        plot_v2.get_annual_stats(
            df,
            year_col="year",
            ensemble_col="ensemble",
            duration_col="n_extreme_cases",
            length_col="length",
            years=years,
            months_col="month_number",
            season_start=6,
            season_length=4,
        )

    out1 = plot_v1.get_annual_stats(
        df,
        year_col="year",
        ensemble_col="ensemble",
        duration_col="n_extreme_cases",
        length_col="length",
        years=years,
        months_col="month_number",
        season_start=6,
        season_length=4,
    )
    out2 = plot_v2.get_annual_stats(
        df,
        year_col="year",
        ensemble_col="ensemble",
        duration_col="n_extreme_cases",
        length_col="length",
        years=years,
        months_col="month_number",
        season_start=6,
        season_length=4,
    )
    assert_equal_df(
        out1,
        out2,
        sort_cols=["ensemble", "year"],
        numeric_cols=["n_events", "total_duration", "mean_duration", "max_duration",
                      "total_length", "mean_length", "max_length"],
        name="get_annual_stats",
    )

    t1 = timed_many(run_v1, repeat)
    t2 = timed_many(run_v2, repeat)
    return BenchResult("get_annual_stats", t1, t2, (t1 / t2) if t2 > 0 else math.inf)


def bench_get_hovmoller_data(df, repeat):
    years = list(range(int(df["year"].min()), int(df["year"].max()) + 1))
    ens = sorted(df["ensemble"].unique())

    def run_v1():
        plot_v1.get_hovmoller_data(
            df,
            ensemble_col="ensemble",
            year_col="year",
            case_col="n_extreme_cases",
            length_col="length",
            years=years,
            ensembles=ens,
            extra_cols="auto",
        )

    def run_v2():
        plot_v2.get_hovmoller_data(
            df,
            ensemble_col="ensemble",
            year_col="year",
            case_col="n_extreme_cases",
            length_col="length",
            years=years,
            ensembles=ens,
            extra_cols="auto",
        )

    out1 = plot_v1.get_hovmoller_data(
        df,
        ensemble_col="ensemble",
        year_col="year",
        case_col="n_extreme_cases",
        length_col="length",
        years=years,
        ensembles=ens,
        extra_cols="auto",
    )
    out2 = plot_v2.get_hovmoller_data(
        df,
        ensemble_col="ensemble",
        year_col="year",
        case_col="n_extreme_cases",
        length_col="length",
        years=years,
        ensembles=ens,
        extra_cols="auto",
    )

    # For list-valued durations we compare a reduced numeric projection.
    chk1 = out1.drop(columns=["durations"]).copy()
    chk2 = out2.drop(columns=["durations"]).copy()
    assert_equal_df(
        chk1,
        chk2,
        sort_cols=["ensemble", "year"],
        numeric_cols=["n_events", "n_extreme_cases", "max_duration", "mean_duration",
                      "mean_length", "n_extreme_cases_1", "n_extreme_cases_2",
                      "n_coincident_cases"],
        name="get_hovmoller_data",
    )

    t1 = timed_many(run_v1, repeat)
    t2 = timed_many(run_v2, repeat)
    return BenchResult("get_hovmoller_data", t1, t2, (t1 / t2) if t2 > 0 else math.inf)


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark threshold_detector v1 vs v2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeat", type=int, default=6,
                   help="timing repeats per benchmark")
    p.add_argument("--n-steps", type=int, default=2_500_000,
                   help="series length for 1-D detector benchmarks")
    p.add_argument("--n-years", type=int, default=180,
                   help="years for make_season_blocks benchmark")
    p.add_argument("--n-events", type=int, default=250_000,
                   help="rows in synthetic events dataframe")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print("Benchmark config:")
    print(f"  seed={args.seed}")
    print(f"  repeat={args.repeat}")
    print(f"  n_steps={args.n_steps}")
    print(f"  n_years={args.n_years}")
    print(f"  n_events={args.n_events}")
    print()

    events_df = _synthetic_events_df(rng, n_rows=args.n_events)

    results = []
    results.append(bench_flag_extreme_events(rng, args.n_steps, args.repeat))
    results.append(bench_make_season_blocks(args.n_years, args.repeat))
    results.append(bench_detect_compound_events_bivariate(rng, args.n_steps, args.repeat))
    results.append(bench_get_annual_stats(events_df, args.repeat))
    results.append(bench_get_hovmoller_data(events_df, args.repeat))

    print("Results (lower is better):")
    print(f"{'Function':36s} {'v1 mean (s)':>12s} {'v2 mean (s)':>12s} {'speedup':>10s}")
    print("-" * 76)
    for r in results:
        print(f"{r.name:36s} {r.v1_mean_s:12.4f} {r.v2_mean_s:12.4f} {r.speedup:10.2f}x")

    gm_v1 = statistics.geometric_mean([r.v1_mean_s for r in results])
    gm_v2 = statistics.geometric_mean([r.v2_mean_s for r in results])
    overall = gm_v1 / gm_v2 if gm_v2 > 0 else math.inf
    print("-" * 76)
    print(f"{'geometric-mean overall':36s} {gm_v1:12.4f} {gm_v2:12.4f} {overall:10.2f}x")


if __name__ == "__main__":
    main()
