"""Compound-event classification.

Definition (confirmed by maintainer, 2026-08): a wet event is **compound** if
at least one other wet event lies within the ECA coincidence window
(lags tau .. tau+delT) before OR after it, within the same (year, season)
block. Equivalently: the union of the self-ECA precursor and trigger indices.
Events with no such neighbour are **single**.

Example (tau=1, delT>=2): series [0,1,0,0,1,1,0,1,0,0,0,0,0,1]
-> compound = positions {1, 4, 5, 7} (one episode of 4 extremes),
   single   = {13} (its nearest event is 6 days back, outside the window).

This supersedes the legacy meteo_window de-duplication + forward-expansion
logic from the original driver, which (a) missed the first event of every
compound episode and (b) used a linkage window (meteo_window) inconsistent
with the ECA window. The legacy functions are kept below, clearly marked,
only for comparison runs against old outputs.
"""

from __future__ import annotations

import numpy as np

from .eca import eca_blocked

__all__ = [
    "compound_membership",
    "compound_episodes",
    "classify_days",
    # legacy (pre-fix) behaviour, for comparison only:
    "legacy_dedup_compound_indices",
    "legacy_expand_compound_days",
]


def compound_membership(wet_series: np.ndarray,
                        blocks: "list[np.ndarray]",
                        delT: int,
                        tau: int) -> np.ndarray:
    """Positions (original-series indices) of compound events: events with a
    neighbouring event at lag in [tau, tau+delT] before or after, within the
    same block."""
    res = eca_blocked(wet_series, wet_series, blocks, delT=delT, tau=tau,
                      sigtest=None)
    return np.union1d(res.prec_indices, res.trigg_indices)


def compound_episodes(wet_series: np.ndarray,
                      blocks: "list[np.ndarray]",
                      delT: int,
                      tau: int) -> "list[np.ndarray]":
    """Group compound events into episodes: connected components of the graph
    linking two events iff their lag is in [tau, tau+delT] (per block).
    Returns components with >= 2 events, each an array of original-series
    positions, chronological. For tau <= 1 this equals maximal runs of
    consecutive events with gaps <= tau+delT."""
    wet_series = np.asarray(wet_series)
    episodes: list[np.ndarray] = []
    for blk in blocks:
        pos = blk[np.flatnonzero(wet_series[blk] == 1)]
        if len(pos) < 2:
            continue
        parent = list(range(len(pos)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        j0 = 0
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                gap = pos[j] - pos[i]
                if gap > tau + delT:
                    break
                if gap >= tau:
                    union(i, j)
        comps: dict[int, list[int]] = {}
        for i in range(len(pos)):
            comps.setdefault(find(i), []).append(pos[i])
        for members in comps.values():
            if len(members) >= 2:
                episodes.append(np.array(sorted(members), dtype=int))
    episodes.sort(key=lambda a: a[0])
    return episodes


def classify_days(all_dates: np.ndarray,
                  wet_event_dates: np.ndarray,
                  compound_dates: np.ndarray):
    """Split dates into (single, compound, no_events). Singles are wet-event
    dates not in the compound set; no-events are all remaining dates.
    Operates on date labels (yyyymmdd strings), so multi-region pooling by
    date works exactly as in the original driver."""
    all_dates = np.unique(np.asarray(all_dates))
    wet_event_dates = np.asarray(wet_event_dates)
    compound_dates = (np.unique(np.asarray(compound_dates))
                      if len(compound_dates)
                      else np.empty(0, dtype=all_dates.dtype))
    single = np.setdiff1d(wet_event_dates, compound_dates)
    events = np.union1d(compound_dates, single)
    no_events = np.setdiff1d(all_dates, events)
    return np.sort(single), np.sort(compound_dates), np.sort(no_events)


# --------------------------------------------------------------------------- #
# LEGACY (pre-fix) behaviour -- reproduces the original script's
# classification, including its known defects: the first event of each
# episode is misclassified as single, and linkage uses meteo_window rather
# than the ECA window. Retained ONLY for comparing against historical output.
# --------------------------------------------------------------------------- #
def legacy_dedup_compound_indices(compound_indices, blocks, meteo_window):
    kept = []
    compound_indices = np.sort(np.asarray(compound_indices, dtype=int))
    for blk in blocks:
        in_blk = compound_indices[np.isin(compound_indices, blk)]
        prev = -np.inf
        for idx in in_blk:
            if idx - prev > meteo_window:
                kept.append(idx)
                prev = idx
    return np.array(kept, dtype=int)


def legacy_expand_compound_days(deduped_indices, wet_series, blocks,
                                meteo_window):
    wet_series = np.asarray(wet_series)
    block_end = {}
    for blk in blocks:
        for p in blk:
            block_end[p] = blk[-1]
    out = []
    for idx in deduped_indices:
        upper = min(idx + meteo_window, block_end[idx] + 1)
        window = wet_series[idx:upper]
        out.extend(idx + np.flatnonzero(window == 1))
    return np.unique(np.array(out, dtype=int))