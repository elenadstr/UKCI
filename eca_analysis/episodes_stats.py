"""Per-compound-event statistics.

For each compound episode (a maximal set of extreme-rain days linked through
the ECA window, within one region and one JJAS block -- see
:func:`eca_analysis.compound.compound_episodes`) this computes:

* ``start_yyyymmdd`` / ``end_yyyymmdd`` -- first and last extreme-rain day
* ``duration``  -- number of extreme-rain days in the episode
                   (the "how many rainfall days" count)
* ``length``    -- span of the episode in days, ``end - start + 1``
                   (the "how long was the time period")

``duration <= length`` always; they are equal only when every day in the span
is an extreme day. For the maintainer's worked example
``[0,1,0,0,1,1,0,1,0,0,0,0,0,1]`` (tau=1, delT>=2) the single episode has
duration=4 and length=7.

Design notes
------------
* ``length`` is computed from time-step **indices**, not from the yyyymmdd
  labels. On the UKCP 360-day calendar, subtracting date labels is only
  accidentally correct inside JJAS (all months = 30 days) and would break for
  any other season; index arithmetic is exact for any contiguous daily block.
* ``length`` is **inclusive** (start and end days both counted): a one-day
  span is length 1, and the example above is 7 not 6. If your downstream
  convention is the exclusive gap (``end - start``), set
  ``length_inclusive=False``.
* Episodes never cross a (year, JJAS) block boundary, because the blocks
  passed in already enforce that; a September event and the following June
  cannot land in the same episode.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .compound import compound_episodes

__all__ = ["episode_stats", "national_series_from_dates"]


def national_series_from_dates(all_dates, wet_dates):
    """Build a UK-wide daily series from per-region date labels.

    Parameters
    ----------
    all_dates : iterable of every in-season day label seen across regions
                (duplicates fine -- one per region per day).
    wet_dates : iterable of day labels on which *any* region had an extreme
                (duplicates fine).

    Returns ``(national_wet, axis, blocks, years)`` where ``axis`` is the
    sorted unique day labels (the national daily time axis), ``national_wet``
    is 1 where that day is wet in >=1 region, ``blocks`` groups axis indices
    by calendar year, and ``years`` is the per-step year.

    A day is nationally wet if ANY region is wet: this is what lets an event
    that is in Scotland one day and Wales the next count as a single UK-wide
    event. Because the axis is the full contiguous in-season calendar, index
    adjacency equals day adjacency, so the ECA window (delT, tau) links
    consecutive days correctly; blocks stop linkage crossing a year boundary.
    """
    axis = np.array(sorted(set(all_dates)))
    if len(axis) == 0:
        return (np.empty(0, int), axis, [], np.empty(0, int))
    wet_set = set(wet_dates)
    national_wet = np.fromiter((lbl in wet_set for lbl in axis),
                               dtype=int, count=len(axis))
    years = np.array([int(lbl[:4]) for lbl in axis])
    # contiguous run of equal years -> one block (axis is sorted, in-season)
    blocks = [np.flatnonzero(years == y) for y in np.unique(years)]
    return national_wet, axis, blocks, years


def episode_stats(
    wet_series: np.ndarray,
    yyyymmdd: Sequence[str],
    blocks: "list[np.ndarray]",
    delT: int,
    tau: int,
    years: Optional[Sequence[int]] = None,
    length_inclusive: bool = True,
    include_rain_days: bool = True,
    include_singles: bool = True,
) -> pd.DataFrame:
    """Return one row per extreme-rain event with start/end/duration/length.

    An ``event_type`` column distinguishes:
    * ``"compound"`` -- a linked episode of >= 2 extreme days (duration >= 2).
    * ``"single"``   -- an isolated extreme day with no linked neighbour;
                        start == end, duration == 1, length == 1 (inclusive).

    Parameters
    ----------
    wet_series : 0/1 array over the full (unblocked) series.
    yyyymmdd   : date label per time step, same length as ``wet_series``.
    blocks     : list of index arrays (one per year/JJAS block), as produced
                 by :func:`eca_analysis.jjas_year_blocks`.
    delT, tau  : ECA window parameters (must match the run that produced the
                 compound classification).
    years      : optional year label per time step; if given, the event's
                 block year is recorded in the ``year`` column.
    length_inclusive : count both endpoints in ``length`` (default True).
    include_rain_days : add a ``rain_days`` column listing the member dates
                 (semicolon-joined) for traceability/QA.
    include_singles : also emit the isolated extreme days as duration-1 rows
                 (default True). Set False for compound episodes only.

    Columns: [year?] event_type, start_index, end_index, start_yyyymmdd,
             end_yyyymmdd, duration, length[, rain_days]. Rows are in
             chronological order.

    A single is an extreme day not belonging to any compound episode, i.e. a
    wet day whose ECA-window neighbourhood contains no other wet day. For the
    maintainer's example ``[0,1,0,0,1,1,0,1,0,0,0,0,0,1]`` (tau=1, delT>=2)
    that is the last day (position 13): one compound event {1,4,5,7} and one
    single {13}.
    """
    wet_series = np.asarray(wet_series)
    yyyymmdd = np.asarray(yyyymmdd)
    years = np.asarray(years) if years is not None else None
    inc = 1 if length_inclusive else 0
    rows = []

    compound_members = set()
    for episode in compound_episodes(wet_series, blocks, delT=delT, tau=tau):
        start, end = int(episode[0]), int(episode[-1])
        compound_members.update(int(p) for p in episode)
        row = {
            "event_type": "compound",
            "start_index": start,
            "end_index": end,
            "start_yyyymmdd": yyyymmdd[start],
            "end_yyyymmdd": yyyymmdd[end],
            "duration": len(episode),                       # rain-day count
            "length": end - start + inc,
        }
        if years is not None:
            row = {"year": int(years[start]), **row}
        if include_rain_days:
            row["rain_days"] = ";".join(yyyymmdd[episode])
        rows.append(row)

    if include_singles:
        # extreme days inside any block that are not part of a compound episode
        block_pos = set()
        for blk in blocks:
            block_pos.update(int(p) for p in blk)
        wet_pos = [int(p) for p in np.flatnonzero(wet_series == 1)]
        for p in wet_pos:
            if p not in block_pos or p in compound_members:
                continue
            row = {
                "event_type": "single",
                "start_index": p,
                "end_index": p,
                "start_yyyymmdd": yyyymmdd[p],
                "end_yyyymmdd": yyyymmdd[p],
                "duration": 1,
                "length": inc,           # 1 if inclusive else 0
            }
            if years is not None:
                row = {"year": int(years[p]), **row}
            if include_rain_days:
                row["rain_days"] = yyyymmdd[p]
            rows.append(row)

    cols = (["year"] if years is not None else []) + [
        "event_type", "start_index", "end_index", "start_yyyymmdd",
        "end_yyyymmdd", "duration", "length",
    ] + (["rain_days"] if include_rain_days else [])
    df = pd.DataFrame(rows, columns=cols)
    return df.sort_values("start_index").reset_index(drop=True)