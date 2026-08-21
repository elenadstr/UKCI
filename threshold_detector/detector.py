"""
threshold_detector/detector.py

All counting maths delegates to the vendored engine in :mod:`eca_analysis`
(see ``eca_analysis/VENDORED.md``); this module adds convenience and the
UKCI extensions (percentile thresholds, tolerant N-day events, two-variable
sequential / co-occurring modes).

Event criteria: per variable, then compound
-------------------------------------------
Each variable gets its own definition of "an extreme event" at the flagging
stage (:func:`flag_extreme_events`): threshold, direction, ``N`` consecutive
days, and optionally ``min_days`` (a tolerance: at least ``min_days`` of the
``N`` days extreme, e.g. a 20-day drought with >= 19 dry days). The compound
stage then links flagged events with the paper's rule and applies a minimum
number of flagged days per variable within one event
(``min_duration`` for single-variable; ``min_duration_1`` /
``min_duration_2`` for the two-variable modes).

Two flagged days link iff their gap ``g`` satisfies
``tau <= g <= tau + delT`` AND both days lie in the same (year, season)
block. With the paper defaults ``tau=1, delT=4`` the maximum linkage gap is
therefore **5 days** (the paper's "four-day window" refers to ``delT``; the
coincidence window has length ``delT + 1``, Eq. 1's ``TOL``). No window
ever spans a season boundary.

Seasons (modular)
-----------------
A season is ``(season_start, season_length)``: the start month and how many
months it lasts, wrapping the calendar year if needed. ``(6, 4)`` is
June-September (JJAS, the default); ``(8, 2)`` is August-September;
``(10, 5)`` is October-February, wrapping into the next calendar year. For
wrapped seasons the season-year label is the START year: October 1980 -
February 1981 is season-year 1980 (see :func:`season_year_labels`).
"""

from __future__ import annotations

import numpy as np

from eca_analysis import compound_episodes


# --------------------------------------------------------------------------- #
# defensive validation
# --------------------------------------------------------------------------- #
def _check_int(name, value, minimum):
    if not (np.isscalar(value) and float(value) == int(value)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _check_linkage(delT, tau):
    delT = _check_int("delT", delT, 0)
    tau = _check_int("tau", tau, 1)   # tau >= 1: a day never coincides with
    return delT, tau                  # itself; max linkage gap = tau + delT


def _check_binary(name, series):
    series = np.asarray(series)
    if series.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {series.shape}")
    bad = ~np.isin(series, (0, 1))
    if bad.any():
        raise ValueError(f"{name} must be binary (0/1); found "
                         f"{np.unique(series[bad])[:5]} at e.g. index "
                         f"{int(np.flatnonzero(bad)[0])}")
    return series.astype(int)


def _check_season(season_start, season_length):
    season_start = _check_int("season_start", season_start, 1)
    if season_start > 12:
        raise ValueError(f"season_start must be 1-12, got {season_start}")
    season_length = _check_int("season_length", season_length, 1)
    if season_length > 12:
        raise ValueError(f"season_length must be 1-12, got {season_length}")
    return season_start, season_length


# --------------------------------------------------------------------------- #
# modular seasons
# --------------------------------------------------------------------------- #
def season_months(season_start, season_length):
    """Months of a modular season: start month + duration, wrapping the year.

    >>> season_months(8, 2)   # August -> September
    (8, 9)
    >>> season_months(10, 5)  # October -> February (wraps)
    (10, 11, 12, 1, 2)
    """
    season_start, season_length = _check_season(season_start, season_length)
    return tuple((season_start - 1 + k) % 12 + 1 for k in range(season_length))


def season_year_labels(years, months, season_start):
    """Season-year label per step: the calendar year the season STARTED in.

    For non-wrapping seasons this equals the calendar year. For wrapping
    seasons, months before ``season_start`` belong to the season that began
    the previous calendar year: with ``season_start=10``, Oct-Dec 1980 and
    Jan-Feb 1981 all get label 1980 -- so every season-year is one contiguous
    block and the labels cover all valid years.
    """
    years = np.asarray(years, dtype=int)
    months = np.asarray(months, dtype=int)
    if years.shape != months.shape:
        raise ValueError("years and months must have the same shape")
    return np.where(months >= season_start, years, years - 1)


def make_season_blocks(years=None, months=None, season_start=6,
                       season_length=4, n_steps=None):
    """Build per-season index blocks for a daily series.

    Parameters
    ----------
    years : array-like of int, aligned to the series. Required unless the
        series is a single contiguous block (then pass ``n_steps`` instead).
    months : array-like of int, optional. If omitted, the series is assumed
        already season-extracted with non-wrapping seasons, and blocks are
        the runs of equal ``years`` values. REQUIRED for wrapping seasons
        (``season_start + season_length > 13``), where calendar years alone
        cannot identify the season boundaries.
    season_start, season_length : the modular season spec (default JJAS).
    n_steps : series length, only for the ``years=None`` single-block case.

    Returns
    -------
    list of np.ndarray -- index arrays, one per season-year, chronological.
    Steps whose month is outside the season are in no block (and therefore
    can never be part of an event).
    """
    season_start, season_length = _check_season(season_start, season_length)
    if years is None:
        if n_steps is None:
            raise ValueError("pass years=..., or n_steps=... for a single "
                             "contiguous block")
        return [np.arange(_check_int("n_steps", n_steps, 1))]
    years = np.asarray(years, dtype=int)
    wraps = season_start + season_length > 13
    if months is None:
        if wraps:
            raise ValueError(
                f"season ({season_start}, {season_length}) wraps the "
                "calendar year, so months= is required to build blocks -- "
                "consecutive calendar years alone cannot mark where one "
                "season ends and the next begins.")
        labels = years
        in_season = np.ones(len(years), dtype=bool)
    else:
        months = np.asarray(months, dtype=int)
        if months.shape != years.shape:
            raise ValueError("years and months must have the same shape")
        in_season = np.isin(months, season_months(season_start,
                                                  season_length))
        labels = season_year_labels(years, months, season_start)
    blocks = []
    for lab in np.unique(labels[in_season]):
        blocks.append(np.flatnonzero((labels == lab) & in_season))
    return blocks


def _resolve_blocks(n_steps, blocks, years, months, season_start,
                    season_length, func_name):
    """Shared block resolution; refuses to silently run unblocked."""
    if blocks is None and years is None:
        raise ValueError(
            f"{func_name} needs season blocks so that events cannot pair "
            "across a season boundary (e.g. September -> next June). Pass "
            "blocks=..., or years= (and months= if the series is not already "
            "season-extracted, or if the season wraps the calendar year). "
            "For a genuinely contiguous single-block series, pass "
            "blocks='contiguous' explicitly.")
    if isinstance(blocks, str):
        if blocks == "contiguous":
            return [np.arange(n_steps)]
        raise ValueError(f"unknown blocks={blocks!r}")
    if blocks is not None:
        return [np.asarray(b, dtype=int) for b in blocks]
    return make_season_blocks(years, months, season_start, season_length)


# --------------------------------------------------------------------------- #
# extreme-event flagging (per-variable event criteria)
# --------------------------------------------------------------------------- #
def flag_extreme_events(timeseries, threshold, N=1, min_days=None,
                        direction='above', flag='last'):
    """Flag extreme events in a timeseries.

    An event is flagged when, within a window of ``N`` consecutive days, at
    least ``min_days`` of them are on the extreme side of ``threshold``
    (strict inequality). ``min_days`` defaults to ``N`` (all days required,
    the strict spell); setting ``min_days < N`` tolerates interruptions --
    e.g. ``N=20, min_days=19`` is "a 20-day drought with at least 19 dry
    days".

    Parameters
    ----------
    timeseries : array-like
        Input timeseries of any climate variable.
    threshold : float
        Threshold defining an extreme.
    N : int, optional
        Window length in days. Default 1.
    min_days : int, optional
        Minimum extreme days within the N-day window. Default: ``N``
        (uninterrupted spell). Must satisfy ``1 <= min_days <= N``.
    direction : {'above', 'below'}, optional
        'above' flags values > threshold (heavy rain, heat);
        'below' flags values < threshold (drought, cold). Default 'above'.
    flag : {'last', 'first', 'all'}, optional
        Which day(s) of each qualifying window to flag. Irrelevant when
        ``N == 1`` (every extreme day is flagged regardless). For ``N > 1``:

        - ``'last'`` (default): one flag on the day each qualifying window
          COMPLETES. One flag = one whole event, so downstream
          ``n_extreme_cases`` counts *events* and ``min_duration=2`` means
          "at least 2 spells close together". Matches the paper engine's
          ``flag_wet_events`` exactly when ``min_days == N``. Use when the
          N-day spell is itself the unit you want to count (heatwaves,
          dry spells).
        - ``'first'``: as ``'last'`` but the flag sits on the window START.
          Same counts, event dates shifted ``N - 1`` days earlier. Use when
          onset timing matters (e.g. sequencing against a second variable).
        - ``'all'``: re-marks the extreme DAYS inside qualifying windows.
          Downstream counts are in raw extreme days and ``min_duration``
          means "at least that many extreme days". Use when you want
          day-resolved durations, at the cost of the flag series no longer
          being "one flag = one event".

        Caveats: ``'last'``/``'first'`` mark the window edge, which with
        ``min_days < N`` may itself be a non-extreme day; ``'all'`` leaves
        a spell tail shorter than a full window unflagged. See
        ``docs/choosing_flag_and_windows.md`` for a worked example, and
        :func:`compare_flag_options` to view all three on your own data.

        The scan jumps past each qualifying window (``i += N``): overlapping
        qualifying windows are not re-flagged, so one long spell yields one
        flag per N days, matching the engine's event counting.

    Returns
    -------
    numpy.ndarray
        Binary array, 1 = extreme event, 0 = not.
    """
    timeseries = np.asarray(timeseries, dtype=float)
    if timeseries.ndim != 1:
        raise ValueError(f"timeseries must be 1-D, got shape "
                         f"{timeseries.shape}")
    if not np.isscalar(threshold):
        raise ValueError(f"threshold must be a number, got {threshold!r}")
    N = _check_int("N", N, 1)
    min_days = N if min_days is None else _check_int("min_days", min_days, 1)
    if min_days > N:
        raise ValueError(f"min_days ({min_days}) cannot exceed N ({N})")
    if direction not in ('above', 'below'):
        raise ValueError(f"direction must be 'above' or 'below', got "
                         f"'{direction}'")
    if flag not in ('last', 'first', 'all'):
        raise ValueError(f"flag must be 'last', 'first' or 'all', got "
                         f"'{flag}'")

    exceeds = (timeseries > threshold) if direction == 'above' \
        else (timeseries < threshold)
    events = np.zeros(len(timeseries), dtype=int)

    if N == 1:
        events[exceeds] = 1
        return events

    i = 0
    while i <= len(timeseries) - N:
        window = exceeds[i:i + N]
        if int(window.sum()) >= min_days:
            if flag == 'last':
                events[i + N - 1] = 1
            elif flag == 'first':
                events[i] = 1
            else:  # 'all': mark the extreme days inside the window
                events[i:i + N][window] = 1
            i += N
        else:
            i += 1
    return events


def flag_extreme_events_percentile(timeseries, percentile, N=1, min_days=None,
                                   direction='above', flag='last',
                                   reference=None):
    """Flag extremes using a percentile threshold rather than a fixed value.

    For the paper's thermodynamic adjustment (Section 2.3): compute the
    percentile rank of the impact threshold in the *baseline* window, then
    call this once per rolling window with ``reference=that_window`` -- or
    use :func:`thermodynamic_thresholds` which does the loop for you.

    Parameters as :func:`flag_extreme_events`, plus:

    percentile : float
        Percentile (0-100) used as the threshold.
    reference : array-like, optional
        Data the percentile is computed from. Default: the timeseries itself.

    Returns
    -------
    (numpy.ndarray, float)
        Binary event array and the threshold value applied.
    """
    if not np.isscalar(percentile) or not 0 <= percentile <= 100:
        raise ValueError(f"percentile must be in [0, 100], got {percentile!r}")
    timeseries = np.asarray(timeseries, dtype=float)
    ref = np.asarray(reference, dtype=float) if reference is not None \
        else timeseries
    threshold = float(np.percentile(ref, percentile))
    binary = flag_extreme_events(timeseries, threshold, N=N,
                                 min_days=min_days, direction=direction,
                                 flag=flag)
    return binary, threshold


def compare_flag_options(timeseries, threshold, N=1, min_days=None,
                         direction='above', show=None):
    """Side-by-side view of what ``flag='last' / 'first' / 'all'`` each
    produce on the datra.

    Returns a DataFrame with one row per timestep and columns:

    - ``value``     : the input value
    - ``extreme``   : 1 where the raw value is on the extreme side of
      ``threshold`` (before any windowing)
    - ``last``, ``first``, ``all`` : the binary flag series each option
      yields, exactly as :func:`flag_extreme_events` would return it

    Notes to read off the table
    ---------------------------
    * With ``N == 1`` all three columns are identical (``flag`` is
      irrelevant).
    * With ``N > 1``, ``'last'``/``'first'`` produce ONE flag per
      qualifying window -- so a long spell becomes a few sparse flags,
      spaced ``N`` apart. Whether two spells then link in
      :func:`detect_compound_events` depends on the gap between *flags*,
      not between raw extreme days.
    * With ``min_days < N``, an ``'last'``/``'first'`` flag can sit on a
      day whose ``extreme`` column is 0 (it marks the window edge, not an
      extreme day).
    * ``'all'`` re-marks the extreme days inside qualifying windows, so
      counts downstream are in raw extreme days -- but a spell tail shorter
      than a full window stays unflagged.

    Parameters as :func:`flag_extreme_events`. ``show`` optionally slices
    the returned table to ``show`` rows around the first flag (handy for
    long series).
    """
    import pandas as pd

    timeseries = np.asarray(timeseries, dtype=float)
    exceeds = (timeseries > threshold) if direction == 'above' \
        else (timeseries < threshold)
    out = {"value": timeseries, "extreme": exceeds.astype(int)}
    for fl in ("last", "first", "all"):
        out[fl] = flag_extreme_events(timeseries, threshold, N=N,
                                      min_days=min_days,
                                      direction=direction, flag=fl)
    table = pd.DataFrame(out)
    table.index.name = "idx"
    if show is not None:
        flagged = np.flatnonzero(table[["last", "first", "all"]].values.any(
            axis=1))
        centre = int(flagged[0]) if len(flagged) else 0
        lo = max(0, centre - show // 4)
        table = table.iloc[lo:lo + show]
    return table


def thermodynamic_thresholds(data_by_window, baseline_key, fixed_threshold):
    """Per-window thresholds matching the baseline percentile of a fixed
    impact threshold (paper Section 2.3; mirrors
    ``eca_analysis.compound_eca_windows.analyse_region``).

    Parameters
    ----------
    data_by_window : dict[label -> array-like]
        Raw (not binarised) data per rolling window.
    baseline_key : label of the baseline window (e.g. '1980-2010').
    fixed_threshold : float, the impact threshold (e.g. 20 mm/hr).

    Returns
    -------
    dict[label -> float] -- the adjusted threshold per window. The baseline
    window's value equals ``fixed_threshold`` up to percentile inversion.
    """
    from eca_analysis import get_percentile, value_from_percentile
    if baseline_key not in data_by_window:
        raise ValueError(f"baseline_key {baseline_key!r} not in "
                         f"data_by_window keys {list(data_by_window)}")
    base_pct = get_percentile(fixed_threshold, data_by_window[baseline_key])
    return {lbl: value_from_percentile(base_pct, d)
            for lbl, d in data_by_window.items()}


# --------------------------------------------------------------------------- #
# compound-event identification
# --------------------------------------------------------------------------- #
def detect_compound_events(binary_series, delT=4, tau=1, min_duration=2,
                           blocks=None, years=None, months=None,
                           season_start=6, season_length=4):
    """Single-variable mode: sequential extremes of ONE variable (the
    paper's method, Section 2.1).

    Implemented on :func:`eca_analysis.compound_episodes`, so the events
    returned here are exactly the episodes implied by the self-ECA
    coincidence indices -- the counting and the significance test can never
    disagree.

    Parameters
    ----------
    binary_series : array-like
        Binary timeseries from :func:`flag_extreme_events`.
    delT : int, optional
        ECA coincidence window (paper default 4).
    tau : int, optional
        ECA minimum lag, must be >= 1 (paper default 1). Max linkage gap =
        ``tau + delT``.
    min_duration : int, optional
        Minimum number of flagged days per event. Default 2 (single days
        excluded).
    blocks : list of index arrays, or 'contiguous', optional
        Season blocks. Events never link across block boundaries.
    years, months : array-like, optional
        Alternative to ``blocks``: per-step year (and month); blocks are
        built with :func:`make_season_blocks`. ``months`` is required if the
        series is not already season-extracted or if the season wraps the
        calendar year.
    season_start, season_length : int, optional
        Modular season spec (default 6, 4 = JJAS). ``(10, 5)`` = Oct-Feb.

    One of ``blocks`` / ``years`` is REQUIRED -- running unblocked on a
    season-extracted series silently links the last month of one season to
    the first month of the next, which the counting method forbids.

    Returns
    -------
    pd.DataFrame
        Columns: start_idx, end_idx, length, n_extreme_cases; one row per
        event. ``length = end_idx - start_idx + 1`` (span in steps),
        ``n_extreme_cases`` is the paper's "duration".

        A **case** is one flagged timestep of the input binary series -- a
        day for daily data, but the detector is timestep-agnostic (sub-daily
        or monthly series work identically; the column was previously named
        ``n_extreme_days``). Note that with ``N > 1`` at the flagging stage
        each flag represents one qualifying N-step window (under
        ``flag='last'``/``'first'``), so ``n_extreme_cases`` counts flagged
        *cases*, not raw extreme days -- see
        ``docs/choosing_flag_and_windows.md``.
    """
    import pandas as pd

    binary_series = _check_binary("binary_series", binary_series)
    delT, tau = _check_linkage(delT, tau)
    min_duration = _check_int("min_duration", min_duration, 1)
    blks = _resolve_blocks(len(binary_series), blocks, years, months,
                           season_start, season_length,
                           "detect_compound_events")
    episodes = compound_episodes(binary_series, blks, delT=delT, tau=tau)
    rows = [{"start_idx": int(ep[0]),
             "end_idx": int(ep[-1]),
             "length": int(ep[-1] - ep[0] + 1),
             "n_extreme_cases": int(len(ep))}
            for ep in episodes if len(ep) >= min_duration]

def _cluster_within_blocks(indices, blocks, max_gap):
    """Group sorted event indices into clusters: consecutive members are in
    the same block and separated by <= max_gap. Returns list of lists."""
    block_id = {}
    for bi, blk in enumerate(blocks):
        for p in blk:
            block_id[int(p)] = bi
    groups, current = [], []
    for idx in indices:
        idx = int(idx)
        if idx not in block_id:
            continue  # outside all blocks: never part of an event
        if current and (block_id[idx] == block_id[current[-1]]
                        and idx - current[-1] <= max_gap):
            current.append(idx)
        else:
            if current:
                groups.append(current)
            current = [idx]
    if current:
        groups.append(current)
    return groups


def detect_compound_events_bivariate(binary_1, binary_2, delT=4, tau=1,
                                     min_duration_1=1, min_duration_2=1,
                                     blocks=None, years=None, months=None,
                                     season_start=6, season_length=4):
    """Sequential mode: extremes from TWO variables within the linkage
    window of each other (e.g. a dry spell followed by extreme rain).

    UKCI extension (NOT the paper's single-variable method): clusters the
    union of both flagged series with the same linkage rule as
    :func:`detect_compound_events` (gap in ``[tau, tau + delT]`` within one
    season block) and keeps clusters that contain at least
    ``min_duration_1`` flagged days from variable 1 AND ``min_duration_2``
    from variable 2 -- each variable has its own event criterion, symmetric
    with the per-variable flagging stage.

    Example: "a 20-day drought with >= 19 dry days, followed by 3 days of
    rain" = flag variable 1 with ``N=20, min_days=19, direction='below'``,
    flag variable 2 daily (``N=1``), then detect with
    ``min_duration_1=1, min_duration_2=3``.

    Returns
    -------
    pd.DataFrame with columns start_idx, end_idx, length, n_extreme_cases,
    n_extreme_cases_1, n_extreme_cases_2 (``n_extreme_cases`` = sum of both).
    """
    import pandas as pd

    b1 = _check_binary("binary_1", binary_1)
    b2 = _check_binary("binary_2", binary_2)
    if len(b1) != len(b2):
        raise ValueError(f"binary_1 (length {len(b1)}) and binary_2 (length "
                         f"{len(b2)}) must be the same length.")
    delT, tau = _check_linkage(delT, tau)
    min_duration_1 = _check_int("min_duration_1", min_duration_1, 1)
    min_duration_2 = _check_int("min_duration_2", min_duration_2, 1)
    blks = _resolve_blocks(len(b1), blocks, years, months, season_start,
                           season_length, "detect_compound_events_bivariate")

    union_idx = np.flatnonzero(np.clip(b1 + b2, 0, 1) == 1)
    cols = ["start_idx", "end_idx", "length", "n_extreme_cases",
            "n_extreme_cases_1", "n_extreme_cases_2"]
    events = []
    for grp in _cluster_within_blocks(union_idx, blks, tau + delT):
        n1 = int(b1[grp].sum())
        n2 = int(b2[grp].sum())
        if n1 >= min_duration_1 and n2 >= min_duration_2:
            events.append({"start_idx": grp[0], "end_idx": grp[-1],
                           "length": grp[-1] - grp[0] + 1,
                           "n_extreme_cases": n1 + n2,
                           "n_extreme_cases_1": n1, "n_extreme_cases_2": n2})
    return pd.DataFrame(events, columns=cols)


def detect_compound_events_coincident(binary_1, binary_2, delT=4, tau=1,
                                      min_duration=1, blocks=None,
                                      years=None, months=None,
                                      season_start=6, season_length=4):
    """Co-occurring mode: TWO variables simultaneously extreme on the same
    day(s) (e.g. heat + drought).

    UKCI extension (NOT the paper's method): finds days where BOTH flagged
    series are 1, then merges coincident days separated by
    <= ``tau + delT`` within one season block. ``min_duration`` is the
    minimum number of coincident days per event -- since a coincident day is
    by definition extreme in both variables, per-variable minima coincide
    with it (``n_extreme_cases_1/_2`` >= ``n_coincident_cases`` always).

    Returns
    -------
    pd.DataFrame with columns start_idx, end_idx, length, n_extreme_cases,
    n_extreme_cases_1, n_extreme_cases_2, n_coincident_cases.
    ``n_extreme_cases_1/_2`` count all flagged days of each variable within
    the event span.
    """
    import pandas as pd

    b1 = _check_binary("binary_1", binary_1)
    b2 = _check_binary("binary_2", binary_2)
    if len(b1) != len(b2):
        raise ValueError(f"binary_1 (length {len(b1)}) and binary_2 (length "
                         f"{len(b2)}) must be the same length.")
    delT, tau = _check_linkage(delT, tau)
    min_duration = _check_int("min_duration", min_duration, 1)
    blks = _resolve_blocks(len(b1), blocks, years, months, season_start,
                           season_length, "detect_compound_events_coincident")

    coin_idx = np.flatnonzero((b1 == 1) & (b2 == 1))
    cols = ["start_idx", "end_idx", "length", "n_extreme_cases",
            "n_extreme_cases_1", "n_extreme_cases_2", "n_coincident_cases"]
    events = []
    for grp in _cluster_within_blocks(coin_idx, blks, tau + delT):
        if len(grp) < min_duration:
            continue
        start, end = grp[0], grp[-1]
        events.append({"start_idx": start, "end_idx": end,
                       "length": end - start + 1,
                       "n_extreme_cases": len(grp),
                       "n_extreme_cases_1": int(b1[start:end + 1].sum()),
                       "n_extreme_cases_2": int(b2[start:end + 1].sum()),
                       "n_coincident_cases": len(grp)})
    return pd.DataFrame(events, columns=cols)


# --------------------------------------------------------------------------- #
# loading / ensemble summaries
# --------------------------------------------------------------------------- #
def load_region_ensemble(base_dir, region, ensembles,
                         season_start=6, season_length=4,
                         filename_pattern="p110{ensemble}_{region}.nc",
                         return_time=False):
    """Load a variable for all ensemble members of one region, extracted to
    a modular season.

    Parameters
    ----------
    base_dir : str
        Root directory containing per-region subfolders.
    region : str
        Region name, e.g. 'Wales' -- used for both subfolder and filename.
    ensembles : list of str
        Ensemble member IDs, e.g. ['0000', '1113', ...].
    season_start, season_length : int, optional
        Modular season spec: start month + duration in months, wrapping the
        calendar year if needed. Default (6, 4) = June-September. (10, 5) =
        October-February.
    filename_pattern : str, optional
        Filename template with {ensemble} and {region} placeholders.
    return_time : bool, optional
        If True, values are ``(data, years, months)`` tuples instead of bare
        arrays -- pass ``years``/``months`` straight to the detection
        functions for correct season blocking (required for wrapping
        seasons).

    Returns
    -------
    dict
        {ensemble_id: data or (data, years, months)}. Members that fail to
        load are skipped with a printed warning rather than raising.
    """
    import os
    import warnings as _warnings
    import iris

    # opt in to microsecond date precision, silencing iris's FutureWarning
    # about legacy date precision at the source
    try:
        iris.FUTURE.date_microseconds = True
    except AttributeError:
        pass

    month_set = set(season_months(season_start, season_length))
    in_season = iris.Constraint(
        time=lambda cell: cell.point.month in month_set)

    member_data = {}
    for ensemble in ensembles:
        path = os.path.join(base_dir, region,
                            filename_pattern.format(ensemble=ensemble,
                                                    region=region))
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", FutureWarning)
                cube = iris.load(path)[0]
                cube = cube.extract(in_season)
                if return_time:
                    t = cube.coord("time")
                    dates = t.units.num2date(t.points)
                    years = np.array([d.year for d in dates])
                    months = np.array([d.month for d in dates])
                    member_data[ensemble] = (cube.data, years, months)
                else:
                    member_data[ensemble] = cube.data
        except Exception as e:
            print(f"  Warning: could not load ensemble {ensemble} for "
                  f"{region}: {e}")
    return member_data


def summarise_ensemble_events(events_by_member):
    """Collapse per-member compound event DataFrames into an ensemble summary
    (spread of counts per year across members).

    Parameters
    ----------
    events_by_member : dict
        {ensemble_id: DataFrame from detect_compound_events}. Each DataFrame
        must have a 'year' column.

    Returns
    -------
    pd.DataFrame with columns year, mean, min, max, median, n_members.
    """
    import pandas as pd

    per_member_counts = {}
    for ensemble, df in events_by_member.items():
        if len(df) == 0:
            continue
        per_member_counts[ensemble] = df.groupby('year').size()

    if not per_member_counts:
        return pd.DataFrame(columns=['year', 'mean', 'min', 'max', 'median',
                                     'n_members'])

    counts_table = pd.DataFrame(per_member_counts).fillna(0)
    return pd.DataFrame({
        'year': counts_table.index,
        'mean': counts_table.mean(axis=1).values,
        'min': counts_table.min(axis=1).values,
        'max': counts_table.max(axis=1).values,
        'median': counts_table.median(axis=1).values,
        'n_members': (counts_table > 0).sum(axis=1).values,
    }).reset_index(drop=True)