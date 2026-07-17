"""
threshold_detector/detector.py

functions for detecting extreme events in a timeseries.
Works for any variable and any threshold direction (above or below).
"""

import numpy as np

def load_region_ensemble(base_dir, region, ensembles,
                         lower_month=6, higher_month=9,
                         filename_pattern="p110{ensemble}_{region}.nc"):
    """
    Load a variable for all ensemble members of one region.

    Builds paths of the form {base_dir}/{region}/{filename_pattern} and loads
    each into a numpy array, extracting a chosen month range.

    Parameters
    ----------
    base_dir : str
        Root directory containing per-region subfolders.
    region : str
        Region name, e.g. 'Wales' — used for both subfolder and filename.
    ensembles : list of str
        Ensemble member IDs, e.g. ['0000', '1113', ...].
    lower_month, higher_month : int, optional
        Inclusive month range to extract. Default 6–9 (June–September).
    filename_pattern : str, optional
        Filename template with {ensemble} and {region} placeholders.

    Returns
    -------
    dict
        {ensemble_id: numpy array of daily values}. Members that fail to
        load are skipped with a printed warning rather than raising.

    Examples
    --------
    >>> data = load_region_ensemble(BASE_DIR, 'Wales', ENSEMBLES)
    >>> data['0000'].shape
    (3050,)
    """
    import os
    import iris
    from iris.time import PartialDateTime

    member_data = {}
    month_range = iris.Constraint(
        time=lambda cell: PartialDateTime(month=lower_month) <= cell.point<= PartialDateTime(month=higher_month))

    for ensemble in ensembles:
        path = os.path.join(base_dir, region,filename_pattern.format(ensemble=ensemble, region=region))
        try:
            cube = iris.load(path)[0]
            cube = cube.extract(month_range)
            member_data[ensemble] = cube.data
        except Exception as e:
            print(f"  Warning: could not load ensemble {ensemble} for {region}: {e}")

    return member_data

def flag_extreme_events(timeseries, threshold, N=1, direction='above', flag='all'):
    '''
    Flag extreme events in a timeseries.

    An event is flagged when N consecutive values are on the extreme side of the threshold.
      The flag position is determined by the 'flag' parameter.

    Parameters:
    timeseries : array-like
        Input timeseries of any climate variable.
    threshold : float
        The threshold value that defines an extreme.
    N : int, optional
        Number of consecutive days that must exceed the threshold to constitute an event. Default is 1 (any single exceedance).
    direction : str, optional
        'above' flags values greater than threshold (e.g. heavy rain, heat).
        'below' flags values less than threshold (e.g. drought, cold).
        Default is 'above'.
    flag : str, optional
        Which days in the consecutive window to flag.
        'last' flags only the last day of the window.
        'first' flags only the first day of the window.
        'all' flags all days in the window (default).

    Returns
    numpy.ndarray
        Binary array of same length as timeseries. 1 = extreme event, 0 = not.
    '''

    timeseries = np.array(timeseries, dtype=float)
    events = np.zeros(len(timeseries), dtype=int)

    if direction == 'above':
        exceeds = timeseries > threshold
    elif direction == 'below':
        exceeds = timeseries < threshold
    else:
        raise ValueError(f"direction must be 'above' or 'below', got '{direction}'")

    if N == 1:
        events[exceeds] = 1
        return events

    i = 0
    while i <= len(timeseries) - N:
        window = exceeds[i:i + N]
        if np.all(window):
            if flag == 'last':
                events[i + N - 1] = 1
            elif flag == 'first':
                events[i] = 1
            elif flag == 'all':
                events[i:i + N] = 1
            else:
                raise ValueError(f"flag must be 'all', 'first', or 'last', got '{flag}'")
            i += N
        else:
            i += 1

    return events


def summarise_ensemble_events(events_by_member):
    """
    Collapse per-member compound event DataFrames into an ensemble summary,
    giving the spread (range of projections) across members.

    Parameters
    ----------
    events_by_member : dict
        {ensemble_id: DataFrame from detect_compound_events}. Each DataFrame
        must have a 'year' column.

    Returns
    -------
    pd.DataFrame
        One row per year, columns:
        year, mean, min, max, median, n_members
        giving the ensemble spread of compound event counts per year.

    Examples
    --------
    >>> summary = summarise_ensemble_events(events_by_member)
    >>> summary[['year', 'min', 'mean', 'max']].head()
    """
    import pandas as pd
    import numpy as np

    # Build a year-by-member count table
    per_member_counts = {}
    for ensemble, df in events_by_member.items():
        if len(df) == 0:
            continue
        per_member_counts[ensemble] = df.groupby('year').size()
 
    if not per_member_counts:
        return pd.DataFrame(columns=['year', 'mean', 'min', 'max', 'median', 'n_members'])

    counts_table = pd.DataFrame(per_member_counts).fillna(0)

    summary = pd.DataFrame({
        'year':      counts_table.index,
        'mean':      counts_table.mean(axis=1).values,
        'min':       counts_table.min(axis=1).values,
        'max':       counts_table.max(axis=1).values,
        'median':    counts_table.median(axis=1).values,
        'n_members': (counts_table > 0).sum(axis=1).values,}).reset_index(drop=True)

    return summary

def flag_extreme_events_percentile(timeseries, percentile, N=1, direction='above', flag='all', reference=None):
    """
    Flag extreme events using a percentile threshold rather than a fixed value.

    Useful for thermodynamic adjustment — compute the threshold from a reference
    period, then apply it to any period.

    Parameters
    timeseries : array-like
        Input timeseries to flag events in.
    percentile : float
        Percentile to use as threshold (0-100).
    N : int, optional
        Number of consecutive days that must exceed the threshold to
        constitute an event. Default is 1 (any single exceedance).
    direction : str, optional
        'above' or 'below'. Default 'above'.
    flag : str, optional
        Which days in the consecutive window to flag.
        'last' flags only the last day of the window.
        'first' flags only the first day of the window.
        'all' flags all days in the window (default).
    reference : array-like, optional
        Data to compute the percentile from. If None, uses timeseries itself.
        Pass your baseline period data here for thermodynamic adjustment.

    Returns
    numpy.ndarray
        Binary array. 1 = extreme event, 0 = not.
    float
        The threshold value that was computed and applied.
    """

    timeseries = np.array(timeseries, dtype=float)
    ref = np.array(reference, dtype=float) if reference is not None else timeseries
    threshold = np.percentile(ref, percentile)
    binary = flag_extreme_events(timeseries, threshold, N=N, direction=direction, flag=flag)
    return binary, threshold


def detect_compound_events(binary_series, delT=4, min_duration=2):
    '''
    Detect compound events from a binary timeseries.

    A compound event is a sequence of extreme days where no gap between
    consecutive extreme days exceeds delT. Returns a DataFrame with one
    row per event.

    Parameters
    ----------
    binary_series : array-like
        Binary timeseries (1 = extreme day, 0 = not).
    delT : int, optional
        Maximum gap in days between extreme days that still counts as part
        of the same compound event. Default 4.
    min_duration : int, optional
        Minimum number of extreme days to qualify as a compound event.
        Default 2 (single days are excluded).

    Returns
    -------
    pd.DataFrame
        Columns: start_idx, end_idx, length, n_extreme_days
        One row per compound event.

    '''
    import pandas as pd

    binary_series = np.array(binary_series, dtype=int)
    extreme_indices = np.where(binary_series == 1)[0]

    if len(extreme_indices) == 0:
        return pd.DataFrame(columns=['start_idx', 'end_idx', 'length', 'n_extreme_days'])

    events = []
    current_event = [extreme_indices[0]]

    for idx in extreme_indices[1:]:
        if idx - current_event[-1] <= delT:
            current_event.append(idx)
        else:
            if len(current_event) >= min_duration:
                events.append({
                    'start_idx':     current_event[0],
                    'end_idx':       current_event[-1],
                    'length':        current_event[-1] - current_event[0] + 1,
                    'n_extreme_days': len(current_event)})
            current_event = [idx]

    # Don't forget last event
    if len(current_event) >= min_duration:
        events.append({
            'start_idx':      current_event[0],
            'end_idx':        current_event[-1],
            'length':         current_event[-1] - current_event[0] + 1,
            'n_extreme_days': len(current_event)})

    return pd.DataFrame(events)


def detect_compound_events_bivariate(binary_1, binary_2, delT=4, min_duration=1):
    '''
    Detect compound events that require extremes from two separate timeseries.

    A compound event is a cluster of extreme days drawn from the union of both
    binary series, where every adjacent pair of extreme days (from either series)
    is separated by at most delT days, AND the cluster contains at least one
    extreme day from each series.

    This mirrors the logic of detect_compound_events but enforces the
    "bivariate" requirement that both variables must contribute to each event.

    Parameters
    ----------
    binary_1 : array-like
        Binary timeseries for variable 1 (1 = extreme, 0 = not).
    binary_2 : array-like
        Binary timeseries for variable 2 (1 = extreme, 0 = not).
        Must be the same length as binary_1.
    delT : int, optional
        Maximum gap in days between extreme days (from either series) that
        still counts as part of the same compound event. Default 4.
    min_duration : int, optional
        Minimum total number of extreme days (summed across both series) to
        qualify as a compound event. Default 1 (all bivariate pairs kept).

    Returns
    -------
    pd.DataFrame
        Columns: start_idx, end_idx, length, n_extreme_days,
                 n_extreme_days_1, n_extreme_days_2
        One row per compound event. Only events where both series contribute
        at least one extreme day are included.

    Examples
    --------
    >>> dry = flag_extreme_events(pr, threshold=1, direction='below', N=10)
    >>> wet = flag_extreme_events(pr, threshold=20, direction='above', N=1)
    >>> events = detect_compound_events_bivariate(dry, wet, delT=4, min_duration=1)
    '''
    import pandas as pd

    b1 = np.array(binary_1, dtype=int)
    b2 = np.array(binary_2, dtype=int)

    if len(b1) != len(b2):
        raise ValueError(
            f"binary_1 (length {len(b1)}) and binary_2 (length {len(b2)}) must be the same length."
        )

    union = np.clip(b1 + b2, 0, 1)
    extreme_indices = np.where(union == 1)[0]

    if len(extreme_indices) == 0:
        return pd.DataFrame(columns=[
            'start_idx', 'end_idx', 'length',
            'n_extreme_days', 'n_extreme_days_1', 'n_extreme_days_2',
        ])

    # Group extreme days from the union series using the same delT logic as
    # detect_compound_events
    groups = []
    current_group = [extreme_indices[0]]
    for idx in extreme_indices[1:]:
        if idx - current_group[-1] <= delT:
            current_group.append(idx)
        else:
            groups.append(current_group)
            current_group = [idx]
    groups.append(current_group)

    events = []
    for group in groups:
        n1 = int(np.sum(b1[group]))
        n2 = int(np.sum(b2[group]))
        n_total = n1 + n2
        # Require at least one extreme from each series AND min_duration total
        if n1 >= 1 and n2 >= 1 and n_total >= min_duration:
            events.append({
                'start_idx':      group[0],
                'end_idx':        group[-1],
                'length':         group[-1] - group[0] + 1,
                'n_extreme_days': n_total,
                'n_extreme_days_1': n1,
                'n_extreme_days_2': n2,
            })

    return pd.DataFrame(events)


def detect_compound_events_coincident(binary_1, binary_2, delT=4, min_duration=1):
    '''
    Detect compound events where two variables are simultaneously extreme.

    Unlike detect_compound_events_bivariate (sequential pairing), this function
    requires at least one day where BOTH timeseries flag an extreme on the same
    day. Adjacent coincident days separated by at most delT are merged into one
    event.

    Typical use-case: a heat extreme occurring during a drought, where the two
    variables may come from completely different datasets (e.g. tasmax and a
    soil-moisture index).

    Parameters
    ----------
    binary_1 : array-like
        Binary timeseries for variable 1 (1 = extreme, 0 = not).
    binary_2 : array-like
        Binary timeseries for variable 2 (1 = extreme, 0 = not).
        Must be the same length as binary_1.
    delT : int, optional
        Maximum gap in days between coincident extreme days to merge into
        the same event. Default 4.
    min_duration : int, optional
        Minimum number of simultaneously extreme days (days where both series
        are 1 on the same day) to qualify as a compound event. Default 1.

    Returns
    -------
    pd.DataFrame
        Columns: start_idx, end_idx, length, n_extreme_days,
                 n_extreme_days_1, n_extreme_days_2, n_coincident_days
        n_extreme_days equals n_coincident_days (the count of overlap days).
        n_extreme_days_1 / _2 count all extreme days from each series within
        the event span (start_idx to end_idx inclusive).

    Examples
    --------
    >>> hot  = flag_extreme_events(tasmax, threshold=30, direction='above', N=1)
    >>> dry  = flag_extreme_events(precip, threshold=1,  direction='below', N=5)
    >>> events = detect_compound_events_coincident(hot, dry, delT=4, min_duration=1)
    '''
    import pandas as pd

    b1 = np.array(binary_1, dtype=int)
    b2 = np.array(binary_2, dtype=int)

    if len(b1) != len(b2):
        raise ValueError(
            f"binary_1 (length {len(b1)}) and binary_2 (length {len(b2)}) must be the same length."
        )

    coincident_indices = np.where((b1 == 1) & (b2 == 1))[0]

    if len(coincident_indices) == 0:
        return pd.DataFrame(columns=[
            'start_idx', 'end_idx', 'length',
            'n_extreme_days', 'n_extreme_days_1', 'n_extreme_days_2', 'n_coincident_days',
        ])

    # Cluster coincident days using the same delT gap logic
    groups = []
    current_group = [coincident_indices[0]]
    for idx in coincident_indices[1:]:
        if idx - current_group[-1] <= delT:
            current_group.append(idx)
        else:
            groups.append(current_group)
            current_group = [idx]
    groups.append(current_group)

    events = []
    for group in groups:
        n_coincident = len(group)
        if n_coincident < min_duration:
            continue
        start = group[0]
        end = group[-1]
        n1 = int(np.sum(b1[start:end + 1]))
        n2 = int(np.sum(b2[start:end + 1]))
        events.append({
            'start_idx':         start,
            'end_idx':           end,
            'length':            end - start + 1,
            'n_extreme_days':    n_coincident,   # coincident days drive the count
            'n_extreme_days_1':  n1,
            'n_extreme_days_2':  n2,
            'n_coincident_days': n_coincident,
        })

    return pd.DataFrame(events)


def process_binary_series(binary_series, meteo_window=4, max_events_per_window=1):
    """
    Remove excess flags within a meteorological window to prevent overcounting.

    If multiple extreme flags fall within meteo_window days of each other,
    only the first max_events_per_window are kept.

    Parameters
    binary_series : array-like
        Binary timeseries of extreme events.
    meteo_window : int, optional
        Window size in days. Default 4.
    max_events_per_window : int, optional
        Maximum number of events allowed within the window. Default 1.

    """
    series = np.array(binary_series, dtype=int)
    idx = np.where(series == 1)[0]

    for i in idx:
        if series[i] == 1:
            window_end = min(i + meteo_window + 1, len(series))
            ones_in_window = np.where(series[i:window_end] == 1)[0] + i
            if len(ones_in_window) > max_events_per_window:
                excess_start = ones_in_window[max_events_per_window]
                series[excess_start:window_end] = 0

    return series.tolist()