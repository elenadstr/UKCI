"""
threshold_detector/plotting.py

plotting functions for extreme event detection and ECA results.
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def _add_ipcc_title(fig, heading, sub_heading, rect_top=0.88):
    """
    Add an IPCC-style figure heading and descriptive sub-heading.

    The heading is bold and left-aligned at the top of the figure.
    The sub-heading is smaller and italic, placed immediately below.
    tight_layout is called with rect adjusted to reserve space at the top.
    """
    fig.suptitle(heading, fontsize=11, fontweight='bold',
                 x=0.01, ha='left', y=0.99)
    fig.text(0.01, 0.93, sub_heading, fontsize=9, style='italic',
             ha='left', va='top', color='#555555')
    plt.tight_layout(rect=[0, 0, 1, rect_top])



def plot_event_timeseries(compound_dates, single_dates=None, rolling_window=10, compound_colour='orangered', single_colour='royalblue', title=None, figsize=(12, 5)):
    '''
    Plot annual counts of compound (and optionally single) extreme events with a rolling mean overlay.

    Parameters
    ----------
    compound_dates : array-like of datetime-like
        Dates of compound extreme events. Each date counts once per year.
    single_dates : array-like of datetime-like, optional
        Dates of single (isolated) extreme events. If None, only compound events are plotted.
    rolling_window : int, optional
        Window size in years for the rolling mean. Default 10.
    compound_colour : str, optional
        Colour for compound event line. Default 'orangered'.
    single_colour : str, optional
        Colour for single event line. Default 'royalblue'.
    title : str, optional
        Plot title. Auto-generated if None.
    '''
    compound_dates = pd.to_datetime(compound_dates)
    compound_per_year = compound_dates.dt.year.value_counts().sort_index()
    compound_rolling = compound_per_year.rolling(window=rolling_window,center=True).mean()

    fig, ax = plt.subplots(figsize=figsize)

    if single_dates is not None:
        single_dates = pd.to_datetime(single_dates)
        single_per_year = single_dates.dt.year.value_counts().sort_index()
        single_rolling = single_per_year.rolling(window=rolling_window, center=True).mean()

        ax.plot(single_per_year.index, single_per_year.values, marker='o', linestyle='-', color=single_colour, markersize=3, alpha=0.4, label='Single')
        ax.plot(single_rolling.index, single_rolling.values, linestyle='-', color=single_colour, linewidth=2.5, label=f'Single ({rolling_window}-yr mean)')

    ax.plot(compound_per_year.index, compound_per_year.values, marker='o', linestyle='-', color=compound_colour, markersize=3, alpha=0.4, label='Compound')
    ax.plot(compound_rolling.index, compound_rolling.values, linestyle='-', color=compound_colour, linewidth=2.5, label=f'Compound ({rolling_window}-yr mean)')

    ax.set_xlabel('Year')
    ax.set_ylabel('Number of events')
    ax.legend()
    _add_ipcc_title(
        fig,
        heading=title or 'Annual frequency of extreme events over time',
        sub_heading=(
            f'Annual counts of compound'
            f'{" and single" if single_dates is not None else ""} extreme events per year, '
            f'with a {rolling_window}-year rolling mean overlay to highlight multi-decadal trends.'
        )
    )
    return fig



def _resolve_case_col(events_df, duration_col):
    """Back-compat shim: accept DataFrames still carrying the pre-rename
    ``n_extreme_days`` column (renamed ``n_extreme_cases``)."""
    import warnings
    if duration_col is None or duration_col in events_df.columns:
        return duration_col
    if duration_col == 'n_extreme_cases' and 'n_extreme_days' \
            in events_df.columns:
        warnings.warn("column 'n_extreme_days' was renamed "
                      "'n_extreme_cases'; using the old column. Re-run "
                      "detection to get the new name.", DeprecationWarning,
                      stacklevel=3)
        return 'n_extreme_days'
    raise KeyError(f"column {duration_col!r} not in events_df "
                   f"(columns: {list(events_df.columns)})")


def get_hovmoller_data(events_df, ensemble_col='ensemble', year_col='year',
                       case_col='n_extreme_cases', length_col='length',
                       years=None, ensembles=None, extra_cols='auto'):
    """The data behind the Hovmoller plots, as a tidy DataFrame: one row per
    (ensemble member, year) cell, zeros filled for empty cells.

    Works for all three detection modes -- pass the concatenated events
    DataFrame from :func:`detect_compound_events` (single-variable),
    ``detect_compound_events_bivariate`` or
    ``detect_compound_events_coincident`` (with ``year``/``ensemble``
    columns attached, as in the demo notebook). Per-variable columns
    (``n_extreme_cases_1/_2``, ``n_coincident_cases``) are summed
    automatically when present.

    This is the single source of truth for
    :func:`plot_ensemble_hovmoller` -- use it directly to drive an
    interactive front-end (hover tooltips, plotly, etc.) or to export the
    numbers, guaranteed to match the figures.

    Parameters
    ----------
    events_df : pd.DataFrame
        One row per event; must contain ``ensemble_col`` and ``year_col``.
    ensemble_col, year_col, case_col, length_col : str
        Column names. ``case_col='n_extreme_cases'`` (falls back to the
        pre-rename ``n_extreme_days`` with a DeprecationWarning);
        pass ``None`` to skip the duration statistics.
    years, ensembles : array-like, optional
        Full grids for the axes. Default: the values present in
        ``events_df``; pass e.g. ``years=range(1981, 2080)`` so years with
        no events anywhere still appear as zero rows.
    extra_cols : 'auto' or list of str
        Additional per-event columns to sum per cell. ``'auto'`` picks up
        ``n_extreme_cases_1``, ``n_extreme_cases_2`` and
        ``n_coincident_cases`` if present.

    Returns
    -------
    pd.DataFrame with one row per (ensemble, year) and columns:

    - ``ensemble``, ``year``
    - ``n_events``       : number of events in the cell
    - ``n_extreme_cases``: total flagged cases (timesteps) across events
    - ``max_duration``   : largest single-event ``case_col`` value
    - ``mean_duration``  : mean ``case_col`` per event (NaN if no events)
    - ``mean_length``    : mean span (``length_col``) per event (NaN if none)
    - ``durations``      : list of per-event ``case_col`` values, in event
      order -- ready for a tooltip like "3 events lasting 2, 4 and 7 cases"
    - any ``extra_cols`` (summed per cell)
    """
    import warnings

    case_col = _resolve_case_col(events_df, case_col)
    if length_col is not None and length_col not in events_df.columns:
        raise KeyError(f"length_col {length_col!r} not in events_df")

    if extra_cols == 'auto':
        extra_cols = [c for c in ('n_extreme_cases_1', 'n_extreme_cases_2',
                                  'n_coincident_cases',
                                  'n_extreme_days_1', 'n_extreme_days_2',
                                  'n_coincident_days')
                      if c in events_df.columns]
    else:
        extra_cols = list(extra_cols or [])

    if ensembles is None:
        ensembles = sorted(events_df[ensemble_col].unique())
    if years is None:
        years = sorted(events_df[year_col].unique())
    years = [int(y) for y in years]

    grouped = events_df.groupby([ensemble_col, year_col])
    rows = []
    for ens in ensembles:
        for yr in years:
            try:
                g = grouped.get_group((ens, yr))
            except KeyError:
                g = None
            row = {'ensemble': ens, 'year': yr,
                   'n_events': 0 if g is None else len(g)}
            if case_col is not None:
                if g is None or len(g) == 0:
                    row.update({'n_extreme_cases': 0, 'max_duration': 0,
                                'mean_duration': np.nan,
                                'durations': []})
                else:
                    d = g[case_col].to_numpy()
                    row.update({'n_extreme_cases': int(d.sum()),
                                'max_duration': int(d.max()),
                                'mean_duration': float(d.mean()),
                                'durations': [int(x) for x in d]})
            if length_col is not None:
                row['mean_length'] = (np.nan if g is None or len(g) == 0
                                      else float(g[length_col].mean()))
            for c in extra_cols:
                row[c] = 0 if g is None else int(g[c].sum())
            rows.append(row)
    return pd.DataFrame(rows)


def _hovmoller_hover_text(cell):
    """One-line hover/tooltip description of a (ensemble, year) cell of
    :func:`get_hovmoller_data`."""
    if cell['n_events'] == 0:
        return (f"{cell['ensemble']} | {cell['year']}: no events")
    dur = cell.get('durations')
    dur_txt = ''
    if isinstance(dur, list) and dur:
        dur_txt = (f" lasting {', '.join(str(d) for d in dur)} "
                   f"case{'s' if max(dur) != 1 else ''}")
    return (f"{cell['ensemble']} | {cell['year']}: "
            f"{cell['n_events']} event{'s' if cell['n_events'] != 1 else ''}"
            f"{dur_txt}")


def plot_ensemble_hovmoller(events_df, ensemble_col='ensemble',
                            year_col='year', count_col=None,
                            duration_col='n_extreme_cases',
                            length_col='length',
                            cmap='Blues', figsize=(14, 5),
                            hover=True, data=None):
    """
    Hovmoller-style heatmaps (ensemble member x year) of the compound-event
    record.

    Produces up to FIVE figures, returned as a dict:

    - ``'events'``        : number of events per member per year
    - ``'cases'``         : total extreme cases (flagged timesteps) per
                            member per year (this key was ``'days'`` before
                            the ``n_extreme_days`` -> ``n_extreme_cases``
                            rename)
    - ``'max_duration'``  : the maximum total duration (extreme cases) of
                            any single event, per member per year
    - ``'mean_duration'`` : mean event duration (extreme cases) per member
                            per year
    - ``'mean_length'``   : mean event length (span, end - start + 1)
                            per member per year

    The duration panels need ``duration_col`` and the length panel needs
    ``length_col``; pass ``None`` to skip (the corresponding dict entry is
    then absent).

    All panels are built from :func:`get_hovmoller_data`, so the figures
    and the exported table can never disagree. With ``hover=True`` (and an
    interactive matplotlib backend, e.g. ``%matplotlib widget``), moving
    the mouse over a cell shows "member | year: k events lasting a, b, c
    cases" -- in the toolbar/status area always, and as a cursor tooltip
    too when the optional ``mplcursors`` package is installed.

    Parameters
    ----------
    events_df : pd.DataFrame
        One row per event. Must contain ensemble_col and year_col.
    ensemble_col, year_col : str
        Column names. Defaults 'ensemble', 'year'.
    count_col : str, optional
        If provided, the 'events' panel sums this column instead of counting
        rows (for pre-aggregated inputs).
    duration_col : str or None, optional
        Column with each event's total duration in extreme cases. Default
        'n_extreme_cases' (accepts the pre-rename 'n_extreme_days' with a
        DeprecationWarning).
    length_col : str or None, optional
        Column with each event's length (span). Default 'length'.
    cmap : str, optional
    figsize : tuple, optional
    hover : bool, optional
        Attach the per-cell hover text (default True; needs an interactive
        backend to be visible).
    data : pd.DataFrame, optional
        A precomputed :func:`get_hovmoller_data` table; computed from
        ``events_df`` if omitted.

    Returns
    -------
    dict of {name: matplotlib Figure}
    """
    duration_col = _resolve_case_col(events_df, duration_col)
    if data is None:
        data = get_hovmoller_data(events_df, ensemble_col=ensemble_col,
                                  year_col=year_col, case_col=duration_col,
                                  length_col=length_col)
    ensembles = sorted(data['ensemble'].unique())
    all_years = sorted(data['year'].unique())
    n_ens = len(ensembles)

    def _pivot(col, fill=0.0):
        return (data.pivot(index='ensemble', columns='year', values=col)
                .reindex(index=ensembles, columns=all_years)
                .fillna(fill).to_numpy(dtype=float))

    n_events = _pivot('n_events')
    if count_col is None:
        event_matrix = n_events
    else:
        # pre-aggregated input: sum count_col per cell
        agg = (events_df.groupby([ensemble_col, year_col])[count_col].sum()
               .unstack().reindex(index=ensembles, columns=all_years)
               .fillna(0))
        event_matrix = agg.to_numpy(dtype=float)

    # order members by total event count (ascending), shared by all panels
    sort_order = np.argsort(event_matrix.sum(axis=1))
    labels = [ensembles[i] for i in sort_order]

    # hover text per (sorted-row, year) cell
    hover_lookup = {(r['ensemble'], r['year']): _hovmoller_hover_text(r)
                    for _, r in data.iterrows()}

    def _make(matrix, heading, sub_heading, cbar_label):
        fig, ax = plt.subplots(figsize=figsize)
        c = ax.pcolormesh(all_years, np.arange(n_ens), matrix[sort_order],
                          shading='nearest', cmap=cmap)
        ax.set_xticks([y for y in all_years if y % 10 == 0])
        ax.set_yticks(np.arange(n_ens))
        ax.set_yticklabels(labels)
        ax.set_xlabel('Year')
        ax.set_ylabel('Ensemble member')
        fig.colorbar(c, ax=ax, label=cbar_label)
        _add_ipcc_title(fig, heading, sub_heading)

        if hover:
            year_arr = np.asarray(all_years)

            def _fmt(x, y, _default=ax.format_coord):
                yi = int(round(y))
                xi = int(np.argmin(np.abs(year_arr - x)))
                if 0 <= yi < n_ens and abs(year_arr[xi] - x) <= 0.5:
                    key = (labels[yi], int(year_arr[xi]))
                    txt = hover_lookup.get(key)
                    if txt:
                        return txt
                return _default(x, y)

            ax.format_coord = _fmt  # status-bar hover, no dependencies
            try:                    # optional richer tooltip
                import mplcursors

                cur = mplcursors.cursor(c, hover=True)

                @cur.connect("add")
                def _(sel):
                    j, i = divmod(sel.index, len(year_arr))
                    key = (labels[j], int(year_arr[i]))
                    sel.annotation.set_text(
                        hover_lookup.get(key, ''))
            except ImportError:
                pass
        return fig

    figs = {'events': _make(
        event_matrix,
        'Ensemble-year Hovmoller diagram of event frequency',
        'Number of compound events per year for each ensemble member, '
        'ordered by total event count.', 'Events')}

    if duration_col is not None:
        figs['cases'] = _make(
            _pivot('n_extreme_cases'),
            'Ensemble-year Hovmoller diagram of extreme case totals',
            'Total number of extreme cases (flagged timesteps) in compound '
            'events per year for each ensemble member.', 'Extreme cases')
        figs['max_duration'] = _make(
            _pivot('max_duration'),
            'Ensemble-year Hovmoller diagram of maximum event duration',
            'Maximum total duration (extreme cases) of any single compound '
            'event in each year, per ensemble member.',
            'Max duration (cases)')
        figs['mean_duration'] = _make(
            _pivot('mean_duration'),
            'Ensemble-year Hovmoller diagram of mean event duration',
            'Mean duration (extreme cases per event) of compound events in '
            'each year, per ensemble member.', 'Mean duration (cases)')

    if length_col is not None:
        figs['mean_length'] = _make(
            _pivot('mean_length'),
            'Ensemble-year Hovmoller diagram of mean event length',
            'Mean length (span in timesteps, end - start + 1) of compound '
            'events in each year, per ensemble member.',
            'Mean length (timesteps)')

    return figs

def plot_ensemble_spread(events_by_member, rolling_window=10,
                         mean_colour='black', cmap='tab20',
                         year_range=None, title=None, figsize=(13, 5.5)):
    """
    Plot every ensemble member as its own coloured line of annual compound
    event counts, with the ensemble mean overlaid in a distinct colour.

    Parameters
    ----------
    events_by_member : dict
        {ensemble_id: events DataFrame with a 'year' column}, as built in
        the demo notebook. (For backwards convenience a summary DataFrame
        from summarise_ensemble_events is NOT accepted -- per-member counts
        are needed to draw the member lines.)
    rolling_window : int, optional
        Years for the rolling mean applied to every line. Default 10.
        Set to 1 for raw annual counts.
    mean_colour : str, optional
        Colour of the ensemble-mean line (distinct from the members).
    cmap : str, optional
        Qualitative colormap cycled across members. Default 'tab20'.
    year_range : (int, int), optional
        Inclusive year axis; default spans the years present in the data.
        Years with no events count as 0 (important for the mean).
    title : str, optional
    figsize : tuple, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    members = sorted(events_by_member)
    if not members:
        raise ValueError("events_by_member is empty")
    all_years = pd.concat(
        [df['year'] for df in events_by_member.values() if len(df) > 0])
    if year_range is None:
        year_range = (int(all_years.min()), int(all_years.max()))
    years = np.arange(year_range[0], year_range[1] + 1)

    counts = pd.DataFrame(index=years)
    for ens in members:
        df = events_by_member[ens]
        c = df.groupby('year').size() if len(df) else pd.Series(dtype=float)
        counts[ens] = c.reindex(years).fillna(0)

    smoothed = counts.rolling(window=rolling_window, center=True,
                              min_periods=1).mean()
    mean_line = smoothed.mean(axis=1)

    colours = mpl.colormaps[cmap](np.linspace(0, 1, len(members)))
    fig, ax = plt.subplots(figsize=figsize)
    for colour, ens in zip(colours, members):
        ax.plot(years, smoothed[ens], color=colour, lw=1.2, alpha=0.85,
                label=ens)
    ax.plot(years, mean_line, color=mean_colour, lw=3.0, zorder=5,
            label='Ensemble mean')

    ax.set_xlabel('Year')
    ax.set_ylabel('Compound events per year')
    ax.legend(title='Member', ncols=2, fontsize=8, loc='center left',
              bbox_to_anchor=(1.01, 0.5))
    _add_ipcc_title(
        fig,
        heading=title or 'Ensemble projection of annual compound event frequency',
        sub_heading=(
            f'Each coloured line is one ensemble member '
            f'({rolling_window}-year rolling mean of annual compound event '
            f'counts); the thick {mean_colour} line is the ensemble mean.'
        )
    )
    fig.tight_layout()
    return fig

def plot_duration_and_counts(events_df, ensemble_col='ensemble', year_col='year', length_col='length', year_range=(1979, 2079), cmap='YlGnBu_r', figsize=(16, 10)):
    """
    Two-panel figure showing event duration distribution and per-ensembl annual event counts.

    Top panel: for each year, how many events of each duration occurred (summed across all ensemble members).
    Bottom panel: for each ensemble member, how many events occurred per year.

    Parameters
    events_df : pd.DataFrame
        One row per event. Must contain ensemble_col, year_col, length_col.
    ensemble_col : str
        Column name for ensemble member ID. Default 'ensemble'.
    year_col : str
        Column for year of event. Default 'year'.
    length_col : str
        Column for event duration in days. Default 'length'.
    year_range : tuple of (int, int), optional
        (start_year, end_year) inclusive. Default (1979, 2079).
    cmap : str, optional
        Matplotlib colormap. Default 'YlGnBu_r'.
    figsize : tuple, optional
        Figure size. Default (16, 10).

    """
    start_yr, end_yr = year_range
    years = np.arange(start_yr, end_yr + 1)
    n_years = len(years)

    ensembles = sorted(events_df[ensemble_col].unique())
    n_ensembles = len(ensembles)
    ens_to_idx = {e: i for i, e in enumerate(ensembles)}

    max_length = int(events_df[length_col].max())

    # duration matrix: rows = duration, cols = year, values = count across all ensembles
    duration_matrix = np.zeros((max_length + 1, n_years))

    # count matrix: rows = ensemble, cols = year, values = number of events
    count_matrix = np.zeros((n_ensembles, n_years), dtype=int)

    for _, row in events_df.iterrows():
        yr = row[year_col]
        if start_yr <= yr <= end_yr:
            yr_idx = yr - start_yr
            L = int(row[length_col])
            duration_matrix[L, yr_idx] += 1

            ei = ens_to_idx.get(row[ensemble_col])
            if ei is not None:
                count_matrix[ei, yr_idx] += 1

    duration_masked = np.ma.masked_equal(duration_matrix, 0)
    count_masked = np.ma.masked_equal(count_matrix, 0)

    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)

    im1 = axs[0].imshow(duration_masked, aspect='auto', origin='lower', extent=[years.min(), years.max(), 0, max_length], cmap=cmap)
    axs[0].set_ylabel('Duration (days)')
    axs[0].set_title('(a) Duration distribution across all ensembles', loc='left', fontsize=9)
    fig.colorbar(im1, ax=axs[0], label='Count')

    im2 = axs[1].imshow(count_masked, aspect='auto', origin='lower', extent=[years.min(), years.max(), 0, n_ensembles], cmap=cmap)
    axs[1].set_xlabel('Year')
    axs[1].set_ylabel('Ensemble member')
    axs[1].set_yticks(np.arange(n_ensembles) + 0.5)
    axs[1].set_yticklabels(ensembles)
    axs[1].set_title('(b) Annual event counts per ensemble member', loc='left', fontsize=9)
    fig.colorbar(im2, ax=axs[1], label='Count')

    _add_ipcc_title(
        fig,
        heading='Event duration distribution and annual counts across ensemble members',
        sub_heading=(
            '(a) Number of events of each duration (days) summed across all ensemble members per year. '
            '(b) Number of events per year for each individual ensemble member. '
            'Both panels share a common time axis.'
        ),
        rect_top=0.85
    )
    return fig

def save_figure(fig, path, dpi=150, **savefig_kwargs):
    """Save a figure, creating the output directory first if it does not
    exist.

    Parameters
    ----------
    fig : matplotlib Figure
    path : str -- output file path; parent directories are created with
        ``os.makedirs(..., exist_ok=True)``.
    dpi : int, optional
    **savefig_kwargs : forwarded to ``fig.savefig`` (``bbox_inches='tight'``
        is applied unless overridden).

    Returns
    -------
    str -- the path written.
    """
    import os

    out_dir = os.path.dirname(path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    savefig_kwargs.setdefault('bbox_inches', 'tight')
    fig.savefig(path, dpi=dpi, **savefig_kwargs)
    return path