"""
threshold_detector/plotting.py

plotting functions for extreme event detection and ECA results.
"""

import numpy as np
import pandas as pd
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

    Parametres@ 
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
    ax.set_ylabel('Number of days')
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



def plot_ensemble_hovmoller(events_df, ensemble_col='ensemble', year_col='year', count_col=None, extreme_days_col=None, cmap='Blues', figsize=(14, 5)):
    """
    Plot Hovmöller-style heatmaps of ensemble spread over time.

    Produces TWO separate figures:
      1. Number of events per ensemble per year
      2. Number of extreme days per ensemble per year (if extreme_days_col given)

    Parameters
    ----------
    events_df : pd.DataFrame
        One row per event. Must contain ensemble_col and year_col.
    ensemble_col : str
        Column name for ensemble member ID. Default 'ensemble'.
    year_col : str
        Column name for year. Default 'year'.
    count_col : str, optional
        If provided, use this column to sum values rather than counting rows.
        Useful if events_df is already aggregated.
    extreme_days_col : str, optional
        Column containing the number of extreme days in each event (e.g. 'n_extreme_days').
        If provided, a second figure is produced showing total extreme days per year.
    cmap : str, optional
        Matplotlib colormap name. Default 'Blues'.
    figsize : tuple, optional
        Size of each individual figure. Default (14, 5).

    Returns
    fig_events : Figure
        Number of events per ensemble per year.
    fig_days : Figure or None
        Number of extreme days per ensemble per year. None if extreme_days_col mnot provided.
    """
    ensembles = sorted(events_df[ensemble_col].unique())
    all_years = sorted(events_df[year_col].unique())
    n_ens = len(ensembles)
    n_years = len(all_years)
    ens_to_idx = {e: i for i, e in enumerate(ensembles)}
    yr_to_idx  = {y: i for i, y in enumerate(all_years)}

    # count matrix (number of events)
    event_matrix = np.zeros((n_ens, n_years))
    for _, row in events_df.iterrows():
        ei = ens_to_idx.get(row[ensemble_col])
        yi = yr_to_idx.get(row[year_col])
        if ei is not None and yi is not None:
            event_matrix[ei, yi] += 1 if count_col is None else row[count_col]

    #order ensembles by total event count (ascending)
    total_counts = event_matrix.sum(axis=1)
    sort_order = np.argsort(total_counts)
    event_matrix = event_matrix[sort_order]
    sorted_ensemble_labels = [ensembles[i] for i in sort_order]

    def _make_hovmoller(matrix, labels, ylabel_text, heading, sub_heading, figsize, cmap):
        fig, ax = plt.subplots(figsize=figsize)
        c = ax.pcolormesh(all_years, np.arange(n_ens), matrix, shading='nearest', cmap=cmap)
        ax.set_xticks([y for y in all_years if y % 10 == 0])
        ax.set_yticks(np.arange(n_ens))
        ax.set_yticklabels(labels)
        ax.set_xlabel('Year')
        ax.set_ylabel(ylabel_text)
        fig.colorbar(c, ax=ax, label='Count')
        _add_ipcc_title(fig, heading, sub_heading)
        return fig

    fig_events = _make_hovmoller(
        event_matrix, sorted_ensemble_labels, 'Ensemble member',
        'Ensemble-year Hovmöller diagram of event frequency',
        'Number of extreme events per year for each ensemble member, ordered by total event count. Colour intensity indicates higher event frequency.',
        figsize, cmap
    )

    fig_days = None
    if extreme_days_col is not None:
        days_matrix = np.zeros((n_ens, n_years))
        for _, row in events_df.iterrows():
            ei = ens_to_idx.get(row[ensemble_col])
            yi = yr_to_idx.get(row[year_col])
            if ei is not None and yi is not None:
                days_matrix[ei, yi] += row[extreme_days_col]
        days_matrix = days_matrix[sort_order]  # same sort order as event matrix

        fig_days = _make_hovmoller(
            days_matrix, sorted_ensemble_labels, 'Ensemble member',
            'Ensemble-year Hovmöller diagram of extreme day totals',
            'Total number of extreme days per year for each ensemble member, ordered by total event count. Colour intensity indicates a greater number of extreme days.',
            figsize, cmap
        )

    return fig_events, fig_days

def plot_ensemble_spread(summary_df, rolling_window=10,
                         line_colour='darkblue', band_colour='cornflowerblue',
                         title=None, figsize=(12, 5)):
    """
    plolt the ensemble mean compound event count per year with a shaded
    min–max band showing the projection range.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of summarise_ensemble_events — needs columns
        year, mean, min, max.
    rolling_window : int, optional
        Years for the rolling mean of the central line. Default 10.
    line_colour : str, optional
        Colour of the ensemble mean line.
    band_colour : str, optional
        Colour of the min–max shaded band.
    title : str, optional
    figsize : tuple, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.pyplot as plt

    s = summary_df.sort_values('year')
    mean_rolling = s['mean'].rolling(window=rolling_window, center=True).mean()

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(s['year'], s['min'], s['max'],
                    color=band_colour, alpha=0.4,
                    label='Ensemble range (min–max)')
    ax.plot(s['year'], s['mean'], color=line_colour, alpha=0.3, lw=1,
            label='Ensemble mean (annual)')
    ax.plot(s['year'], mean_rolling, color=line_colour, lw=2.5,
            label=f'Ensemble mean ({rolling_window}-yr)')

    ax.set_xlabel('Year')
    ax.set_ylabel('Compound events per year')
    ax.legend()
    _add_ipcc_title(
        fig,
        heading=title or 'Ensemble projection of annual compound event frequency',
        sub_heading=(
            f'Ensemble mean compound event count per year with a {rolling_window}-year rolling '
            f'mean (solid line) and a shaded band spanning the full ensemble range '
            f'(minimum to maximum across members).'
        )
    )
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