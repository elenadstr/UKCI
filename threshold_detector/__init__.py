'''
threshold_detector

Tools for detecting extreme events, identifying compound episodes, testing
clustering significance, and visualising ensemble spread. The counting/ECA
maths is the vendored paper engine (``eca_analysis``, see
``eca_analysis/VENDORED.md``); this package is the user-facing layer.

Typical workflow
----------------
1. Detect extremes (any variable, either direction, fixed or percentile
   threshold):
   >>> from threshold_detector import flag_extreme_events
   >>> binary = flag_extreme_events(data, threshold=20, direction='above')

2. Identify compound events (paper semantics: linkage gap in [tau, tau+delT],
   never across a season boundary -- so year/month info is required):
   >>> from threshold_detector import detect_compound_events
   >>> events = detect_compound_events(binary, delT=4, tau=1,
   ...                                 years=years, months=months)

3. Test statistical clustering with ECA (same blocking):
   >>> from threshold_detector import run_eca, summary_table
   >>> result = run_eca(binary, binary, delT=4, tau=1,
   ...                  years=years, months=months)
   >>> print(summary_table(result))
   >>> compound_positions = sorted(set(result.prec_indices)
   ...                             | set(result.trigg_indices))

4. Sliding-window analysis with null bands (paper Fig. 4): use
   ``eca_analysis.WindowConfig`` + ``run_window_analysis``, or
   ``coincidence.eca_null_band`` for a single window.

5. Visualise:
   >>> from threshold_detector import plot_event_timeseries
   >>> fig = plot_event_timeseries(compound_dates, rolling_window=10)
'''

from .detector import (
    flag_extreme_events,
    season_months,
    season_year_labels,
    flag_extreme_events_percentile,
    thermodynamic_thresholds,
    detect_compound_events,
    detect_compound_events_bivariate,
    detect_compound_events_coincident,
    make_season_blocks,
    load_region_ensemble,
    summarise_ensemble_events,
)
from .coincidence import (
    run_eca,
    run_eca_rolling,
    eca_null_band,
    summary_table,
)
from .plotting import (
    save_figure,
    plot_event_timeseries,
    plot_ensemble_hovmoller,
    plot_duration_and_counts,
    plot_ensemble_spread,
)

# paper-method building blocks re-exported from the vendored engine
from eca_analysis import (
    classify_days,
    episode_stats,
    national_series_from_dates,
    compound_membership,
    compound_episodes,
)

__all__ = [
    # Detection
    'flag_extreme_events',
    'flag_extreme_events_percentile',
    'thermodynamic_thresholds',
    'detect_compound_events',
    'detect_compound_events_bivariate',
    'detect_compound_events_coincident',
    'make_season_blocks',
    'season_months',
    'season_year_labels',
    # ECA
    'run_eca',
    'run_eca_rolling',
    'eca_null_band',
    'summary_table',
    # Paper building blocks (vendored engine)
    'classify_days',
    'episode_stats',
    'national_series_from_dates',
    'compound_membership',
    'compound_episodes',
    # IO / ensemble
    'load_region_ensemble',
    'summarise_ensemble_events',
    # Plotting
    'save_figure',
    'plot_event_timeseries',
    'plot_ensemble_hovmoller',
    'plot_duration_and_counts',
    'plot_ensemble_spread',
]
