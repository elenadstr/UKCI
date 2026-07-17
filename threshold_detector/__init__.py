'''
threshold_detector

Tools for detecting extreme events, identifying compound episodes, and visualising ensemble spread. 
Designed to feed into
EventCoincidenceAnalysis but usable standalone.

Typical workflow
----------------
1. Detect extremes from your timeseries:
   >>> from threshold_detector import flag_extreme_events
   >>> binary = flag_extreme_events(data, threshold=20, direction='above')

2. Identify compound events:
   >>> from threshold_detector import detect_compound_events
   >>> events = detect_compound_events(binary, delT=4, min_duration=2)

3. Test statistical clustering with ECA:
   >>> from threshold_detector import run_eca
   >>> result = run_eca(binary, binary, delT=4, tau=1)
   >>> print(result.summary_table())

4. Visualise:
   >>> from threshold_detector import plot_event_timeseries
   >>> fig = plot_event_timeseries(compound_dates, rolling_window=10)
'''

from .detector import (
    flag_extreme_events,
    flag_extreme_events_percentile,
    detect_compound_events,
    detect_compound_events_bivariate,
    detect_compound_events_coincident,
    process_binary_series,
    load_region_ensemble,
    summarise_ensemble_events,)
from .coincidence import (run_eca,run_eca_rolling,)

from .plotting import (plot_event_timeseries,plot_ensemble_hovmoller,plot_duration_and_counts,plot_ensemble_spread,)

__all__ = [
    # Detection
    'flag_extreme_events',
    'flag_extreme_events_percentile',
    'detect_compound_events',
    'detect_compound_events_bivariate',
    'detect_compound_events_coincident',
    'process_binary_series',
    # ECA wrappers
    'run_eca',
    'run_eca_rolling',
    # Plotting
    'plot_event_timeseries',
    'plot_ensemble_hovmoller',
    'plot_duration_and_counts',
    'load_region_ensemble',
    'summarise_ensemble_events',
    'plot_ensemble_spread',]