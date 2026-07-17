# threshold_detector

Tools for detecting extreme events in climate timeseries, identifying compound events, and visualising ensemble spread.

Designed to be used standalone or to feed binary event series into `EventCoincidenceAnalysis`.

## What this module does

Given any climate timeseries (rainfall, temperature, soil moisture, wind speed, etc.), this module lets you:

1. **Detect extremes** using a fixed threshold or a percentile-based threshold that can be anchored to a reference period (thermodynamic adjustment, see Section 2.3 of ADDPAPER)
2. **Identify compound events** — sequences of extreme days where no gap exceeds a user-defined window
3. **Test clustering** by running ECA on the resulting binary series
4. **Visualise** event counts, ensemble spread, and event duration distributions

## Functions

### Detection (`detector.py`)

function: purpose 
`flag_extreme_events(timeseries, threshold, N, direction)`: Binary flag for extremes above or below a threshold 
`flag_extreme_events_percentile(timeseries, percentile, direction, reference)`: Same, but threshold is computed as a percentile of a reference period 
`detect_compound_events(binary_series, delT, min_duration)`: Returns a DataFrame of compound events with start, end, length 
`process_binary_series(binary_series, meteo_window, max_events_per_window)`: Remove overcounting within a meteorological window 

### ECA wrappers (`coincidence.py`)

function: purpose
`run_eca(seriesA, seriesB, delT, tau)`: Run ECA on two binary series 
'run_eca_rolling(series_list, delT, tau)`: Run ECA across a list of time periods 

### Visualisation (`plotting.py`)

function: purpose
`plot_event_timeseries(compound_dates, single_dates, ...)`: Annual counts with rolling mean
`plot_ensemble_hovmoller(events_df, ...)`: Two Hovmöller figures: event counts and extreme days
`plot_duration_and_counts(events_df, ...)`: Two-panel: duration distribution and per-ensemble counts

## Usage examples

### Detect extremes above a fixed threshold (e.g extreme rainfall)

```python
from threshold_detector import flag_extreme_events

binary = flag_extreme_events(rainfall_array, threshold=20, direction='above')
```

### Detect extremes below a threshold (e.g. dry days leading to a drought)

```python
binary = flag_extreme_events(soil_moisture, threshold=5, direction='below')
```

### Thermodynamic adjustment

Compute the threshold from a baseline period and apply it to future data.
This removes the effect of general warming on event frequency.

```python
from threshold_detector import flag_extreme_events_percentile

# baseline_data and future_data are numpy arrays
binary_future, threshold_used = flag_extreme_events_percentile( timeseries=future_data, percentile=95, reference=baseline_data)
print(f"Threshold applied: {threshold_used:.1f} mm/hr")
```

### Identify compound events

```python
from threshold_detector import detect_compound_events

events_df = detect_compound_events(binary, delT=4, min_duration=2)
print(events_df.head())
#    start_idx  end_idx  length  n_extreme_days
# 0        142      148       7               3
# 1        301      305       5               2
```

### Test whether clustering exceeds random chance

```python
from threshold_detector import run_eca

result = run_eca(binary, binary, delT=4, tau=1)
print(result.summary_table())
```

### Visualise ensemble spread

```python
from threshold_detector import plot_ensemble_hovmoller

# events_df has columns: ensemble, year, n_extreme_days
fig_events, fig_days = plot_ensemble_hovmoller(events_df,extreme_days_col='n_extreme_days',cmap='Blues')
fig_events.savefig('hovmoller_events.png', dpi=300)
fig_days.savefig('hovmoller_days.png', dpi=300)
```

## Parameters reference

Parameter | Used in | Meaning 
`threshold` | `flag_extreme_events` | Value above/below which a day is extreme |
`direction` | `flag_extreme_events` | `'above'` or `'below'` |
`N` | `flag_extreme_events` | Consecutive days required to flag an event |
`percentile` | `flag_extreme_events_percentile` | Percentile of reference distribution |
`reference` | `flag_extreme_events_percentile` | Data to compute percentile from (e.g. baseline period) |
`delT` | `detect_compound_events`, `run_eca` | Max gap in days within a compound event / ECA window |
`min_duration` | `detect_compound_events` | Minimum extreme days to qualify as compound |
`tau` | `run_eca` | Minimum lag in ECA (set to 1 to avoid same-day self-coincidence) |
`meteo_window` | `process_binary_series` | Window for removing overcounting |

## Input data format

All detection functions accept:
- A **numpy array** or any array-like (list, pandas Series) of numeric values
- One value per timestep (typically one per day)
- No time coordinate required — pass the data array directly

The `detect_compound_events` function returns a DataFrame with integer indices
(`start_idx`, `end_idx`). If you have a date array, align with:

```python
import numpy as np
dates = np.array(my_date_array)
events_df['start_date'] = dates[events_df['start_idx']]
events_df['end_date']   = dates[events_df['end_idx']]
```