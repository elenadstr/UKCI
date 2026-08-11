
# UKCI — UK Climate Information API

A Python API for detecting and analysing compound and single extreme events from the UKCP18 
CPM runs.

## Overview
The repository contains the modules:

- **`eca_analysis`** — the vendored counting/ECA engine (single source of
  truth for the maths; copied self-contained from the paper repository, see
  `eca_analysis/VENDORED.md` — UKCI has no dependency on that repo).

- **`threshold_detector`** — the user-facing layer: detect extreme events in
  any climate timeseries (fixed or percentile thresholds, either direction),
  identify compound episodes, run ECA, and visualise ensemble spread.

- **`EventCoincidenceAnalysis`** — the regional sliding-window analysis
  notebook (paper Figs. 4-5) and the regional agreement map plotting.

## Method (paper Sections 2.1-2.3)

Two extreme days are linked iff their gap `g` satisfies
`tau <= g <= tau + delT` **and** both days lie in the same (year, season)
block — the maximum linkage gap is `tau + delT` days (**5** with the paper
defaults `tau=1, delT=4`; the "four-day window" of the paper refers to
`delT`, and Eq. 1's coincidence window has length `delT + 1`). Compound
events are the connected components of that linkage with at least
`min_duration` (default 2) extreme days; no window ever spans a season
boundary. Significance uses Eq. (1)'s binomial null
(`p = 1 - (1 - (delT+1)/(T - tau))**N`), band = 2.5-97.5% with K indexed
from 0; `null_model="blocked"` gives the exact per-season Poisson-binomial
as an advanced alternative.

Because `detect_compound_events` is built on the engine's
`compound_episodes` (= the self-ECA coincidence indices), the event counts
and the significance test can never disagree.

## Detection modes

Three modes, demonstrated end-to-end in
`notebooks/demo_threshold_detector.ipynb` (edit one parameter cell, run all):

1. **single-variable** — sequential extremes of one variable
   (`detect_compound_events`); this is the paper's method.
2. **sequential** — extremes from two variables within the linkage window of
   each other, e.g. a dry spell followed by extreme rain
   (`detect_compound_events_bivariate`). UKCI extension.
3. **co-occurring** — both variables extreme on the same day(s), e.g. heat +
   drought (`detect_compound_events_coincident`). UKCI extension.

Event criteria are set **per variable** at the flagging stage: any variable,
either direction (`above`/`below`), fixed or percentile thresholds, `N`-day
windows with an optional `min_days` tolerance (e.g. a 20-day drought with
>= 19 dry days). The compound stage then applies per-variable minima within
each event (`min_duration_1`/`min_duration_2` in sequential mode). Seasons
are modular — `(season_start, season_length)`, wrapping the calendar year if
needed, with wrapped seasons labelled by their start year — and all modes are
season-blocked (`years=`/`blocks=` required).

UKCI is pre-release; `CHANGELOG.md` records the method corrections made
during development so earlier numbers can be reconciled.


## Setup

### 1. Clone the repository

```bash
git clone <i-need-to-add-this>
cd UKCI
```

### 2. Create your config file

```bash
cp config/config.template.yaml config/config.yaml
```

Edit `config/config.yaml` with your local data paths.
This file is gitignored and will never be committed.

### 3. Install dependencies

```bash
conda env create -f environment.yml
conda activate ukci
```

## Repository structure

UKCI/
├── threshold_detector/          # event detection and visualisation
│   ├── detector.py              # flag_extreme_events, detect_compound_events
│   ├── coincidence.py           # ECA wrappers
│   ├── plotting.py              # timeseries, Hovmöller, duration plots
│   └── README.md
├── EventCoincidenceAnalysis/    # core ECA implementation
│   ├── ECA_analysis.ipynb       # end-to-end analysis notebook
│   ├── functions/
│   │   ├── eca_core.py          # eventCoincidence class
│   │   ├── event_detection.py
│   │   ├── statistics.py
│   │   ├── thresholds.py
│   │   ├── iris_utils.py
│   │   └── plotting.py
│   └── README.md
├── config/
│   ├── config.template.yaml     # committed — fill in and copy to config.yaml
│   └── config.yaml              # gitignored — your local paths
└── outputs/                     # gitignored — generated figures and CSVs


## Data format

Input data is expected as NetCDF files with a daily time dimension, one file
per ensemble member per region. Filename convention: p110{ensemble}_{region_code}.nc   #FOR NOW, TO MODIFY: BUT NEED TO WAIT FOR DAVID TO CONFIRM
ex: `p1100000_Wales.nc`

## References

Donges, J.F. et al. (2016). Event coincidence analysis for quantifying
statistical interrelationships between event time series.
*European Physical Journal Special Topics*, 225, 471–487.

Our paper! TBD
