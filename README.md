
# UKCI — UK Climate Information API

A Python API for detecting and analysing compound and single extreme events from the UKCP18 
CPM runs.

## Overview
The repository contains the  modules:

- **`threshold_detector`** — detect extreme events in any climate timeseries,
  identify compound episodes, and visualise ensemble spread. Designed to be
  useful standalone or as input to EventCoincidenceAnalysis.

- **`EventCoincidenceAnalysis`** — test whether the temporal clustering of
  extreme events exceeds random chance, using Event Coincidence Analysis (ECA) following Donges et al. (2016).


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
