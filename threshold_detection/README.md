# threshold_detection
this modeule implements compound event detection and event coincidence analysis (ECA), developed by Donges et al., 2016
following methodology in Dauster et al.(XXXX).

## what this module does
Given a regional variable for extreme at a regular time step (here we use daily maximum rainfall data from the CPM), it

1. Detects extreme precipitation events using either a fixed impact threshold 
   (20 mm/hr) or a thermodynamically-adjusted percentile-based threshold
2. Applies ECA to test whether temporal clustering of extreme events exceeds random chance
3. Produces the two main diagnostic figures from the paper:
   - A 4-panel Wales case study (single member, both thresholds, fixed and ensemble)
   - A 3-panel UK regional map showing ensemble agreement across historical and 
     future periods

## setup
Before running the notebook, copy the config template and fill in your paths:

    cp ../config/config.template.yaml ../config/config.yaml

Then edit `config.yaml` with:
- paths to your NetCDF rainfall data (one file per ensemble member per region)
- path to the regional GeoJSON mask
- output directory for figures and CSVs
- path to the R script `CoinCalc.R`, copied from: https://github.com/JonatanSiegmund/CoinCalc

Expected NetCDF filename convention: `p110{ensemble}_{region_code}.nc`  
Example: `p1100000_Wales.nc`

## runnig the notebook

    jupyter notebook ECA_analysis.ipynb

Run cells top to bottom. Cell 1 loads and validates your config. If `config.yaml` is missing, it will let you 
know it's not happy.

## structure

threshold_detection/
├── ECA_analysis.ipynb      # Main notebook
├── functions/
│   ├── event_detection.py  # flag_wet_events, process_timeseries
│   ├── eca_core.py         # ECA class, calculate_ECA_30y
│   ├── statistics.py       # binomial_Kt, binomial_Kp, decompose_years_30y
│   ├── thresholds.py       # get_percentile, get_value_from_percentile
│   ├── iris_utils.py       # add_yyyymmdd
│   └── plotting.py         # plot_wales_case_study, plot_regional_ensemble_agreement
└── R/
└── CoinCalc.R


## dependencies

- Python: iris, numpy, scipy, pandas, geopandas, matplotlib, rpy2
- R: the CoinCalc package used by CoinCalc.R : https://github.com/JonatanSiegmund/CoinCalc

See `../environment.yml` for the full pinned environment.

## params

All analysis parameters are set in `config.yaml` under the `analysis` key:

| Parametre      | default | maning                                       |
|----------------|---------|----------------------------------------------|
| wet_threshold  | 20      | mm/hr fixed threshold for extreme events     |
| lower_month    | 6       | Start of summer season (June)                |
| higher_month   | 9       | End of summer season (September)             |
| delT           | 4       | Coincidence window size (days)               |
| tau            | 1       | Lag to avoid same-day self-coincidence       |
| len_wet        | 1       | Minimum consecutive wet days                 |
| meteo_window   | 4       | Window for overflow processing               |