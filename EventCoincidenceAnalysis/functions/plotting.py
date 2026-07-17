import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import iris
from iris.time import PartialDateTime

from .event_detection import flag_wet_events
from .eca_core import EventCoincidence, calculate_ECA_30y
from .statistics import binomial_Kt, decompose_years_30y
from .thresholds import get_percentile, get_value_from_percentile


PERIOD_TITLE = ('1980-2010', '1990-2020', '2000-2030', '2010-2040',
    '2020-2050', '2030-2060', '2040-2070', '2050-2080')


def _compute_binomial_bounds(wet_timeseries, delT, tau):
    '''
    for each period in wet_timeseries, compute the 2.5th and 97.5th percentile of the binomial null distribution. 
    returns lists of lower/upper bounds.
    '''
    num_periods = len(wet_timeseries)
    lower_bounds, upper_bounds = [], []

    for period_index in range(num_periods):
        seriesA = wet_timeseries[period_index]
        binom_probs = []
        cumsum = 0
        KT = 0
        while cumsum < 1 - 1e-6:
            p = binomial_Kt(seriesA, seriesA, KT, delT=delT, tau=tau)
            binom_probs.append(p)
            cumsum += p
            KT += 1
        y_values = np.arange(1, len(binom_probs) + 1)
        cumulative_probs = np.cumsum(binom_probs)
        upper_bounds.append(np.interp(0.975, cumulative_probs, y_values))
        lower_bounds.append(np.interp(0.025, cumulative_probs, y_values))

    return lower_bounds, upper_bounds


def _plot_single_eca_panel(ax, lower_bounds, upper_bounds, observed_coincidences,
                            title, num_periods):
    '''draws one ECA panel (scatter + confidence band) onto ax.'''
    ax.plot(range(num_periods), lower_bounds, color='royalblue', lw=0.7)
    ax.plot(range(num_periods), upper_bounds, color='royalblue', lw=0.7)
    ax.fill_between(range(num_periods), lower_bounds, upper_bounds,color='cornflowerblue', alpha=0.5, label='2.5-97.5 Percentile')
    ax.scatter(range(num_periods), observed_coincidences,label='Observed coincidences', color='#E63946', edgecolor='black', s=80)
    ax.set_xticks(range(num_periods))
    ax.set_xticklabels(PERIOD_TITLE, rotation=45)
    ax.set_ylabel("Number of coincidences", fontsize=16)
    ax.set_title(title, fontsize=16)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.legend(loc='upper left', fontsize=9)


def _compute_exceedance_heatmap(base_dir, region_code, ensembles_dict, lower_month, higher_month, wet_threshold, len_wet, delT, tau, use_thermo=False):
    '''
    Loops over all ensemble members for one region. Returns a 2D numpy array
    (n_ensembles x n_periods) of 0/1 exceedance flags.
    '''
    num_periods = len(PERIOD_TITLE)
    results = {ens: [] for ens in ensembles_dict}

    for ensemble in ensembles_dict:
        try:
            cube = iris.load(f'{base_dir}/{region_code}/p110{ensemble}_{region_code}.nc')[0]
            month_range = iris.Constraint( time=lambda cell: PartialDateTime(month=lower_month) <= cell.point <= PartialDateTime(month=higher_month))
            month_extraction_cube = cube.extract(month_range)
            climatology = { f'cube_{yr}': decompose_years_30y(month_extraction_cube)[j].data for j, yr in enumerate(PERIOD_TITLE)}

            if use_thermo:
                percentile_ref = get_percentile(wet_threshold, climatology['cube_1980-2010'])
                wet_thresholds = [get_value_from_percentile(percentile_ref, climatology[f'cube_{yr}']) for yr in PERIOD_TITLE]
                wet_ts = [flag_wet_events(climatology[key], len_wet, wt) for key, wt in zip(climatology.keys(), wet_thresholds)]
            else:
                wet_ts = [flag_wet_events(climatology[key], len_wet, wet_threshold) for key in climatology]

            period_ECA = calculate_ECA_30y(wet_ts, wet_ts, delT=delT, tau=tau)
            N_trig = [eca.get_coincidences[1] for eca in period_ECA]
            _, upper_bounds = _compute_binomial_bounds(wet_ts, delT, tau)

            for period_index in range(num_periods):
                results[ensemble].append(1 if N_trig[period_index] > upper_bounds[period_index] else 0)

        except Exception as e:
            print(f"  Error processing ensemble {ensemble}: {e}")
            results[ensemble] = [0] * num_periods

    heatmap = np.zeros((len(ensembles_dict), num_periods))
    for ens_idx, ensemble in enumerate(ensembles_dict):
        for period_idx in range(num_periods):
            if len(results[ensemble]) > period_idx:
                heatmap[ens_idx, period_idx] = results[ensemble][period_idx]
    return heatmap


def plot_wales_case_study(base_dir, region_code, ensemble, ensembles_dict, wet_threshold, lower_month, higher_month, delT, tau, len_wet, meteo_window):
    '''
    produces the 4-panel Wales case study figure:
      a) Fixed threshold, single member
      b) Thermodynamic threshold, single member
      c) Fixed threshold, all ensemble members (heatmap)
      d) Thermodynamic threshold, all ensemble members (heatmap)

    '''

    num_periods = len(PERIOD_TITLE)

    # load data once, reuse for both panels
    cube = iris.load(f'{base_dir}/{region_code}/p110{ensemble}_{region_code}.nc')[0]
    month_range = iris.Constraint(time=lambda cell: PartialDateTime(month=lower_month) <= cell.point <= PartialDateTime(month=higher_month))
    month_extraction_cube = cube.extract(month_range)
    climatology = {f'cube_{yr}': decompose_years_30y(month_extraction_cube)[j].data for j, yr in enumerate(PERIOD_TITLE)}

    # --- fixed threshold timeseries and ECA ---
    wet_ts_fixed = [flag_wet_events(climatology[key], len_wet, wet_threshold) for key in climatology]
    period_ECA_fixed = calculate_ECA_30y(wet_ts_fixed, wet_ts_fixed, delT=delT, tau=tau)
    N_trig_fixed = [eca.get_coincidences[1] for eca in period_ECA_fixed]
    lower_fixed, upper_fixed = _compute_binomial_bounds(wet_ts_fixed, delT, tau)

    # --- thermo threshold timeseries and ECA ---
    percentile_ref = get_percentile(wet_threshold, climatology['cube_1980-2010'])
    wet_thresholds = [get_value_from_percentile(percentile_ref, climatology[f'cube_{yr}'])for yr in PERIOD_TITLE]

    wet_ts_thermo = [flag_wet_events(climatology[key], len_wet, wt) for key, wt in zip(climatology.keys(), wet_thresholds)]
    period_ECA_thermo = calculate_ECA_30y(wet_ts_thermo, wet_ts_thermo, delT=delT, tau=tau)
    N_trig_thermo = [eca.get_coincidences[1] for eca in period_ECA_thermo]
    lower_thermo, upper_thermo = _compute_binomial_bounds(wet_ts_thermo, delT, tau)

    # --- Heatmaps ---
    heatmap_fixed = _compute_exceedance_heatmap(base_dir, region_code, ensembles_dict,lower_month, higher_month, wet_threshold, len_wet, delT, tau,use_thermo=False)
    heatmap_thermo = _compute_exceedance_heatmap(base_dir, region_code, ensembles_dict,lower_month, higher_month, wet_threshold, len_wet, delT, tau,use_thermo=True)

    # --- Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    _plot_single_eca_panel(axes[0, 0], lower_fixed, upper_fixed, N_trig_fixed,f'a) Fixed threshold (20 mm/hr)\n{region_code}, p110{ensemble}',num_periods )
    _plot_single_eca_panel(axes[0, 1], lower_thermo, upper_thermo, N_trig_thermo,f'b) Thermodynamic threshold\n{region_code}, p110{ensemble}',num_periods )

    for ax, heatmap, label in [(axes[1, 0], heatmap_fixed,  'c) Fixed threshold - Ensemble exceedances'),(axes[1, 1], heatmap_thermo, 'd) Thermodynamic threshold - Ensemble exceedances'),]:
        ax.imshow(heatmap, cmap='Blues', aspect='auto', vmin=0, vmax=1,interpolation='nearest')
        ax.set_xticks(range(num_periods))
        ax.set_xticklabels(PERIOD_TITLE, rotation=45, ha='right')
        ax.set_yticks(range(len(ensembles_dict)))
        ax.set_yticklabels(list(ensembles_dict.keys()))
        ax.set_ylabel('Ensemble member', fontsize=16)
        ax.set_title(f'{label}\n{region_code}', fontsize=16)
        ax.set_xticks(np.arange(num_periods) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(ensembles_dict)) - 0.5, minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.3)

    plt.tight_layout()
    return fig


def plot_regional_ensemble_agreement(summary_df, regions_gdf,
                                      geo_region_col='geo_region'):
    """
    Produces the 3-panel regional UK map (Figure 5 in the paper) showing
    ensemble agreement for historical fixed, future fixed, and future
    thermodynamic thresholds.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of the regional loop — columns:
        'Region', '1980-2010 Fixed Threshold',
        '2050-2080 Fixed Threshold', '2050-2080 Thermo Threshold'
    regions_gdf : GeoDataFrame
        Regional shapefile/GeoJSON with a column matching geo_region_col
    geo_region_col : str
        Column in regions_gdf to join on (matched to 'Region' in summary_df)

    Returns
    -------
    fig : matplotlib Figure
    """
    merged = regions_gdf.merge(summary_df, left_on=geo_region_col, right_on='Region')
    plot_cols = [
        '1980-2010 Fixed Threshold',
        '2050-2080 Fixed Threshold',
        '2050-2080 Thermo Threshold']
    merged_clean = merged.dropna(subset=plot_cols)

    boundaries = list(range(0, 14))
    norm = mpl.colors.BoundaryNorm(boundaries, ncolors=256)

    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.01, hspace=0)
    axes = [fig.add_subplot(gs[0, j]) for j in range(3)]
    panel_labels = ['a.', 'b.', 'c.']

    for ax, col, label in zip(axes, plot_cols, panel_labels):
        merged_clean.plot(column=col, ax=ax, cmap='managua',
                          edgecolor='black', aspect=1, norm=norm, legend=False)
        ax.axis('off')
        ax.set_aspect('equal')
        ax.text(-0.08, 0.95, label, transform=ax.transAxes,
                fontsize=16, va='center', ha='center')

    sm = mpl.cm.ScalarMappable(cmap='managua', norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical',
                        fraction=0.03, pad=0.04, spacing='proportional')
    cbar.set_label('Number of ensemble members')

    return fig