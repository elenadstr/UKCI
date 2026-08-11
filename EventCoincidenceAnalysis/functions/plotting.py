"""Plotting for the regional ECA analysis.

The per-window band/heatmap figures now come from the vendored engine
(:func:`eca_analysis.run_window_analysis`, ``plot_region_panels``,
``plot_region_4panel``), which uses the corrected null band (Eq. 1 success
probability, K indexed from 0).

This module provides the paper's Figure-5-style regional agreement map and adds
:func:`ensemble_agreement_summary` to build its input from the engine's tidy
window DataFrame.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

PERIOD_TITLE = ('1980-2010', '1990-2020', '2000-2030', '2010-2040',
    '2020-2050', '2030-2060', '2040-2070', '2050-2080')


def ensemble_agreement_summary(windows_df, region_names=None,
                               hist_period='1980-2010',
                               fut_period='2050-2080'):
    """Build the 3-column summary for :func:`plot_regional_ensemble_agreement`
    from the tidy DataFrame returned by
    :func:`eca_analysis.run_window_analysis` (columns: ensemble, region,
    period, regime, significant, ...).

    Counts, per region, the number of ensemble members whose observed
    coincidences exceed the null band (``significant == 1``).
    """
    def _count(period, regime):
        sub = windows_df[(windows_df.period == period)
                         & (windows_df.regime == regime)]
        return sub.groupby('region')['significant'].sum()

    hist_fixed = _count(hist_period, 'fixed')
    fut_fixed = _count(fut_period, 'fixed')
    fut_thermo = _count(fut_period, 'thermodynamic')
    regions = sorted(set(windows_df.region))
    names = region_names or {}
    return pd.DataFrame({
        'Region': [names.get(r, r) for r in regions],
        f'{hist_period} Fixed Threshold':
            [int(hist_fixed.get(r, 0)) for r in regions],
        f'{fut_period} Fixed Threshold':
            [int(fut_fixed.get(r, 0)) for r in regions],
        f'{fut_period} Thermo Threshold':
            [int(fut_thermo.get(r, 0)) for r in regions],
    })


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