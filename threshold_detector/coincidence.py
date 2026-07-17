"""
this is a convenience wrappers around EventCoincidenceAnalysis for users who want
to go from binary series straight to ECA results without importing the ECA module directly.
"""
import sys
import os

# Add the repo root to path so EventCoincidenceAnalysis is findable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from EventCoincidenceAnalysis.functions.eca_core import (EventCoincidence,calculate_ECA_30y,calculate_ECA_whole_ts,)

def run_eca(seriesA, seriesB, delT=4, tau=1):
    '''
    Run Event Coincidence Analysis on two binary timeseries.

    Parameters
    seriesA : array-like
        Binary timeseries for event type A (precursor).
    seriesB : array-like
        Binary timeseries for event type B (trigger).
        Pass the same series as seriesA to test self-clustering.
    delT : int, optional
        Coincidence window in days. Default 4.
    tau : int, optional
        Minimum lag in days. Default 1 (avoids same-day self-coincidence).

    Returns
    EventCoincidence
        ECA result object. Call .summary_table() to see all statistics, or .get_coincidences to get (Kp, Kt) directly.
    '''
    return EventCoincidence(seriesA, seriesB, delT=delT, tau=tau)


def run_eca_rolling(series_list, delT=4, tau=1):
    '''
    Run ECA across a list of periods (e.g. rolling 30-year windows).

    Parameters
    series_list : list of array-like
        One binary timeseries per period. For self-clustering, pass the
        same list for both arguments.
    delT : int, optional (Default 4).
    tau : int, optional (Default 1).

    Returns
    list of EventCoincidence
        One result object per period.
        '''
    return calculate_ECA_30y(series_list, series_list, delT=delT, tau=tau)