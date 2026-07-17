'''
Event Coincidence Analysis (ECA) - Pure Python Implementation

This module provides a pure Python implementation of Event Coincidence Analysis
following the method of Donges et al. (2016), python version: https://pypi.org/project/event-analysis/

Uses inclusive boundary condition: tau <= |t_i - t_j| <= tau + delT, which is in the R implemetnation but niot the python one
The inclusive boundary means: with tau=1, delT=4, it checks gaps of {1, 2, 3, 4, 5} days, not {2, 3, 4, 5}.

References:
Donges, Jonathan F., et al. "Event coincidence analysis for quantifying statistical interrelationships between event time 
series: on the role of flood events as triggers of epidemic outbreaks."
 The European Physical Journal Special Topics 225.3 (2016): 471-487.
'''

import numpy as np
import pandas as pd
from scipy.stats import binom

def eca_coincidence_count(seriesA, seriesB, delT, tau):
    """
    calculate event coincidence counts between two binary time series.
    
    Uses inclustve boundary condition: tau <= |t_i - t_j| <= tau + delT
    
    Parametres
    seriesA : array-like, binary time series for event A (precursor)
    seriesB : array-like, binary time series for event B (trigger)
    delT : int, max time difference for coincidence (days)
    tau : int, min time difference for coincidence (days)
        
    Returns:
    N_A : int: Number of events in series A
    N_B : int: number of events in series B
    Kt : int: number of trigger coincidences (A followed by B)
    Kp : int: number of precursor coincidences (B preceded by A)
    """
    seriesA = np.array(seriesA, dtype=int)
    seriesB = np.array(seriesB, dtype=int)
    
    # Find event indices
    events_A = np.where(seriesA == 1)[0]
    events_B = np.where(seriesB == 1)[0]
    
    N_A = len(events_A)
    N_B = len(events_B)
    
    # Count trigger coincidences (A -> B)
    Kt = 0
    for t_a in events_A:
        # Check if any B event occurs within [tau, tau+delT] after this A event
        # CRITICAL: Use >= for inclusive boundary (not > for strict)
        diffs = events_B - t_a
        valid = (diffs >= tau) & (diffs <= tau + delT)
        if np.any(valid):
            Kt += 1
    
    # Count precursor coincidences (A <- B)  
    Kp = 0
    for t_b in events_B:
        # Check if any A event occurs within [tau, tau+delT] before this B event
        diffs = t_b - events_A
        valid = (diffs >= tau) & (diffs <= tau + delT)
        if np.any(valid):
            Kp += 1
    
    return N_A, N_B, Kt, Kp


def binomial_Kt(Kt, N_A, N_B, T, delT, tau):
    '''
    calculate binomial null hypothesis probability for trigger coincidences.
    
    Parametres
    Kt : int, Observed number of trigger coincidences
    N_A : int, Number of precursor events
    N_B : int, Number of trigger events
    T : int, Total length of time series
    delT : int, Time window for coincidence
    tau : int, Time lag for coincidence
        
    Returns
    p_value : float
        probs of observing Kt or more coincidences under null hypothesis
    '''
    # prob that a B event falls in coincidence window after an A event
    p = (delT + 1) * N_B / T  # +1 because range is inclusive [tau, tau+delTres  P(X >= Kt) where X ~ Binomial(N_A,
    p_value = 1 - binom.cdf(Kt - 1, N_A, p)
    
    return p_value


# ---------------------------------- ECA class ----------------------------------
class EventCoincidence():
    '''
    puter python ECA class
    '''

    def __init__(self, seriesA, seriesB, delT, tau, sym=False, dates=None, 
                 seriesAname='Event Series A', seriesBname='Event Series B'):
        self._seriesA = np.array(seriesA, dtype=int)
        self._seriesB = np.array(seriesB, dtype=int)
        self._delT = delT
        self._tau = tau
        self._sym = sym
        self._dates = dates
        self._seriesAname = seriesAname
        self._seriesBname = seriesBname
        
        # Calculate coincidence counts using pure Python
        self._N_A, self._N_B, self._Kt, self._Kp = eca_coincidence_count(
            self._seriesA, self._seriesB, delT, tau
        )
        
        # Calculate binomial probabilities
        T = len(seriesA)
        self._p_value_trig = binomial_Kt(self._Kt, self._N_A, self._N_B, T, delT, tau)
        self._p_value_prec = binomial_Kt(self._Kp, self._N_B, self._N_A, T, delT, tau)
        
        # Calculate coincidence rates
        self._trig_coin_rate = self._Kt / self._N_A if self._N_A > 0 else 0
        self._prec_coin_rate = self._Kp / self._N_B if self._N_B > 0 else 0

    @property
    def get_events(self):
        """Returns (N_A, N_B) - number of events in each series"""
        return self._N_A, self._N_B
    
    @property
    def get_coincidences(self):
        """Returns (Kp, Kt) - precursor and trigger coincidence counts"""
        return self._Kp, self._Kt
    
    @property
    def get_coincidence_indices(self):
        """Returns indices of coincident events (not yet implemented)"""
        # TODO: implement if needed
        return None, None

    @property
    def get_poisson_values(self):
        """
        Returns binomial test statistics.
        
        Returns (NH_prec, NH_trig, p_value_prec, p_value_trig, 
                 prec_coin_rate, trig_coin_rate)
        """
        # NH values (expected coincidences under null hypothesis)
        T = len(self._seriesA)
        p_trig = (self._delT + 1) * self._N_B / T
        p_prec = (self._delT + 1) * self._N_A / T
        NH_trig = self._N_A * p_trig
        NH_prec = self._N_B * p_prec
        
        return (NH_prec, NH_trig, 
                self._p_value_prec, self._p_value_trig,
                self._prec_coin_rate, self._trig_coin_rate)
        
    def calc_probs(self):
        """Calculate probability of events in each series"""
        p_A = np.sum(self._seriesA) / len(self._seriesA)
        p_B = np.sum(self._seriesB) / len(self._seriesB)
        return p_A, p_B

    def summary_table(self):
        """Returns summary DataFrame of ECA results"""
        NH_prec, NH_trig, p_val_prec, p_val_trig, prec_rate, trig_rate = self.get_poisson_values
        p_A, p_B = self.calc_probs()
        
        df = pd.DataFrame({
            'Value': [ NH_prec, NH_trig,  p_val_prec, p_val_trig, prec_rate, trig_rate, p_A, p_B]}, 
            index=[ 'NH precursor', 'NH trigger', 'p-value precursor', 'p-value trigger', 'precursor coincidence rate', 'trigger coincidence rate', 'p(A)', 'p(B)'])
        
        return df



#------------------------------------------------
def calculate_ECA_30y(wet_list, dry_list, delT, tau):
    """
    Calculate ECA for 30-year periods.
    
    Pure Python implementation - no R dependencies.

    Parameters
    wet_list : list of np arrays
        List of binary time series for the "wet" events, one for each period.
    dry_list : list of np arrays
        List of binary time series for the "dry" events, one for each period.
    delT : int
        Time window for coincidence (in days).
    tau : int
        Time lag for coincidence (in days).

    Returns
    list of EventCoincidence objects
        List of ECA results for each period.
    """
    eca_objects = []
    for wet, dry in zip(wet_list, dry_list):
        eca = EventCoincidence(wet, dry, delT, tau, sym=False)
        eca_objects.append(eca)
    return eca_objects


#---------------------------------------------------------------
def calculate_ECA_whole_ts(wet_list, dry_list, delT, tau):
    """
    calc ECA for the whole time series.
    
    """
    eca_results = EventCoincidence(wet_list, dry_list, delT, tau)
    return eca_results