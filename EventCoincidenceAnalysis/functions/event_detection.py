import numpy as np


def flag_wet_events(timeseries, N, wet_threshold):
    '''
    flags wet events in a timeseries.

    input:
    timeseries (array-like): The input timeseries of rainfall data.
    N (int): The window size to check for wet events.
    wet_threshold (float): The threshold above which a day is considered wet.

    returns:
    numpy.ndarray: An array of the same length as timeseries, with 1 indicating a wet event and 0 otherwise.
    '''
    wet_events = np.zeros(len(timeseries), dtype=int)
    
    i = 0
    while i <= len(timeseries) - N:
        window = timeseries[i:i+N]
        
        #ceck all days in the window are below the dry threshold
        if np.all(window > wet_threshold):
            #mark last day of this window with a 1
            wet_events[i+N-1] = 1
            i += N
        else:
            i += 1
    
    return wet_events

#---------------------------------------------------------------
def process_timeseries(series, meteo_window, len_wet):
    '''
    Processes a timeseries to remove excess consecutive wet events.

    input:
    series (array-like): The input timeseries of wet events (1s and 0s).
    meteo_window (int): The window size to check for consecutive wet events.
    len_wet (int): The maximum allowed number of consecutive wet events.

    returns:
    list: The processed timeseries with excess consecutive wet events removed.
    '''
    series = np.array(series) 
    idx = np.where(series == 1)[0]  # find indices of all 1s
    
    for i in idx:
        if series[i] == 1: 
            window_end = min(i + meteo_window + 1, len(series))
            ones_positions = np.where(series[i:window_end] == 1)[0] + i 
            
            if len(ones_positions) > (len_wet):
                excess_start = ones_positions[len_wet]  # the index of the (len_wet + 1)th 1
                series[excess_start:window_end] = 0  # Set all excess 1s to 0
                
    return series.tolist()


#----------------------------------------------------------------------
def find_consecutive_ones_indices(data, N):
    '''
    finds the starting indices of consecutive ones in a timeseries (compounds).

    input:
    data (array-like): The input timeseries of 1s and 0s.
    N (int): the min number of consecutive ones to look for.

    returns:
    list of starting indices of consecutive ones of length N or more.
    '''
    indices = []
    count = 0
    for i in range(len(data)):
        if data[i] == 1:
            count += 1
            if count >= N:
                indices.append(i - N + 1)
        else:
            count = 0
    return indices