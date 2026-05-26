''' percentuile and corresponding value code '''
def get_percentile(value, precipitation_cube):
    precip_data = precipitation_cube.data.flatten()
    percentile = stats.percentileofscore(precip_data, value, kind="rank")
    
    return percentile

def get_value_from_percentile(percentile, precipitation_cube):
    '''
    Returns the precipitation value that corresponds to a given percentile.
    ''' 
    
    precip_data = precipitation_cube.data.flatten()
    
    value = np.percentile(precip_data, percentile)
    
    return value