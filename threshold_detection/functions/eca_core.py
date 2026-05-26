# ---------------------------------- ECA class ----------------------------------

class EventCoincidence():

    def __init__(self, seriesA, seriesB, delT, tau, sym=False, dates=None, seriesAname='Event Series A', seriesBname='Event Series B'):
        self._seriesA = seriesA
        self._seriesB = seriesB
        #if not len(seriesA)==len(seriesB):
        #    raise ValueError('Time series A and B are not the same length')
        self._delT = delT
        self._tau = tau
        self._sym = sym
        self._dates = dates
        self._seriesAname = seriesAname
        self._seriesBname = seriesBname
        self._poisson = self.perform_ECA('poisson') 
        self._shuffle = None
        self._n_wet_coincidences = self._poisson.rx2('N wet coincidences')[0]
        self._n_dry_coincidences = self._poisson.rx2('N dry coincidences')[0]
        self._n_prec_coincidences = self._poisson.rx2('N precursor')[0]
        self._n_trig_coincidences = self._poisson.rx2('N trigger')[0]
        self._trigg_indices = self._poisson.rx2('trigger indices')
        self._prec_indices = self._poisson.rx2('precursor indices')

                                                   
    @property
    def get_events(self):
        N_A = self._n_wet_coincidences
        N_B = self._n_dry_coincidences

        return N_A, N_B
    
    @property
    def get_coincidences(self):
        N_prec = self._n_prec_coincidences
        N_trig = self._n_trig_coincidences

        return N_prec, N_trig
    
    @property
    def get_coincidence_indices(self):
        prec_ind = self._trigg_indices
        trigg_ind = self._prec_indices

        return prec_ind, trigg_ind


    @property
    def get_poisson_values(self):
        p_nh_prec = self._poisson.rx2('NH precursor')[0]
        p_nh_trig = self._poisson.rx2('NH trigger')[0] 
        p_pvalue_prec = self._poisson.rx2('p-value precursor')[0]
        p_pvalue_trig = self._poisson.rx2('p-value trigger')[0]
        p_prec_coin_rate = self._poisson.rx2('precursor coincidence rate')[0]
        p_trig_coin_rate = self._poisson.rx2('trigger coincidence rate')[0]

        return p_nh_prec, p_nh_trig, p_pvalue_prec, p_pvalue_trig, p_prec_coin_rate, p_trig_coin_rate

    #----------------------------calculate event coincidence-------------------
    def perform_ECA(self, sigtest):
        ra = rpy2.robjects.vectors.IntVector(self._seriesA)         # convert np array to vetcor for R
        rb = rpy2.robjects.vectors.IntVector(self._seriesB)

        ev_def = 'threshold'
        thres = 1

        abin = r['CC.binarize'](ra, ev_def, thres)                  #binarise time series for ECA function to work
        bbin = r['CC.binarize'](rb, ev_def, thres)

        ECA_output = r['CC.eca.ts'](abin, bbin, delT=self._delT, tau=self._tau, sym=self._sym, sigtest=sigtest)
        return ECA_output
        
    def calc_probs(self):
        p_wet = np.sum(self._seriesA) / len(self._seriesA)
        p_dry = np.sum(self._seriesB) / len(self._seriesB)

        return p_wet, p_dry
        

    def do_shuffle_test(self):
        if self._shuffle is None: 
            self._shuffle = self.perform_ECA('shuffle.surrogate')

    def summary_table(self):
        if self._shuffle is None:
            p_nh_prec = self._poisson.rx2('NH precursor')[0]
            p_nh_trig = self._poisson.rx2('NH trigger')[0] 
            p_pvalue_prec = self._poisson.rx2('p-value precursor')[0]
            p_pvalue_trig = self._poisson.rx2('p-value trigger')[0]
            p_prec_coin_rate = self._poisson.rx2('precursor coincidence rate')[0]
            p_trig_coin_rate = self._poisson.rx2('trigger coincidence rate')[0]

            df = pd.DataFrame([[p_nh_prec],
                           [p_nh_trig],
                           [p_pvalue_prec],
                           [p_pvalue_trig],
                           [p_prec_coin_rate],
                           [p_trig_coin_rate],
                           [p_dry],
                           [p_wet]],
                          index=['NH precursor', 
                                 'NH trigger',
                                 'p-value precursor', 
                                 'p-value trigger',
                                 'precursor coincidence rate',
                                 'trigger coincidence rate',
                                 'p(dry)',
                                 'p(wet)'],  
                          columns=['Poisson', 'Shuffle'])


        if self._shuffle is not None:
            p_nh_prec = self._poisson.rx2('NH precursor')[0] 
            s_nh_prec = self._shuffle.rx2('NH precursor')[0] 
            p_nh_trig = self._poisson.rx2('NH trigger')[0] 
            s_nh_trig = self._shuffle.rx2('NH trigger')[0]
            p_pvalue_prec = self._poisson.rx2('p-value precursor')[0]
            s_pvalue_prec = self._shuffle.rx2('p-value precursor')[0]
            p_pvalue_trig = self._poisson.rx2('p-value trigger')[0]
            s_pvalue_trig = self._shuffle.rx2('p-value trigger')[0]
            p_prec_coin_rate = self._poisson.rx2('precursor coincidence rate')[0]
            s_prec_coin_rate = self._shuffle.rx2('precursor coincidence rate')[0]
            p_trig_coin_rate = self._poisson.rx2('trigger coincidence rate')[0]
            s_trig_coin_rate = self._shuffle.rx2('trigger coincidence rate')[0]

            df = pd.DataFrame([[p_nh_prec, s_nh_prec],
                               [p_nh_trig, s_nh_trig],
                               [p_pvalue_prec, s_pvalue_prec],
                               [p_pvalue_trig, s_pvalue_trig],
                               [p_prec_coin_rate, s_prec_coin_rate],
                               [p_trig_coin_rate, s_trig_coin_rate],
                               [p_dry, None],
                               [p_wet, None]],
                              index=['NH precursor', 
                                 'NH trigger',
                                 'p-value precursor', 
                                 'p-value trigger',
                                 'precursor coincidence rate',
                                 'trigger coincidence rate',
                                 'p_dry',
                                 'p_wet'],  
                              columns=['Poisson', 'Shuffle'])

        return df
    


#------------------------------------------------
def calculate_ECA_30y(wet_list, dry_list, delT, tau):
    ''' calculates ECA for 30-year periods.

    Parameters
    wet_list : list of np arrays
        List of binary time series for the "wet" events, one for each ensemble member.
    dry_list : list of np arrays
        List of binary time series for the "dry" events, one for each ensemble member.
    delT : int
        Time window for coincidence (in years).
    tau : int       Time lag for coincidence 


    eturns
    -------  list of EventCoincidence objects
         List of ECA results for each ensemble member.  
    '''
    eca_objects = []
    for wet, dry in zip(wet_list, dry_list):
        eca = EventCoincidence(wet, dry, delT,tau, sym=False)
        eca_objects.append(eca)
    return eca_objects


#---------------------------------------------------------------
def calculate_ECA_whole_ts(wet_list, dry_list, delT, tau):
    ''' calculates ECA for the whole time series.'''
    eca_results = EventCoincidence(wet_list, dry_list, delT, tau)

    return eca_results