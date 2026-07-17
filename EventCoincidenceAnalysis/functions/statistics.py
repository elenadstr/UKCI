import numpy as np
import math
import iris
from iris.time import PartialDateTime

''' Binomial distribution of trigger coincidences (Kt)'''
def binomial_Kt(seriesA, seriesB, KT, delT=4, tau =0):
    TOL = delT + 1
    T = len(seriesA)        #number of obervations
    NA = np.sum(seriesA)    #number of precursor events
    NB = np.sum(seriesB)    #number of trigger events

    bracket = (1 - TOL/(T-tau))**NA
    combination = math.comb(NB, KT)

    p = combination * (1-bracket)**KT * bracket**(NB-KT)

    return p 

''' Binomial distribution of precursr coincidences (Kp)'''

def binomial_Kp(seriesA, seriesB, KP, delT = 4, tau = 0):
    TOL = delT + 1
    T = len(seriesA)        #number of obervations
    NA = np.sum(seriesA)    #number of precursor events
    NB = np.sum(seriesB)    #number of trigger events

    bracket = (1 - TOL/(T-tau))**NB
    combination = math.comb(NA, KP)

    prob_Kp = []
    for i in range(T):
        p = combination * (1-bracket)**KP * bracket**(NA-KP)
        prob_Kp.append(p)

    return prob_Kp

def decompose_years_30y(cube):
    ''' decomposes a cube into 30-year periods, starting with 1980-2010 and shifting the period by 10 years until 2050-2080.'''
    period_cubes = [cube.extract(iris.Constraint(time=lambda cell: PartialDateTime(year=1980) <= cell.point <= PartialDateTime(year=2010))),
               cube.extract(iris.Constraint(time=lambda cell: PartialDateTime(year=1990) <= cell.point <= PartialDateTime(year=2020))),
               cube.extract(iris.Constraint(time=lambda cell: PartialDateTime(year=2000) <= cell.point <= PartialDateTime(year=2030))),
               cube.extract(iris.Constraint(time=lambda cell: PartialDateTime(year=2010) <= cell.point <= PartialDateTime(year=2040))),
               cube.extract(iris.Constraint(time=lambda cell: PartialDateTime(year=2020) <= cell.point <= PartialDateTime(year=2050))),
               cube.extract(iris.Constraint(time=lambda cell: PartialDateTime(year=2030) <= cell.point <= PartialDateTime(year=2060))),
               cube.extract(iris.Constraint(time=lambda cell: PartialDateTime(year=2040) <= cell.point <= PartialDateTime(year=2070))),
               cube.extract(iris.Constraint(time=lambda cell: PartialDateTime(year=2050) <= cell.point <= PartialDateTime(year=2080)))
    ]
    return period_cubes