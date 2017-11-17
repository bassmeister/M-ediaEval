# -*- coding: utf-8 -*-
"""----------------------------------------------------------------------------
Created on Wed Nov  8 11:22:24 2017

@author: Cedric
"""

"""============================================================================
    Packages
============================================================================"""

import re
import pandas as pd
import numpy as np

'''============================================================================
    Dico
============================================================================'''

target_categs = {'1001':'Autos and vehicule',
                 '1009':'Food and drink',
                 '1011':'Health',
                 '1013':'Movies and television',
                 '1014':'Litterature',
                 '1016':'Politics',
                 '1017':'Religion',
                 '1019':'Sports'} 

"""============================================================================
    Main Functions
============================================================================"""

def ClearEnvir():
    allvars = [var for var in globals() if var[0] != "_"]
    for var in allvars:
        globals()[var]
        continue

## Test
# vid = str(31).zfill(6)
# '000031'

def Unique(ls):
    """
        Descr :
            get unique values in a list
        In :
            - txt : a list of values
        Out :
            A list of unique values
    """
    res = list(set(ls))
    return res
    

def StringToBool(txt):
    """
        Descr :
            Conversion of a string in boolean
        In :
            - txt : a string
        Out :
            A boolean:
                - True if "txt" is assimiled as true
                - False else
    """
    res = (txt.lower() in ("yes", "true", "t", "1"))
    return res



def Lod_to_DataFrame(lod):
    """
        Descr :
            Conversion of a string in boolean
        In :
            - txt : a string
        Out :
            A boolean:
                - True if "txt" is assimiled as true
                - False else
    """
    orderkeys = [k for k in lod[0]]
    resDf = pd.DataFrame(lod)[orderkeys]
    return resDf

"""============================================================================
    Analyse Words
============================================================================"""

def ValidWord(w, min_char = 3, stop_words = []):
    """ Valid a word """
    w = w.lower()
    w = re.sub('\s+', '', w)
    w = re.sub('[^a-z0-9]+', '', w)
    ok = (len(w) >= min_char and w.lower() not in stop_words)
    return ok


def Count_Occurences(data, groups):
    """
    """
    sizes = data.groupby(groups).size().rename('count')
    dfGrouped = pd.DataFrame(sizes)
    dfGrouped.reset_index(inplace = True)
    if (len(groups) >= 2):
        dfCounts = dfGrouped.pivot(index = groups[0],
                                   columns = groups[1],
                                   values = 'count')
        dfCounts = dfCounts.fillna(0)
        dfCounts = dfCounts.astype(int)
        resDf = dfCounts
    else:
        resDf = dfGrouped
    return resDf


def take_n_greatest(x, n = 10):
    """ take the n greatest values """
    y = x[x != 0].sort_values(ascending = False)
    if len(y) >= n:
        res = [(y.index[i], int(y[i])) for i in range(n)]
    else:
        res = [(y.index[i], int(y[i])) for i in range(len(y))]
        res += [np.nan for i in range(len(y), n)]
    return res



