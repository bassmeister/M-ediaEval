# -*- coding: utf-8 -*-
"""----------------------------------------------------------------------------
Created on Wed Nov  8 11:22:24 2017

@author: Cedric
"""

"""============================================================================
    Packages
============================================================================"""

import pandas as pd

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




