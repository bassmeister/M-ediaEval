# -*- coding: utf-8 -*-
"""-------------------------------------------------------
Created on Thu Nov  9 14:00:43 2017

@author: Cedric Bezy

Analysis of metadata : tags
-------------------------------------------------------"""

"""============================================================================
    Import Packages
============================================================================"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

import utils_media as utils
from target_categories import target_categories as target_categs

"""============================================================================
    Functions
============================================================================"""

def ValidWord(w, min_char = 3, stop_words = []):
    """
        Valid a word
    """
    w = w.lower()
    w = re.sub("\s+", "", w)
    w = re.sub("[^a-z0-9]+", "", w)
    ok = (len(w) >= min_char and w.lower() not in stop_words)
    return ok


def Count_Occurences(data, groups):
    """
    """
    sizes = data.groupby(groups).size().rename("count")
    dfGrouped = pd.DataFrame(sizes)
    dfGrouped.reset_index(inplace = True)
    if (len(groups) >= 2):
        dfCounts = dfGrouped.pivot(index = groups[0],
                                   columns = groups[1],
                                   values = "count")
        dfCounts = dfCounts.fillna(0)
        resDf = dfCounts
    else:
        resDf = dfGrouped
    return resDf 


def take_n_greatest(x, n = 10):
    y = x[x != 0].sort_values(ascending = False)
    if len(y) >= n:
        res = [(y.index[i], int(y[i])) for i in range(n)]
    else:
        res = [(y.index[i], int(y[i])) for i in range(len(y))]
        res += [np.nan for i in range(len(y), n)]
    return res


"""============================================================================
    Main program
============================================================================"""

path_proj = "D:/Ced/Documents/UNIVERSITE/Cours/2017_M2-SID/mediaeval/"
AllkeywordsData = pd.read_csv(path_proj + "data/database/keywords.csv", sep = ";")

dfCateg = pd.read_csv(path_proj + "data/database/Liste_Videos.csv", sep = ";")
dfCateg["categ"] = [target_categs[str(cat)] for cat in dfCateg["categorie"]]

stopwords_ls = stopwords.words("english")

"""----------------------------------------------
    dfKeywords
----------------------------------------------"""

dfKeywords = AllkeywordsData[[
        ValidWord(k,
                  min_char = 4,
                  stop_words = stopwords_ls) for k in AllkeywordsData["keyword"]
]]
dfKeywords = pd.merge(dfKeywords,
                      right = dfCateg[["iddoc", "categ"]],
                      on = "iddoc")
dfKeywords = dfKeywords.sort_values(["keyword"])

"""----------------------------------------------
    countDf
----------------------------------------------"""

countKW = Count_Occurences(dfKeywords, ["keyword"])

countKW_docs = pd.concat([pd.Series(dfCateg["iddoc"], index = dfCateg.index),
                          pd.Series(dfCateg["categ"], index = dfCateg.index),
                          Count_Occurences(data = dfKeywords,
                                           groups = ["iddoc", "keyword"])],
                          axis = 1).fillna(0)

countKW_categs = Count_Occurences(dfKeywords, ["keyword", "categ"])

"""----------------------------------------------
    10 first words
----------------------------------------------"""

regularWords_categs = countKW_categs.apply(take_n_greatest)
regularWords_categs = pd.DataFrame.from_dict(regularWords_categs.to_dict(),
                                             orient = "index")

"""----------------------------------------------
    PCA
----------------------------------------------"""

from Build_PCA import Build_PCA

pca = Build_PCA(countKW_categs, 5)

pca.plot([1, 2])
pca.plot([1, 3])

"""----------------------------------------------
    Random Forest
----------------------------------------------"""










"""============================================================================
    Brouillon
============================================================================"""




'''
lemmatizer = WordNetLemmatizer()
subKeysDf_df["keyword"] = [lemmatizer.lemmatize(w) for w in subKeysDf_df["keyword"]]
'''


