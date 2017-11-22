# -*- coding: utf-8 -*-
"""-------------------------------------------------------
Created on Thu Nov  9 14:00:43 2017

@author: Cedric Bezy

Analysis of metadata : tags
-------------------------------------------------------"""

"""============================================================================
    Import Packages
============================================================================"""

import re
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import utils_media as utils


"""============================================================================
    Paths and Importation
============================================================================"""

## Paths
path_proj = 'D:/Ced/Documents/UNIVERSITE/Cours/2017_M2-SID/mediaeval/'
path_data = path_proj + 'database/'

## meta
path_meta = path_proj + "extraction/DEV_M2SID_METADATA/"

"""============================================================================
    Import and remake data
============================================================================"""
"""------------------------------------------------------------
    import data
------------------------------------------------------------"""
## Tags
metadata = pd.read_csv(path_data + 'metadata.csv',
                       sep = ';',
                       dtype = {"categorie": "category"})

## List of Categories Labels
categs_lab = [utils.target_categs[k] for k in utils.target_categs]

## Replace categories in dfCateg
fcateg = pd.Categorical(
        [utils.target_categs[str(cat)] for cat in metadata['categorie']],
        categories = categs_lab
)

metadata['categ_lab'] = fcateg

"""============================================================================
     Analysis
============================================================================"""



## TAGS
tags = [' '.join(re.findall('\'(\w+)\'', tags)) for tags in metadata["vd_tags"]]
tags = pd.Series(tags, index = metadata.index)
tfidf_tags, nvaltags = utils.GetTFIDF(tags, 2)

## TITLES
titles = metadata["vd_title"]
tfidf_titles, nvaltitles  = utils.GetTFIDF(titles, 2)

## DESCRIPTIONS
descriptions = metadata["vd_descr"]
tfidf_descr, nvaldescr = utils.GetTFIDF(descriptions, 2)

## TOUT
alltext = titles + " // " + descriptions + " // " + tags
tfidf_all, nvalall = utils.GetTFIDF(alltext, 2)


"""============================================================================
     Random Forest
============================================================================"""

## Make random train and test data
dfX = tfidf_all
dfY = metadata["categ_lab"]

Xtrain, Xtest, Ytrain, Ytrue = train_test_split(dfX, dfY,
                                                test_size = 0.21)


rfc = RandomForestClassifier(n_estimators = 10,
                             criterion = "gini")

rffit = rfc.fit(Xtrain, Ytrain)
Ypred = pd.Series(rffit.predict(Xtest), index = Ytrue.index)


Ypred_proba = pd.DataFrame(rffit.predict_proba(Xtest),
                           index = Xtest.index,
                           columns = categs_lab)


rfconfus = confusion_matrix(Ytrue, Ypred, labels = dfY.unique())

dfConfus = pd.DataFrame(rfconfus,
                        index = dfY.unique(),
                        columns = dfY.unique())

rfscore = rffit.score(Xtest, Ytrue)
print(rfscore)

"""============================================================================
     Brouillon
============================================================================"""

# allwords = ['2010', '1h5', 'hector', 'http://www.ex.org', 'http://www.e3.org']


## DEBUG
# savetext = corpus[73]
# utils.Tokenize(corpus[0], 3)

"""
Tokenize(corpus[0], 3)

cv, cvfit, tfidf, tfidf_fit  = Tf_Idf(corpus)

print(cvfit)
print(tfidf_fit)

vocab = cv.vocabulary_


mat = pd.DataFrame(tfidf_fit.todense(),
                   index = corpus.index,
                   columns = cv.get_feature_names())

"""





