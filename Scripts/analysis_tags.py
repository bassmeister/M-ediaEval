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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from Build_PCA import Build_PCA
import utils_media as utils

"""============================================================================
    Paths and Importation
============================================================================"""

## Paths
path_proj = 'D:/Ced/Documents/UNIVERSITE/Cours/2017_M2-SID/mediaeval/'
path_data = path_proj + 'database/'

"""============================================================================
    Import
============================================================================"""

## Tags
AllTagsData = pd.read_csv(path_data + 'tags.csv', sep = ';')

## Categories
dfCateg = pd.read_csv(path_data + 'Liste_Videos.csv', sep = ';')

## Stopwords
stopwords_ls = stopwords.words('english')

"""============================================================================
     Remake data
============================================================================"""
"""------------------------------------------------------------
    dfCategs
------------------------------------------------------------"""
## List of Categories Labels
categs_lab = [utils.target_categs[k] for k in utils.target_categs]

## Replace categories in dfCateg
fcateg = [utils.target_categs[str(cat)] for cat in dfCateg['categorie']]
dfCateg["categ"] = fcateg

"""------------------------------------------------------------
    dfTags
------------------------------------------------------------"""

words_ls = [utils.ValidWord(k, 4, stopwords_ls) for k in AllTagsData['tag']]

dfTags = pd.merge(left = AllTagsData[words_ls],
                  right = dfCateg[['iddoc', 'categ']],
                  on = 'iddoc')

dfTags = dfTags.sort_values(['tag'])


"""------------------------------------------------------------
    countDf
------------------------------------------------------------"""

countTags = utils.Count_Occurences(dfTags, ['tag'])

scateg = pd.Series(dfCateg['categ'], index = dfCateg['iddoc'])
scount = utils.Count_Occurences(data = dfTags, groups = ['iddoc', 'tag'])

countTags_docs = pd.concat([scateg, scount], axis = 1, join = 'outer')
countTags_docs = countTags_docs.fillna(0)
countTags_docs = countTags_docs.sort_values(["categ"])

countTags_docs.to_csv(path_proj + 'output/count_tags_docs.csv',
                    sep = ";",
                    index = True)

countTags_categs = utils.Count_Occurences(dfTags, ['tag', 'categ'])

countTags_categs.to_csv(path_proj + 'output/count_tags_categs.csv',
                        sep = ";",
                        index = True)



"""------------------------------------------------------------
    10 first words
------------------------------------------------------------"""

regularTags_categs = countTags_categs.apply(utils.take_n_greatest)

regularTags_categs = pd.DataFrame.from_dict(regularTags_categs.to_dict(),
                                             orient = 'index')

regularTags_categs.to_csv(path_proj + 'output/10words_bycateg.csv',
                           sep = ";",
                           index = True)

"""------------------------------------------------------------
    PCA
------------------------------------------------------------"""


pca = Build_PCA(countTags_categs, 5)

inertie = pca.fit.explained_variance_ratio_
inertie


pca.plot([1, 2])
pca.plot([1, 3])


"""------------------------------------------------------------
    Random Forest
------------------------------------------------------------"""

## Make random train and test data
dfX = countTags_docs.drop(["categ"], axis = 1)
dfY = countTags_docs["categ"]

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


"""
lemmatizer = WordNetLemmatizer()
subKeysDf_df['tag'] = [lemmatizer.lemmatize(w) for w in subKeysDf_df['tag']]
"""


