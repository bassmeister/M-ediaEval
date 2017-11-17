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

import utils_media as utils

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix




"""============================================================================
    Main program
============================================================================"""

path_proj = 'D:/Ced/Documents/UNIVERSITE/Cours/2017_M2-SID/mediaeval/'
AllkeywordsData = pd.read_csv(path_proj + 'data/database/keywords.csv', sep = ';')

dfCateg = pd.read_csv(path_proj + 'data/database/Liste_Videos.csv',
                      sep = ';')

categs_lab = [utils.target_categs[k] for k in utils.target_categs]

fcateg = [utils.target_categs[str(cat)] for cat in dfCateg['categorie']]

dfCateg["categ"] = fcateg


stopwords_ls = stopwords.words('english')

"""------------------------------------------------------------
    dfKeywords
------------------------------------------------------------"""

dfKeywords = AllkeywordsData[[
        utils.ValidWord(k,
                        min_char = 4,
                        stop_words = stopwords_ls)
        for k in AllkeywordsData['keyword']
]]

dfKeywords = pd.merge(dfKeywords,
                      right = dfCateg[['iddoc', 'categ']],
                      on = 'iddoc')

dfKeywords = dfKeywords.sort_values(['keyword'])

"""------------------------------------------------------------
    countDf
------------------------------------------------------------"""

countKW = utils.Count_Occurences(dfKeywords, ['keyword'])

scateg = pd.Series(dfCateg['categ'], index = dfCateg['iddoc'])
scount = utils.Count_Occurences(data = dfKeywords, groups = ['iddoc', 'keyword'])

countKW_docs = pd.concat([scateg, scount], axis = 1, join = 'outer')
countKW_docs = countKW_docs.fillna(0)
countKW_docs = countKW_docs.sort_values(["categ"])

countKW_docs.to_csv(path_proj + 'output/count_words_docs.csv',
                    sep = ";",
                    index = True)

countKW_categs = utils.Count_Occurences(dfKeywords, ['keyword', 'categ'])

"""------------------------------------------------------------
    10 first words
------------------------------------------------------------"""

regularWords_categs = countKW_categs.apply(utils.take_n_greatest)

regularWords_categs = pd.DataFrame.from_dict(regularWords_categs.to_dict(),
                                             orient = 'index')

regularWords_categs.to_csv(path_proj + 'output/10words_bycateg.csv',
                           sep = ";",
                           index = True)



"""------------------------------------------------------------
    PCA
------------------------------------------------------------"""

from Build_PCA import Build_PCA

pca = Build_PCA(countKW_categs, 5)

pca.plot([1, 2])
pca.plot([1, 3])


"""------------------------------------------------------------
    Random Forest
------------------------------------------------------------"""

## Make random train and test data
dfX = countKW_docs.drop(["categ"], axis = 1)
dfY = countKW_docs["categ"]

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
subKeysDf_df['keyword'] = [lemmatizer.lemmatize(w) for w in subKeysDf_df['keyword']]
"""


