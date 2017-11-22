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
from unidecode import unidecode

## NLTK
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

## SKLEARN
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

'''============================================================================
    Dico
============================================================================'''

target_categs = {'1001':'Autos_vehicule',
                 '1009':'Food_drink',
                 '1011':'Health',
                 '1013':'Movies_television',
                 '1014':'Litterature',
                 '1016':'Politics',
                 '1017':'Religion',
                 '1019':'Sports'} 

"""============================================================================
    Functions for simple types
============================================================================"""
"""----------------------------------------------------
    Contains, begins by and ends_by
----------------------------------------------------"""
def Contains(pattern, x):
    """ Is a pattern contained in x string ? """
    ok = bool(re.search(str(pattern), str(x)))
    return ok

def BeginsBy(pattern, x):
    """ Does pattern begin x string ? """
    ok = Contains('^' + pattern, x)
    return ok

def EndsBy(pattern, x):
    """ Does pattern ends x string ? """
    ok = Contains(pattern + '$', x)
    return ok

def DeleteSpaces(text):
    """ Suppress double, starting, ending spaces"""
    res = text
    res = re.sub(" +", " ", res)
    res = re.sub("^\s+", "", res)
    res = re.sub("\s+$", "", res)
    return res

"""============================================================================
    Main Functions
============================================================================"""
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


"""============================================================================
    Tokenize
============================================================================"""

"""----------------------------------------------

----------------------------------------------"""

def Tokenize(text, len_min = 3):
    """
        Descr:
             Tokenisation : Analyse lexicale d'un texte
             On cherche les différents mots.
        In:
            - text (string): text to tokenize
            - len_min = minimal length of th e words (number of character).
        Out:
            - res
    """
    ## Clear Balises
    text = text.lower()
    ## Clear Balises
    text = re.sub("<.*?>", " ", text)
    text = re.sub(" -- ", "", text)
    ## Signes bizarres
    text = re.sub("\&apos;", "\'", text)
    text = re.sub("\&\#\w+;", "", text)
    text = re.sub("â\xa0", "", text)
    ## replace accents
    text = unidecode(text)
    ## For each word
    allwords = re.findall("\S+", text)
        
    words_ls = []
    for w in allwords:
        islink = BeginsBy("http[s]?://", w) or Contains("www\.", w)
        if islink:
            wlink = re.sub("(https?://)?", "", w)
            wlink = re.findall("(^.*?\.\w+)/?$", wlink)
            words_ls += wlink
        else:
            wkeep = re.sub("[\.,\']", "", w)
            wkeep = re.sub("^[-_]+", "", wkeep)
            wkeep = re.sub("[-_]$", "", wkeep)
            wkeep = re.sub("[\W^-]", " ", wkeep)
            wkeep = re.sub("^ ", "", wkeep)
            wkeep = re.sub(" $", "", wkeep)
            if len(wkeep) >= 1:
                words_ls.append(wkeep)
            else:
                pass
        continue
    
    if len(words_ls) >= 1:
        ## Collecte des mots
        text = ' '.join(words_ls)
        ## tokenisation
        tokens = word_tokenize(text)
        ## lemmatisation
        lemmatizer = WordNetLemmatizer()
        lemmes = [lemmatizer.lemmatize(token) for token in tokens]
        words_ls = [l for l in lemmes if len(l) >= len_min]
    else:
        words_ls = []
    ## Result : list of words
    return words_ls


"""----------------------------------------------

----------------------------------------------"""

def Tf_Idf(corpus, language = 'english'):
    """
        Descr:
            From a textual corpus, return tfidf matrix for each words.
        In:
            corpus: Series(!!!) of text
            language: language for disable  stopwords
        Out:
            
    """
    corpus_ls = list(corpus)
    #trans_tf_dfs = []
    cv = CountVectorizer(tokenizer = Tokenize,
                         stop_words = language)
    cv_fit = cv.fit_transform(corpus_ls)  
    # TF-IDF
    tfidf = TfidfTransformer()
    tfidf_fit = tfidf.fit_transform(cv_fit)
    
    ## Matrice tf_idf
    resDf = pd.DataFrame(tfidf_fit.todense(),
                         index = corpus.index,
                         columns = cv.get_feature_names())
    # result    
    return resDf


"""----------------------------------------------
    Get TFIDF
----------------------------------------------"""

def nn0(x):
    return (x!=0).sum()

def GetTFIDF(series, ndocsmin):
    tfidf = Tf_Idf(series)
    nvalues = tfidf.apply(nn0).sort_values(ascending = False)
    nvalues_sup2 = nvalues[(nvalues >= ndocsmin)]
    tfidf = tfidf[nvalues_sup2.index]
    return tfidf, nvalues

    

