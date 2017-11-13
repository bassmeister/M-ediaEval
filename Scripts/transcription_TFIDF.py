#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 08:52:18 2017

@author: nabil
"""
from sklearn.feature_extraction.text import TfidfVectorizer

from lxml import etree
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from scipy.sparse import csr_matrix


    
    
"""
 # Importation des données
    
"""

def extraction_words(doc):
    
    soup =  BeautifulSoup(doc)
    # Transcription extraction
    words = ""   
    for p in soup.find_all('word'):
        if(p.get_text() != " {fw} " and len(p.get_text())>3):  
            words=words+" "+p.get_text()
    
    return words 


def open_doc(path):
    ### Corpus qui contient tout les documents
    corpus = []
    
    files= os.listdir(path)
    
    for file in files:  # files  == target['nom']
        chemin = path+file
        doc = open(chemin).read()
        words = extraction_words(doc)
        corpus.append(words)
           
    return corpus  

# Extracting features from text files
def run(corpus):

    count_vect = CountVectorizer()
    X = count_vect.fit_transform(corpus)
        
    # TF-IDF
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X)
            
    return X_tfidf

"""
Probleme de l'extraction 

"""
def export_transTFIDF(d,path):

    sdf = pd.SparseDataFrame(d)
    #Export Data
    path_output = path
    ## videos
    sdf.to_csv(path_output + "transTFIDF.csv",sep = ";",index = False)
 

# MAIN
           
path = '/Users/nabil/Desktop/TER_Challenge/DEV_M2SID_LIMSI_ASR/'
corpus = open_doc(path)
X_TFId = run(corpus)

path_out = '/Users/nabil/Desktop/TER_Challenge/'

export_transTFIDF(X_TFId,path_out)





