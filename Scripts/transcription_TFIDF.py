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
import nltk
from nltk.stem import WordNetLemmatizer
import string


    
    
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



def tokenize(text):
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    lemmes = [lemmatizer.lemmatize(token) for token in tokens]
    return [l for l in lemmes if len(l) > 2]

# Extracting features from text files
def run(corpus):
    #trans_tf_dfs = []
    count_vect = CountVectorizer(tokenizer=tokenize, stop_words='english')
    X = count_vect.fit_transform(corpus)
        
    # TF-IDF
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X)
    #trans_tf_dfs      
    return X_tfidf,X

"""
Probleme de l'extraction 

"""
def export_transTFIDF(data,path_out):

    #Export Data
    path_output = path_out
    ## videos
    data.to_csv(path_output + "transTFIDF.csv",sep = ";")
 

# MAIN
           
path = '/Users/nabil/Desktop/TER_Challenge/DEV_M2SID_LIMSI_ASR/'
corpus = open_doc(path)
X_TfIdf , Vect = run(corpus)


files= os.listdir(path)
X = X_TfIdf.todense()
trans_TFIDF = pd.DataFrame(X,index=(f for f in files))#, columns = Vect.get_feature_names())#,columns=Vect.todense().get_feature_names())
            

path_out = '/Users/nabil/Desktop/TER_Challenge/'
trans_TFIDF.to_csv(path_out + "transTFIDF.csv",sep = ";")



