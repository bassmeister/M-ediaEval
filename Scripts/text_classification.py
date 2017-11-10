#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 08:52:18 2017

@author: nabil
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from lxml import etree
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd


target = pd.read_csv('/Users/nabil/Desktop/TER_Challenge/Liste_Videos.csv',sep=';')

# Importation des données
path = '/Users/nabil/Desktop/TER_Challenge/DEV_M2SID_LIMSI_ASR/'
files= os.listdir(path)
documents=[]

for file in target['nom']:
    chemin =path+file
    tree = open(chemin).read()
    soup = BeautifulSoup(tree)
    words = ""   
    for p in soup.find_all('word'):
        if(p.get_text() != " {fw} " and len(p.get_text())>3):  
            words=words+" "+p.get_text()
    documents.append(words)  
    
# Definir la variable cible des elements de documents
#target_documents = {'Autos and vehicule:1001','Health:1011','Movies and television:1013','Religion:1017','Food and drink:1009','Litterature:1014','Sports:1019','Politics:1016'}

# Extracting features from text files

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(documents)
X_train_counts.shape

# TF-IDF
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape
"""
FAIRE DE TRAIN ET DE TEST 

"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, target['categorie'], test_size=0.3, random_state=42)


# Machine Learning
# Training Naive Bayes (NB) classifier on training data.
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train,y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy =accuracy_score(y_test, y_pred)

# SVM
from sklearn import svm
clf = svm.SVC(decision_function_shape='ovo', probability=True) # probability = True afin de pouvoir recuperer la probabilite d'appartenance de chaque instance a une classe
clf.fit(X_train, y_train)
y_pred_svm = clf.predict(X_test)
accuracy_svm =accuracy_score(y_test, y_pred_svm)
prob1 = clf.predict_proba(X_test)