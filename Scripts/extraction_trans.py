#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 21:10:13 2017

@author: nabil
"""

import os
import numpy as np
from bs4 import BeautifulSoup
import pandas as pd



#from .base import Base
#from .util import apply_func_dict_values
#from .util import filename_without_extension
#from .util import remove_extra_html_tags
#from .util import tokenize


transFeatures = []
path = '/Users/nabil/Desktop/TER_Challenge/DEV_M2SID_LIMSI_ASR/'
documents=[]
transDic ={}
files= os.listdir(path)


for file in files:  # files  == target['nom']
    nom = file
    chemin = path+file
    tree = open(chemin).read()
    #soup = BeautifulSoup(tree)
    soup = BeautifulSoup(tree, 'lxml')
    for  sp in soup.find_all("speakerlist"):
        for p in soup.find_all("speaker"):
            transDic = {}
            dur = p['dur']
            gender = p['gender']
            spkid = p['spkid']
            transDic = {"filename": nom ,"duree":dur,"gender":gender,"spkid":spkid}  
            transFeatures += [transDic]     

## Make Data
transDf = df = pd.DataFrame(data=transFeatures)

#Export Data
path_output = path
## videos
transDf.to_csv(path_output + "trans_feature.csv",
                sep = ";",
                index = False)

    
"""        
  Extraction du deuxieme fichier 
— entropy,indicateursurladistributiondelaparole(prochede1siladistri-
   bution est homogène, proche de 0 si une personne parle plus souvent que les autres par exemple)
— nb_M,nombredespeakershomme
— nb_F,nombredespeakersfemme
— Freq_M/F,fréquencehomme/femme   

"""
def nbHomme(transFeatures):
    per = transFeatures['spkid']
    existe = []
    m = 0
    for p in per:
        #print(p)
        if not p in existe:
            if p[0]=='M':
                m+=1
                #print(p)
                existe.append(p)
            existe.append(p)
            if p[0]=='F':
                #print(p)
                existe.append(p)
 
    return m

def nbFemme(transFeatures):
    per = transFeatures['spkid']
    existe = []
    f = 0
    for p in per:
        #print(p)
        if not p in existe:
            if p[0]=='M':
                existe.append(p)
            existe.append(p)
            if p[0]=='F':
                f+=1
                #print(p)
                existe.append(p)
 
    return f
        #transDf = pd.concat([transDf, transFeatures])
## Make Data
def nbFemmeHomme(transFeatures):
    per = transFeatures['spkid']
    existe = []
    m = 0
    f = 0
    for p in per:
        #print(p)
        if not p in existe:
            if p[0]=='M':
                m+=1
                #print(p)
                existe.append(p)
            existe.append(p)
            if p[0]=='F':
                f+=1
                #print(p)
                existe.append(p)
 
    return f,m

def periode(transFeatures):
    per = transFeatures['spkid']
    dur = np.array(transFeatures['duree'])
    dure = [float(x) for x in dur]
    dic = {}
    cmpt=0
    periode_totale = 0
    for w in per:
        if w in dic.keys():
            dic[w]+=dure[cmpt]
            periode_totale+=dure[cmpt]
            cmpt+=1
        else :
            dic[w]=dure[cmpt]
            periode_totale+=dure[cmpt]
            cmpt+=1
    val_max = max({i:j for i, j in dic.items()}.values())
    return val_max/periode_totale

def run(transFeatures):
    features ={}
    female,male = nbFemmeHomme(transFeatures)
    freqHF=0
    if male != 0:
        freqHF= float(female/male)
    features = {"filename": nom ,"Femme":female,"Homme":male,"Freq_HF":freqHF,"distr":periode(transFeatures)}
    return features






transFeatures = []
path = '/Users/nabil/Desktop/TER_Challenge/DEV_M2SID_LIMSI_ASR/'
documents=[]
transDic ={}
files= os.listdir(path)
test = []
for f in files:
    #transFeatures = []
    tree = open(path+f).read()
    #soup = BeautifulSoup(tree)
    soup = BeautifulSoup(tree, 'lxml')
    for  sp in soup.find_all("speakerlist"):
        for p in soup.find_all("speaker"):
            transDic ={}
            dur = p['dur']
            gender = p['gender']
            spkid = p['spkid']
            transDic = {"filename": f ,"duree":dur,"gender":gender,"spkid":spkid}
            transFeatures += [transDic]
    tr = pd.DataFrame(transFeatures)
   
    #tr = pd.DataFrame(transFeatures)
    #tr_glob = pd.concat(tr_glob,tr)
    features ={}
    female,male = nbFemmeHomme(tr)

    freqHF=0
    if male != 0: 
        freqHF= (nbFemme(tr)/float(nbHomme(tr)))+0.0
    features = {"filename": f ,"Femme":nbFemme(tr),"Homme":nbHomme(tr),"Freq_HF":freqHF,"distr":periode(tr)}
    
    test.append(features)



test = pd.DataFrame(data=test)



#transFeatures = []
def extract_trans(paths):
    tree = open(paths).read()
    transFeatures = []
    #soup = BeautifulSoup(tree)
    soup = BeautifulSoup(tree, 'lxml')
    for  sp in soup.find_all("speakerlist"):
        for p in soup.find_all("speaker"):
            transDic ={}
            dur = p['dur']
            gender = p['gender']
            spkid = p['spkid']
            transDic = {"filename": f ,"duree":dur,"gender":gender,"spkid":spkid}
            transFeatures += [transDic]
    tr = pd.DataFrame(transFeatures)
       
    #tr = pd.DataFrame(transFeatures)
    #tr_glob = pd.concat(tr_glob,tr)
    features ={}
    female,male = nbFemmeHomme(tr)
    
    freqHF=0
    if male != 0: 
        freqHF= (nbFemme(tr)/float(nbHomme(tr)))+0.0
    features = {"filename": f ,"Femme":nbFemme(tr),"Homme":nbHomme(tr),"Freq_HF":freqHF,"distr":periode(tr)}
    
    return features
    
transFeatures = []
path = '/Users/nabil/Desktop/TER_Challenge/DEV_M2SID_LIMSI_ASR/'
documents=[]
transDic ={}
transfeatures = []
files= os.listdir(path)
test = []
for f in files:
    try:
        val = extract_trans(path+f)
        transfeatures.append(val)
    except:
        print(f)


transfeatures = pd.DataFrame(transfeatures)
path_out = '/Users/nabil/Desktop/TER_Challenge/'
transfeatures.to_csv(path_output + "trans_featureNew.csv",
                sep = ";",
                index = False)