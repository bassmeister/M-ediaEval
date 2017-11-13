#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 21:10:13 2017

@author: nabil
"""

import os

from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


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