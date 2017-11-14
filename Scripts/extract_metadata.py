# -*- coding: utf-8 -*-
"""-------------------------------------------------------
Created on Mon Nov  6 14:45:23 2017

@author: Cedric Bezy
-------------------------------------------------------"""

"""============================================================================
    Importation of Packages
============================================================================"""

import utils_media as utils
import os
import re
import pandas as pd
from bs4 import BeautifulSoup

"""============================================================================
    Main functions
============================================================================"""

"""------------------------------------------------
    Extract informations from soup and Make Data
------------------------------------------------"""

def ExtractFromSoup(path_meta, filesDf):
    """
        Descr :
            This function make dataframes from videos features
            contained in a collection of XML files
        Warning :
            ExtractFromSoup is used, where Structure is defined !
        In :
            - path_meta : folder where XML files are localised
        Out :
            4 dataframes :
                - video features
                - users features
                - licenses features
                - tags (keywords)
    """
    
    ## Make Data List
    shortFiles = os.listdir(path_meta)
    files_ls = [(path_files + f) for f in shortFiles]
    
    """
        Initialisation des listes
        Les listes contiendront les dictionnaires de données,
        ce afin de faciliter la formation de dataframes.
    """
    videosFeatures_ls = []
    us_ids = []
    usersFeatures_ls = []
    lc_ids = []
    liceFeatures_ls = []
    tagsFeatures_ls = []
    
    for i in range(len(files_ls)):
        """-------------
            Pour un fichier donné, on recueille le contenu,
            qu'on structure en object "soup"
        -------------"""
        iFile = files_ls[i]
        with open(iFile) as fp:
            iSoup = BeautifulSoup(fp.read(), "lxml")
        
        """-------------
            On recueille les informations nécessaires
        -------------"""
        ## Extract Data
        v_title = iSoup.title.get_text()
        v_descr = iSoup.description.get_text()
        v_explicit = utils.StringToBool(iSoup.explicit.get_text())
        v_duration = int(iSoup.duration.get_text())
        v_url = iSoup.url.get_text()
        
        ## File Name
        file_name = iSoup.file.filename.get_text()
        file_link = iSoup.file.link.get_text()
        file_size = iSoup.file.size.get_text()
        
        subId = dfIds[(dfIds["nom"] == file_name + ".ogv.xml")]
        file_id = [x for x in subId["iddoc"]][0]
        """-------------
            Définition de la structure des 4 dicos,
            correspondants à celles des dataframes souhaitées
        -------------"""
        ## Tags
        v_tags = iSoup.tags.find_all("string")
        for tag in v_tags:
            tDict = {"iddoc": file_id,
                     "keyword": tag.get_text()}
            tagsFeatures_ls += [tDict]
            continue
        
        ## Uploader
        upl_user_id = int(iSoup.uploader.uid.get_text())
        if not upl_user_id in us_ids:
            upl_user_login = iSoup.uploader.login.get_text()
            uDict = {"user_id": upl_user_id,
                     "user_login": upl_user_login}
            usersFeatures_ls += [uDict]
            us_ids += [upl_user_id]
        
        ## Licenses
        license_id = int(iSoup.license.id.get_text())
        if not (license_id in lc_ids):
            license_type = iSoup.license.type.get_text()
            lDict =  {"license_id": license_id,
                      "license_type": license_type}
            liceFeatures_ls += [lDict]
            lc_ids += [license_id]
        
        ## Videos
        vDict = {"vd_id": i,
                 "vd_title": v_title,
                 "vd_descr": v_descr,
                 "vd_explicit": v_explicit,
                 "vd_duration": v_duration,
                 "vd_url": v_url,
                 "license_id": license_id,
                 "upl_user_id": upl_user_id,
                 "file_name": file_name,
                 "file_link": file_link,
                 "file_size": file_size,
                 "iddoc": file_id}
        
        videosFeatures_ls += [vDict]
        continue
    
    """-------------
        Constitution des 4 dataframes à partir des listes :
           A partir des listes, il suffit de concatener tous les dicos.
           Afin de conserver automatiquement l'ordre des colonnes,
           la fonction 'Lod_to_DataFrame' du module 'utils' est utilisée.
    -------------"""
    ## Make Data
    vidDf = utils.Lod_to_DataFrame(videosFeatures_ls)
    usDf = utils.Lod_to_DataFrame(usersFeatures_ls)
    liceDf = utils.Lod_to_DataFrame(liceFeatures_ls)
    tagsDf = utils.Lod_to_DataFrame(tagsFeatures_ls)
    
    ## Sort Users and Licenses
    vidDf.sort_values("vd_id")
    usDf.sort_values("user_id")
    liceDf.sort_values("license_id")
    tagsDf.sort_values(["iddoc", "keyword"])
    
    ## results
    return vidDf, usDf, liceDf, tagsDf


"""============================================================================
    Main Program
============================================================================"""

"""---------------------------------------
    Files Path
---------------------------------------"""

## Path of Project
path_proj = "D:/Ced/Documents/UNIVERSITE/Cours/2017_M2-SID/mediaeval/"

## Path Files
path_files = path_proj + "data/extraction/DEV_M2SID_METADATA/"
shortFiles = os.listdir(path_files)
files_ls = [(path_files + f) for f in shortFiles]

"""---------------------------------------
    Make Data
---------------------------------------"""

## Import Data
dfIds = pd.read_csv(path_proj + "data/database/Liste_Videos.csv", sep = ";")

## Data Frame
dfVideos, dfUsers, dfLicenses, dfKeywords = ExtractFromSoup(path_files, dfIds)


"""---------------------------------------
    Export
---------------------------------------"""

path_output = path_proj + "/data/database/"
## videos
dfVideos.to_csv(path_output + "videos.csv",
                sep = ";",
                index = False)
## users
dfUsers.to_csv(path_output + "users.csv",
               sep = ";",
               index = False)
## licenses
dfLicenses.to_csv(path_output + "licenses.csv",
                  sep = ";",
                  index = False)
## tags
dfKeywords.to_csv(path_output + "keywords.csv",
                  sep = ";",
                  index = False)

