# -*- coding: utf-8 -*-
"""-------------------------------------------------------
Created on Mon Nov 13 11:26:58 2017

@author: Cedric
-------------------------------------------------------"""
"""============================================================================
    Import Package
============================================================================"""

from sklearn import decomposition
import matplotlib.pyplot as plt

"""============================================================================
    Class
============================================================================"""

class Build_PCA:
    def __init__(self, X, ncp = 5):
        pca = decomposition.PCA(n_components = ncp)
        pcafit = pca.fit(X)
        self.pca = pca
        self.fit = pcafit
        self.data = X
        self.ncp = ncp
    
    def __repr__(self):
        res = "PCA with {} components."
        return res.format(self.ncp)
    
    def __getattr__(self, name):
        print("Error ! There is no attribute \"{}\" !".format(name))
    
    def coords(self):
        coords = self.fit.transform(self.data)
        return coords
        
    def plot(self, axis = [1, 2], color = "b"):
        matcoords = self.coords()
        x = matcoords[:,axis[0]]
        y = matcoords[:,axis[1]]
        return plt.scatter(x, y, c = color)
    
    