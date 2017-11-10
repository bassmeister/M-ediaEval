# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 08:56:39 2017

@author: mouster
"""
#import k-NN classifier for image classificationPython
# import the necessary packages

import cv2
import matplotlib.pyplot as plt


img=cv2.imread('4797.jpg')
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.plot(hist)
plt.show()