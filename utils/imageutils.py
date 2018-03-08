# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 15:38:28 2018

@author: USUARIO
"""

import numpy as np

'''
Funcion para obtener array en formato numpy del histograma normalizado
'''
def getNormHist(img, excluding = None):
    histogramValues = np.zeros(256)
    totalValues = len(img) * len(img[0])
#    print(histogramValues)
#    print(totalValues)
    for y in range(0, len(img)):
        for x in range(0, len(img[y])):
            if excluding is None:
                histogramValues[int(img[y,x])] += 1
            elif int(img[y,x]) != excluding:
                histogramValues[int(img[y,x])] += 1
#    print(histogramValues)
    return histogramValues / totalValues