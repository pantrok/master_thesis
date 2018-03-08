# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 09:06:25 2018

@author: Daniel Sanchez Ruiz
Script que toma imagenes de un paciente, extrae caracteristicas
y crea u ocupa un archivo como vector de caracteristicas
"""
from extractores import extractores as e
from utils import imageutils as iu
from skimage import io
import os

'''
MAIN
'''
if __name__ == "__main__":
    
    rootdir = "F:/base"
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if file.endswith(".jpg") and \
                (file.find('_SLB') != -1 or file.find('_SRB') != -1):
                print(os.path.join(subdir, file).replace("\\","/"))
                directory_in_str = subdir.replace("\\","/")
                absFilePath = os.path.join(subdir, file).replace("\\","/")
                print(absFilePath)
                # Lectura de ambas imagenes y de matriz de temperatura
                img = io.imread(absFilePath)
                histNorm = iu.getNormHist(img[...,0], excluding = 255)
                print(type(histNorm))
                CarPriOrd = e.ExtractorCarPrimerOrden(directory_in_str, histNorm)
                # Extraccion de caracteristicas de primer orden (clase)
                CarPriOrd.extractAllFirstOrderFeatures()
                CarText = e.ExtractorCarTexture(directory_in_str, img[...,0])
                CarText.extractAllTextureFeatures()
                
                
                