# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:45:13 2018

@author: USUARIO
"""
import numpy as np
import csv
from scipy import stats
from extractores import Textures

class Extractor:
    
    def __init__(self):
        self.firOrdFeaLeft = np.array();
        self.firOrdFeaRight = np.array();
        self.secOrdFeaLeft = np.array();
        self.secOrdFeaRight = np.array();
    
    # Implementar escritura de archivo de vector, ultimo paso
    # tener primero todos los arreglos de caracteristicas
    def saveFeaturesToCSV(self, path, features):
        absolutePath = "{}/{}".format(path, "features.csv")
        with open(absolutePath, "a") as out:
            writer = csv.writer(out, lineterminator='\n')
            writer.writerow(features)
#        np.savetxt(absolutePath, features, delimiter=',')

class ExtractorCarPrimerOrden(Extractor):
    
    def __init__(self, absolutePath, data, typeData = 'Hist'):
#        super().__init__(self)
        self.absolutePath = absolutePath
        self.typeData = typeData
        self.data = data
        # Arreglo de valores de caracteristicas
        # orden de caracteristicas media, mediana, desviacion, curtosis, asimetria
        self.featuresArray = np.zeros(5)
        
    #implementar las formulas de cada una de las caracteristicas
    def extractMean(self):
        self.featuresArray[0] = np.mean(self.data)
    
    def extractMedian(self):
        self.featuresArray[1] = np.median(self.data)
    
    def extractStandardDeviation(self):
        self.featuresArray[2] = np.std(self.data)
    
    def extractKurtosis(self):
        self.featuresArray[3] = stats.kurtosis(self.data)
    
    def extractSkewness(self):
        self.featuresArray[4] = stats.skew(self.data)
        
    def extractAllFirstOrderFeatures(self):
        self.extractMean()
        self.extractMedian()
        self.extractStandardDeviation()
        self.extractKurtosis()
        self.extractSkewness()
        Extractor.saveFeaturesToCSV(self, self.absolutePath, self.featuresArray)

#Implementar caracteristicas de segundo orden
class ExtractorCarTexture(Extractor):
    
    def __init__(self, absolutePath, img):
#        super().__init__(self)
        self.absolutePath = absolutePath
        self.glcm = Textures.CooccurenceMatrixTextures(img, windowRadius = 1)
        self.corr, self.var, self.mean = self.glcm.getCorrVarMean()
        # Arreglo de valores de caracteristicas
        # orden de caracteristicas media, mediana, desviacion, curtosis, asimetria
        self.featuresArray = np.zeros(5)
        
    #implementar las formulas de cada una de las caracteristicas
    def extractDissimilarity(self):
        self.featuresArray[0] = np.sum(self.glcm.getDissimlarity())
    
    def extractEntropy(self):
        self.featuresArray[1] = np.sum(self.glcm.getEntropy())
    
    def extractEnergy(self):
        self.featuresArray[2] = np.sum(self.glcm.getASM())
    
    def extractMean(self):
        self.featuresArray[3] = np.sum(self.mean)
    
    def extractCorrelation(self):
        self.featuresArray[4] = np.sum(self.corr)
        
    def extractAllTextureFeatures(self):
        self.extractDissimilarity()
        self.extractEntropy()
        self.extractEnergy()
        self.extractMean()
        self.extractCorrelation
        Extractor.saveFeaturesToCSV(self, self.absolutePath, self.featuresArray)