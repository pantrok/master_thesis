# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 16:50:13 2018

@author: USUARIO
"""

import os
import csv

'''
Funcion para generar vector de caracteristicas
con formato de weka
'''
def createFeaVecWeka():
    file = open("C:/Users/USUARIO/Documents/Python Scripts/Tesis/vectorescaracteristicas/feature-vector-weka.arff","w") 
    #Agregamos encabezados que requiera weka
    file.write("@RELATION thermal-breast-cancer \n\n")
    file.write("@ATTRIBUTE meanL REAL\n")
    file.write("@ATTRIBUTE medianL REAL\n")
    file.write("@ATTRIBUTE standardDeviationL REAL\n")
    file.write("@ATTRIBUTE kurtosisL REAL\n")
    file.write("@ATTRIBUTE skewnessL REAL\n")
    file.write("@ATTRIBUTE dissimilarityL REAL\n")
    file.write("@ATTRIBUTE entropyL REAL\n")
    file.write("@ATTRIBUTE energyDeviationL REAL\n")
    file.write("@ATTRIBUTE meanGLCML REAL\n")
    file.write("@ATTRIBUTE correlationL REAL\n")
    file.write("@ATTRIBUTE meanR REAL\n")
    file.write("@ATTRIBUTE medianR REAL\n")
    file.write("@ATTRIBUTE standardDeviationR REAL\n")
    file.write("@ATTRIBUTE kurtosisR REAL\n")
    file.write("@ATTRIBUTE skewnessR REAL\n")
    file.write("@ATTRIBUTE dissimilarityR REAL\n")
    file.write("@ATTRIBUTE entropyR REAL\n")
    file.write("@ATTRIBUTE energyDeviationR REAL\n")
    file.write("@ATTRIBUTE meanGLCMR REAL\n")
    file.write("@ATTRIBUTE correlationR REAL\n")
    file.write("@ATTRIBUTE class {Healthy, Sick}\n\n")
    file.write("@DATA\n")
    
    rootdir = "F:/base"
    for subdir, dirs, files in os.walk(rootdir):
        for f in files:
            if f.endswith(".csv"):
              absFilePath = os.path.join(subdir, f).replace("\\","/")
              arrayRows = []
              with open(absFilePath, "r") as csvFile:
                reader = csv.reader(csvFile)
                for row in reader:
                    for item in row:
                        arrayRows.append(format(float(item), 'f'))
              if subdir.find("Healthy") != -1:
                  arrayRows.append("Healthy")
              else:
                  arrayRows.append("Sick")
              allFeatures = ",".join(arrayRows)
              print(allFeatures)
              file.write(allFeatures + "\n")
    file.close()
    
'''
MAIN
'''
if __name__ == "__main__":
    createFeaVecWeka()