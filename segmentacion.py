# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 17:35:55 2017

@author: USUARIO
Ajuste de contraste por regiones para encontrar
umbrales y definir umbral global para umbralización
dinámica
"""

import matplotlib.pyplot as plt
from skimage import io, filters, exposure, color, feature, draw
import numpy.polynomial.polynomial as poly
from scipy.stats import norm
from scipy import interpolate
import numpy as np
import seaborn as sns
import queue
import os

class Cluster:

    def __init__(self, xleft, xright, yup, ydown):
        self.xleft = xleft
        self.xright = xright
        self.yup = yup
        self.ydown = ydown
    
    def __str__(self):
        return 'Cluster xleft %d, xright %d, yup %d, ydown %d' % (self.xleft,self.xright, self.yup, self.ydown)

'''
Histograma de imagen en grises excluyendo valor
'''
def histogramExcluding(img, excluded = 0):
    allValues = []
    histogramValues = np.zeros(256)
    histogramBins = np.zeros(256)
    for binV in range(0,256):
        histogramBins[binV] = binV
    indexValues = 0
    for y in range(0, len(img)):
        for x in range(0, len(img[y])):
            if img[y,x] != excluded:
                allValues.append(int(img[y,x]))
                histogramValues[int(img[y,x])] += 1
            indexValues += 1
    #print(histogramValues2)
    #print(histogramBins2)
    return histogramBins, histogramValues, allValues

'''
Histograma acumulativo
'''
def greyCumulativeHistogram(bins, histoV, cumulative):
    if bins == 0:
        cumulative[0] = histoV[0]
        return cumulative
    else:
        aux = greyCumulativeHistogram(bins - 1, histoV, cumulative)
        cumulative[bins] = aux[bins - 1] + histoV[bins]
        return cumulative

'''
Contraste automatico
'''
def automaticContrast(img, histoV, amin, amax):
    imgCopy = img.copy()
    alow = 0
    ahigh = 0;
    for index in range(0, 256):
        if histoV[index] != 0:
            alow = index;
            break;
    index = len(img)
    while index >= 0:
        if histoV[index] != 0:
            ahigh = index;
            break;
        index -= 1
            
    rangeV = ahigh - alow;
    alow = int(alow + (rangeV * 0.05))
    ahigh = int(ahigh - (rangeV * 0.05))
    generatedValues = np.zeros(256)
#    print('alow %d, ahigh %d' % (alow, ahigh))
    for index in range(0, 256):
        pixelValue = int(amin + (index - alow) * ((amax - amin) / (ahigh - alow)))
        if (pixelValue < amin):
            pixelValue = amin
        if (pixelValue > amax):
            pixelValue = amax
        generatedValues[index] = pixelValue
    for y in range(0, len(img)):
        for x in range(0, len(img[y])):
            imgCopy[y, x] = generatedValues[img[y, x]]
            
    return imgCopy

'''
Contraste automatico acumulativo
'''
def automaticCumulativeContrast(img, histoV, q, amin, amax):
    imgCopy = img.copy()
    alow = 0
    ahigh = 0
    cumulative = np.zeros(256)
    histoAcumulado = greyCumulativeHistogram(255, histoV, cumulative)
    size = len(img) * len(img[0]);
    i = 0;
    lowCutValue = (size * q) / size;
    while histoAcumulado[i] < lowCutValue:
        i += 1
    alow = i
    i = 255
    highCutValue = histoAcumulado[len(histoAcumulado) - 1] - ((size * q) / size)
    while histoAcumulado[i] > highCutValue:
        i -= 1
    ahigh = i;
    generatedValues = np.zeros(256)
#    print('alow %d, ahigh %d' % (alow, ahigh))
    for index in range(0, 256):
        pixelValue = int(amin + (index - alow) * ((amax - amin) / (ahigh - alow)))
        if (pixelValue < amin):
            pixelValue = amin
        if (pixelValue > amax):
            pixelValue = amax
        generatedValues[index] = pixelValue
    for y in range(0, len(img)):
        for x in range(0, len(img[y])):
            if img[y, x] != 255:
                imgCopy[y, x] = generatedValues[img[y, x]]
            
    return imgCopy

'''
Tomamos la imagen, solo tomamos de abajo hacia arriba las dos terceras partes
de la imagen, en esa region encontramos min x y max x, son los puntos viniendo
de izquierda y derecha donde se pasa de negro a algun pixel distinto
con ese dato y tomando que en y tenemos los puntos
dividimos el eje de las x en tres y el eje de las y en dos
en regiones equitativas, guardamos esa informacion para definir un cluster
'''
def findClusters(img):
    height = len(img) #Altura total de la imagen
    #El primer tercio se ignora pues es la region del cuello y hasta brazos de forma total/parcial
    topLimit = len(img) - round((2/3) * len(img)) 
    #Altura de cada cluster
    heightSize = round((height - topLimit) / 2)
    #Se ubican los limites en x
    xmin = 999
    xmax = 0
    for y in range(topLimit, height):
        for x in range(0, len(img[y])):
            if img[y, x, 1] != 255 and x < xmin:
                xmin = x
            elif img[y, x, 1] != 255 and x > xmax:
                xmax = x
    #Con los limites se calcula el ancho de cada cluster
    widthSize = round((xmax - xmin) / 3)
    #Se generan los seis cluster y se retorna la lista de los objetos
    clustersList = []
    yup = topLimit
    ydown = topLimit + heightSize
    for yAxis in range(0,2):
        xleft = xmin
        xright = xmin + widthSize
        for xAxis in range(0,3):
            cluster = Cluster(xleft, xright, yup, ydown)
            clustersList.append(cluster)
            xleft = xright + 1
            if xAxis == 2:
                xright = xmax
            else:
                xright = xleft + widthSize
        yup = ydown + 1
        ydown = height
    return clustersList

def createCopyImgAfterThres(img, thesh):
    imgCopy = np.zeros((len(img),len(img[0])), dtype='uint8')
    for y in range(0, len(img)):
        for x in range(0, len(img[y])):
            if img[y, x] >= thesh:
                imgCopy[y, x] = img[y, x]
            else:
                imgCopy[y, x] = 255
    return imgCopy

'''
Hacemos el ajuste de contraste sobre un cluster ignorando negros y 
generamos histograma de subimagen y la guardamos,
asi como la imagen, 
normalizamos histograma y encontramos zona de umbral
guardamos coordenadas en una lista de esta zona

'''
def contrastAdjustCluster(img, cluster):
    #Obtenemos subimagen con datos de cluster
    clusterImg = img[cluster.yup:cluster.ydown, cluster.xleft:cluster.xright]
#    print(clusterImg.shape)
    clusterImg_gray = clusterImg[...,1]
#    with sns.axes_style('dark'):
#        norm2 = clusterImg_gray / clusterImg_gray.max()
#        io.imshow(norm2)
#        io.imsave('cluster_original.jpg',norm2)
#    io.imshow(clusterImg_gray)
#    io.show()
    #Calculamos histograma ignorando cualquier negro que aun sobreviva
    histoB, histoV, allValues = histogramExcluding(clusterImg_gray, excluded = 255) 
#    plt.plot(histoB, histoV)
    #Hacemos ajuste de contraste automatico
    clusterImg_adjContr = automaticCumulativeContrast(clusterImg_gray, histoV, 0.2 ,0, 255)
#    sig = exposure.adjust_sigmoid(clusterImg_gray, cutoff = 0.7, gain = 10)
    #clusterImg_adjContr = automaticContrast(clusterImg_gray, histoV, 1, 255)
#    clusterImg_adjContr = exposure.equalize_hist(clusterImg_gray) * 255
#    clusterImg_adjContr = exposure.adjust_sigmoid(clusterImg_gray, cutoff = 0.5, gain = 10)
#    print(clusterImg_adjContr)
#    print(clusterImg_adjContr.shape)
#    for y in range(0, len(clusterImg_adjContr)):
#        for x in range(0, len(clusterImg_adjContr[y])):
#            clusterImg_adjContr[y, x] = round(clusterImg_adjContr[y, x])
#            
    histoB, histoV, allValues = histogramExcluding(clusterImg_adjContr, excluded = 255)
#    indices = pu.indexes(histoV, thres=0.5, min_dist=0.1)
#    plt.plot(histoB, histoV)
#    plt.hist(allValues, histoB)
#    indices = pu.indexes(histoV, thres=0.02/max(histoV), min_dist=0.1)
#    print(indices)
#    plt.savefig('histograma_cluster.png')
    otsu_img_adjContr = filters.threshold_otsu(clusterImg_adjContr)
#    print('Valor de otsu %f' % (otsu_img_adjContr))
    imgFirstFilter = createCopyImgAfterThres(clusterImg_adjContr, otsu_img_adjContr)
#    with sns.axes_style('dark'):
#        norm2 = imgFirstFilter / imgFirstFilter.max()
#        io.imshow(norm2)
#        io.imsave('imgFirstFilter.jpg',norm2)
    histoB, histoV, allValues = histogramExcluding(imgFirstFilter, excluded = 255) 
#    plt.plot(histoB, histoV)
#    io.imshow(imgFirstFilter)
#    io.show()
    clusterImg_adjContr2 = automaticCumulativeContrast(imgFirstFilter, histoV, 0.2 ,0, 255)
    histoB, histoV, allValues = histogramExcluding(clusterImg_adjContr2, excluded = 255) 
#    plt.plot(histoB, histoV)
#    plt.savefig('histogramaS_cluster.png')
    otsu_img_adjContr2 = filters.threshold_otsu(clusterImg_adjContr2)
#    print('Valor de otsu %f' % (otsu_img_adjContr2))
#    io.imshow(clusterImg_adjContr2)
#    io.show()
#    with sns.axes_style('dark'):
#        norm2 = clusterImg_adjContr2 / clusterImg_adjContr2.max()
#        io.imshow(norm2)
#        io.imsave('cluster_ajusC.jpg',norm2)
#    io.imshow(clusterImg_adjContr)
#    io.show()
#    io.imshow(sig)
#    io.show()
    #Guardamos coordenadas de valor de umbral con un delta +- 5
    coordinatesList = []
#    print('Empezando impresion de valores en imagen con ajuste de contraste')
    for y in range(0, len(clusterImg_adjContr2)):
        for x in range(0, len(clusterImg_adjContr2[y])):
            if clusterImg_adjContr2[y, x] >= round(otsu_img_adjContr2 - 5) and \
            clusterImg_adjContr2[y, x] <= round(otsu_img_adjContr2 + 5):
                coordinatesList.append([y, x])
             
#    print('Empezando impresion de valores en imagen original en grises')
    minT = 999
    maxT = -1
    for key in range(0, len(coordinatesList)):
        coordinate = coordinatesList[key]
        if key == 0:
            minT = clusterImg_gray[coordinate[0], coordinate[1]]
            maxT = clusterImg_gray[coordinate[0], coordinate[1]]
        else:
            if clusterImg_gray[coordinate[0], coordinate[1]] < minT:
                minT = clusterImg_gray[coordinate[0], coordinate[1]]
            elif clusterImg_gray[coordinate[0], coordinate[1]] > maxT:
                maxT = clusterImg_gray[coordinate[0], coordinate[1]]

    return [minT,maxT]
#    return [1,1]

'''
Funcion para encontrar la media entre todos los valores de umbral que se encontraron
'''
def findThresholdsMean(thresholdList):
    thresholdsCount = 0
    meanThresholds = 0
    for thresholdInterval in thresholdList:
        meanThresholds += round((int(thresholdInterval[0]) + int(thresholdInterval[1]))/2)
        thresholdsCount += 1
    meanThresholds = round(meanThresholds/thresholdsCount)
    
    return meanThresholds

'''
Funcion para eliminar ruido
'''
def removeNoise(img):
    imgCopy = img.copy()
    print(imgCopy.shape)
    #Primero quitamos cualquier valor que exista en el primer tercio
    height = len(imgCopy)
    limit = round((1/3) * height) 
    #Altura de cada cluster
    for y in range(0, limit):
        for x in range(0, len(imgCopy[y])):
            if imgCopy[y, x, 0] == True:
                for channel in range(0,3):
                    imgCopy[y, x, channel] = False
               
    #Generamos lista de coordenadas de clusters, iteramos de abajo hacia arriba
    #Ordenamos la lista de clusters de mayor a menor longitud
    #Comparamos primero con segundo, si no hay una diferencia tan grande, consideramos segundo
    #En las coordenadas de los clusters restantes removemos ruido
    clusterList = []
    countY = height - 1
    while countY >= limit:
        for x in range(0, len(imgCopy[countY])):
            if imgCopy[countY, x, 0] == True:
                point = [countY, x]
#                print(point)
                clusterList = makeCluster(imgCopy, point, clusterList)
        
#        countY = limit - 1
        countY -= 1
    
    maxCluster = 0
    secondMaxCluster = 0
    for cluster in clusterList:
        print(len(cluster))
        if len(cluster) > maxCluster:
            maxCluster = len(cluster)
            
    for cluster in clusterList:
        if len(cluster) > secondMaxCluster and len(cluster) < maxCluster:
            secondMaxCluster = len(cluster)
    
    print("MaxCluster {}".format(maxCluster))
    print("SecondMaxCluster {}".format(secondMaxCluster))
    relationMaxsCluster = (secondMaxCluster/maxCluster) * 100
    print("Relacion MaxCluster y SecondMaxCLuster {}".format(relationMaxsCluster))
    for cluster in clusterList:
        if relationMaxsCluster >= 40:
            if len(cluster) == maxCluster or len(cluster) == secondMaxCluster:
                continue
            else:
                for point in cluster:
                    for channel in range(0,3):
                        imgCopy[point[0], point[1], channel] = False
        else:
            if len(cluster) != maxCluster:
                for point in cluster:
                    for channel in range(0,3):
                        imgCopy[point[0], point[1], channel] = False
    
    return imgCopy

def makeCluster(img, point, clusterList):
    for c in clusterList:
        if point in c:
#            print('se cumple condicion de paro')
            return clusterList
    
    neighQueue = queue.Queue()
    cluster = [point] 
    #Checamos los vecinos del primer punto
    for y in range(-1,2):
        for x in range(-1,2):
#            if ((point[0] + y) < len(img) and (point[1] + x) < len(img[0])):
#                print("Checando vecino [{}, {}] de [{}, {}] = {}".format((point[0] + y),(point[1] + x),point[0], point[1],img[(point[0] + y),(point[1] + x),0]))
            if x == 0 and y == 0:
                continue
            elif ((point[0] + y) < len(img) and (point[1] + x) < len(img[0])) \
                and (img[(point[0] + y),(point[1] + x),0] == True):
                    p = [(point[0] + y),(point[1] + x)]
#                    print("Vecino de [{}, {}] -> [{}, {}]".format(point[0], point[1], p[0], p[1]))
                    cluster.append(p)
                    neighQueue.put(p)
    checkedPoints = [point]
    #Si se agregaron vecinos en la cola, iteramos en checar vecinos de cada punto
    #cuando estos  no existan, se agrega al cluster pero tambien a la cola
    while True:
        try:
            newPoint = neighQueue.get_nowait()
        except queue.Empty:
            # Ya se han retornado todos los ítems
            break
        else:
            if newPoint not in checkedPoints:
                for y in range(-1,2):
                    for x in range(-1,2):
                        if (y == 0 and x == 0):
                            continue
                        elif ((newPoint[0] + y) < len(img) and (newPoint[1] + x) < len(img[0])) \
                            and (img[(newPoint[0] + y),(newPoint[1] + x),2] == True):
                                p = [(newPoint[0] + y),(newPoint[1] + x)]
                                if p not in cluster:
                                    cluster.append(p)
                                neighQueue.put(p)
                checkedPoints.append(newPoint)

    clusterList.append(cluster)
    return clusterList

'''
Funcion para adelgazar la region de blancos 
'''
def thinWhiteRegion(img):
    imgCopy = img.copy()
    for x in range(0, len(imgCopy[0])):
        whiteDetecteted = False
        for y in range(0, len(imgCopy)):
            if imgCopy[y, x, 2] == True:
                if whiteDetecteted == False:
                    whiteDetecteted = True
                else:
                    for channel in range(0,3):
                        imgCopy[y, x, channel] = False
    return imgCopy
    
'''
Funcion para detectar las coordenadas de la curva o curvas
'''
def saveCurvePoints(img):
    curvePoints = []
    twoCurvesCase = False
    #curve2Points = []
    for x in range(0, len(img[0])):
        for y in range(0, len(img)):
            if img[y, x, 0] > 230 and img[y, x, 1] > 230 and img[y, x, 2] > 230:
#            if img[y, x, 2] is True:
                p = [y, x]
                curvePoints.append(p)
                break
            
    #Buscamos la diferencia maxima entre dos puntos continuos, 
    maxDiferenceTwoCurves = 10
    maxDiference = 0
    indexP1 = 0
    indexP2 = 0
    indexHighestYPoint = 0
    highYPoint = 500
    for count in range(0,(len(curvePoints)-1)):
        p1 = curvePoints[count]
        p2 = curvePoints[count + 1]
        if p1[0] < highYPoint:
            highYPoint = p1[0]
            indexHighestYPoint = count
        if p2[0] < highYPoint:
            highYPoint = p2[0]
            indexHighestYPoint = count + 1
        if (p2[1] - p1[1]) > maxDiference:
            indexP1 = count
            indexP2 = count + 1
            maxDiference = (p2[1] - p1[1])
            
    #Creamos las dos listas con la informacion
    #Aqui habria que checar si la mayor diferencia es menor o igual a 5 digamos, si es asi seria solo
    #una curva, habria que encontrar el punto maximo y tomarlo para dividir ambas curvas
    if maxDiference >= maxDiferenceTwoCurves:
        curve1Points = curvePoints[0:(indexP1+1)]
        curve2Points = curvePoints[indexP2:len(curvePoints)]
        twoCurvesCase = True
        print('caso dos curvas')
    else:
        print('caso una curva')
        print(highYPoint)
        print(indexHighestYPoint)
        #Una curva, identificar el punto mas alto y con el partir la zona en dos zonas
        curve1Points = curvePoints[0:(indexHighestYPoint)]
        curve2Points = curvePoints[(indexHighestYPoint):len(curvePoints)]
        
    return curve1Points, curve2Points, twoCurvesCase

def checkNeigOnCurve(point, points):
    leftN = False
    rightN = False
    for ydelta in range(-1,2):
        if [point[0]+ydelta, point[1]-1] in points:
            leftN = True
            break
    for ydelta in range(-1,2):
        if [point[0]+ydelta, point[1]+1] in points:
            rightN = True
            break
        
    return leftN and rightN

'''
Funcion para obtener el polinomio de la curva
'''
def getPolynomialCurve(curvePoints, oriX = None, oriY = None):
    #Hacer traslacion de ejes
    pointsTranslated, originX, originY = pointsTraslation(curvePoints, oriX, oriY)
    #Encontrar cuatro puntos
    indexMinY = 0
    minX = 500
    indexMaxY = 0
    maxX = 0
    for count in range(0,len(pointsTranslated)):
        if pointsTranslated[count][1] < minX and \
            checkNeigOnCurve(pointsTranslated[count], pointsTranslated) is True:
            minX = pointsTranslated[count][1]
            indexMinY = count
        elif pointsTranslated[count][1] > maxX and \
            checkNeigOnCurve(pointsTranslated[count], pointsTranslated) is True:
            maxX = pointsTranslated[count][1]
            indexMaxY = count
#    interval = int(abs(indexMaxY - indexMinY) / 4)
#    interval = int(len(pointsTranslated) / 3)
    interval = int(abs(indexMaxY - indexMinY))
    #Seleccionamos cuatro puntos de la lista de coordenadas
    if indexMinY < indexMaxY:
#        pointSelected = [pointsTranslated[indexMinY],pointsTranslated[indexMinY+interval],pointsTranslated[indexMinY+interval+interval],pointsTranslated[indexMaxY]]
        pointSelected = [pointsTranslated[indexMinY],pointsTranslated[indexMinY+int(interval*.3)],pointsTranslated[indexMaxY-int(interval*.3)],pointsTranslated[indexMaxY]]
    else:
#        pointSelected = [pointsTranslated[indexMaxY],pointsTranslated[indexMaxY+interval],pointsTranslated[indexMaxY+interval+interval],pointsTranslated[indexMinY]]
        pointSelected = [pointsTranslated[indexMaxY],pointsTranslated[indexMaxY+int(interval*.3)],pointsTranslated[indexMinY-int(interval*.3)],pointsTranslated[indexMinY]]
#    pointSelected = [pointsTranslated[0],pointsTranslated[interval],pointsTranslated[-1]]
    print(pointSelected)
    #Pasar los puntos a la funcion de lagrange
    #Poner los datos en la forma que los espera la funcion, arreglo de x's y de y's
    x = [pointSelected[0][1], pointSelected[1][1], pointSelected[2][1], pointSelected[3][1]]
#    y = [pointSelected[0][0], pointSelected[1][0], pointSelected[2][0]]
    y = [pointSelected[0][0], pointSelected[1][0], pointSelected[2][0], pointSelected[3][0]]
    print(x)
    print(y)
    #Retornar polinomio o coeficientes
    lagrange_pylonomial = interpolate.lagrange(x, y)
    print(lagrange_pylonomial)
    return originX, originY, lagrange_pylonomial, pointsTranslated
    
'''
Funcion para obtener nuevos puntos haciendo traslacion de ejes
'''
def pointsTraslation(curvePoints, originX = None, originY = None):
    pointsTranslated = []
    y = 0 #solo son etiquetas para los indices
    x = 1
    #Encontrar punto mas a la izquierda y mas hacia abajo, que seran el nuevo origen
    if originX is None and originY is None:
        originX = 500
        originY = 0
        for coordinate in curvePoints:
            if coordinate[y] > originY:
                originY = coordinate[y]
            if coordinate[x] < originX:
                originX = coordinate[x]
    print(originY, originX)
    #Con los nuevos origines, calculamos la traslacion de los puntos
    for coordinate in curvePoints:
        translatedPoint = [originY - coordinate[y], coordinate[x] - originX]
        pointsTranslated.append(translatedPoint)
#    print(curvePoints)
#    print(pointsTranslated)
    return pointsTranslated, originX, originY

'''
Funcion que encuentra y muestra limites inferiores de la region de interes
'''
def findBottomBoundaries(img):
    img_copy = img.copy()
    #Encontrar puntos de las curvas inferiores de los senos
    curve1Points, curve2Points, twoCurvesCase = saveCurvePoints(img)
    #Aplicar lagrange
    oriX1, oriY1, poly1, pT1 = getPolynomialCurve(curve1Points)
    oriX2, oriY2, poly2, pT2 = getPolynomialCurve(curve2Points, oriX1, oriY1)
    
    firstXPoint1 = int((pT1[0][1] + oriX1) / 4)
    firstXPoint1 *= 3
    lastXPoint2 = int((len(img_copy[0]) - (pT2[-1][1] + oriX1))  / 4)
    lastXPoint2 *= 3
    #extraPT1 = np.linspace((pT1[0][1]-30), (pT1[-1][1]+30), num=((pT1[-1][1]+31)-(pT1[0][1]-30)),dtype=int)
    extraPT1 = np.linspace((pT1[0][1] - firstXPoint1), (pT1[-1][1]+30), num=((pT1[-1][1]+31)-(pT1[0][1] - firstXPoint1)),dtype=int)
    #extraPT2 = np.linspace((pT2[0][1]-30), (pT2[-1][1]+30), num=((pT2[-1][1]+31)-(pT2[0][1]-30)),dtype=int)
    extraPT2 = np.linspace((pT2[0][1]-30), (pT2[-1][1]+lastXPoint2), num=((pT2[-1][1]+lastXPoint2+1)-(pT2[0][1]-30)),dtype=int)
#    [[pT1[0][0],(pT1[0][1]-60)]]+[[pT1[0][0],(pT1[0][1]-30)]]+pT1+[[pT1[-1][0],(pT1[-1][1]+30)]]+[[pT1[-1][0],(pT1[-1][1]+60)]]
#    extraPT2 = [[pT2[0][0],(pT2[0][1]-30)]]+[[pT2[0][0],(pT2[0][1]-15)]]+pT2+[[pT2[-1][0],(pT2[-1][1]+15)]]+[[pT2[-1][0],(pT2[-1][1]+30)]]
    
    polyLeftPoints = []
    for curvePoint in extraPT1:
        y = (oriY1 - int(poly1(curvePoint)))
        if y >= len(img_copy):
            y = len(img_copy) - 1
        if y < 0:
            y = 0
        x = curvePoint + oriX1
        if x >= len(img_copy[0]):
            x = len(img_copy[0]) - 1
        polyLeftPoints.append([y, x])
        img_copy[y, x, 0] = 255
        img_copy[y, x, 1] = 0
        img_copy[y, x, 2] = 255
    
    polyRightPoints = []    
    for curvePoint in extraPT2:
        y = (oriY1 - int(poly2(curvePoint)))
        if y >= len(img_copy):
            y = len(img_copy) - 1
        if y < 0:
            y = 0
        x = curvePoint + oriX1
        if x >= len(img_copy[0]):
            x = len(img_copy[0]) - 1
        polyRightPoints.append([y, x])
        img_copy[y, x, 0] = 255
        img_copy[y, x, 1] = 0
        img_copy[y, x, 2] = 255
    
    #Igualar polinomios a cero para encontrar punto de interseccion
    #Extrapolar con otro color cada curva hasta el punto de inteseccion
    topIntersecPY = 500
    topIntersecPX = 0
    if twoCurvesCase is True:
        #si son dos curvas iterar sobre eje x con la lista de puntos de final left e inicial right
        #en sentido contrario hasta que lleguen a un mismo valor, tomar el valor mas alto en y de los dos
        lastLeftP = pT1[-1][1]
        firstRightP = pT2[0][1]
#        print(lastLeftP, firstRightP)
        leftCount = 1
        rightCount = 1
        insertectionHappen = False
        while insertectionHappen is False:
            if (lastLeftP + leftCount) == (firstRightP - rightCount):
#                print((lastLeftP + leftCount),(firstRightP - rightCount))
#                print(oriX1)
                topIntersecPX = (lastLeftP + leftCount) + oriX1
                yRight = (oriY1 - int(poly2((lastLeftP + leftCount))))
                yLeft = (oriY1 - int(poly1((lastLeftP + leftCount))))
#                print(yLeft, yRight)
                if yLeft < yRight:
                    topIntersecPY = yLeft
                else:
                    topIntersecPY = yRight
                break
            elif abs((lastLeftP + leftCount) - (firstRightP - rightCount)) == 1:
                rightCount += 1
            else:
                leftCount += 1
                rightCount += 1
    else:
        diff = poly1 - poly2
        print(diff)
        r = diff.roots
        print(r)
        r = r[~np.iscomplex(r)]
        for eRoot in r:
            print(eRoot.real)
            print(diff(eRoot.real))
            intersecPX =  int(eRoot.real + oriX1)
            intersecPY = int(oriY1 - poly1(eRoot.real))
            #Solo quedarme con el punto mas alto en y, checar si tengo una curva o dos
            if intersecPY < len(img_copy) and intersecPY >= 0 \
                and intersecPX < len(img_copy[0]) and intersecPX >= 0 \
                and intersecPY < topIntersecPY:
                topIntersecPY = intersecPY
                topIntersecPX = intersecPX
        
    rr, cc = draw.circle_perimeter(topIntersecPY, topIntersecPX, 5)
    img_copy[rr, cc, 0] = 255
    img_copy[rr, cc, 1] = 0
    img_copy[rr, cc, 2] = 0
   
    #retornar la imagen ya modificada
    return img_copy, topIntersecPX, topIntersecPY, polyLeftPoints, polyRightPoints

'''
Funcion para encontrar limites superiores
'''
def findUpperBoundaries(img, intersectPX, intersectPY):
    img_copy = img.copy()
    #Definir dos arreglos con los valores de las x
    leftXValues = None
    rightXValues = None
    #Iteramos de abajo hacia arriba a lo largo de toda una fila
    #Preguntamos cuando aparezca el primer cambio de color de negro a otro color
    #Donde se dio el cambio guardamos el valor de la coordenada
    #Cuando el arreglo esta vacio solo se ingresa la info, cuando no, se calcula la desviacion estandar
    #y la media, se toma la media y se agrega o resta la desviacion y se checa si el nuevo valor esta dentro del rango
    #si no lo esta, se guarda la coordenada en y para cada lado
    #cuando haya valor de y para cada lado se termina el proceso
    leftYCoor = None
    rightYCoor = None
    y = (len(img) - 1)
    leftDone = False
    rightDone = False
    breakCycle = False
    caseLeftChangeCount = 0
    oneThirdWidth = int(len(img[y]) / 3)
    while y >= 0:
        caseLeftChangeCount = 0
        changeColorFromLeft = False
        for x in range(0, len(img[y])):
            if img[y, x, 1] == 0 and changeColorFromLeft is False: #Checar si funciona con esta comparacion
                changeColorFromLeft = True
                if x == 0:
                    breakCycle = True
                else:
                    if caseLeftChangeCount == 0:
                        if leftXValues is None:
                            leftXValues = np.array([x],dtype=np.int64)
                        else:
                            leftXValues = np.append(leftXValues, [x])
                        caseLeftChangeCount += 1
            if img[y, x, 1] == 255 and changeColorFromLeft is True and breakCycle is False:
                if caseLeftChangeCount == 1 and x > (oneThirdWidth + oneThirdWidth):
                    changeColorFromLeft = False
                    if rightXValues is None:
                        rightXValues = np.array([x-1],dtype=np.int64)
                    else:
                        rightXValues = np.append(rightXValues, [x])
        if breakCycle is True:
            break
        y -= 1
        
    print("leftV {}".format(leftXValues))
    print("rightV {}".format(rightXValues))
    meanL = np.mean(leftXValues)
    stdL = np.std(leftXValues)
    print("mean, std, interval left values {},{},{}".format(meanL,stdL,((meanL-stdL)-1)))
    meanR = np.mean(rightXValues)
    stdR = np.std(rightXValues)
    print("mean, std, interval right values {},{},{}".format(meanR,stdR,((meanR+stdR)-1)))
    y = (int(len(img) / 2) - 1)
    while y >= 0:
        changeColorFromLeft = False
        for x in range(0, len(img[y])):
            if img[y, x, 1] == 0 and changeColorFromLeft is False: #Checar si funciona con esta comparacion
                changeColorFromLeft = True
                if x < ((meanL - stdL) - 1) and leftDone is False:
                    leftDone = True
                    leftYCoor = y
            if img[y, x, 1] == 255 and changeColorFromLeft is True:
                changeColorFromLeft = False
                if (x-1) > ((meanR + stdR) - 1) and rightDone is False:
                    #print(x)
                    rightYCoor = y
                    rightDone = True
            if leftDone is True and rightDone is True:
                break
        y -= 1
    print("leftYC, rightYC {},{}".format(leftYCoor,rightYCoor))
    #Con las coordenadas se pintan lineas sobre toda la fila para checar
    img_copy[leftYCoor,:,:] = 255
    img_copy[rightYCoor,:,:] = 255
    
    #Con la imagen modificada, ahora buscamos punto de inflexion tomando la coordenada
    #mas grande en y, los bordes de la imagen y el punto de interseccion
    inflectionRowL = 0
    inflectionRowR = 0
    xLeft = 0
    xRight = 700
    upLimit = 0
    if leftYCoor > rightYCoor:
        upLimit = leftYCoor
    else:
        upLimit = rightYCoor
    bottomLimit = int((intersectPY+upLimit)/2)
    leftInflValues = []
    rightInflValues = []
    print("up, bottom limit {},{}".format(upLimit, bottomLimit))
    for y in range(upLimit, bottomLimit):
        for x in range(0, len(img[0])):
            if img[y, x, 1] == 0:
                leftInflValues.append([y,x])
                break
            
    for y in range(upLimit, bottomLimit):
        x = len(img[0]) - 1
        while x >= 0:
            if img[y, x, 1] == 0:
                rightInflValues.append([y,x])
                break
            x -= 1
    
    leftXValue = 0
    for index in range(0, len(leftInflValues)):
#        print(leftInflValues[index][0],leftInflValues[index][1])
        if leftInflValues[index][1] > leftXValue:
            leftXValue = leftInflValues[index][1]
            
    for index in range(0, len(leftInflValues)):
        if leftInflValues[index][1] == leftXValue:
            inflectionRowL = leftInflValues[index][0]
            break
        
    rightXValue = 700
    for index in range(0, len(rightInflValues)):
#        print(rightInflValues[index][0],rightInflValues[index][1])
        if rightInflValues[index][1] < rightXValue:
            rightXValue = rightInflValues[index][1]
            
    for index in range(0, len(rightInflValues)):
        if rightInflValues[index][1] == rightXValue:
            inflectionRowR = rightInflValues[index][0]
            break
    
    print(inflectionRowL, inflectionRowR)
    if inflectionRowL < inflectionRowR:
        img_copy[inflectionRowL,:,:] = 200
        upLimit = inflectionRowL
    else:
        img_copy[inflectionRowR,:,:] = 200
        upLimit = inflectionRowR
    
    return img_copy, upLimit
   
'''
Funcion para segmentar imagenes
'''
def segmentImg(originalImg, intPX, intPY, upLimit, polyLP, polyRP):    
    original_copy = originalImg.copy()
    
    maxYCoor = 0
    for point in polyLP:
        for y in range(point[0], len(original_copy)):
            if y > intPY and point[1] < intPX:
                original_copy[y, point[1], :] = 255
    
    maxYCoor = 0
    for point in polyRP:
        for y in range(point[0], len(original_copy)):
            if y > intPY and point[1] >= intPX:
                original_copy[y, point[1], :] = 255
    
    img_leftBreast = original_copy[upLimit:len(originalImg), 0:intPX]
    img_rightBrest = original_copy[upLimit:len(originalImg), intPX:len(originalImg[0])]
    for y in range(0, len(img_leftBreast)):
        for x in range(0, len(img_leftBreast[y])):
            if img_leftBreast[y, x, 1] == 255 and img_leftBreast[y, x, 0] == 0 \
                and img_leftBreast[y, x, 2] == 0:
                img_leftBreast[y, x, 0] = 255
                img_leftBreast[y, x, 2] = 255
    
    for y in range(0, len(img_rightBrest)):
        for x in range(0, len(img_rightBrest[y])):
            if img_rightBrest[y, x, 1] == 255 and img_rightBrest[y, x, 0] == 0 \
                and img_rightBrest[y, x, 2] == 0:
                img_rightBrest[y, x, 0] = 255
                img_rightBrest[y, x, 2] = 255
    
    return img_leftBreast, img_rightBrest

'''
Funcion que umbraliza, detecta zonas de calor
y adelgaza region densa
'''
def clusteringAndThinning(img, img_3channel, directory_in_str, filename):
    nameImageOT = "{}/{}_OT.jpg".format(directory_in_str,filename)
    with sns.axes_style('dark'):
        norm2 = img_3channel / img_3channel.max()
#        #        io.imshow(norm2)
        io.imsave(nameImageOT,norm2)
#        #    io.imshow(img_3channel)
#        #    io.show()
            
#        #    with sns.axes_style('dark'):
#        #        norm2 = mult / mult.max()
#        #        io.imshow(norm2)
#        #    histoB, histoV = histogramExcluding(img)
#        #    plt.plot(histoB, histoV)
#        #    plt.savefig('histograma_global.png')
    #Calcular las regiones o clusters
    clusters = findClusters(img_3channel)
    thresholdList = []
    countCluster = 0
    #Solamente nos quedamos con los cluster impares, que son de la zona media superios y de las orillas de la zona inferior
    #pues son las que presentan mayor informacion de interes
    for cluster in clusters:
        if countCluster % 2 != 0:
            thresholdVCluster = contrastAdjustCluster(img, cluster)
            thresholdList.append(thresholdVCluster)
        countCluster += 1
#        #    contrastAdjustCluster(img_3channel, clusters[3])
#        #    
    print(thresholdList)
    thresholdValue = findThresholdsMean(thresholdList)
    print(thresholdValue)
    binary2 = img_grey > thresholdValue
    img_3channel2 = color.gray2rgb(binary2)
    for y in range(0, len(img_3channel)):
        for x in range(0, len(img_3channel[y])):
            if img_3channel[y, x, 1] == 255:
                img_3channel2[y, x, 1] = 255
#        #    with sns.axes_style('dark'):
#        #        norm2 = img_3channel2 / img_3channel2.max()
#        #        io.imshow(norm2)
#        #        io.imsave('T0034.1.1.D.2012-11-29.01_final.jpg',norm2)
#        #    io.imshow(img_3channel2)
#        #    io.show()
    nameImageWNoise = "{}/{}_WN.jpg".format(directory_in_str,filename)
    imgWNoise = removeNoise(img_3channel2)
    with sns.axes_style('dark'):
        norm2 = imgWNoise / imgWNoise.max()
#                io.imshow(norm2)
        io.imsave(nameImageWNoise,norm2)
    nameImageThin = "{}/{}_TH.jpg".format(directory_in_str,filename)
    imgThin = thinWhiteRegion(imgWNoise)
    with sns.axes_style('dark'):
        norm2 = imgThin / imgThin.max()
#                io.imshow(norm2)
        io.imsave(nameImageThin,norm2)

'''
Funcion que tiene la logica de encontrar limites superiores
e inferiores y que segmenta
'''
def findBoundariesAndSegment(img, directory_in_str, filename):
    imgWBottomBounds, intPX, intPY, polyLP, polyRP = findBottomBoundaries(img)
#            imgWBottomBounds, intPX, intPY, polyLP, polyRP = findBottomBoundaries(img)
    nameImageLB = "{}/{}_LB.jpg".format(directory_in_str,filename)
#            io.imsave(nameImageLB,imgWBottomBounds)
    with sns.axes_style('dark'):
        norm2 = imgWBottomBounds / imgWBottomBounds.max()
#                io.imshow(norm2)
        io.imsave(nameImageLB,norm2)
    nameImageUB = "{}/{}_UB.jpg".format(directory_in_str,filename)
    imgWUpperBounds, upLimit = findUpperBoundaries(imgWBottomBounds, intPX, intPY)
    with sns.axes_style('dark'):
        norm2 = imgWUpperBounds / imgWUpperBounds.max()
#                io.imshow(norm2)
        io.imsave(nameImageUB,norm2)
            
    imgLeftBreast, imgRightBreast = segmentImg(img_3channel, intPX, intPY, upLimit, polyLP, polyRP)
    nameImageLB = "{}/{}_SLB.jpg".format(directory_in_str,filename)
    with sns.axes_style('dark'):
        norm2 = imgLeftBreast / imgLeftBreast.max()
#                io.imshow(norm2)
        io.imsave(nameImageLB,norm2)

    nameImageRB = "{}/{}_SRB.jpg".format(directory_in_str,filename)
    with sns.axes_style('dark'):
        norm2 = imgRightBreast / imgRightBreast.max()
#                io.imshow(norm2)
        io.imsave(nameImageRB,norm2)

'''
MAIN
'''
if __name__ == "__main__":

    directory_in_str = "F:/base/Healthy/195/T0195 1 1 S 2013-10-07"
    directory = os.fsencode(directory_in_str)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
#        if filename.endswith(".jpg"):
        if filename.endswith(".jpg") and filename.find('_TH') != -1:
            absFilePath = "{}/{}".format(directory_in_str, filename)
            print(absFilePath)
            img = io.imread(absFilePath)
            absFilePath = "{}/{}".format(directory_in_str, filename[:filename.find('_')])
            print(absFilePath)
            originalImg = io.imread(absFilePath)
                
            print(img.shape)
#            img_grey = img[...,0]
            img_grey = originalImg[...,0]
            hist, bins = exposure.histogram(img_grey)
#            plt.plot(hist)
#            plt.savefig('histograma.png')
            thresh = filters.threshold_otsu(img_grey)
            binary = img_grey > thresh
            mult = binary * img_grey
            img_3channel = color.gray2rgb(mult)

            for y in range(0, len(img_3channel)):
                for x in range(0, len(img_3channel[y])):
                    if img_3channel[y, x, 1] == 0:
                        img_3channel[y, x, 1] = 255
            
#            clusteringAndThinning(img, img_3channel, directory_in_str, filename)
            findBoundariesAndSegment(img, directory_in_str, filename)

    