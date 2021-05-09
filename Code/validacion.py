import numpy as np   # Libreria de matrices numeros enteros
import matplotlib.pyplot as plt  #libreria Graficas
import tensorflow as tf  #libreria de inteligencia artificial 
from tensorflow import keras  #libreria para el modelo 
from tensorflow.keras import layers #modulo de capas 
from keras.layers import Dense
import pywt          #libreria wavelet
import math
import scipy.fftpack as fourier

matriz_entre=np.loadtxt('Matriz.txt')

matriz_entre=matriz_entre[:,0:5000]#recorte de la matriz

vec_pa=np.zeros((40,1295)) #Vector de 2 filas x 15000 columnas, para el entrenamiento

def wcv(matriz_entre):
    for i in range(matriz_entre.shape[0]):
        #Transformada de Wavelet
        cA, cD = pywt.dwt(matriz_entre[i,:], 'dmey')#dmey:aproximacion FIR
        cA1,cD1= pywt.dwt(cD,'dmey')
        cA2,cD2= pywt.dwt(cD1,'dmey')
        cA3,cD3= pywt.dwt(cD2,'dmey')
        cA4,cD4= pywt.dwt(cD3,'dmey')
        cA5,cD5= pywt.dwt(cD4,'dmey')
        cA6,cD6= pywt.dwt(cD5,'dmey')
        cA7,cD7= pywt.dwt(cD6,'dmey')
    #plt.plot(cD1)
    #plt.show()
        vec_pa[i,0:len(cD1)+0]=cD1
        
wcv(matriz_entre)
y=(vec_pa[39,0:1295]) #selecciono la fila 
y=y.reshape(1,y.shape[0])

modeloT = keras.models.load_model('prueba1.h5')

x=modeloT.predict(y)

if(x<0.2 ):
    print("Sin fatiga")
    
if(x>0.8):
    print("Fatiga")


