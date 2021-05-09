import numpy as np   # Libreria de matrices numeros enteros
import matplotlib.pyplot as plt  #libreria Graficas
import tensorflow as tf  #libreria de inteligencia artificial 
from tensorflow import keras  #libreria para el modelo 
from tensorflow.keras import layers #modulo de capas 
from keras.layers import Dense
import pywt          #libreria wavelet
import math
import scipy.fftpack as fourier
#import scipy.fftpack as spfft #llamado de la funcion para la DCT


matriz_entre=np.loadtxt('Matriz.txt')

matriz_entre=matriz_entre[:,0:5000]#recorte de la matriz

vec_pa=np.zeros((68,1250)) #Vector de 2 filas x 15000 columnas, para el entrenamiento

"""
def maxi(matriz_entre):#Función para hallar los maximos
    for i in range(matriz_entre.shape[0]):
        vec_pa[i,0]=max(matriz_entre[i,:])#guarda en la columna 0 los maximos

def mini(matriz_entre):#Función para hallar los minimos
    for i in range(matriz_entre.shape[0]):
        vec_pa[i,1]=min(matriz_entre[i,:])#guarda en la columna 1 los minimos"""
  
"""        
def des(matriz_entre):
    for i in range(matriz_entre.shape[0]):
        vec_pa[i,0]=np.std(matriz_entre[i,:])#guarda en la columna 2 la desviacion estandar"""
"""        
def vari(matriz_entre):
    for i in range(matriz_entre.shape[0]):
        vec_pa[i,1]=np.var(matriz_entre[i,:])#guarda en la columna 3 la varianza

#Posiciones del vector de la señal
def raiz_cuadrada(matriz_entre):
    for k in range(matriz_entre.shape[0]):
        N=0
        pot=0
        RMS_senal=0
        for i in matriz_entre[k,:]:
            N += 1  
            #Suma de las posiciones del vector de la señal  RMS  
            suma = 0
        for l in matriz_entre[k,:]:
            suma = suma + l    
        pot=math.pow(suma,2)
        RMS_senal=math.sqrt(pot/N) #Raiz media cuadrada
        vec_pa[k,196]=RMS_senal"""

"""#ENERGIA
def energia(matriz_entre):
    for i in range(matriz_entre.shape[0]):    
            ventana=25
            p=0
            E=[]
            sumae=0
            for f in range (len(matriz_entre[i,:])):
                if(p<ventana):
                    sumae+=matriz_entre[i,f]**2
                    p=p+1
                else:
                    E.append(sumae)
                    p=0
                    sumae=0
            vec_pa[i,0:len(E)+0]=normalizar(E)
                        
def normalizar(X):
    X = X-min(X)
    return   X/max(X)"""

"""#Transformada de Fourier
def fou(matriz_entre):
    for i in range(matriz_entre.shape[0]):
        sp = np.real(fourier.fft(matriz_entre[i,:]))
    vec_pa[i,198:len(sp)+198]=sp
    plt.plot(sp)
    plt.show()"""

   
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
        vec_pa[i,0:len(cD2)+0]=cD2
        

eti = np.zeros((40,1)) #etiquetas

for i in range(40):
    e1 = 1
    e2 =2
    s=0
    
    if i <20:
        s=e1
    else:
        s=e2
    eti[i]=s

#maxi(matriz_entre)
#mini(matriz_entre)
#des(matriz_entre)
#vari(matriz_entre)
#energia(matriz_entre)
wcv(matriz_entre)
#plt.style.use('classic')
fig, ax = plt.subplots()
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
plt.grid('on')
#ax.plot(vec_pa[11,0:677], '-b', label='Sin Fatiga')
ax.plot(vec_pa[64,0:677], '-r', label='Con Fatiga')
#ax.axis('equal')
leg = ax.legend();
#plt.plot(vec_pa[1,0:570])
#plt.plot(vec_pa[33,0:570])

#plt.show()
"""
#Creacion del modelo perceptron simple 
model = keras.Sequential([
    layers.Dense(150,activation ='relu',input_shape=[621]), #Primera capa 5 neuronas
    layers.Dense(300,activation ='relu'),  #Segunda capa 20 neuronas
    layers.Dense(1,activation ='sigmoid')]) #Funcion de salida 

#adam optimizador,loss error cuadratico medio, acuracy precision 
model.compile(optimizer='adam',loss='mse',metrics=['binary_accuracy'])
model.fit(vec_pa,eti,epochs=500,verbose=1)
model.save("./modelos/prueba1.h5")
#signal1=signal1.reshape(1,signal1.shape[0])
#print(model.predict(vec))"""