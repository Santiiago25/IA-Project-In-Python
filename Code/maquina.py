########## LIBRERÍAS A UTILIZAR ##########
#Se importan la librerias a utilizar
import numpy as np   # Libreria de matrices numeros enteros
import matplotlib.pyplot as plt  #libreria Graficas 
import pywt          #libreria wavelet
import joblib
from sklearn.svm import SVC

matriz_entre=np.loadtxt('Matriz.txt')

matriz_entre=matriz_entre[:,1500:4000]#recorte de la matriz

vec_pa=np.zeros((68,670)) #Vector de 2 filas x 15000 columnas, para el entrenamiento

def wcv(matriz_entre):
    for i in range(matriz_entre.shape[0]):
        #Transformada de Wavelet
        cA, cD = pywt.dwt(matriz_entre[i,:],'dmey')#dmey:aproximacion FIR
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
        
eti = np.zeros((68,1)) #etiquetas

for i in range(68):
    e1 = 1
    e2 =2
    s=0
    
    if i <34:
        s=e1
    else:
        s=e2
    eti[i]=s

wcv(matriz_entre)
plt.style.use('classic')
fig, ax = plt.subplots()
plt.xlabel('Muestras')
plt.ylabel('Amplitud')
ax.plot(vec_pa[2,0:2500], '-b', label='Sin Fatiga')
ax.plot(vec_pa[32,0:2500], '-r', label='Con Fatiga')
leg = ax.legend();


#Seleccionamos todas las columnas
X = vec_pa
#Defino los datos correspondientes a las etiquetas
y = eti

"""########## IMPLEMENTACIÓN DE MAQUINAS VECTORES DE SOPORTE ##########
from sklearn.model_selection import train_test_split
#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)"""

#Defino el algoritmo a utilizar

algoritmo = SVC(kernel = 'linear')
#Entreno el modelo
algoritmo.fit(X, y)
#Realizo una predicción
y_pred = algoritmo.predict(X)

joblib.dump(algoritmo, 'modelo_entrenado.pkl') # Guardo el modelo.

"""#Verifico la matriz de Confusión
from sklearn.metrics import confusion_matrix
matriz = confusion_matrix(y_test, y_pred)
print('Matriz de Confusión:')
print(matriz)"""

'''
#Calculo la precisión del modelo
from sklearn.metrics import precision_score
precision = precision_score(X[0,:], y_pred)
print('Precisión del modelo:')
print(precision)'''