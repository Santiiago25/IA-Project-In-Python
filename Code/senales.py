import numpy as np
import matplotlib.pyplot as plt  #libreria Graficas

sig=np.zeros((68,5000)) #Vector de 2 filas x 15000 columnas, para el entrenamiento

sig[0,:] = np.loadtxt('Señal1.txt') #Cargar señal .txt
sig[1,:] = np.loadtxt('Señal2.txt') #Cargar señal .txt
sig[2,:] = np.loadtxt('Señal3.txt') #Cargar señal .txt
sig[3,:] = np.loadtxt('Señal4.txt') #Cargar señal .txt
sig[4,:] = np.loadtxt('Señal5.txt') #Cargar señal .txt
sig[5,:] = np.loadtxt('Señal6.txt') #Cargar señal .txt
sig[6,:] = np.loadtxt('Señal7.txt') #Cargar señal .txt
sig[7,:] = np.loadtxt('Señal8.txt') #Cargar señal .txt
sig[8,:] = np.loadtxt('Señal9.txt') #Cargar señal .txt
sig[9,:] = np.loadtxt('Señal10.txt') #Cargar señal .txt
sig[10,:] = np.loadtxt('Señal11.txt') #Cargar señal .txt
sig[11,:] = np.loadtxt('Señal12.txt') #Cargar señal .txt
sig[12,:] = np.loadtxt('Señal13.txt') #Cargar señal .txt
sig[13,:] = np.loadtxt('Señal14.txt') #Cargar señal .txt
sig[14,:] = np.loadtxt('Señal15.txt') #Cargar señal .txt
sig[15,:] = np.loadtxt('Señal16.txt') #Cargar señal .txt
sig[16,:] = np.loadtxt('Señal17.txt') #Cargar señal .txt
sig[17,:] = np.loadtxt('Señal18.txt') #Cargar señal .txt
sig[18,:] = np.loadtxt('Señal19.txt') #Cargar señal .txt
sig[19,:] = np.loadtxt('Señal20.txt') #Cargar señal .txt
sig[20,:] = np.loadtxt('Señal21.txt') #Cargar señal .txt
sig[21,:] = np.loadtxt('Señal22.txt') #Cargar señal .txt
sig[22,:] = np.loadtxt('Señal23.txt') #Cargar señal .txt
sig[23,:] = np.loadtxt('Señal24.txt') #Cargar señal .txt
sig[24,:] = np.loadtxt('Señal25.txt') #Cargar señal .txt
sig[25,:] = np.loadtxt('Señal26.txt') #Cargar señal .txt
sig[26,:] = np.loadtxt('Señal27.txt') #Cargar señal .txt
sig[27,:] = np.loadtxt('Señal28.txt') #Cargar señal .txt
sig[28,:] = np.loadtxt('Señal29.txt') #Cargar señal .txt
sig[30,:] = np.loadtxt('Señal30.txt') #Cargar señal .txt
sig[31,:] = np.loadtxt('Señal31.txt') #Cargar señal .txt
sig[32,:] = np.loadtxt('Señal32.txt') #Cargar señal .txt
sig[33,:] = np.loadtxt('Señal33.txt') #Cargar señal .txt
sig[34,:] = np.loadtxt('Señal34.txt') #Cargar señal .txt

sig[35,:] = np.loadtxt('Señal41.txt') #Cargar señal .txt
sig[36,:] = np.loadtxt('Señal42.txt') #Cargar señal .txt
sig[37,:] = np.loadtxt('Señal43.txt') #Cargar señal .txt
sig[38,:] = np.loadtxt('Señal44.txt') #Cargar señal .txt
sig[39,:] = np.loadtxt('Señal45.txt') #Cargar señal .txt
sig[40,:] = np.loadtxt('Señal46.txt') #Cargar señal .txt
sig[41,:] = np.loadtxt('Señal47.txt') #Cargar señal .txt
sig[42,:] = np.loadtxt('Señal48.txt') #Cargar señal .txt
sig[43,:] = np.loadtxt('Señal49.txt') #Cargar señal .txt
sig[44,:] = np.loadtxt('Señal50.txt') #Cargar señal .txt
sig[45,:] = np.loadtxt('Señal51.txt') #Cargar señal .txt
sig[46,:] = np.loadtxt('Señal52.txt') #Cargar señal .txt
sig[47,:] = np.loadtxt('Señal53.txt') #Cargar señal .txt
sig[48,:] = np.loadtxt('Señal54.txt') #Cargar señal .txt
sig[49,:] = np.loadtxt('Señal55.txt') #Cargar señal .txt
sig[50,:] = np.loadtxt('Señal56.txt') #Cargar señal .txt
sig[51,:] = np.loadtxt('Señal57.txt') #Cargar señal .txt
sig[52,:] = np.loadtxt('Señal58.txt') #Cargar señal .txt
sig[53,:] = np.loadtxt('Señal59.txt') #Cargar señal .txt
sig[54,:] = np.loadtxt('Señal60.txt') #Cargar señal .txt
sig[55,:] = np.loadtxt('Señal61.txt') #Cargar señal .txt
sig[56,:] = np.loadtxt('Señal62.txt') #Cargar señal .txt
sig[57,:] = np.loadtxt('Señal63.txt') #Cargar señal .txt
sig[58,:] = np.loadtxt('Señal64.txt') #Cargar señal .txt
sig[59,:] = np.loadtxt('Señal65.txt') #Cargar señal .txt
sig[60,:] = np.loadtxt('Señal66.txt') #Cargar señal .txt
sig[61,:] = np.loadtxt('Señal67.txt') #Cargar señal .txt
sig[62,:] = np.loadtxt('Señal68.txt') #Cargar señal .txt
sig[63,:] = np.loadtxt('Señal69.txt') #Cargar señal .txt
sig[64,:] = np.loadtxt('Señal70.txt') #Cargar señal .txt
sig[65,:] = np.loadtxt('Señal71.txt') #Cargar señal .txt
sig[66,:] = np.loadtxt('Señal72.txt') #Cargar señal .txt
sig[67,:] = np.loadtxt('Señal73.txt') #Cargar señal .txt

np.savetxt("Matriz.txt",sig)

