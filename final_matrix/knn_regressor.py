import socket
import sys
import csv
import xlwt
import time
import pandas as pd
import numpy as np
from numpy import matrix
import xlwt
import matplotlib.pyplot as plt
from scipy import signal
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
# ML parameters

# retrieving data...
mydata=pd.read_excel("secondindex==.xls")
mydata1=mydata.iloc[:,:37]
#print(np.shape(labels))
mydata1.as_matrix()  #converting the dataframe to Matrix
data2=matrix(mydata1).transpose()[36].getA()[0][:, np.newaxis]
data1=matrix(mydata1).transpose()[0].getA()[0][:, np.newaxis]
for i in range(35):
    col1=matrix(mydata1).transpose()[i+1].getA()[0][:, np.newaxis]
    data1=np.hstack((data1,col1))

X=data1[:,:]
print(np.shape(X))
Xt=data2[:,:]
print(np.shape(Xt))
Y=data1[:,:] 
print(np.shape(Y))

neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(X, Xt) 
Yt=neigh.predict(Y)
print(np.shape(Yt))
print(Yt)
plt.plot(Xt,'ro',color='m',label='orig')

plt.plot(Yt,'ro',color='b',label='knn')
plt.title('knn_regressor')
plt.grid()
plt.legend()
plt.show()
