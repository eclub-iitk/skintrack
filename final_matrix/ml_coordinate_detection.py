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
# ML parameters
gamma=0.1

# retrieving data...
mydata=pd.read_excel("secondindex==.xls")
mydata1=mydata.iloc[:,:37]
#print(np.shape(labels))
mydata1.as_matrix()  #converting the dataframe to Matrix
data1=matrix(mydata1).transpose()[0].getA()[0][:, np.newaxis]
for i in range(35):
    col1=matrix(mydata1).transpose()[i+1].getA()[0][:, np.newaxis]
    data1=np.hstack((data1,col1))

X=data1[:,:]
print(np.shape(X))
#print(X[0,:])
Xt=matrix(mydata1).transpose()[36].getA()[0][:, np.newaxis]
Y=data1[:,:]

print(np.shape(X))
print(np.shape(Xt))
print(np.shape(Y))
svr_rbf = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)

svr_rbf.fit(X, Xt)
y_rbf =svr_rbf.predict(Y)
print(y_rbf)

plt.plot(y_rbf,'ro',color='r',label='rbf')

plt.plot(Xt,'ro',color='g',label='orignal')

plt.grid()
plt.legend()
plt.show()
