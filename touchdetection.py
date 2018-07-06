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
gamma=0.3

# retrieving data...
mydata=pd.read_excel("5wristtap1_.xls")
mydata1=mydata.iloc[:,:3]

mydata1.as_matrix()  #converting the dataframe to Matrix

col1=matrix(mydata1).transpose()[0].getA()[0]
col2=matrix(mydata1).transpose()[1].getA()[0]
col3=matrix(mydata1).transpose()[2].getA()[0]

col1=np.array((col1),float)[:, np.newaxis]
col2=np.array((col2),float)[:, np.newaxis]
col3=np.array((col3),float)[:, np.newaxis]
# organizing the raw data

table=np.hstack((col1,col2))

data = np.array(table[50:len(table)-50,:], float)
data2 = np.array(col3[50:len(table)-50,:], float)
plt.plot(data,label='COPY')
#print(np.shape(data2))
X=data[:,:2]
Xt=data2[:,:]
X=np.hstack((X,X[:,:1]-X[:,1:2]))
Y=data[:,:2]
Y=np.hstack((Y,Y[:,:1]-Y[:,1:2]))
"""
# adding some more features data mean 
median1 
max 
min 
range=
sum
avg = np.average(data[],0)
std = np.std(data,0)
"""
# preparing the variables

#svr_rbf =svm.SVR(kernel='rbf', C=1e3, gamma=0.3)
print(np.shape(X),np.shape(Xt))
#svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.3)

#y_rbf = svr_rbf.fit(X, Xt).predict(X)
svr_rbf = svm.SVR(kernel='rbf', C=1e3, gamma=0.3)
#svr_lin = svm.SVR(kernel='linear', C=1e3)
#svr_poly = svm.SVR(kernel='poly', C=1e3, degree=2)
y_rbf = svr_rbf.fit(X, Xt).predict(X)
#y_lin = svr_lin.fit(X, Xt).predict(X)
#y_poly = svr_poly.fit(X, Xt).predict(X)

plt.plot(y_rbf,label='rbf')
#plt.plot(y_lin,label='linear')
#plt.plot(y_poly,label='poly')
plt.grid()
plt.legend()
plt.show()
