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
from sklearn.kernel_ridge import KernelRidge
# ML parameters
gamma=0.1

# retrieving data...
mydata=pd.read_excel("x=.xls")
mydata1=mydata.iloc[:,:37]
#print(np.shape(labels))
mydata1.as_matrix()  #converting the dataframe to Matrix
data1=matrix(mydata1).transpose()[0].getA()[0][:, np.newaxis]
for i in range(35):
    col1=matrix(mydata1).transpose()[i+1].getA()[0][:, np.newaxis]
    data1=np.hstack((data1,col1))

X=data1
Xt=matrix(mydata1).transpose()[36].getA()[0][:, np.newaxis]
Y=data1

clf = KernelRidge(alpha = 3)
clf.fit(X,Xt)
Yt=clf.predict(Y)

plt.plot(Yt,'ro',color='g',label='predict')
plt.plot(Xt,'ro',color='r',label='orignal')

plt.grid()
plt.legend()
plt.show()
