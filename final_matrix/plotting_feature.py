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

a=['mean','mean','mean','mean','max','max','max','max','iqr','iqr','iqr','iqr','min','min','min','min','median','median','median','median','range','range','range','range','std','std','std','std','sum','sum','sum','sum','rms','rms','rms','rms']
# retrieving data...
mydata=pd.read_excel("secondindex==.xls")
mydata1=mydata.iloc[:,:37]
#print(np.shape(labels))
mydata1.as_matrix()  #converting the dataframe to Matrix
data2=matrix(mydata1).transpose()[36].getA()[0][:, np.newaxis]
for i in range(36):
    col1=matrix(mydata1).transpose()[i].getA()[0][:, np.newaxis]
    plt.grid()
    plt.scatter(data2,col1 , c='red')
    plt.title('feature %s.xls'%(a[i]))
    plt.show()


