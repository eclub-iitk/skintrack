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


# retrieving data...
mydata=pd.read_excel("final7.xls")
mydata1=mydata.iloc[:,:69]
#print(np.shape(labels))
mydata1.as_matrix()  #converting the dataframe to Matrix
data2=matrix(mydata1).transpose()[68].getA()[0][:, np.newaxis]
for i in range(68):
    col1=matrix(mydata1).transpose()[i].getA()[0][:, np.newaxis]
    plt.grid()
    plt.plot(col1)
    plt.title('feature%d.xls'%(i))
    plt.show()

