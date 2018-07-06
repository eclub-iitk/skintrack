import pandas as pd
import numpy as np
from numpy import matrix
import xlwt
import myfeat
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, lfilter 
from scipy.fftpack import fft

mydata=pd.read_excel('final_features_hover.xls')
mydata1=mydata.iloc[:,:36]
mydata1.as_matrix() 
mydata2=pd.read_excel('final_features_touch.xls')
mydata3=mydata2.iloc[:,:36]
mydata3.as_matrix()

for i in range(36):
    col1=matrix(mydata1).transpose()[i].getA()[0]
    plt.plot(col1,label = 'hover')
    col2=matrix(mydata3).transpose()[i].getA()[0]
    plt.plot(col2,label = 'touch')
    plt.legend()
    plt.grid()
    plt.show()
