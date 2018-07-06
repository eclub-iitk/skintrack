import pandas as pd
import numpy as np
from numpy import matrix
import xlwt
import myfeat
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, lfilter 
from scipy.fftpack import fft

mydata=pd.read_excel('features.xls')
mydata1=mydata.iloc[:,:36]
mydata1.as_matrix() 
k = 0
i=1
"""
col=[[0] * 36] * 15
for i in range(36):
    col[:,i:i+1]=matrix(mydata1).transpose()[1].getA()[0][:,:]
"""
col1=matrix(mydata1).transpose()[i].getA()[0]
"""
col2=matrix(mydata1).transpose()[1].getA()[0]
col3=matrix(mydata1).transpose()[2].getA()[0]
col4=matrix(mydata1).transpose()[1].getA()[0]
col5=matrix(mydata1).transpose()[1].getA()[0]
col6=matrix(mydata1).transpose()[1].getA()[0]
col7=matrix(mydata1).transpose()[1].getA()[0]
col8=matrix(mydata1).transpose()[1].getA()[0]
col9=matrix(mydata1).transpose()[1].getA()[0]
col10=matrix(mydata1).transpose()[1].getA()[0]
col11=matrix(mydata1).transpose()[1].getA()[0]
col12=matrix(mydata1).transpose()[1].getA()[0]
col13=matrix(mydata1).transpose()[1].getA()[0]
col14=matrix(mydata1).transpose()[1].getA()[0]
col15=matrix(mydata1).transpose()[1].getA()[0]
col16=matrix(mydata1).transpose()[1].getA()[0]
col17=matrix(mydata1).transpose()[1].getA()[0]
col18=matrix(mydata1).transpose()[1].getA()[0]
col19=matrix(mydata1).transpose()[1].getA()[0]
col20=matrix(mydata1).transpose()[1].getA()[0]
col21=matrix(mydata1).transpose()[1].getA()[0]
col22=matrix(mydata1).transpose()[1].getA()[0]
col23=matrix(mydata1).transpose()[1].getA()[0]
col24=matrix(mydata1).transpose()[1].getA()[0]
col25=matrix(mydata1).transpose()[1].getA()[0]
col26=matrix(mydata1).transpose()[1].getA()[0]
col27=matrix(mydata1).transpose()[1].getA()[0]
col28=matrix(mydata1).transpose()[1].getA()[0]
col29=matrix(mydata1).transpose()[1].getA()[0]
col30=matrix(mydata1).transpose()[1].getA()[0]
col31=matrix(mydata1).transpose()[1].getA()[0]
col32=matrix(mydata1).transpose()[1].getA()[0]
col33=matrix(mydata1).transpose()[1].getA()[0]
col34=matrix(mydata1).transpose()[1].getA()[0]
col35=matrix(mydata1).transpose()[1].getA()[0]
col36=matrix(mydata1).transpose()[1].getA()[0]
"""
"""
mydata=pd.read_excel('features2.xls')
mydata1=mydata.iloc[:,:36]
k=-1
mydata1.as_matrix()
tcol=[]
for i in range(36):
    tcol[i]=matrix(mydata1).transpose()[1].getA()[0]
"""
tcol1=matrix(mydata1).transpose()[i].getA()[0]
"""
tcol2=matrix(mydata1).transpose()[1].getA()[0]
tcol3=matrix(mydata1).transpose()[1].getA()[0]
tcol4=matrix(mydata1).transpose()[1].getA()[0]
tcol5=matrix(mydata1).transpose()[1].getA()[0]
tcol6=matrix(mydata1).transpose()[1].getA()[0]
tcol7=matrix(mydata1).transpose()[1].getA()[0]
tcol8=matrix(mydata1).transpose()[1].getA()[0]
tcol9=matrix(mydata1).transpose()[1].getA()[0]
tcol10=matrix(mydata1).transpose()[1].getA()[0]
tcol11=matrix(mydata1).transpose()[1].getA()[0]
tcol12=matrix(mydata1).transpose()[1].getA()[0]
tcol13=matrix(mydata1).transpose()[1].getA()[0]
tcol14=matrix(mydata1).transpose()[1].getA()[0]
tcol15=matrix(mydata1).transpose()[1].getA()[0]
tcol16=matrix(mydata1).transpose()[1].getA()[0]
tcol17=matrix(mydata1).transpose()[1].getA()[0]
tcol18=matrix(mydata1).transpose()[1].getA()[0]
tcol19=matrix(mydata1).transpose()[1].getA()[0]
tcol20=matrix(mydata1).transpose()[1].getA()[0]
tcol21=matrix(mydata1).transpose()[1].getA()[0]
tcol22=matrix(mydata1).transpose()[1].getA()[0]
tcol23=matrix(mydata1).transpose()[1].getA()[0]
tcol24=matrix(mydata1).transpose()[1].getA()[0]
tcol25=matrix(mydata1).transpose()[1].getA()[0]
tcol26=matrix(mydata1).transpose()[1].getA()[0]
tcol27=matrix(mydata1).transpose()[1].getA()[0]
tcol28=matrix(mydata1).transpose()[1].getA()[0]
tcol29=matrix(mydata1).transpose()[1].getA()[0]
tcol30=matrix(mydata1).transpose()[1].getA()[0]
tcol31=matrix(mydata1).transpose()[1].getA()[0]
tcol32=matrix(mydata1).transpose()[1].getA()[0]
tcol33=matrix(mydata1).transpose()[1].getA()[0]
tcol34=matrix(mydata1).transpose()[1].getA()[0]
tcol35=matrix(mydata1).transpose()[1].getA()[0]
tcol36=matrix(mydata1).transpose()[1].getA()[0]
"""
plt.plot(col1,label = 'hover')
plt.plot(tcol1,label = 'touch')
plt.legend()
plt.grid()
plt.show()