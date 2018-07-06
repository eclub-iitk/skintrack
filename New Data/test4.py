import pandas as pd
import numpy as np
from numpy import matrix
import xlwt
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, lfilter 
from scipy.fftpack import fft

for i in range(5):
    mydata=pd.read_excel('20hover%d.xls'%(i))
    mydata1=mydata.iloc[:,:4]

    #mydata1=mydata1.rolling(20).mean() #moving average

    mydata1.as_matrix()  #converting the dataframe to Matrix

    #breaking the matrix into column vectors {IMU}
    col1=matrix(mydata1).transpose()[0].getA()[0]
    col2=matrix(mydata1).transpose()[1].getA()[0]
    col3=matrix(mydata1).transpose()[2].getA()[0]
    col4=matrix(mydata1).transpose()[3].getA()[0]
    col1=signal.medfilt(col1[:],7)
    col2=signal.medfilt(col2[:],7)
    col3=signal.medfilt(col3[:],7)
    col4=signal.medfilt(col4[:],7)
    col1=col1[20:]
    col2=col2[20:]
    col3=col3[20:]
    col4=col4[20:]
    col1 = signal.savgol_filter(col1,401,3)
    col2 = signal.savgol_filter(col2,401,3)
    col3 = signal.savgol_filter(col3,401,3)
    col4 = signal.savgol_filter(col4,401,3)

    plt.subplot(2,2,1)
    plt.plot(col1,label = 'phase1')
    plt.legend()
    plt.grid()
    #plt.ylim(80,130)

    plt.subplot(2,2,2)
    plt.plot(col2,label = 'mag1')
    plt.legend()
    plt.grid()
    #plt.ylim(4,10)

    plt.subplot(2,2,3)
    plt.plot(col3,label='phase2')
    plt.legend()
    plt.grid()
    #plt.ylim(10,15)

    plt.subplot(2,2,4)
    plt.plot(col4,label = 'mag2')
    plt.legend()
    plt.grid()
    #plt.ylim(-28,-20)

    plt.show()

