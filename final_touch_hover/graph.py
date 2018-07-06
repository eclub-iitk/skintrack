import pandas as pd
import numpy as np
from numpy import matrix
import xlwt
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, lfilter 
from scipy.fftpack import fft
a=['hover','touch']
for j in range(2):
    for i in range(50):
        mydata=pd.read_excel('%s%d.xls'%(a[j],i))
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
        col1 = signal.savgol_filter(col1,401,3)
        col2 = signal.savgol_filter(col2,401,3)
        col3 = signal.savgol_filter(col3,401,3)
        col4 = signal.savgol_filter(col4,401,3)
        def butter_lowpass(cutOff, fs, order=1):
            nyq = 0.5 * fs
            normalCutoff = cutOff / nyq
            b, a = butter(order, normalCutoff, btype='low', analog = True)
            return b, a

        def butter_lowpass_filter(data, cutOff, fs, order=4):
            b, a = butter_lowpass(cutOff, fs, order=order)
            y = lfilter(b, a, data)
            return y

        cutOff =  5#cutoff frequency in rad/s
        fs = 500/2.08 #sampling frequency in rad/s
        order = 3 #order of filter
        col1 = butter_lowpass_filter(col1, cutOff, fs, order)
        col2 = butter_lowpass_filter(col2, cutOff, fs, order)
        col3 = butter_lowpass_filter(col3, cutOff, fs, order)
        col4 = butter_lowpass_filter(col4, cutOff, fs, order)

        col1 = signal.savgol_filter(col1,201,3)
        col2 = signal.savgol_filter(col2,201,3)
        col3 = signal.savgol_filter(col3,201,3)
        col4 = signal.savgol_filter(col4,201,3)


      
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
        plt.title('%s%d.xls'%(a[j],i))
        plt.show()

