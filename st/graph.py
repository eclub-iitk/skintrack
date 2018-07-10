import pandas as pd
import numpy as np
from numpy import matrix
import xlwt
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, lfilter 
from scipy.fftpack import fft

def butter_lowpass(cutOff, fs, order=1):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog = True)
    return b, a

def butter_lowpass_filter(data, cutOff, fs, order=4):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = lfilter(b, a, data)
    return y

a=['rl']
for j in range(1):
    for i in range(10):
        mydata=pd.read_excel('%s,%d.xls'%(a[j],i+2))
       
        
        mydata1=mydata.iloc[:,:4]



        mydata1.as_matrix()  #converting the dataframe to Matrix

        #breaking the matrix into column vectors {IMU}
        col1=matrix(mydata1).transpose()[0].getA()[0]
        col2=matrix(mydata1).transpose()[1].getA()[0]
        col3=matrix(mydata1).transpose()[2].getA()[0]
        col4=matrix(mydata1).transpose()[3].getA()[0]

        col1=signal.medfilt(col1,7)
        col2=signal.medfilt(col2,7)
        col3=signal.medfilt(col3,7)
        col4=signal.medfilt(col4,7)


        cutOff =  5#cutoff frequency in rad/s
        fs = 238 #sampling frequency in rad/s
        order = 3 #order of filter



        col1 = butter_lowpass_filter(col1, cutOff, fs, order)
        col2 = butter_lowpass_filter(col2, cutOff, fs, order)
        col3 = butter_lowpass_filter(col3, cutOff, fs, order)
        col4 = butter_lowpass_filter(col4, cutOff, fs, order)

        print(np.shape(col1))

        col1 = signal.savgol_filter(col1,201,3)
        col2 = signal.savgol_filter(col2,201,3)
        col3 = signal.savgol_filter(col3,201,3)
        col4 = signal.savgol_filter(col4,201,3)

        col1=col1-np.mean(col1)
        col2=col2-np.mean(col2)
        col3=col3-np.mean(col3)
        col4=col4-np.mean(col4)

        plt.subplot(2,2,1)
        plt.plot(col1,label = 'phase1')
        plt.legend()
        plt.grid()
        #plt.ylim(0.0148,0.0162)

        plt.subplot(2,2,2)
        plt.plot(col2,label = 'mag1')
        plt.legend()
        plt.grid()
        #plt.ylim(0.0036,0.0052)

        plt.subplot(2,2,3)
        plt.plot(col3,label='mag2')
        plt.legend()
        plt.grid()
        #plt.ylim(0.0134,0.0144)

        plt.subplot(2,2,4)
        plt.plot(col4,label = 'phase2')
        plt.legend()
        plt.grid()
        #plt.ylim(0.0215,0.0245)
        plt.title('%s,%d.xls'%(a[j],i+1))

        plt.show()
