import pandas as pd
import numpy as np
from numpy import matrix
import xlwt
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats
from scipy.fftpack import fft
from scipy.signal import butter, lfilter

def butter_lowpass(cutOff, fs, order=5):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog = True)
    return b, a

def butter_lowpass_filter(data, cutOff, fs, order=4):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = lfilter(b, a, data)
    return y

mydata=pd.read_excel('repeatedslides.xls')
mydata1=mydata.iloc[:,0:2]

mydata1=mydata1.rolling(20).mean() #moving average
#mydata1=mydata1.rolling(20).mean()
mydata1.as_matrix()  #converting the dataframe to Matrix

#breaking the matrix into column vectors {IMU}
col1=matrix(mydata1).transpose()[0].getA()[0]
col2=matrix(mydata1).transpose()[1].getA()[0]
col1=signal.medfilt(col1,5)
col2=signal.medfilt(col2,5)


cutOff = 0.25 #cutoff frequency in rad/s
fs = 4500/8.349 #sampling frequency in rad/s
order = 2
N = 4500
# sample spacing
T = 8.349 / 4500

yf = fft(col1)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

#plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))

#col1 = butter_lowpass_filter(col1, cutOff, fs, order)


plt.plot(col1)

plt.show()