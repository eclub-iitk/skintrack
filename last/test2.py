import pandas as pd
import numpy as np
from numpy import matrix
import xlwt
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, lfilter 

mydata=pd.read_excel('hover7.xls')
mydata1=mydata.iloc[:,:4]

#mydata1=mydata1.rolling(20).mean() #moving average

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

N = 500
T = 2.08 / 500
"""x = np.linspace(0.0, N*T, N)
yf = fft(col1)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]),'g',label=' one')

x = np.linspace(0.0, N*T, N)
yf = fft(col2)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]),'r',label='two')

x = np.linspace(0.0, N*T, N)
yf = fft(col3)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]),'b',label='three')

x = np.linspace(0.0, N*T, N)
yf = fft(col4)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]),'m',label='four')
plt.legend()
plt.show()
"""


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
'''



col1=col1[20:]
col2=col2[20:]
col3=col3[20:]
col4=col4[20:]

col1 = signal.savgol_filter(col1,701,3)
col2 = signal.savgol_filter(col2,701,3)
col3 = signal.savgol_filter(col3,701,3)
col4 = signal.savgol_filter(col4,701,3)


diff_p1 = max(col1)-min(col1)
diff_m1 = max(col2)-min(col2)
diff_p2 = max(col3)-min(col3)
diff_m2 = max(col4)-min(col4)
print(diff_p1 ,diff_p2 ,diff_m1, diff_m2)
'''

plt.subplot(2,2,1)
plt.plot(col1,label = 'phase1')
plt.legend()
plt.grid()
#plt.ylim(-0.0015,0.0020)

plt.subplot(2,2,2)
plt.plot(col2,label = 'mag1')
plt.legend()
plt.grid()
#plt.ylim(-0.0007,0.0020)

plt.subplot(2,2,3)
plt.plot(col3,label='mag2')
plt.legend()
plt.grid()
#plt.ylim(-0.004,0.003)

plt.subplot(2,2,4)
plt.plot(col4,label = 'phase2')
plt.legend()
plt.grid()
#plt.ylim(-0.003,0.005)
#plt.ylim(-1,1)

plt.show()
