# This python script listens on UDP port 3333
# for messages from the ESP32 board and prints them
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
from scipy.fftpack import fft
from scipy.signal import butter, lfilter 
ip="192.168.43.127"
socketno=3333
                                                  #no of diff inputs
s =  0                                    #no of diff value of each inputs
v =  3   #time wait for different action
                        
try :                                                          #AF_INET (IPv4 protocol) , AF_INET6 (IPv6 protocol)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)       #error -1 is returned
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
    # listen(queuewaiting_reciever), accept ,connect not used as udp mode
                                                               #Prevents error such as: “address already in use”.
except socket.error :

    print('Failed to create socket. Error Code : '  + ' Message ') 
    sys.exit()
try:
    s.bind((ip, socketno))
except socket.error:
    print('Bind failed. Error: '  ': ')
    sys.exit()

print('Server listening')


book = xlwt.Workbook(encoding="utf-8")
sheet1 = book.add_sheet("Sheet",cell_overwrite_ok=True)
t1 = time.time()
k = 0
while(k<500):
    data,add = s.recvfrom(1024)#recv also if u use different version of python
    d1=data.split()
    w=int(d1[1][:])
    x=int(d1[2][:])
    y=int(d1[3][:])
    z=int(d1[4][:])
    print(k,w,x,y,z)
    sheet1.write(k,0,w)
    sheet1.write(k,1,x)
    sheet1.write(k,2,y)
    sheet1.write(k,3,z)
    k=k+1  
t2=time.time() 
t=t2-t1 
print(t)         
book.save('trial%d.xls'%(v))
s.close()

mydata=pd.read_excel('trial%d.xls'%(v))
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
