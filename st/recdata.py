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

values=500

def butter_lowpass(cutOff, fs, order=1):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog = True)
    return b, a

def butter_lowpass_filter(data, cutOff, fs, order=4):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = lfilter(b, a, data)
    return y



ip="192.168.43.127"
socketno=3333



                                                  #no of diff inputs
j = 3 #no of diff value of each inputs
                                          #time wait for different action
                        
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
sheet1.write(k,0,0)
sheet1.write(k,1,1)
sheet1.write(k,2,2)
sheet1.write(k,3,3)


while(k<values):
    data,add = s.recvfrom(1024)#recv also if u use different version of python
    d1=data.split()
    w=int(d1[1][:])
    x=int(d1[2][:])
    y=int(d1[3][:])
    z=int(d1[4][:])
    print(k)
    sheet1.write(k+1,0,w)
    sheet1.write(k+1,1,x)
    sheet1.write(k+1,2,y)
    sheet1.write(k+1,3,z)
    k=k+1  
t2=time.time() 
t=t2-t1 
print(t)         
book.save('rl,%d.xls'%(j))
s.close()



mydata=pd.read_excel('rl,%d.xls'%(j))
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
'''
col1=col1[100:400]
col2=col2[100:400]
col3=col3[100:400]
col4=col4[100:400]
'''

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


plt.show()
