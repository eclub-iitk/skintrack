import socket
import sys
import time
import xlwt
import myfeat
import pandas as pd
import numpy as np
import train_first_index
import train_second_index
from scipy.stats import kurtosis, skew
from numpy import matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.stats import kurtosis, skew
import pickle
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft
from scipy.signal import butter, lfilter
mydata=pd.read_excel('secondindex==.xls')
feat=mydata.iloc[:,0:36]
feat.as_matrix()
label=mydata.iloc[:,36]
label.as_matrix()

x = np.array(feat)
y = np.array(label)

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2)
model =KNeighborsClassifier(n_neighbors=3)
model.fit(x_train,y_train.ravel())
predictions=model.predict(x_test)
accuracy=accuracy_score(y_test,predictions) 



ip="192.168.43.127"
socketno=3333
y=[]
my_feat=np.zeros((36,1))
try :
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

except socket.error :
    print ('Failed to create socket. Error Code : ')
    sys.exit()

try:
    s.bind((ip, 3333))
   
except socket.error:
    print ('Bind failed. Error:"')
    sys.exit()


i=0
h=0

w1=np.zeros((500))
x1=np.zeros((500))
y1=np.zeros((500))
z1=np.zeros((500))
t1=time.time()
while(i<600):
    data,add = s.recvfrom(1024)#recv also if u use different version of python
    d1=data.split()
    w=int(d1[1][:])
    x=int(d1[2][:])
    y=int(d1[3][:])
    z=int(d1[4][:])
    if(i>99):
        w1[i-100]=w
        x1[i-100]=x
        y1[i-100]=y
        z1[i-100]=z
    #print i
    i=i+1

print (time.time()-t1)
while(1):

    w111 = np.zeros(550)
    x111 = np.zeros(550)
    y111 = np.zeros(550)
    z111 = np.zeros(550)

    w11=np.zeros((50))
    x11=np.zeros((50))
    y11=np.zeros((50))
    z11=np.zeros((50))
    i=0
    t2=time.time()
    while(i<50):
        data,add = s.recvfrom(1024)
        d1=data.split()
        w=int(d1[1][:])
        x=int(d1[2][:])
        y=int(d1[3][:])
        z=int(d1[4][:])

        w11[i]=w
        x11[i]=x
        y11[i]=y
        z11[i]=z
        #print i
        i=i+1
    w111=np.concatenate((w1, w11), axis=0)
    w1=w111[50:550]
    x111=np.concatenate((x1,x11), axis=0)
    x1=x111[50:550]
    y111=np.concatenate((y1, y11), axis=0)
    y1=y111[50:550]
    z111=np.concatenate((z1, z11), axis=0)
    z1=z111[50:550]
    col1=w1
    col2=x1
    col3=y1
    col4=z1

    col1=signal.medfilt(col1,7)
    col2=signal.medfilt(col2,7)
    col3=signal.medfilt(col3,7)
    col4=signal.medfilt(col4,7)

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

   ##print (col1,"  ",col2,"  ",col3,"  ",col4,"  ",col5,"  ",col6,"  ",col7,"  ",col8,"  ",col9)
    arr_p1 = col1
    arr_p2 = col2
    arr_m1 = col3
    arr_m2 = col4
#mean

    mean_p1 = np.mean(arr_p1)
    mean_p2 = np.mean(arr_p2)
    mean_m1 = np.mean(arr_m1)
    mean_m2 = np.mean(arr_m2)

    my_feat[0]=mean_p1
    my_feat[1]=mean_p2
    my_feat[2]=mean_m1
    my_feat[3]=mean_m2

#max

    max_p1 = np.amax(arr_p1)
    max_p2 = np.amax(arr_p2)
    max_m1 = np.amax(arr_m1)
    max_m2 = np.amax(arr_m2)

    my_feat[4]=max_p1
    my_feat[5]=max_p2
    my_feat[6]=max_m1
    my_feat[7]=max_m2

#iqr
    iqr_p1 = myfeat.IQR(arr_p1)
    iqr_p2 = myfeat.IQR(arr_p2)
    iqr_m1 = myfeat.IQR(arr_m1)
    iqr_m2 = myfeat.IQR(arr_m2)

    my_feat[8]=iqr_p1
    my_feat[9]=iqr_p2
    my_feat[10]=iqr_m1
    my_feat[11]=iqr_m2

#min
    min_p1 = np.amin(arr_p1)
    min_p2 = np.amin(arr_p2)
    min_m1 = np.amin(arr_m1)
    min_m2 = np.amin(arr_m2)

    my_feat[12]=min_p1
    my_feat[13]=min_p2
    my_feat[14]=min_m1
    my_feat[15]=min_m2

#median

    med_p1 = np.median(arr_p1)
    med_p2 = np.median(arr_p2)
    med_m1 = np.median(arr_m1)
    med_m2 = np.median(arr_m2)

    my_feat[16]=med_p1
    my_feat[17]=med_p2
    my_feat[18]=med_m1
    my_feat[19]=med_m2


#range

    range_p1 = max_p1-min_p1
    range_p2 = max_p2-min_p2
    range_m1 = max_m1-min_m1
    range_m2 = max_m2-min_m2

    my_feat[20]=range_p1
    my_feat[21]=range_p2
    my_feat[22]=range_m1
    my_feat[23]=range_m2


#standard deviation

    std_p1 = np.std(arr_p1)
    std_p2 = np.std(arr_p2)
    std_m1 = np.std(arr_m1)
    std_m2 = np.std(arr_m2)

    my_feat[24]=std_p1
    my_feat[25]=std_p2
    my_feat[26]=std_m1
    my_feat[27]=std_m2

#sum

    sum_p1 = np.sum(arr_p1)
    sum_p2 = np.sum(arr_p2)
    sum_m1 = np.sum(arr_m1)
    sum_m2 = np.sum(arr_m2)

    my_feat[28]=sum_p1
    my_feat[29]=sum_p2
    my_feat[30]=sum_m1
    my_feat[31]=sum_m2

#rms
    rms_p1 = myfeat.rms(arr_p1)
    rms_p2 = myfeat.rms(arr_p2)
    rms_m1 = myfeat.rms(arr_m1)
    rms_m2 = myfeat.rms(arr_m2)

    my_feat[32]=rms_p1
    my_feat[33]=rms_p2
    my_feat[34]=rms_m1
    my_feat[35]=rms_m2

    result = model.predict(my_feat.T)
    print(result)
    print(h)
    h=h+1
    