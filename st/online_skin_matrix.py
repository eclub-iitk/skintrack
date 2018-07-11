import socket
import sys
import time
import myfeat
from scipy.stats import kurtosis, skew
from numpy import matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import numpy as np
import xlwt
from scipy.signal import butter, lfilter 
from scipy.fftpack import fft
import detect_peaks
from scipy.stats import iqr
from numpy import trapz

def slope_sign_change(sig,mph,mpd):
    sig=np.absolute(sig)
    n=len(peaks_indices(sig,mph,mpd))
    return n
def butter_lowpass(cutOff, fs, order=1):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog = True)
    return b, a

def butter_lowpass_filter(data, cutOff, fs, order=4):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def mad(arr):
    arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.absolute(arr - med))


def peaks_indices(sig,mph,mpd):
    indexes = detect_peaks.detect_peaks(sig, mph, mpd)
    return indexes


def mfft(sig,fs=100):
    y1=np.fft.fft(sig)
    N = len(y1)
    y1 = y1[0:N//2]
    fr = np.linspace(0,fs/2,N//2)
    return y1,fr


def max_freq(sig):
    n=len(sig)
    mf,fr=mfft(sig)
    return np.argmax(np.absolute(mf))
def butter_bandpass(sig,lowcut, highcut, fs, order=2):
   nyq = 0.5 * fs
   low = lowcut / nyq
   high = highcut / nyq
   b, a = butter(order, [low, high], btype='band')
   y=lfilter(b,a,sig)
   return y


def a2m(sig):
    n=len(sig)
    t=0
    H=[]
    while(t<n):
        H.append(float(sig[t]))
        t=t+1
    return H

def flex_feat(sig,lpass,hpass):

    k=0
    maxi=0
    while(k<len(sig)):
        if (sig[k]>hpass):
            j=0
        elif (sig[k]>maxi):
            maxi=sig[k]
        k=k+1

    if (maxi>lpass):
        return 2000
    return 0



def rms(sig):
    n=len(sig)
    summ=0
    i=0
    while (i<n):
        summ = summ + sig[i]**2
        i=i+1
    summ = summ**(0.5)
    return int((summ*1000)/n)


def peaks(sig,mph,mpd):
    indi=peaks_indices(sig,mph,mpd)
    n=len(indi)
    i=0
    peak=[]
    while (i<n):
        peak.append(sig[indi[i]])
        i=i+1
    return peak

def rang(sig):
    a=int(100*(max(sig)-min(sig)))
    return a

def check_predict(sig):
    i=1
    count=0
    while (i<3):
        if (sig[i]==sig[i-1]):

            count=count+1

        i=i+1
    if (count==2):
        return 1
    return 0

def IQR(sig):
    return iqr(sig)

def entropy(sig):
    ts = sig
    std_ts = np.std(ts)
    sample_entropy = ent.sample_entropy(ts, 4, 0.2 * std_ts)
    return sample_entropy


mydata=pd.read_excel('final3.xls')
feat=mydata.iloc[:,0:68]
feat.as_matrix()
label=mydata.iloc[:,68]
label.as_matrix()

x = np.array(feat)
y = np.array(label)

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.1)
model =KNeighborsClassifier(n_neighbors=5)
model =model.fit(x_train,y_train.ravel())
predictions=model.predict(x_test)
accuracy=accuracy_score(y_test,predictions) 
print ("Accuracy = ", accuracy)

svr_rbf = svm.SVR(kernel='rbf', C=1e3, gamma=0.3)
y_rbf = svr_rbf.fit(x, y)


ip="192.168.43.127"
socketno=3333
y=[]
my_feat=np.zeros((68,1))
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
    fs = 238 #sampling frequency in rad/s
    order = 3 #order of filter

    col1 = butter_lowpass_filter(col1, cutOff, fs, order)
    col2 = butter_lowpass_filter(col2, cutOff, fs, order)
    col3 = butter_lowpass_filter(col3, cutOff, fs, order)
    col4 = butter_lowpass_filter(col4, cutOff, fs, order)

    col1 = signal.savgol_filter(col1,201,3)
    col2 = signal.savgol_filter(col2,201,3)
    col3 = signal.savgol_filter(col3,201,3)
    col4 = signal.savgol_filter(col4,201,3)

    col1=col1[100:]
    col2=col2[100:]
    col3=col3[100:]
    col4=col4[100:]

    col1=col1-np.mean(col1)
    col2=col2-np.mean(col2)
    col3=col3-np.mean(col3)
    col4=col4-np.mean(col4)

    

   ##print (col1,"  ",col2,"  ",col3,"  ",col4,"  ",col5,"  ",col6,"  ",col7,"  ",col8,"  ",col9)
    arr_p1 = col1
    arr_p2 = col2
    arr_m1 = col3
    arr_m2 = col4

    max_p1 = np.amax(arr_p1)
    max_p2 = np.amax(arr_p2)
    max_m1 = np.amax(arr_m1)
    max_m2 = np.amax(arr_m2)
    my_feat[0] = max_p1
    my_feat[1] = max_p2
    my_feat[2] = max_m1
    my_feat[3] = max_m2
  
    
    min_p1 = np.amin(arr_p1)

    min_p2 = np.amin(arr_p2)

    min_m1 = np.amin(arr_m1)

    min_m2 = np.amin(arr_m2)
    my_feat[4] = max_p1
    my_feat[5] = max_p2
    my_feat[6] = max_m1
    my_feat[7] = max_m2
  

    
#slope sign change
    mph=0.0001
    mpd=30
    nslopep1=slope_sign_change(arr_p1, mph, mpd)
    nslopep2=slope_sign_change(arr_p1, mph, mpd)
    nslopem1=slope_sign_change(arr_p1, mph, mpd)
    nslopem2=slope_sign_change(arr_p1, mph, mpd)
    my_feat[8] = nslopep1
    my_feat[9] = nslopep2
    my_feat[10] = nslopem1
    my_feat[11] = nslopem2
    

#median np.absoluteolute deviation
    f81=mad(col1)
    f82=mad(col2)
    f83=mad(col3)
    f84=mad(col4)
    my_feat[12] = f81
    my_feat[13] = f82
    my_feat[14] = f83
    my_feat[15] = f84
    
#iqr

    f91=iqr(col1)
    f92=iqr(col2)
    f93=iqr(col3)
    f94=iqr(col4)
    my_feat[16] = f91
    my_feat[17] = f92
    my_feat[18] = f93
    my_feat[19] = f94
    


#skew
    f101=skew(col1)
    f102=skew(col2)
    f103=skew(col3)
    f104=skew(col4)
    my_feat[20] = f101
    my_feat[21] = f102
    my_feat[22] = f103
    my_feat[23] = f104
    


#kurtosis
    f111=kurtosis(col1)
    f112=kurtosis(col2)
    f113=kurtosis(col3)
    f114=kurtosis(col4)
    my_feat[24] = f111
    my_feat[25] = f112
    my_feat[26] = f113
    my_feat[27] = f114
    
    
#skew_freq_domain
    f121=np.absolute(skew(col1))
    f122=np.absolute(skew(col2))
    f123=np.absolute(skew(col3))
    f124=np.absolute(skew(col4))
    my_feat[28] = f121
    my_feat[29] = f122
    my_feat[30] = f123
    my_feat[31] = f124
    

#kurtosis_freq_domain
    f131=np.absolute(kurtosis(col1))
    f132=np.absolute(kurtosis(col2))
    f133=np.absolute(kurtosis(col3))
    f134=np.absolute(kurtosis(col4))
    my_feat[32] = f131
    my_feat[33] = f132
    my_feat[34] = f133
    my_feat[35] = f134
    


    med_p1 = np.median(arr_p1)

    med_p2 = np.median(arr_p2)

    med_m1 = np.median(arr_m1)

    med_m2 = np.median(arr_m2)
    my_feat[36] = med_p1
    my_feat[37] = med_p2
    my_feat[38] = med_m1
    my_feat[39] = med_m2
    
    

#standard deviation
    
    std_p1 = np.std(arr_p1)

    std_p2 = np.std(arr_p2)

    std_m1 = np.std(arr_m1)

    std_m2 = np.std(arr_m2)
    my_feat[40] = std_p1
    my_feat[41] = std_p2
    my_feat[42] = std_m1
    my_feat[43] = std_m2
    
#sum

    sum_p1 = np.sum(arr_p1)

    sum_p2 = np.sum(arr_p2)

    sum_m1 = np.sum(arr_m1)

    sum_m2 = np.sum(arr_m2)
    my_feat[44] = sum_p1
    my_feat[45] = sum_p2
    my_feat[46] = sum_m1
    my_feat[47] = sum_m2
    
    w=np.sum(peaks(col1,0.0001,30))
    x=np.sum(peaks(col1,0.0001,30))
    y=np.sum(peaks(col1,0.0001,30))
    z=np.sum(peaks(col1,0.0001,30))
    
    my_feat[48] = w
    my_feat[49] = x
    my_feat[50] = y
    my_feat[51] = z     

    
    w=np.size(peaks_indices(col1,0.0001,30))
    x=np.size(peaks_indices(col2,0.0001,30))
    y=np.size(peaks_indices(col3,0.0001,30))
    z=np.size(peaks_indices(col4,0.0001,30))
      
    my_feat[52] = w
    my_feat[53] = x
    my_feat[54] = y
    my_feat[55] = z     

    w=rang(col1)
    x=rang(col2)
    y=rang(col3)
    z=rang(col4)
      
    my_feat[56] = w
    my_feat[57] = x
    my_feat[58] = y
    my_feat[59] = z    

    #zero crossing
    def zcr(sig):
        
        c=0
        for x in np.nditer(sig):
            if ((x<0 and x>0) or (x>0 and x<0)):
                c=c+1
        return c

    w=zcr(col1)
    x=zcr(col2)
    y=zcr(col3)
    z=zcr(col4)
    
    my_feat[60] = w
    my_feat[61] = x
    my_feat[62] = y
    my_feat[63] = z   

    w=np.sum(np.absolute(col1))
    x=np.sum(np.absolute(col2))
    y=np.sum(np.absolute(col3))
    z=np.sum(np.absolute(col4))

    

    my_feat[64] = w
    my_feat[65] = x
    my_feat[66] = y
    my_feat[67] = z   
    a=['Single Tap','Double Tap','Still']  
    g=1
   
    result = sum(model.predict(my_feat.T))
    

    print(a[result])
    h=h+1
    