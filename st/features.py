import pandas as pd
import numpy as np
from numpy import matrix
import xlwt
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, lfilter 
from scipy.fftpack import fft
import detect_peaks
from scipy.stats import iqr
from scipy.stats import kurtosis, skew
from numpy import trapz

def slope_sign_change(sig,mph,mpd):
    sig=abs(sig)
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
    return np.median(np.abs(arr - med))


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
    return np.argmax(abs(mf))
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
    a=(100*(max(sig)-min(sig)))
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


book = xlwt.Workbook(encoding="utf-8")
sheet1 = book.add_sheet("Sheet",cell_overwrite_ok=True)
a=['nt','2t','hover']
for j in range(90):
    sheet1.write(0,j,j)
for j in range(3):
    for i in range(50):
        mydata=pd.read_excel('%s,%d.xls'%(a[j],i+1))
        mydata1=mydata.iloc[:,:4]
        mydata1.as_matrix() 
        
        col1=matrix(mydata1).transpose()[0].getA()[0]
        col2=matrix(mydata1).transpose()[1].getA()[0]
        col3=matrix(mydata1).transpose()[2].getA()[0]
        col4=matrix(mydata1).transpose()[3].getA()[0]
        col1=signal.medfilt(col1[:],7)
        col2=signal.medfilt(col2[:],7)
        col3=signal.medfilt(col3[:],7)
        col4=signal.medfilt(col4[:],7)        

        
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


        col1=col1[100:len(col1)]
        col2=col2[100:len(col1)]
        col3=col3[100:len(col1)]
        col4=col4[100:len(col1)]


        col1=col1-np.mean(col1)
        col2=col2-np.mean(col2)
        col3=col3-np.mean(col3)
        col4=col4-np.mean(col4)

        
        arr_p1 = col1
        arr_p2 = col2
        arr_m1 = col3
        arr_m2 = col4


        max_p1 = np.amax(arr_p1)

        max_p2 = np.amax(arr_p2)

        max_m1 = np.amax(arr_m1)

        max_m2 = np.amax(arr_m2)
        sheet1.write(i+1+50*j,0,max_p1)
        sheet1.write(i+1+50*j,1,max_p2)
        sheet1.write(i+1+50*j,2,max_m1)
        sheet1.write(i+1+50*j,3,max_m2)

        
        min_p1 = np.amin(arr_p1)

        min_p2 = np.amin(arr_p2)

        min_m1 = np.amin(arr_m1)

        min_m2 = np.amin(arr_m2)
        sheet1.write(i+1+50*j,4,min_p1)
        sheet1.write(i+1+50*j,5,min_p2)
        sheet1.write(i+1+50*j,6,min_m1)
        sheet1.write(i+1+50*j,7,min_m2)


#slope sign change
        nslopep1=slope_sign_change(arr_p1, 0.0001, 30)
        nslopep2=slope_sign_change(arr_p1, 0.0001, 30)
        nslopem1=slope_sign_change(arr_p1, 0.0001, 30)
        nslopem2=slope_sign_change(arr_p1, 0.0001, 30)
        sheet1.write(i+1+50*j,8,nslopep1)
        sheet1.write(i+1+50*j,9,nslopep2)
        sheet1.write(i+1+50*j,10,nslopem1)
        sheet1.write(i+1+50*j,11,nslopem2)

#median absolute deviation
        f81=mad(col1)
        f82=mad(col2)
        f83=mad(col3)
        f84=mad(col4)
        sheet1.write(i+1+50*j,12,f81)
        sheet1.write(i+1+50*j,13,f82)
        sheet1.write(i+1+50*j,14,f83)
        sheet1.write(i+1+50*j,15,f84)

#iqr

        f91=iqr(col1)
        f92=iqr(col2)
        f93=iqr(col3)
        f94=iqr(col4)
        sheet1.write(i+1+50*j,16,f91)
        sheet1.write(i+1+50*j,17,f92)
        sheet1.write(i+1+50*j,18,f93)
        sheet1.write(i+1+50*j,19,f94)


#skew
        f101=skew(col1)
        f102=skew(col2)
        f103=skew(col3)
        f104=skew(col4)
        sheet1.write(i+1+50*j,20,f101)
        sheet1.write(i+1+50*j,21,f102)
        sheet1.write(i+1+50*j,22,f103)
        sheet1.write(i+1+50*j,23,f104)


#kurtosis
        f111=kurtosis(col1)
        f112=kurtosis(col2)
        f113=kurtosis(col3)
        f114=kurtosis(col4)
        sheet1.write(i+1+50*j,24,f111)
        sheet1.write(i+1+50*j,25,f112)
        sheet1.write(i+1+50*j,26,f113)
        sheet1.write(i+1+50*j,27,f114)

        
#skew_freq_domain
        f121=abs(skew(col1))
        f122=abs(skew(col2))
        f123=abs(skew(col3))
        f124=abs(skew(col4))
        sheet1.write(i+1+50*j,28,f121)
        sheet1.write(i+1+50*j,29,f122)
        sheet1.write(i+1+50*j,30,f123)
        sheet1.write(i+1+50*j,31,f124)
        

#kurtosis_freq_domain
        f131=abs(kurtosis(col1))
        f132=abs(kurtosis(col2))
        f133=abs(kurtosis(col3))
        f134=abs(kurtosis(col4))
        sheet1.write(i+1+50*j,32,f131)
        sheet1.write(i+1+50*j,33,f132)
        sheet1.write(i+1+50*j,34,f133)
        sheet1.write(i+1+50*j,35,f134)


        med_p1 = np.median(arr_p1)

        med_p2 = np.median(arr_p2)

        med_m1 = np.median(arr_m1)

        med_m2 = np.median(arr_m2)
        sheet1.write(i+1+50*j,36,med_p1)
        sheet1.write(i+1+50*j,37,med_p2)
        sheet1.write(i+1+50*j,38,med_m1)
        sheet1.write(i+1+50*j,39,med_m2)
    
    #range
        
    
    #standard deviation
         
        std_p1 = np.std(arr_p1)

        std_p2 = np.std(arr_p2)

        std_m1 = np.std(arr_m1)

        std_m2 = np.std(arr_m2)
        sheet1.write(i+1+50*j,40,std_p1)
        sheet1.write(i+1+50*j,41,std_p2)
        sheet1.write(i+1+50*j,42,std_m1)
        sheet1.write(i+1+50*j,43,std_m2)

    #sum

        sum_p1 = np.sum(arr_p1)

        sum_p2 = np.sum(arr_p2)

        sum_m1 = np.sum(arr_m1)

        sum_m2 = np.sum(arr_m2)
        sheet1.write(i+1+50*j,44,sum_p1)
        sheet1.write(i+1+50*j,45,sum_p2)
        sheet1.write(i+1+50*j,46,sum_m1)
        sheet1.write(i+1+50*j,47,sum_m2)
        
        w=sum(peaks(col1,0.0001,30))
        x=sum(peaks(col2,0.0001,30))
        y=sum(peaks(col3,0.0001,30))
        z=sum(peaks(col4,0.0001,30))

        sheet1.write(i+1+50*j,48,w)
        sheet1.write(i+1+50*j,49,x)
        sheet1.write(i+1+50*j,50,y)
        sheet1.write(i+1+50*j,51,z)
    
        w=len(peaks_indices(col1,0.0001,30))
        x=len(peaks_indices(col2,0.0001,30))
        y=len(peaks_indices(col3,0.0001,30))
        z=len(peaks_indices(col4,0.0001,30))

        sheet1.write(i+1+50*j,52,w)
        sheet1.write(i+1+50*j,53,x)
        sheet1.write(i+1+50*j,54,y)
        sheet1.write(i+1+50*j,55,z)

        w=rang(col1)
        x=rang(col2)
        y=rang(col3)
        z=rang(col4)

        sheet1.write(i+1+50*j,56,w)
        sheet1.write(i+1+50*j,57,x)
        sheet1.write(i+1+50*j,58,y)
        sheet1.write(i+1+50*j,59,z)

        #zero crossing
        def zcr(sig):
          
            c=0
            for i in range(len(sig)-1):
                if ((sig[i]<0 and sig[i+1]>0) or (sig[i]>0 and sig[i+1]<0)):
                    c=c+1
            return c

        w=zcr(col1)
        x=zcr(col2)
        y=zcr(col3)
        z=zcr(col4)
      

        sheet1.write(i+1+50*j,60,w)
        sheet1.write(i+1+50*j,61,x)
        sheet1.write(i+1+50*j,62,y)
        sheet1.write(i+1+50*j,63,z)

        w=sum(abs(col1))
        x=sum(abs(col2))
        y=sum(abs(col3))
        z=sum(abs(col4))

        

        sheet1.write(i+1+50*j,64,w)
        sheet1.write(i+1+50*j,65,x)
        sheet1.write(i+1+50*j,66,y)
        sheet1.write(i+1+50*j,67,z)




        sheet1.write(i+1+50*j,68,j)

book.save("final3.xls")    
        