import socket
import sys
import time
import xlwt
import myfeat
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from numpy import matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ip="192.168.43.127"
socketno=3333
y=[]

try :
    s1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s1.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

except socket.error, msg :
    print 'Failed to create socket. Error Code : ' + str(msg[0]) + ' Message ' + msg[1]
    sys.exit()

try:
    s1.bind((ip, 3333))
   
except socket.error , msg:
    print 'Bind failed. Error: ' + str(msg[0]) + ': ' + msg[1]
    sys.exit()


i=0

w1=np.zeros((300,1))
x1=np.zeros((300,1))
y1=np.zeros((300,1))
z1=np.zeros((300,1))
t1=time.time()
while(i<350):
    data,add = s.recvfrom(1024)#recv also if u use different version of python
    d1=data.split()
    w=int(d1[1][:])
    x=int(d1[2][:])
    y=int(d1[3][:])
    z=int(d1[4][:])
    if(i>49):
        w1[i-50]=w
        x1[i-50]=x
        y1[i-50]=y
        z1[i-50]=z
    #print i
    i=i+1

print (time.time()-t1)
while(1):

    w11=np.zeros((300,1))
    x11=np.zeros((300,1))
    y11=np.zeros((300,1))
    z11=np.zeros((300,1))
    i=0
    t2=time.time()
    while(i<50):
        d1 = s1.recvfrom(1024)
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
    w1=w111[50:350]
    x111=np.concatenate((x1,x11), axis=0)
    x1=x111[50:350]
    y111=np.concatenate((y1, y11), axis=0)
    y1=y111[50:350]
    z111=np.concatenate((z1, z11), axis=0)
    z1=z111[50:350]
    col1=w111
    col2=x111
    col3=y111
    col4=z111

    col1=signal.medfilt(col1,7)
    col2=signal.medfilt(col2,7)
    col3=signal.medfilt(col3,7)
    col4=signal.medfilt(col4,7)

    col1 = signal.savgol_filter(col1,401,3)
    col2 = signal.savgol_filter(col2,401,3)
    col3 = signal.savgol_filter(col3,401,3)
    col4 = signal.savgol_filter(col4,401,3)

   ##print (col1,"  ",col2,"  ",col3,"  ",col4,"  ",col5,"  ",col6,"  ",col7,"  ",col8,"  ",col9)
"""
    feat_data=np.zeros((36,1))


    tr1=myfeat.flex_feat(col5,9,20)
    tr2=myfeat.flex_feat(col6,7,20)
    tr3=myfeat.flex_feat(col7,13,20)
    tr4=myfeat.flex_feat(col8,9,20)
    tr5=myfeat.flex_feat(col9,100,200)

    #mfft

    y1,fr1=myfeat.mfft(col1)
    y2,fr2=myfeat.mfft(col2)
    y3,fr3=myfeat.mfft(col3)
    y4,fr4=myfeat.mfft(col4)




    #variance
    feat_data[4]=float(np.var(col1,ddof=1))
    feat_data[5]=float(np.var(col2,ddof=1))
    feat_data[6]=float(np.var(col3,ddof=1))
    feat_data[7]=float(np.var(col4,ddof=1))

    #max_freq

    feat_data[0]=1000*myfeat.max_freq(col1)
    feat_data[1]=1000*myfeat.max_freq(col2)
    feat_data[2]=1000*myfeat.max_freq(col3)
    feat_data[3]=1000*myfeat.max_freq(col4)

    #RMS

    feat_data[8]=myfeat.rms(col1)
    feat_data[9]=myfeat.rms(col2)
    feat_data[10]=myfeat.rms(col3)
    feat_data[11]=myfeat.rms(col4)

    #mean

    feat_data[17]=np.mean(col1)
    feat_data[18]=np.mean(col2)
    feat_data[19]=np.mean(col3)
    feat_data[20]=np.mean(col4)

    #sum_peaks

    feat_data[21]=sum(myfeat.peaks(col1,2,10))
    feat_data[22]=sum(myfeat.peaks(col2,2,10))
    feat_data[23]=sum(myfeat.peaks(col3,2,10))
    feat_data[24]=sum(myfeat.peaks(col4,2,10))

    #range
    feat_data[25]=myfeat.range(col1)
    feat_data[26]=myfeat.range(col2)
    feat_data[27]=myfeat.range(col3)
    feat_data[28]=myfeat.range(col4)

    feat_data[12]=tr1
    feat_data[13]=tr2
    feat_data[14]=tr3
    feat_data[15]=tr4
    feat_data[16]=tr5

    feat_data[29]=max(col1)
    feat_data[30]=max(col2)
    feat_data[31]=max(col3)
    feat_data[32]=max(col4)

    feat_data[33]=myfeat.mad(col1)
    feat_data[34]=myfeat.mad(col2)
    feat_data[35]=myfeat.mad(col3)
    feat_data[36]=myfeat.mad(col4)

    feat_data[37]=myfeat.IQR(col1)
    feat_data[38]=myfeat.IQR(col2)
    feat_data[39]=myfeat.IQR(col3)
    feat_data[40]=myfeat.IQR(col4)

    feat_data[41]=skew(col1)
    feat_data[42]=skew(col2)
    feat_data[43]=skew(col3)
    feat_data[44]=skew(col4)

    feat_data[45]=kurtosis(col1)
    feat_data[46]=kurtosis(col2)
    feat_data[47]=kurtosis(col3)
    feat_data[48]=kurtosis(col4)

    feat_data[49]=abs(skew(y1))
    feat_data[50]=abs(skew(y2))
    feat_data[51]=abs(skew(y3))
    feat_data[52]=abs(skew(y4))

    feat_data[53]=abs(kurtosis(y1))
    feat_data[54]=abs(kurtosis(y2))
    feat_data[55]=abs(kurtosis(y3))
    feat_data[56]=abs(kurtosis(y4))

    feat_data[57]=tr1
    feat_data[58]=tr2
    feat_data[59]=tr3
    feat_data[60]=tr4
    feat_data[61]=tr5

    feat_data[62]=min(col1)
    feat_data[63]=min(col2)
    feat_data[64]=min(col3)
    feat_data[65]=min(col4)

"""
    q[0]=int(clf.predict(feat_data.T))

    pv=int(q[0])
    l4=l3
    l3=l2
    l2=l1
    l1=pv

    if (l1==l2 and l2==l3 and l3==l4):
        a=myfeat.printout(l1)
        print a
