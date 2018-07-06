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
#from sklearn import svm
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

list1=[2.34,4.346,4.234]

y=[]

mydata=pd.read_excel('featdata.xls')
feat=mydata.iloc[:,0:66]
feat.as_matrix()
label=mydata.iloc[:, 66]
label.as_matrix()

x = np.array(feat)
y = np.array(label)
p=np.zeros((3,1))
q=np.zeros((3,1))
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.1)
clf =KNeighborsClassifier(n_neighbors=3)
#clf=svm.SVC(kernel='linear',C=1.0)
#clf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
clf=clf.fit(x_train,y_train.ravel())

predictions=clf.predict(x_test)
accuracy=accuracy_score(y_test,predictions)
#print x_test
#print(y_test,predictions)
print ("Accuracy = ", accuracy)


try :
    s1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    s1.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    s2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    s2.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

except socket.error, msg :
    print 'Failed to create socket. Error Code : ' + str(msg[0]) + ' Message ' + msg[1]
    sys.exit()


try:


    s1.bind(('192.168.0.107', 3334))
    s2.bind(('192.168.0.107', 3333))

except socket.error , msg:
    print 'Bind failed. Error: ' + str(msg[0]) + ': ' + msg[1]
    sys.exit()


i=0

p1=np.zeros((300,1))
r1=np.zeros((300,1))
p2=np.zeros((300,1))
r2=np.zeros((300,1))
fl1=np.zeros((300,1))
fl2=np.zeros((300,1))
fl3=np.zeros((300,1))
fl4=np.zeros((300,1))
fl5=np.zeros((300,1))

t1=time.time()
pv=1
l1=6
l2=6
l3=6
l4=6
while(i<300):

    d1 = s1.recvfrom(1024)
    data1 = d1[0]
    d2 = s2.recvfrom(1024)
    data2 = d2[0]

    idf1,str1,pitch1,roll1,flex1,flex2,flex3,flex4,flex5 = data1.split()

    idf2,str2,pitch2,roll2,flex11,flex22,flex33,flex44,flex55 = data2.split()

    p1[i]=float(pitch1)
    r1[i]=float(roll1)
    p2[i]=float(pitch2)
    r2[i]=float(roll2)

    fl1[i]=int(flex1)
    fl2[i]=int(flex2)
    fl3[i]=int(flex3)
    fl4[i]=int(flex4)
    fl5[i]=int(flex5)
    #print i
    i=i+1

print (time.time()-t1)
c=0
while(1):

    p11=np.zeros((300,1))
    r11=np.zeros((300,1))
    p22=np.zeros((300,1))
    r22=np.zeros((300,1))
    fl11=np.zeros((300,1))
    fl22=np.zeros((300,1))
    fl33=np.zeros((300,1))
    fl44=np.zeros((300,1))
    fl55=np.zeros((300,1))
    i=0

    t2=time.time()
    while(i<50):

        d1 = s1.recvfrom(1024)
        data1 = d1[0]
        d2 = s2.recvfrom(1024)
        data2 = d2[0]

        idf1,str1,pitch1,roll1,flex1,flex2,flex3,flex4,flex5 = data1.split()

        idf2,str2,pitch2,roll2,flex11,flex22,flex33,flex44,flex55 = data2.split()

        p11[i]=float(pitch1)
        r11[i]=float(roll1)
        p22[i]=float(pitch2)
        r22[i]=float(roll2)

        fl11[i]=int(flex1)
        fl22[i]=int(flex2)
        fl33[i]=int(flex3)
        fl44[i]=int(flex4)
        fl55[i]=int(flex5)
        #print i
        i=i+1


    p111=np.concatenate((p1, p11), axis=0)
    p1=p111[50:350]
    r111=np.concatenate((r1, r11), axis=0)
    r1=r111[50:350]
    p222=np.concatenate((p2, p22), axis=0)
    p2=p222[50:350]
    r222=np.concatenate((r2, r22), axis=0)
    r2=r222[50:350]

    fl111=np.concatenate((fl1, fl11), axis=0)
    fl1=fl111[50:350]
    fl222=np.concatenate((fl2, fl22), axis=0)
    fl2=fl222[50:350]
    fl333=np.concatenate((fl3, fl33), axis=0)
    fl3=fl333[50:350]
    fl444=np.concatenate((fl4, fl44), axis=0)
    fl4=fl444[50:350]
    fl555=np.concatenate((fl5, fl55), axis=0)
    fl5=fl555[50:350]



    col1=myfeat.a2m(p1)
    col2=myfeat.a2m(r1)
    col3=myfeat.a2m(p2)
    col4=myfeat.a2m(r2)

    col5=myfeat.a2m(fl1)
    col6=myfeat.a2m(fl2)
    col7=myfeat.a2m(fl3)
    col8=myfeat.a2m(fl4)
    col9=myfeat.a2m(fl5)

    ##print (col1,"  ",col2,"  ",col3,"  ",col4,"  ",col5,"  ",col6,"  ",col7,"  ",col8,"  ",col9)


    feat_data=np.zeros((66,1))


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


    q[0]=int(clf.predict(feat_data.T))

    pv=int(q[0])
    l4=l3
    l3=l2
    l2=l1
    l1=pv

    if (l1==l2 and l2==l3 and l3==l4):
        a=myfeat.printout(l1)
        print a
