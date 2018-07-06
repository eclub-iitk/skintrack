#By Ashok Kumar Chaudhary

""" This Code loads data and extract features from them
and label them."""
import myfeat
import pandas as pd
import numpy as np
from numpy import matrix
import xlwt
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt

book = xlwt.Workbook(encoding="utf-8")
sheet1 = book.add_sheet("Sheet 1")

j=0
noc=20
while (j<noc):

    i=1
    nof=50 #no of files

    while(i<=nof):
        #reading from each raw data
        if (i<10):
            l=str(0)+str(i)
        elif (i>=10):
            l=str(i)
        mydata=pd.read_excel('d%d%s.xls' %(j,l))
        print ("d%d%s.xls" %(j,l))

        mydata1=mydata.iloc[:,:10]

        #mydata1=mydata1.rolling(10).mean() #moving average

        mydata1.as_matrix()  #converting the dataframe to Matrix
        #breaking the matrix into column vectors {IMU}
        col1=matrix(mydata1).transpose()[0].getA()[0]
        col2=matrix(mydata1).transpose()[1].getA()[0]
        col3=matrix(mydata1).transpose()[2].getA()[0]
        col4=matrix(mydata1).transpose()[3].getA()[0]
        col5=matrix(mydata1).transpose()[4].getA()[0]
        col6=matrix(mydata1).transpose()[5].getA()[0]
        col7=matrix(mydata1).transpose()[6].getA()[0]
        col8=matrix(mydata1).transpose()[7].getA()[0]
        col9=matrix(mydata1).transpose()[8].getA()[0]

        #==============================================
        tr1=myfeat.flex_feat(col5,9,20)
        tr2=myfeat.flex_feat(col6,7,20)
        tr3=myfeat.flex_feat(col7,12,20)
        tr4=myfeat.flex_feat(col8,9,20)
        tr5=myfeat.flex_feat(col9,100,200)

        #mfft
        y1,fr1=myfeat.mfft(col1)
        y2,fr2=myfeat.mfft(col2)
        y3,fr3=myfeat.mfft(col3)
        y4,fr4=myfeat.mfft(col4)



        #variance
        f11=float(np.var(col1,ddof=1))
        f12=float(np.var(col2,ddof=1))
        f13=float(np.var(col3,ddof=1))
        f14=float(np.var(col4,ddof=1))

        #max_freq

        b1=1000*myfeat.max_freq(col1)
        b2=1000*myfeat.max_freq(col2)
        b3=1000*myfeat.max_freq(col3)
        b4=1000*myfeat.max_freq(col4)


        #RMS

        f31=myfeat.rms(col1)
        f32=myfeat.rms(col2)
        f33=myfeat.rms(col3)
        f34=myfeat.rms(col4)


        #mean

        f41=np.mean(col1)
        f42=np.mean(col2)
        f43=np.mean(col3)
        f44=np.mean(col4)


        #sum_peaks

        f51=sum(myfeat.peaks(col1,2,10))
        f52=sum(myfeat.peaks(col2,2,10))
        f53=sum(myfeat.peaks(col3,2,10))
        f54=sum(myfeat.peaks(col4,2,10))

        #range
        f61=myfeat.range(col1)
        f62=myfeat.range(col2)
        f63=myfeat.range(col3)
        f64=myfeat.range(col4)

        #max
        f71=max(col1)
        f72=max(col2)
        f73=max(col3)
        f74=max(col4)


        #median absolute deviation
        f81=myfeat.mad(col1)
        f82=myfeat.mad(col2)
        f83=myfeat.mad(col3)
        f84=myfeat.mad(col4)



        # interquartile range
        f91=myfeat.IQR(col1)
        f92=myfeat.IQR(col2)
        f93=myfeat.IQR(col3)
        f94=myfeat.IQR(col4)


        #skew
        f101=skew(col1)
        f102=skew(col2)
        f103=skew(col3)
        f104=skew(col4)

        #kurtosis
        f111=kurtosis(col1)
        f112=kurtosis(col2)
        f113=kurtosis(col3)
        f114=kurtosis(col4)


        #skew_freq_domain
        f121=abs(skew(y1))
        f122=abs(skew(y2))
        f123=abs(skew(y3))
        f124=abs(skew(y4))

        #kurtosis_freq_domain
        f131=abs(kurtosis(y1))
        f132=abs(kurtosis(y2))
        f133=abs(kurtosis(y3))
        f134=abs(kurtosis(y4))

        #min

        f141=min(col1)
        f142=min(col2)
        f143=min(col3)
        f144=min(col4)

        #print (int(f141),int(f142),int(f143),int(f144))

        #write
        sheet1.write((j*nof)+(i-1), 0, b1)
        sheet1.write((j*nof)+(i-1), 1, b2)
        sheet1.write((j*nof)+(i-1), 2, b3)
        sheet1.write((j*nof)+(i-1), 3, b4)

        sheet1.write((j*nof)+(i-1), 4, f11)
        sheet1.write((j*nof)+(i-1), 5, f12)
        sheet1.write((j*nof)+(i-1), 6, f13)
        sheet1.write((j*nof)+(i-1), 7, f14)

        sheet1.write((j*nof)+(i-1), 8, f31)
        sheet1.write((j*nof)+(i-1), 9, f32)
        sheet1.write((j*nof)+(i-1), 10, f33)
        sheet1.write((j*nof)+(i-1), 11, f34)

        sheet1.write((j*nof)+(i-1), 12, tr1)
        sheet1.write((j*nof)+(i-1), 13, tr2)
        sheet1.write((j*nof)+(i-1), 14, tr3)
        sheet1.write((j*nof)+(i-1), 15, tr4)
        sheet1.write((j*nof)+(i-1), 16, tr5)

        sheet1.write((j*nof)+(i-1), 17, f41)
        sheet1.write((j*nof)+(i-1), 18, f42)
        sheet1.write((j*nof)+(i-1), 19, f43)
        sheet1.write((j*nof)+(i-1), 20, f44)

        sheet1.write((j*nof)+(i-1), 21, f51)
        sheet1.write((j*nof)+(i-1), 22, f52)
        sheet1.write((j*nof)+(i-1), 23, f53)
        sheet1.write((j*nof)+(i-1), 24, f54)


        sheet1.write((j*nof)+(i-1), 25, f61)
        sheet1.write((j*nof)+(i-1), 26, f62)
        sheet1.write((j*nof)+(i-1), 27, f63)
        sheet1.write((j*nof)+(i-1), 28, f64)

        sheet1.write((j*nof)+(i-1), 29, f71)
        sheet1.write((j*nof)+(i-1), 30, f72)
        sheet1.write((j*nof)+(i-1), 31, f73)
        sheet1.write((j*nof)+(i-1), 32, f74)

        sheet1.write((j*nof)+(i-1), 33, f81)
        sheet1.write((j*nof)+(i-1), 34, f82)
        sheet1.write((j*nof)+(i-1), 35, f83)
        sheet1.write((j*nof)+(i-1), 36, f84)

        sheet1.write((j*nof)+(i-1), 37, f91)
        sheet1.write((j*nof)+(i-1), 38, f92)
        sheet1.write((j*nof)+(i-1), 39, f93)
        sheet1.write((j*nof)+(i-1), 40, f94)

        sheet1.write((j*nof)+(i-1), 41, f101)
        sheet1.write((j*nof)+(i-1), 42, f102)
        sheet1.write((j*nof)+(i-1), 43, f103)
        sheet1.write((j*nof)+(i-1), 44, f104)

        sheet1.write((j*nof)+(i-1), 45, f111)
        sheet1.write((j*nof)+(i-1), 46, f112)
        sheet1.write((j*nof)+(i-1), 47, f113)
        sheet1.write((j*nof)+(i-1), 48, f114)

        sheet1.write((j*nof)+(i-1), 49, f121)
        sheet1.write((j*nof)+(i-1), 50, f122)
        sheet1.write((j*nof)+(i-1), 51, f123)
        sheet1.write((j*nof)+(i-1), 52, f124)

        sheet1.write((j*nof)+(i-1), 53, f131)
        sheet1.write((j*nof)+(i-1), 54, f132)
        sheet1.write((j*nof)+(i-1), 55, f133)
        sheet1.write((j*nof)+(i-1), 56, f134)

        sheet1.write((j*nof)+(i-1), 57, tr1)
        sheet1.write((j*nof)+(i-1), 58, tr2)
        sheet1.write((j*nof)+(i-1), 59, tr3)
        sheet1.write((j*nof)+(i-1), 60, tr4)
        sheet1.write((j*nof)+(i-1), 61, tr5)

        sheet1.write((j*nof)+(i-1), 62, f141)
        sheet1.write((j*nof)+(i-1), 63, f142)
        sheet1.write((j*nof)+(i-1), 64, f143)
        sheet1.write((j*nof)+(i-1), 65, f144)

        sheet1.write((j*nof)+(i-1), 66, j+1)

        i=i+1
    j=j+1

book.save("featdata.xls")
