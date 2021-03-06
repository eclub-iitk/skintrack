import pandas as pd
import numpy as np
from numpy import matrix
import xlwt
import myfeat
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, lfilter 
from scipy.fftpack import fft
book = xlwt.Workbook(encoding="utf-8")
sheet1 = book.add_sheet("Sheet",cell_overwrite_ok=True)
a=['hover','touch']
for j in range(2):
    for i in range(50):
        mydata=pd.read_excel('%s%d.xls'%(a[j],i))
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
        col1=col1[100:len(col1)-100]
        col2=col2[100:len(col1)-100]
        col3=col3[100:len(col1)-100]
        col4=col4[100:len(col1)-100]

        arr_p1 = col1
        arr_p2 = col2

        arr_m1 = col3

        arr_m2 = col4
    #mean

        mean_p1 = np.mean(arr_p1)

        mean_p2 = np.mean(arr_p2)

        mean_m1 = np.mean(arr_m1)

        mean_m2 = np.mean(arr_m2)

        sheet1.write(i+50*j,0,mean_p1)
        sheet1.write(i+50*j,1,mean_p2)
        sheet1.write(i+50*j,2,mean_m1)
        sheet1.write(i+50*j,3,mean_m2)

        #max

        max_p1 = np.amax(arr_p1)

        max_p2 = np.amax(arr_p2)

        max_m1 = np.amax(arr_m1)

        max_m2 = np.amax(arr_m2)
        sheet1.write(i+50*j,4,max_p1)
        sheet1.write(i+50*j,5,max_p2)
        sheet1.write(i+50*j,6,max_m1)
        sheet1.write(i+50*j,7,max_m2)


        iqr_p1 = myfeat.IQR(arr_p1)
        iqr_p2 = myfeat.IQR(arr_p2)
        iqr_m1 = myfeat.IQR(arr_m1)
        iqr_m2 = myfeat.IQR(arr_m2)
        sheet1.write(i+50*j,8,iqr_p1)
        sheet1.write(i+50*j,9,iqr_p2)
        sheet1.write(i+50*j,10,iqr_m1)
        sheet1.write(i+50*j,11,iqr_m2)


        #min

        min_p1 = np.amin(arr_p1)

        min_p2 = np.amin(arr_p2)

        min_m1 = np.amin(arr_m1)

        min_m2 = np.amin(arr_m2)

        sheet1.write(i+50*j,12,min_p1)
        sheet1.write(i+50*j,13,min_p2)
        sheet1.write(i+50*j,14,min_m1)
        sheet1.write(i+50*j,15,min_m2)

        #median

        med_p1 = np.median(arr_p1)

        med_p2 = np.median(arr_p2)

        med_m1 = np.median(arr_m1)

        med_m2 = np.median(arr_m2)
        sheet1.write(i+50*j,0+16,med_p1)
        sheet1.write(i+50*j,1+16,med_p2)
        sheet1.write(i+50*j,2+16,med_m1)
        sheet1.write(i+50*j,3+16,med_m2)


        #range

        range_p1 = max_p1-min_p1
        range_p2 = max_p2-min_p2

        range_m1 = max_m1-min_m1
        range_m2 = max_m2-min_m2
        sheet1.write(i+50*j,0+20,range_p1)
        sheet1.write(i+50*j,1+20,range_p2)
        sheet1.write(i+50*j,2+20,range_m1)
        sheet1.write(i+50*j,3+20,range_m2)


        #standard deviation

        std_p1 = np.std(arr_p1)

        std_p2 = np.std(arr_p2)

        std_m1 = np.std(arr_m1)

        std_m2 = np.std(arr_m2)

        sheet1.write(i+50*j,0+24,std_p1)
        sheet1.write(i+50*j,1+24,std_p2)
        sheet1.write(i+50*j,2+24,std_m1)
        sheet1.write(i+50*j,3+24,std_m2)

        #sum

        sum_p1 = np.sum(arr_p1)

        sum_p2 = np.sum(arr_p2)

        sum_m1 = np.sum(arr_m1)

        sum_m2 = np.sum(arr_m2)

        sheet1.write(i+50*j,0+28,sum_p1)
        sheet1.write(i+50*j,1+28,sum_p2)
        sheet1.write(i+50*j,2+28,sum_m1)
        sheet1.write(i+50*j,3+28,sum_m2)

        #rms
        rms_p1 = myfeat.rms(arr_p1)
        rms_p2 = myfeat.rms(arr_p2)
        rms_m1 = myfeat.rms(arr_m1)
        rms_m2 = myfeat.rms(arr_m2)

        sheet1.write(i+50*j,0+32,rms_p1)
        sheet1.write(i+50*j,1+32,rms_p2)
        sheet1.write(i+50*j,2+32,rms_m1)
        sheet1.write(i+50*j,3+32,rms_m2)
        #IQR
        sheet1.write(i+50*j,36,j)
        

        #entropy
    """
        ent_p1 = myfeat.entropy(arr_p1)
        ent_p2 = myfeat.entropy(arr_p2)
        ent_m1 = myfeat.entropy(arr_m1)
        ent_m2 = myfeat.entropy(arr_m2)
        print(ent_p1)
        sheet1.write(i+50*j,0+36,ent_p1)
        sheet1.write(i+50*j,1+36,ent_p2)
        sheet1.write(i+50*j,2+36,ent_m1)
        sheet1.write(i+50*j,3+36,ent_m2)
    """
    book.save("final_features2.xls")        