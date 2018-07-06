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

mydata=pd.read_excel('5cmarmtap1.xls')
mydata1=mydata.iloc[:,:3]

mydata1.as_matrix()  #converting the dataframe to Matrix

#breaking the matrix into column vectors {IMU}
col1=matrix(mydata1).transpose()[0].getA()[0]
col2=matrix(mydata1).transpose()[1].getA()[0]
col1=signal.medfilt(col1[:],5)
col2=signal.medfilt(col2[:],5)
col1 = signal.savgol_filter(col1,201,3)
col2 = signal.savgol_filter(col2,201,3)
    

book = xlwt.Workbook(encoding="utf-8")
sheet1 = book.add_sheet("Sheet",cell_overwrite_ok=True)
k = 0
key=-1
while(k<4449):#recv also if u use different version of python
    if col1[k]<12:
        key=1
    sheet1.write(k,0,col1[k])
    sheet1.write(k,1,col2[k])
    sheet1.write(k,2,key)
    k=k+1         
    key=-1
book.save("5wristtap2_.xls")