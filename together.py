import pandas as pd
import numpy as np
from numpy import matrix
import xlwt
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, lfilter 
from scipy.fftpack import fft
book = xlwt.Workbook(encoding="utf-8")
sheet1 = book.add_sheet("Sheet",cell_overwrite_ok=True)

a=[1,5,20]
for j in range(3):
    for i in range(4):
        mydata=pd.read_excel('%dhover%d.xls'%(j,i))
        mydata1=mydata.iloc[:,:4]
        mydata1.as_matrix()  #converting the dataframe to Matrix
        #breaking the matrix into column vectors {IMU}
        col1=matrix(mydata1).transpose()[0].getA()[0]
        col2=matrix(mydata1).transpose()[1].getA()[0]
        col3=matrix(mydata1).transpose()[2].getA()[0]
        col4=matrix(mydata1).transpose()[3].getA()[0]
        for h in range(len(col1)):
                sheet1.write(h+(i+j*len(col1))*len(col1),0,col1[h])
                sheet2.write(h+(i+j*len(col2))*len(col2),1,col2[h])
                sheet2.write(h+(i+j*len(col3))*len(col3),2,col3[h])
                sheet2.write(h+(i+j*len(col4))*len(col4),3,col4[h])
book.save("together.xls")

