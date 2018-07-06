#By Ashok Kumar Chaudhary

""" This Code is fow viewing the data"""

import pandas as pd
import numpy as np
from numpy import matrix
import xlwt
import matplotlib.pyplot as plt

i=1501
nof=1550 #no of files

book = xlwt.Workbook(encoding="utf-8")
sheet1 = book.add_sheet("Sheet 1")

while(i<=nof):
    #reading from each raw data
    mydata=pd.read_excel('d%d.xls'%i)
    mydata1=mydata.iloc[:,:9]

    #mydata1=mydata1.rolling(20).mean() #moving average

    mydata1.as_matrix()  #converting the dataframe to Matrix

    #breaking the matrix into column vectors {IMU}
    col1=matrix(mydata1).transpose()[0].getA()[0]
    col2=matrix(mydata1).transpose()[1].getA()[0]
    col3=matrix(mydata1).transpose()[2].getA()[0]


    #breaking the matrix into column vector
    col4=matrix(mydata1).transpose()[3].getA()[0]
    col5=matrix(mydata1).transpose()[4].getA()[0]
    col6=matrix(mydata1).transpose()[5].getA()[0]
    col7=matrix(mydata1).transpose()[6].getA()[0]
    col8=matrix(mydata1).transpose()[7].getA()[0]
    col9=matrix(mydata1).transpose()[8].getA()[0]

    print (i)
    i=i+1

    plt.plot(col1)
    plt.plot(col2)
    plt.plot(col3)
    plt.plot(col4)
    #plt.plot(col9)

    plt.show()
