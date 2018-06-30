import pandas as pd
import numpy as np
from numpy import matrix
import xlwt
import matplotlib.pyplot as plt
from scipy import signal

mydata=pd.read_excel('data%d.xls'%3)
mydata1=mydata.iloc[:,:3]

mydata1=mydata1.rolling(20).mean() #moving average

mydata1.as_matrix()  #converting the dataframe to Matrix

#breaking the matrix into column vectors {IMU}
col1=matrix(mydata1).transpose()[0].getA()[0]
col2=matrix(mydata1).transpose()[1].getA()[0]
col1=signal.medfilt(col1,7)
col2=signal.medfilt(col2,7)

plt.plot(col1)
plt.plot(col2)
plt.show()