# This python script listens on UDP port 3333
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

ip="192.168.43.127"
socketno=3333
                                                       #no of diff inputs
                                                 #no of diff value of each inputs
                                          #time wait for different action
                        
try :                                                          #AF_INET (IPv4 protocol) , AF_INET6 (IPv6 protocol)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)       #error -1 is returned
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
    # listen(queuewaiting_reciever), accept ,connect not used as udp mode
                                                               #Prevents error such as: “address already in use”.
except socket.error :

    print('Failed to create socket. Error Code : '  + ' Message ') 
    sys.exit()
try:
    s.bind((ip, socketno))
except socket.error:
    print('Bind failed. Error: '  ': ')
    sys.exit()

print('Server listening')
book = xlwt.Workbook(encoding="utf-8")
sheet1 = book.add_sheet("Sheet",cell_overwrite_ok=True)
t1 = time.time()
k = 0
while(k<400):
    data,add = s.recvfrom(1024)#recv also if u use different version of python
    d1=data.split()
    x=int(d1[1][:])
    y=int(d1[2][:])
    print(k,x,y)
    sheet1.write(k,0,x)
    sheet1.write(k,1,y)
    k=k+1  
t2=time.time() 
t=t2-t1 
print(t)         
book.save("data1.xls")
s.close()


mydata=pd.read_excel('data%d.xls'%1)
mydata1=mydata.iloc[:,:3]

#mydata1=mydata1.rolling(20).mean() #moving average

mydata1.as_matrix()  #converting the dataframe to Matrix

#breaking the matrix into column vectors {IMU}
col1=matrix(mydata1).transpose()[0].getA()[0]
col2=matrix(mydata1).transpose()[1].getA()[0]

plt.plot(col1)
plt.plot(col2)
plt.show()