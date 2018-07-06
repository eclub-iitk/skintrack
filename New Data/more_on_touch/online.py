import socket
import sys
import time
import xlwt

list1=[2.34,4.346,4.234]
book = xlwt.Workbook(encoding="utf-8")
sheet1 = book.add_sheet("Sheet 1")
try :
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

except socket.error, msg :
    print 'Failed to create socket. Error Code : ' + str(msg[0]) + ' Message ' + msg[1]
    sys.exit()


try:
    s.bind(('192.168.43.242', 3333))

except socket.error , msg:
    print 'Bind failed. Error: ' + str(msg[0]) + ': ' + msg[1]
    sys.exit()

print 'Server listening'
d = s.recvfrom(1024)
data = d[0]
blabla,stri,yaw,pitch,roll = data.split()
pitch=float(pitch)
roll=float(roll)
yaw=float(yaw)
count = 0

i = 0
aa=yaw
t1=time.time()
while (i<50):
    d = s.recvfrom(1024)
    data = d[0]
    blabla,stri,yaw,pitch,roll = data.split()
    pitch=float(pitch)
    roll=float(roll)
    yaw=float(yaw)
    yaw=yaw-aa



while (1):

    d = s.recvfrom(1024)
    data = d[0]
    blabla,stri,yaw,pitch,roll = data.split()
    pitch=float(pitch)
    roll=float(roll)
    yaw=float(yaw)
    yaw=yaw-aa
    sheet1.write(i, 0, pitch)
    sheet1.write(i, 1, roll)

    #sheet1.write(i, 2, yaw)
    #print yaw, pitch, roll

    i=i+1

s.close()

book.save("ash.xls")
