# This python script listens on UDP port 3333

# for messages from the ESP32 board and prints them

import socket

import sys

import time

import csv

import xlwt



list1=[2.34,4.346,4.234]



book = xlwt.Workbook(encoding="utf-8")



sheet1 = book.add_sheet("Sheet 1",cell_overwrite_ok=True)



try :

    s1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    s1.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    s2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    s2.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

except socket.error, msg :

    print 'Failed to create socket. Error Code : ' + str(msg[0]) + ' Message ' + msg[1]

    sys.exit()


try:

    s1.bind(('192.168.0.107', 8001))
    s2.bind(('192.168.0.107', 7999))

except socket.error , msg:

    print 'Bind failed. Error: ' + str(msg[0]) + ': ' + msg[1]

    sys.exit()
j=0
w
while(j<1):

            print "Start Now "

            count = 0
            i = 0

            t1=time.time()
            while (time.time()-t1<3):

                d1 = s1.recvfrom(1024)
                data1 = d1[0]
                d2 = s2.recvfrom(1024)
                data2 = d2[0]

                idf1,str1,pitch1,roll1,flex1,flex2,flex3,flex4,flex5 = data1.split()

                idf2,str2,pitch2,roll2,flex11,flex22,flex33,flex44,flex55 = data2.split()

                pitch1=float(pitch1)
                roll1=float(roll1)

                pitch2=float(pitch2)
                roll2=float(roll2)

                flex1=int(flex1)
                flex2=int(flex2)
                flex3=int(flex3)
                flex4=int(flex4)
                flex5=int(flex5)

                sheet1.write(i, 0, pitch1)
                sheet1.write(i, 1, roll1)

                sheet1.write(i, 2, pitch2)
                sheet1.write(i, 3, roll2)
                #sheet1.write(i, 2, yaw-aa)
                sheet1.write(i, 4, flex1)
                #sheet1.write(i, 5, flex2)
                sheet1.write(i, 6, flex3)
                sheet1.write(i, 7, flex4)
                sheet1.write(i, 8, flex5)
                i=i+1
                count=count+1
                print (count,str1,str2)

            book.save('daaa.xls')
            print (time.time()-t1)
            j=j+1


s1.close()
s2.close()
