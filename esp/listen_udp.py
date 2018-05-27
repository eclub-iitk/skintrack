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
print "Wait for 3 Sec"
time.sleep(3)
j=0
while(j<50):
            print "Start Now"


            count = 0

            d = s.recvfrom(1024)

            data = d[0]

            pitch = data.split()

            t1=time.time()

            while (time.time()-t1<3):

                d = s.recvfrom(1024)

                data = d[0]

                pitch= data.split()

                pitch=float(pitch)

                print (pitch)



            


            j=j+1

book.save('d%d.xls'%(j))



s.close()