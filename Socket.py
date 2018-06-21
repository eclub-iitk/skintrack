# This python script listens on UDP port 3333
# for messages from the ESP32 board and prints them
import socket
import sys
import xlwt
import time

ip = "192.168.43.127"
socketno = 3333
n =                                                        		#no of diff inputs
ndata = []                                                 		#no of diff value of each inputs
t =                                            					#time wait for different action
                        
try :                                                          	#AF_INET (IPv4 protocol) , AF_INET6 (IPv6 protocol)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)       	#error -1 is returned
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
    # listen(queuewaiting_reciever)								# accept ,connect not used as udp mode
except socket.error :											#Prevents error such as: “address already in use”.
    print('Failed to create socket') 
    sys.exit()

try:
    s.bind((ip, socketno))
except socket.error:
    print('Bind failed')
    sys.exit()

print('Server listening')

j = 0
while (j < n):
    print(" starting " + str(j))
    print("wait for some time:" + str(t))
    time.sleep(t)
    k=0
    while(k < ndata[j]):
        data = s.recvfrom(1024) 
        book = xlwt.Workbook(encoding="utf-8")
        sheet1 = book.add_sheet("Sheet",cell_overwrite_ok=True)
        data,add = s.recvfrom(1024)								#recv also if you use different version of python
        d1=data.split()
        for i in range(len(d1)):
            sheet1.write(k,i,d1[i])
        k+=1                   
    book.save("output%d.xls"%(j))
    j+=1
s.close() 
