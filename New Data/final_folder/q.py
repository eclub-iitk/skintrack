import socket
import sys
import time
import xlwt


class Queue:
    def __init__(self):
        self.items=[]

    def isEmpty(self):
        return self.items ==[]

    def enqueue(self,item):
        self.items.insert(0,item)

    def dequeue(self):
        self.items.pop()

    def size(self):
        return len(self.items)

    def printqueue(self):
        for items in self.items:
            print (items)


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
i=0
y=Queue()
p=Queue()
r=Queue()

while (i<400):
    d = s.recvfrom(1024)
    data = d[0]
    blabla,stri,yaw,pitch,roll = data.split()
    pitch=float(pitch)
    roll=float(roll)
    yaw=float(yaw)
    y.enqueue(yaw)
    p.enqueue(pitch)
    r.enqueue(roll)
    i=i+1
p.printqueue()

i=0
while (i<50):
    d = s.recvfrom(1024)
    data = d[0]
    blabla,stri,yaw,pitch,roll = data.split()
    pitch=float(pitch)
    roll=float(roll)
    yaw=float(yaw)
    y.enqueue(yaw)
    r.dequeue()
    p.enqueue(pitch)
    p.dequeue()
    r.enqueue(roll)
    p.dequeue()
    i=i+1


y.printqueue()
