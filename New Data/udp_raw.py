
import udp_class
import time
udp = udp_class.UDP_data()
udp.start("192.168.0.104",3334)
time1 = time.time()
i=0
while(time.time()-time1<20):
    data = udp.update()
    print(data)
    i=i+1

udp.stop()
print("over",i)
