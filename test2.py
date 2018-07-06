import pandas as pd
import numpy as np
from numpy import matrix
import xlwt
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, lfilter 
mydata=pd.read_excel('data%d.xls'%3)
mydata1=mydata.iloc[:,:3]

#mydata1=mydata1.rolling(20).mean() #moving average

mydata1.as_matrix()  #converting the dataframe to Matrix

#breaking the matrix into column vectors {IMU}
col1=matrix(mydata1).transpose()[0].getA()[0]
col2=matrix(mydata1).transpose()[1].getA()[0]


ncol1=np.insert(col1,0,0)
ncol2=np.insert(col2,0,0)
navg1=np.average(ncol1)
nstd1=np.std(ncol1)                                     # 50:np.size(ncol2)-50
nstd2=np.std(ncol2)
ancol1=np.insert(col1,0,0)
ancol2=np.insert(col2,0,0)
print(navg1,navg2)
print(nstd1,nstd2)
print(np.amax(ncol1),np.amin(ncol2))
print (np.amin(ncol1),np.amin(ncol2))

k=1
for i in range(1,len(ncol1)-1):
    if ncol1[i]>navg1+k*nstd1 or ncol1[i]<navg1-k*nstd1:
        ancol1[i]=(ncol1[i-1]+ncol1[i+1])/2
    if ncol2[i]>navg2+k*nstd2 or ncol2[i]<navg2-k*nstd2:
        ancol2[i]=(ncol2[i-1]+ncol2[i+1])/2
    print(ncol1[i],ancol1[i])
#ancol1=signal.medfilt(ancol1,11)                 
#plt.plot(ancol1,'c',label='line two')
"""
col1=signal.medfilt(col1,5)
col2=signal.medfilt(col2,5)
col3=signal.medfilt(col1,101)
col2=signal.medfilt(col2,101)
"""
"""
domain = np.identity(np.shape(col1))
col1=signal.order_filter(col1, domain, 0)

col1=signal.medfilt(col1,101)
col2=signal.medfilt(col2,101)
"""
"""
def running_mean(x, N):
	cumsum = np.cumsum(np.insert(x, 0, 0)) 
	return (cumsum[N:] - cumsum[:-N]) / float(N)
y_ = np.array(ancol1,int)
n=20
result8_ = running_mean(y_,n)
plt.plot(result8_,'g',label='line three')
"""
"""

def getTheta(X, Y, x, bandwidth):
	# X: np vector with x vals (here numbers in sequence)
		#Y: np vector with y vals (here magnitudes)
		#x: point for which y is reqd
        
	T = bandwidth
	Wts = np.exp(-(np.square(X[:,0] - x)/(2*T*T)))
	A = np.linalg.inv(np.einsum('ij,i,ik->jk', X, Wts, X))
	B = np.einsum('ji,j,j->i', X, Wts, Y)
	O = np.matmul(A, B)	
	return O
	
def weightedLinRegr(X, Y, bandwidth):
	#assumed X as ascending 1D array, Y as the corresponding vals 
    
	ylist = []
	X = X[:, np.newaxis]
	X = np.hstack((X, np.ones(X.shape[0])[:, np.newaxis]))
	for xo in X[:,0]:
		O = getTheta(X, Y, xo, bandwidth)
		yo = O[0]* xo + O[1]
		ylist.append(yo)
	return np.array(ylist)
s2 = weightedLinRegr(np.arange(col1.shape[0]), col1, 2)
plt.plot(np.arange(col1.shape[0]), s2 )
s3 = weightedLinRegr(np.arange(col2.shape[0]), col1, 2)
plt.plot(np.arange(col1.shape[0]), s3 )
"""

n=100
col=np.zeros(len(col1))
for i in range(len(ancol1)-int(n//2)):
    col[i+n//2-1]=np.sum(ancol1[i:n+i])/n
plt.plot(col,'r',label='line three')  

"""
def butter_lowpass(cutOff, fs, order=1):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = butter(order, normalCutoff, btype='low', analog = True)
    return b, a

def butter_lowpass_filter(data, cutOff, fs, order=4):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = lfilter(b, a, data)
    return y

cutOff =  500#cutoff frequency in rad/s
fs = 4500/8.349 #sampling frequency in rad/s
order = 1 #order of filter
col1 = butter_lowpass_filter(col1, cutOff, fs, order)
"""
#plt.plot(ncol1,'g',label='line one')
#plt.plot(col2,'c',label='line two')
#plt.plot(col3,'r',label='line three')
#plt.plot(col1,'g',label='line one')

plt.legend()
plt.show()
