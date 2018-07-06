import pandas as pd
import numpy as np
from numpy import matrix
import xlwt
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, lfilter 
from scipy.fftpack import fft
<<<<<<< HEAD

for i in range(5):
mydata=pd.read_excel('touch.xls')
=======
mydata=pd.read_excel('2tap0.xls')
>>>>>>> e9ca89d6705038322a0b31bb50f7a01fe85e379f
mydata1=mydata.iloc[:,:4]

#mydata1=mydata1.rolling(20).mean() #moving average

mydata1.as_matrix()  #converting the dataframe to Matrix

#breaking the matrix into column vectors {IMU}
col1=matrix(mydata1).transpose()[0].getA()[0]
col2=matrix(mydata1).transpose()[1].getA()[0]
col3=matrix(mydata1).transpose()[2].getA()[0]
col4=matrix(mydata1).transpose()[3].getA()[0]
col1=signal.medfilt(col1[:],7)
col2=signal.medfilt(col2[:],7)
col3=signal.medfilt(col3[:],7)
col4=signal.medfilt(col4[:],7)
col1=col1[20:]
col2=col2[20:]
col3=col3[20:]
col4=col4[20:]
col1 = signal.savgol_filter(col1,401,3)
col2 = signal.savgol_filter(col2,401,3)
col3 = signal.savgol_filter(col3,401,3)
col4 = signal.savgol_filter(col4,401,3)
#col1=signal.medfilt(col1, kernel_size=None)
#col2=signal.medfilt(col2, kernel_size=None)
#col1=signal.medfilt(col1,41)
#col2=signal.medfilt(col2,41)
<<<<<<< HEAD

=======
"""
N = 4500
T = 8.349 / 4500
x = np.linspace(0.0, N*T, N)
yf = fft(col1)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]),'g',label='line one')
"""
"""
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
y_ = np.array(col1,int)
n=1
result8_ = running_mean(y_,n)
y__ = np.array(col2,int)
result8__ = running_mean(y__,n)
plt.plot(result8_)
plt.plot(result8__)		"""
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
"""
n=100
col=np.zeros(len(col1))
for i in range(len(col1)-int(n//2)):
    col[i+n//2-1]=np.sum(col1[i:n+i])/n
"""
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

cutOff =  2#cutoff frequency in rad/s
fs = 4500/8.349 #sampling frequency in rad/s
order = 3 #order of filter
col1 = butter_lowpass_filter(col1, cutOff, fs, order)


for i in range(len(col1)):
    print(col1[i])
"""    
>>>>>>> e9ca89d6705038322a0b31bb50f7a01fe85e379f
plt.subplot(2,2,1)
plt.plot(col1,label = 'phase1')
plt.legend()
plt.grid()
#plt.ylim(80,130)

plt.subplot(2,2,2)
plt.plot(col2,label = 'mag1')
plt.legend()
plt.grid()
#plt.ylim(4,10)

plt.subplot(2,2,3)
plt.plot(col3,label='phase2')
plt.legend()
plt.grid()
#plt.ylim(10,15)

plt.subplot(2,2,4)
plt.plot(col4,label = 'mag2')
plt.legend()
plt.grid()
#plt.ylim(-28,-20)

plt.show()

