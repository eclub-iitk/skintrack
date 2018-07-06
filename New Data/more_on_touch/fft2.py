import numpy as np
import matplotlib.pyplot as plt

fs = 30.0
t = np.arange(0,10,1/fs)
x = np.cos(2*np.pi*10*t)

xF = np.fft.fft(x)
N = len(xF)
xF = xF[0:N/2]
fr = np.linspace(0,fs/2,N/2)
plt.plot(fr,abs(xF)**2)
plt.show()
