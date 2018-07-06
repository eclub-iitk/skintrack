
from scipy.signal import butter, lfilter
def butter_bandpass(sig,lowcut, highcut, fs, order=2):
   nyq = 0.5 * fs
   low = lowcut / nyq
   high = highcut / nyq
   b, a = butter(order, [low, high], btype='band')
   y=lfilter(b,a,sig)
   return y
