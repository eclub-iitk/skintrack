import pandas as pd
import numpy as np
from numpy import matrix
import xlwt
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, lfilter


col1=np.arange(10)
print(col1)
print(np.size(col1))
col1=signal.medfilt(col1,3)
print(col1)
print(np.size(col1))