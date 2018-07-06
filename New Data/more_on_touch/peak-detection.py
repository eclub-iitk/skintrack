import numpy as np
vector = [
    0, 8,18,5,7,12,0 ]

import detect_peaks

def plot_peaks(x, indexes, algorithm=None, mph=None, mpd=None):
    """Plot results of the peak dectection."""
    y = []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
        return
    _, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(x, 'b', lw=1)
    if indexes.size:
        label = 'peak'
        label = label + 's' if indexes.size > 1 else label
        for i in range(len(indexes)):
            y.append(x[indexes[i]])
        ax.plot(indexes, y, '+', mfc=None, mec='r', mew=2, ms=8,
                label='%d %s' % (indexes.size, label))
        ax.legend(loc='best', framealpha=.5, numpoints=1)
    ax.set_xlim(-.02*len(x), len(x)*1.02-1)
    ymin, ymax = np.argmax(x), np.argmin(x)
    yrange = ymax - ymin if ymax > ymin else 1
    ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
    ax.set_xlabel('Data #', fontsize=14)
    ax.set_ylabel('Amplitude', fontsize=14)
    ax.set_title('%s (mph=%s, mpd=%s)' % (algorithm, mph, mpd))
    plt.show()


print('Detect peaks with minimum height and distance filters.')
indexes = detect_peaks.detect_peaks(vector, mph=10, mpd=1)
print('Peaks are: %s' % (indexes))
plot_peaks(vector,indexes)
