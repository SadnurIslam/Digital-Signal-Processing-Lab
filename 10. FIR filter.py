'''
Design a low pass FIR filter to remove high-frequency noise from a signal using
convolution.

'''


import numpy as np
import matplotlib.pyplot as plt

# 1. Generate noisy signal
fs = 1200  # Sampling frequency (Hz)
t = np.arange(0, 1, 1/fs)  # 1 second of data

# Signal = low freq (5 Hz) + high freq noise (100 Hz)
x = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*50*t)

# 2. Define Hanning window manually
def hanning(N):
    n = np.arange(N)
    return 0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))

def hamming(N):
    n = np.arange(N)
    return 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))

# 3. Design Low-pass FIR filter (cutoff = 10 Hz)
def lowpass_fir(N, fc, fs):
    n = np.arange(N)
    fc_norm = fc / fs
    h = 2 * fc_norm * np.sinc(2 * fc_norm * (n - (N - 1) / 2))
    h *= hanning(N)  # Apply window
    h /= np.sum(h)   # Normalize
    return h

# 4. Manual convolution function
def convolve(x, h):
    len_x = len(x)
    len_h = len(h)
    len_y = len_x + len_h - 1
    y = []
    for n in range(len_y):
        sum = 0
        for k in range(len_h):
            if 0 <= n - k < len_x:
                sum += h[k] * x[n - k]
        y.append(sum)
    return np.array(y)

# 5. Apply filter
N = 51         # Filter length (odd)
fc = 10        # Cutoff frequency (Hz)
h = lowpass_fir(N, fc, fs)
y = convolve(x, h)

# Time axis for filtered signal
t_y = np.arange(len(y)) / fs

# 6. Plotting

plt.plot(t, x, label='Noisy Signal', color='r')
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.plot(t_y, y, label='Filtered Signal (Low-pass)', color='g')
plt.grid(True)

plt.tight_layout()
plt.show()
