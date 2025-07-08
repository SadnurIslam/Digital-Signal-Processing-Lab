'''
Filter realization using 6-point averaging, 6-point differencing equations.

'''


import numpy as np
import matplotlib.pyplot as plt

# Example signal: slow sine + fast sine (noise-like)
fs = 100  # Sampling frequency
t = np.arange(0, 2, 1/fs)
x = np.sin(2*np.pi*2*t) + 0.4*np.sin(2*np.pi*20*t)  # 2 Hz signal + 20 Hz noise
print(x)
# 6-point averaging filter
def six_point_averaging(x):
    y = [0] * len(x)  # Initialize output array with zeros
    for i in range (0, len(x)):
        s = 0
        for j in range (0,6):
            if( i-j >= 0):
                s += x[i-j]
        y[i] = s / 6
    return y


# 6-point differencing filter
def six_point_differencing(x):
    y = [0] * len(x)  # Initialize output array with zeros
    for i in range (0, len(x)):
        s = 0
        for j in range (0,6):
            if( i-j >= 0):
                if(j&1):
                    s -= x[i-j]
                else:
                    s += x[i-j]
        y[i] = s / 6
    return y

# Apply filters
y_avg = six_point_averaging(x)
y_diff = six_point_differencing(x)

# Plotting

plt.subplot(3, 1, 1)
plt.plot(t, x, label='Original Signal', color='gray')
plt.title('Original Signal (2 Hz + 20 Hz)')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, y_avg, label='6-Point Averaging', color='blue')
plt.title('Low-Pass Output (Smoothing)')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, y_diff, label='6-Point Differencing', color='red')
plt.title('High-Pass Output (Edge Detection)')
plt.grid(True)

plt.tight_layout()
plt.show()
