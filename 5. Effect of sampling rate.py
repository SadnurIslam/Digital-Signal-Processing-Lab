'''
Consider the analog signal: xa(t)=3cos(200πt)+5sin(600πt)+10cos(1200πt).
Show the effect of sampling rate

'''


import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 0.01, 1000)
xt = 3 * np.cos(200 * np.pi * t) + 5 * np.sin(600 * np.pi * t) + 10 * np.cos(1200 * np.pi * t)
# plt.subplot(4, 1, 1)
# plt.plot(t, xt, label='Continuous Signal')
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.title('Given continuous signal')
# plt.grid(True)

def sampling(fs):
    n = np.arange(0, 0.01, 1/fs)
    xn = 3 * np.cos(200 * np.pi * n) + 5 * np.sin(600 * np.pi * n) + 10 * np.cos(1200 * np.pi * n)
    return n, xn

def plotting(x,y,sub,title):
    plt.subplot(3,1,sub)
    plt.plot(t,xt, label='Continuous Signal', color='lightgray')
    plt.stem(x,y, linefmt='r-', markerfmt='ro', label = f'Sampled Signal at {fs} Hz')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.grid(True)
    
fs = [2400, 1000, 800]  # Sampling frequencies
pos = 0
for i in fs:
    pos += 1
    n, xn = sampling(i)
    plotting(n, xn, pos, f'Sampled Signal at {i} Hz')
    
    
plt.tight_layout()
plt.show()