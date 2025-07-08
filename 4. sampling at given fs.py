'''
Consider the continuous-time analog signal x(t)=3cos(100πt). Sample the analog
signal at 200 Hz and 75 Hz. Show the discrete-time signal after sampling. ⟹
realization.

'''


import numpy as np
import matplotlib.pyplot as plt

t_cont = np.linspace(0, 0.1, 1000)
xt_cont = 3*np.cos(100*np.pi*t_cont)

plt.subplot(3,1,1)
plt.plot(t_cont,xt_cont, label='Continuous Signal')
plt.title('Continuous Signal x(t) = 3cos(100πt)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

def getx(fs):
    ts = np.arange(0, 0.1, 1/fs)
    xt = 3 * np.cos(100 * np.pi * ts)
    return ts, xt

fs1 = 200  # Sampling frequency 1
ts1, xt1 = getx(fs1)
plt.subplot(3,1,2)
plt.plot(t_cont,xt_cont, label='Continuous Signal')
plt.stem(ts1, xt1, linefmt='r-', markerfmt='ro', basefmt=' ', label=f'Sampled at {fs1} Hz')
plt.title(f'Sampled Signal at {fs1} Hz')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

fs2 = 75  # Sampling frequency 2
ts2, xt2 = getx(fs2)
plt.subplot(3,1,3)
plt.plot(t_cont,xt_cont, label='Continuous Signal')
plt.stem(ts2, xt2, linefmt='r-', markerfmt='ro', basefmt=' ', label=f'Sampled at {fs2} Hz')
plt.title(f'Sampled Signal at {fs2} Hz')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()