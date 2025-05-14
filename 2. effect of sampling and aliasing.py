import numpy as np
import matplotlib.pyplot as plt

# Continuous signal parameters
f_signal = 9  # Hz (signal frequency)
t_cont = np.linspace(0, 1, 1000)  # High-resolution time axis (simulates continuous time)
x_cont = np.sin(2 * np.pi * f_signal * t_cont)  # Continuous-time signal

# Sampling rates (below and above Nyquist)
sampling_rates = [5, 15, 20]  # Hz (Nyquist rate is 2*f = 18 Hz for f=9 Hz)

plt.figure(figsize=(12, 9))

for i, fs in enumerate(sampling_rates):
    t_samp = np.arange(0, 1, 1/fs)
    x_samp = np.sin(2 * np.pi * f_signal * t_samp)

    plt.subplot(3, 1, i + 1)
    plt.plot(t_cont, x_cont, 'lightgray', label='Original Signal')
    plt.stem(t_samp, x_samp, linefmt='r-', markerfmt='ro', basefmt='k-', label='Sampled Signal')
    plt.title(f"Sampling at {fs} Hz {'(Aliasing)' if fs < 2*f_signal else '(No Aliasing)'}")
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.suptitle("Effect of Sampling and Aliasing", fontsize=16, y=1.02)
plt.show()
