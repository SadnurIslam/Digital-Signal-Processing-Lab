'''
Demonstrates the effect of sampling, aliasing.

'''


import numpy as np
import matplotlib.pyplot as plt

f = 5
t_cont = np.linspace(0,1,1000)
x_cont = np.sin(2*np.pi*f*t_cont)

fs = [f+1,f*2,f*4]
pos=0
for i in fs:
    t_sampled = np.arange(0, 1, 1/i)
    x_sampled = np.sin(2 * np.pi * f * t_sampled)
    pos += 1
    plt.subplot(len(fs),1,pos)
    plt.plot(t_cont, x_cont, label='Continuous Signal', color='blue')
    plt.stem(t_sampled, x_sampled, linefmt='r-', markerfmt='ro', label=f'sampled at {i} Hz')
    # plt.plot(t_sampled, x_sampled, 'r-')  # Connect sampled points with a line
    plt.title(f'Sampling at fs = {i} Hz')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()






























# import numpy as np
# import matplotlib.pyplot as plt

# # Define continuous signal parameters
# f = 5     # Signal frequency (Hz)
# t_cont = np.linspace(0, 1, 1000)  # Continuous time (high resolution)
# x_cont = np.sin(2 * np.pi * f * t_cont)  # Continuous sine wave

# # Different sampling frequencies
# fs_high = 20   # No aliasing (fs > 2f)
# fs_nyquist = 10  # At Nyquist rate (fs = 2f)
# fs_low = 8     # Aliasing (fs < 2f)

# # Sampled signals
# t_high = np.arange(0, 1, 1/fs_high)
# x_high = np.sin(2 * np.pi * f * t_high)

# t_nyq = np.arange(0, 1, 1/fs_nyquist)
# x_nyq = np.sin(2 * np.pi * f * t_nyq)

# t_low = np.arange(0, 1, 1/fs_low)
# x_low = np.sin(2 * np.pi * f * t_low)

# # Plotting
# plt.figure(figsize=(14, 8))

# # High sampling rate
# plt.subplot(3,1,1)
# plt.plot(t_cont, x_cont, label="Original Signal")
# plt.stem(t_high, x_high, linefmt='r-', markerfmt='ro', basefmt=" ", label="Sampled (fs=20Hz)")
# plt.title("Sampling at fs = 20 Hz (No Aliasing)")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.legend()
# plt.grid(True)

# # Nyquist sampling
# plt.subplot(3,1,2)
# plt.plot(t_cont, x_cont, label="Original Signal")
# plt.stem(t_nyq, x_nyq, linefmt='g-', markerfmt='go', basefmt=" ", label="Sampled (fs=10Hz)")
# plt.title("Sampling at fs = 10 Hz (Nyquist Rate)")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.legend()
# plt.grid(True)

# # Undersampling
# plt.subplot(3,1,3)
# plt.plot(t_cont, x_cont, label="Original Signal")
# plt.stem(t_low, x_low, linefmt='m.', markerfmt='mo', basefmt=" ", label="Sampled (fs=8Hz)")
# plt.plot(t_low, x_low)
# plt.title("Sampling at fs = 8 Hz (Aliasing Occurs)")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()
