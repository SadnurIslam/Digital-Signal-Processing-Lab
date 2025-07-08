'''
DFT of xa(t)=sin(2π⋅1000t)+0.5sin(2π⋅2000t+4π). Also IDFT. DFT with
window + window function realization

'''




import numpy as np
import matplotlib.pyplot as plt

# 1. Sampling Parameters
fs = 8000          # Sampling frequency
N = 64             # Number of samples
t = np.arange(N) / fs  # Time array

# 2. Sample the Analog Signal
x = np.sin(2 * np.pi * 1000 * t) + 0.5 * np.sin(2 * np.pi * 2000 * t + 4 * np.pi)

# 3. Apply DFT using built-in
X = np.fft.fft(x)
freq = np.fft.fftfreq(N, d=1/fs)

# 4. Apply IDFT
x_recon = np.fft.ifft(X)

# 5. Apply Hanning Window and compute DFT
w = np.hanning(N)
x_windowed = x * w
X_win = np.fft.fft(x_windowed)

# ----------- Plotting -----------

plt.figure(figsize=(14, 10))

# Original Signal
plt.subplot(4, 1, 1)
plt.plot(t, x, label='Original Signal')
plt.title('Sampled Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# DFT Magnitude
plt.subplot(4, 1, 2)
plt.stem(freq[:N//2], np.abs(X[:N//2]))
plt.title('DFT Magnitude Spectrum (without window)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('|X[k]|')
plt.grid(True)

# DFT with Hanning Window
plt.subplot(4, 1, 3)
plt.stem(freq[:N//2], np.abs(X_win[:N//2]), linefmt='g-', markerfmt='go', basefmt=' ')
plt.title('DFT Magnitude Spectrum (with Hanning window)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('|X[k] with window|')
plt.grid(True)

# IDFT Reconstruction
plt.subplot(4, 1, 4)
plt.plot(t, x, 'r--', label='Original')
plt.plot(t, np.real(x_recon), 'b', label='Reconstructed (IDFT)')
plt.title('Reconstructed Signal using IDFT')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
