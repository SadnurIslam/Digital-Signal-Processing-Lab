import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 64
fs = 8000
n = np.arange(N)
t = n / fs

# Original signal xa(t)
def xa(t):
    return np.sin(2 * np.pi * 1000 * t) + 0.5 * np.sin(2 * np.pi * 2000 * t + 4 * np.pi)

# Corrected Hanning Window
def hanning(N):
    return 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1)))

# DFT Implementation
def dft(x):
    N = len(x)
    print(N)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

# IDFT Implementation
def idft(X):
    N = len(X)
    x = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            x[n] += X[k] * np.exp(2j * np.pi * k * n / N)
        x[n] /= N
    return x

# Signals
x = xa(t)
window = hanning(N)
x_win = x * window

# DFTs
X = dft(x)
X_win = dft(x_win)

# IDFTs
x_recon = idft(X)
x_recon_win = idft(X_win)

# Frequency axis
freq = np.arange(N) * fs / N

# Plotting
plt.figure(figsize=(12, 10))

# Time-domain signals
plt.subplot(3, 2, 1)
plt.plot(t, x, label="Original", color='g')
plt.plot(t, x_win, label="Hanning Applied", color='r')
plt.title("Time Domain Signals")
plt.xlabel("Time (s)")
plt.legend()
plt.grid(True)

# Window
plt.subplot(3, 2, 2)
plt.plot(window, label="Hanning Window")
plt.title("Window Function")
plt.grid(True)
plt.legend()

# Magnitude Spectrum
plt.subplot(3, 2, 3)
plt.stem(freq, np.abs(X), label="DFT |X[k]|")
plt.title("DFT Magnitude (Original)")
plt.xlabel("Frequency (Hz)")
plt.grid(True)
plt.legend()

plt.subplot(3, 2, 4)
plt.stem(freq, np.abs(X_win), label="DFT |X[k]| with Hanning", linefmt='r-', markerfmt='ro')
plt.title("DFT Magnitude (Windowed)")
plt.xlabel("Frequency (Hz)")
plt.grid(True)
plt.legend()

# Time-domain reconstruction
plt.subplot(3, 2, 5)
plt.plot(t, np.real(x_recon), label="IDFT of Original", color='g')
plt.plot(t, np.real(x_recon_win), label="IDFT of Windowed", color='r')
plt.title("Reconstructed Signals")
plt.xlabel("Time (s)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
