'''
DFT of xa(t)=sin(2π⋅1000t)+0.5sin(2π⋅2000t+4π). Also IDFT. DFT with
window + window function realization

'''



import numpy as np
import matplotlib.pyplot as plt

# ---------- Step 1: Sample the Analog Signal ----------
fs = 8000  # Sampling frequency (must be > 4000 Hz)
N = 64     # Number of samples
t = np.arange(N) / fs

# Sampled composite signal: sin(2π⋅1000t) + 0.5sin(2π⋅2000t + 4π)
x = np.sin(2*np.pi*1000*t) + 0.5*np.sin(2*np.pi*2000*t + 4*np.pi)

# ---------- Step 2: Define Manual Hanning Window ----------
def hanning(N):
    return np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)])

# ---------- Step 3: Manual DFT and IDFT ----------
def DFT(x):
    N = len(x)
    X = []
    for k in range(N):
        sum_val = 0
        for n in range(N):
            sum_val += x[n] * np.exp(-2j * np.pi * k * n / N)
        X.append(sum_val)
    return np.array(X)

def IDFT(X):
    N = len(X)
    x_recon = []
    for n in range(N):
        sum_val = 0
        for k in range(N):
            sum_val += X[k] * np.exp(2j * np.pi * k * n / N)
        x_recon.append(sum_val / N)
    return np.array(x_recon)

# ---------- Step 4: Apply DFT With and Without Window ----------
X_plain = DFT(x)

w = hanning(N)
x_win = x * w
X_win = DFT(x_win)

# ---------- Step 5: Frequency Axis ----------
f = np.arange(N) * fs / N  # Frequency axis

# ---------- Step 6: Plot ----------
plt.figure(figsize=(14, 8))

# Original Time Domain Signal
plt.subplot(3, 1, 1)
plt.plot(t, x, label='Original x[n]')
plt.title('Sampled Signal x[n]')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# DFT Magnitude Without Window
plt.subplot(3, 1, 2)
plt.stem(f, np.abs(X_plain))
plt.title('DFT Magnitude (without window)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('|X[k]|')
plt.grid(True)

# DFT Magnitude With Hanning Window
plt.subplot(3, 1, 3)
plt.stem(f, np.abs(X_win), linefmt='g-', markerfmt='go', basefmt=' ')
plt.title('DFT Magnitude (with Hanning window)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('|X[k] with window|')
plt.grid(True)

plt.tight_layout()
plt.show()

# ---------- Step 7: Plot Reconstructed Signal via IDFT ----------
x_recon = IDFT(X_plain)

plt.figure(figsize=(10, 4))
plt.plot(np.real(x_recon), label='Reconstructed (IDFT)', color='blue')
plt.plot(x, '--', label='Original', color='red')
plt.title("Signal Reconstruction using IDFT")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
