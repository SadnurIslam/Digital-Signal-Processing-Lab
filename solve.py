import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 1000
T = 1
f = 10
t = np.linspace(0, T, int(fs*T), endpoint=False)

# Signals
x = np.sin(2 * np.pi * f * t)                     # 10 Hz sine wave
y = np.sign(np.sin(2 * np.pi * f * t))            # 10 Hz square wave

# Manual cross-correlation
def manual_cross_correlation(x, y):
    N = len(x)
    corr = []
    lags = np.arange(-N + 1, N)
    for lag in lags:
        sum_val = 0
        for n in range(N):
            if 0 <= n + lag < N:
                sum_val += x[n] * y[n + lag]
        corr.append(sum_val)
    return np.array(corr), lags

# Compute manually
corr_manual, lags = manual_cross_correlation(x, y)

# Plot
plt.figure(figsize=(12, 5))

plt.subplot(2,1,1)
plt.plot(t, x, label="Sine Wave")
plt.plot(t, y, label="Square Wave")
plt.title("Signals")
plt.legend()
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(lags, corr_manual)
plt.title("Cross-Correlation (Manual)")
plt.xlabel("Lag")
plt.ylabel("Correlation")
plt.grid(True)

plt.tight_layout()
plt.show()
