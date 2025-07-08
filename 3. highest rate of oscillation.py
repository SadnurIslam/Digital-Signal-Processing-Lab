'''
Show that the highest rate of oscillation in a discrete-time sinusoidal is obtained
when ω=π

'''



import numpy as np
import matplotlib.pyplot as plt

n = np.arange(0, 20,1)

# Different frequencies
w1 = np.pi / 4
w2 = np.pi / 2
w3 = np.pi

x1 = np.cos(w1 * n)
x2 = np.cos(w2 * n)
x3 = np.cos(w3 * n)

plt.subplot(3,1,1)
plt.stem(n, x1)
plt.title("x[n] = cos(π/4 * n) - Slow Oscillation")
plt.grid(True)

plt.subplot(3,1,2)
plt.stem(n, x2)
plt.title("x[n] = cos(π/2 * n) - Medium Oscillation")
plt.grid(True)

plt.subplot(3,1,3)
plt.stem(n, x3)
plt.title("x[n] = cos(π * n) = (-1)^n - Fastest Oscillation")
plt.grid(True)

plt.tight_layout()
plt.show()
