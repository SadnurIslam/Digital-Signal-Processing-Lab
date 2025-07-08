import numpy as np
import matplotlib.pyplot as plt

n = np.arange(-10, 11, 1)
xn = np.cos(2 * np.pi * 0.1 * n)
plt.stem(n, xn, label="cosine signal")
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Cosine Signal Function')
plt.grid(True)
plt.legend()
plt.show()