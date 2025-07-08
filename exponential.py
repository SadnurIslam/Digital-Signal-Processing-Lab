import numpy as np
import matplotlib.pyplot as plt

n = np.arange(-10, 11, 1)
xn = np.where(True,0.8**n,0)

plt.stem(n, xn, label="exponential signal")
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Exponential Signal Function')
plt.grid(True)
plt.legend()
plt.show()