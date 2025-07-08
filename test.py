import numpy as np
import matplotlib.pyplot as plt

n = np.arange(-10,11,1)
xn = 0.7**n

plt.stem(n,xn, label = "unit impulse")
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Unit Impulse Function')
plt.grid(True)
plt.legend()
plt.xticks(np.arange(-10,11,1))
plt.tight_layout()
plt.show()