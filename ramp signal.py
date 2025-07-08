import numpy as np 
import matplotlib.pyplot as plt

n = np.arange(-4, 5, 1)
xn = np.where(n>=0,n,0)

plt.plot(n, xn, label="ramp signal")
plt.stem(n, xn)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Ramp Signal Function')
plt.grid(True)
plt.legend()
plt.show()