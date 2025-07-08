import numpy as np
import matplotlib.pyplot as plt

n = np.arange(-4,5,1)
xn = np.where(n>=0,1 ,0)

plt.stem(n,xn,label = "unit step")
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Unit Step Function')
plt.grid(True)
plt.legend()
plt.show()