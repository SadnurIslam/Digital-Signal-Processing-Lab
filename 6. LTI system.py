import numpy as np
import matplotlib.pyplot as plt

def unit_step(n):
    return np.where(n>=0, 1, 0)

def x(n):
    return unit_step(n)

def h(n):
    return unit_step(n) - unit_step(n-5)

n = np.arange(-4, 15)
h1 = unit_step(n) - unit_step(n-5)
x1 = unit_step(n)

plt.subplot(3,1,1)
plt.stem(n, x1, label = 'unit step input', linefmt='b-', basefmt= ' ')
plt.legend()
plt.grid(True)

plt.subplot(3,1,2)
plt.stem(n, h1, label = 'unit impulse response', linefmt='g-', basefmt= ' ')
plt.legend()
plt.grid(True)


y = np.zeros(len(n))

for i in range(len(n)):
    sum = 0
    for k in range(len(n)):
        if i-k >= 0:
            sum += x(k) * h(i-k)
    y[i] = sum


plt.subplot(3,1,3)
plt.stem(n, y, label = 'convulation sum', linefmt='b-', basefmt= ' ')
plt.legend()
plt.grid(True)



plt.tight_layout()
plt.show()
