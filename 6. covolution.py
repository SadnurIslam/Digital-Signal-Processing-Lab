'''
The impulse response of a discrete-time LTI system is h(n)={u(n)−u(n−5)}.
Determine the output of the system for the input x[n]=u(n), using the convolution
sum

'''

import numpy as np
import matplotlib.pyplot as plt

def u(n):
    if(n>=0):
        return 1
    return 0

def h(n):
    return u(n) - u(n-5)

def x(n):
    return u(n)

y = []
min_n = 0
max_n = 20
for n in range(min_n, max_n + 1):
    sum = 0
    for k in range(0, n + 1):
        sum+=x(k)*h(n-k)
        # print(f"n: {n}, k: {k}, h(k): {h(k)}, h(n-k): {h(n-k)}, sum: {sum}")
    y.append(sum)

n = np.arange(min_n, max_n+1, 1)

print(y)   

plt.stem(n,y)
plt.title("Output of the LTI System")
plt.xlabel("n")
plt.ylabel("y[n]")
plt.grid(True)
plt.xticks(np.arange(min_n, max_n+1, 1))
plt.show() 