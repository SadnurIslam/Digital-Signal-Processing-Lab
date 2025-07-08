''' 
consider a differencing equation:
y(n) = (1-a)*y(n-1) + a * x(n)

x(n) = [1, 2,2, 10, 2, 2, 1]
here, 0 < a < 1

observation of y(n) apply diffent value of a

'''

import numpy as np
import matplotlib.pyplot as plt


x = np.array([1, 2, 2, 10, 2, 2, 1])

a = 0.1

y = np.zeros_like(x, dtype=float)
y[0] = a * x[0]



for n in range(1, len(x)):
    y[n] = ((1-a) * y[n-1] + a * x[n])
    
    
n = np.arange(0, len(x)+1)

plt.subplot(2,1,1)
plt.plot(y, label = 'output of the signal')



a_val = [0.1, 0.3, 0.5, 0.7, 0.9]


plt.subplot(2, 1, 1)
plt.title('main output')

for a in a_val:
    y = []
    y_prev = 0  
    
    for n in range(len(x)):
        y_curr = (1 - a) * y_prev + a * x[n]
        y.append(y_curr)
        y_prev = y_curr

    plt.plot(y, label=f'a = {a}')


plt.plot(x, 'lightgray')
plt.grid(True)
plt.legend()


np.random.seed(0)
n = np.linspace(0, 0.01, 500)
input = np.sin( 2 * np.pi * 500 * n) 
x =  np.sin( 2 * np.pi * 500 * n) + 0.5 * np.random.randn(len(n))



y = np.zeros_like(x, dtype=float)
y[0] = a * x[0]

plt.subplot(2,1,2)

for a in a_val:
    y = []
    
    y_prev = 0
    for n in range(len(x)):
        y_curr = (1 - a) * y_prev + a * x[n]
        y.append(y_curr)
        y_prev = y_curr

    plt.plot(y, label=f'a = {a}')

#plt.plot(x, 'lightgray')
plt.plot(x, 'lightgray' )
plt.grid()
plt.legend()
    







plt.tight_layout()
plt.show()






# source: mohon127