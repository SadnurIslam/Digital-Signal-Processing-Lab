
import numpy as np
import matplotlib.pyplot as plt















































































''' 
Generating elementary signals like Unit Step, Ramp, Exponential, Sine, and 
Cosine sequences. 

'''

import numpy as np
import matplotlib.pyplot as plt


def ploting(x, y, title, labeling , sub):
    plt.subplot(3,2,sub)
    plt.title(title)
    plt.stem(x, y, label = labeling)
    plt.legend()
    plt.xlabel('time axis')
    plt.ylabel('amplitude')
    plt.grid(True)
    plt.xticks(np.arange(x[0], x[-1]+1, 1))
    
    
    
    

n = np.arange(-10, 11, 1)
amplitude  = np.where(n>=0, 1, 0)
ploting(n, amplitude, 'unit_step', 'unit_step', 1)

amplitude = 0.8 ** n
ploting(n, amplitude, 'exponential', 'exponential', 2)

amplitude = np.where(n>=0, n, 0)
ploting(n, amplitude, 'unit_ramp', 'unit_ramp', 3)



amplitude = np.sin(2*np.pi*0.1*n)
ploting(n, amplitude, 'sin', 'sin', 4)

amplitude = np.cos(2*np.pi*0.1*n)
ploting(n, amplitude, 'cos', 'cos', 5)



plt.tight_layout()
plt.show()




#2 no
import numpy as np
import matplotlib.pyplot as plt

# Continuous signal parameters
f_signal = 9  # Hz (signal frequency)
t_cont = np.linspace(0, 1, 1000)  # High-resolution time axis (simulates continuous time)
x_cont = np.sin(2 * np.pi * f_signal * t_cont)  # Continuous-time signal

# Sampling rates (below and above Nyquist)
sampling_rates = [5, 15, 20]  # Hz (Nyquist rate is 2*f = 18 Hz for f=9 Hz)

plt.figure(figsize=(12, 9))

for i, fs in enumerate(sampling_rates):
    t_samp = np.arange(0, 1, 1/fs)
    x_samp = np.sin(2 * np.pi * f_signal * t_samp)

    plt.subplot(3, 1, i + 1)
    plt.plot(t_cont, x_cont, 'lightgray', label='Original Signal')
    plt.stem(t_samp, x_samp, linefmt='r-', markerfmt='ro', basefmt='k-', label='Sampled Signal')
    plt.title(f"Sampling at {fs} Hz {'(Aliasing)' if fs < 2*f_signal else '(No Aliasing)'}")
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.suptitle("Effect of Sampling and Aliasing", fontsize=16, y=1.02)
plt.show()


#3 no

import numpy as np
import matplotlib.pyplot as plt

# Sample range
n = np.arange(0, 20)

# Frequencies to test
omegas = [np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]

# Plotting
plt.figure(figsize=(10, 8))

for i, omega in enumerate(omegas):
    x = np.cos(omega * n)
    plt.subplot(2, 2, i + 1)
    plt.stem(n, x)  # Removed 'use_line_collection=True'
    plt.title(f'ω = {omega:.2f} rad/sample')
    plt.xlabel('n')
    plt.ylabel('x[n]')
    plt.grid(True)

plt.tight_layout()
plt.suptitle("Discrete-Time Sinusoids for Different ω", fontsize=16, y=1.02)
plt.show()




'''
Consider the continuous-time analog signal x(t)=3cos(100πt). Sample the analog 
signal at 200 Hz and 75 Hz. Show the discrete-time signal after sampling. ⟹ 
realization. 

'''
import numpy as np
import matplotlib.pyplot as plt

n = np.linspace(0, 0.04, 1000)
y = 3 * np.cos(100*np.pi*n)
plt.plot(n, y, label = 'input signal')
plt.xlabel('time')
plt.ylabel('amplitude')
plt.legend()



n = np.arange(0, 0.04, 1/75)
y = y = 3 * np.cos(100*np.pi*n)
plt.stem(n, y, 'g', label = 'sample 75',  basefmt=" ")
plt.xlabel('time')
plt.ylabel('amplitude')
plt.legend()


n = np.arange(0, 0.04, 1/200)
y = y = 3 * np.cos(100*np.pi*n)
plt.stem(n, y, 'b', label = 'sample 200')
plt.xlabel('time')
plt.ylabel('amplitude')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()






'''
Problem statement: 
Consider the analog signal: xa(t)=3cos(200πt)+5sin(600πt)+10cos(1200πt). 
Show the effect of sampling rate.  

'''

import numpy as np
import matplotlib.pyplot as plt

n = n1 = np.linspace(0, 0.01, 800)


input_signal = 3 * np.cos(200*np.pi*n) + 5*np.sin(600*np.pi*n) + 10*np.cos(1200*np.pi*n)
plt.subplot(4,1,1)
plt.plot(n, input_signal, label = 'input signal')
plt.grid(True)


n = np.arange(0, 0.01, 1/800)
low_sampled = 3 * np.cos(200*np.pi*n) + 5*np.sin(600*np.pi*n) + 10*np.cos(1200*np.pi*n)
plt.subplot(4,1,2)
plt.plot(n1, input_signal,'lightgray')
plt.stem(n, low_sampled, label = 'sampled at low of nyquist')
plt.grid(True)


n = np.arange(0, 0.01, 1/1200)
low_sampled = 3 * np.cos(200*np.pi*n) + 5*np.sin(600*np.pi*n) + 10*np.cos(1200*np.pi*n)
plt.subplot(4,1,3)
plt.plot(n1, input_signal,'lightgray')
plt.stem(n, low_sampled, label = 'sampled at nyquist')
plt.grid(True)

n = np.arange(0, 0.01, 1/2000)
low_sampled = 3 * np.cos(200*np.pi*n) + 5*np.sin(600*np.pi*n) + 10*np.cos(1200*np.pi*n)
plt.subplot(4,1,4)
plt.plot(n1, input_signal,'lightgray')
plt.stem(n, low_sampled, label = 'sampled at high of nyquist')
plt.grid(True)

plt.legend()
plt.tight_layout()
plt.show()






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



'''
Given 
x(n)=[1,3,−2,4] 
y(n)=[2,3,−1,3] 
z(n)=[2,−1,4,−2] 
Find the correlation between x(n) & y(n) and y(n) & z(n). ⟹ observe the 
realization.
'''

import numpy as np
import matplotlib.pyplot as plt

# Define the sequences
x = np.array([1, 3, -2, 4])
y = np.array([2, 3, -1, 3])
z = np.array([2, -1, 4, -2])


# Function to calculate normalized correlation 

def normalized_corr(x, y):
    numerator = np.sum(x * y)
    denominator = np.sqrt(np.sum(x**2)) * np.sqrt(np.sum(y**2))
    return numerator / denominator


# Calculate the normalized correlation
def correlation(x, y):
        N = len(x) + len(y) - 1
        result = np.zeros(N)

        for i in range(N):
              sum = 0
              for k in range(len(x)):
                    if i-k>=0 and i-k<len(y):
                        sum += x[k] * y[i-k]
              result[i] = sum
        return result


r_xy = normalized_corr(x, y) 
r_yz = normalized_corr(y, z)
s = str(r_xy)
s = 'correlation value: ' + s[:5]

r_xy_0 = correlation(x, y[::-1])
r_yz_0 = correlation(y, z[::-1])
lag = np.arange(-len(x) + 1, len(y))

# Display the results

plt.subplot(2, 1, 1)
plt.title('Correlation between x(n) and y(n)')
plt.stem(lag , r_xy_0, label= s, linefmt='b-', basefmt='k-')
plt.legend()
plt.xlabel('Lag')
plt.ylabel('Amplitude')
plt.grid(True)



s = str(r_yz)
s = 'correlation value: ' +s[:5]

plt.subplot(2, 1, 2)
plt.title('Correlation between y(n) and z(n)')
plt.stem(lag, r_yz_0, label= s, linefmt='g-', basefmt='k-')
plt.legend()
plt.xlabel('Lag')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()






'''
Filter realization using 6-point averaging, 6-point differencing equations. 
'''

import numpy as np
import matplotlib.pyplot as plt

# Sample input signal (analog-like)
np.random.seed(0)
n = np.linspace(0, 1, 200)
x = np.sin(2 * np.pi * 5 * n) + 0.5 * np.random.randn(len(n))  # 5 Hz sine + noise

# 6-point Averaging Filter
def avg_filter(x):
    y = np.zeros_like(x)
    for i in range(5, len(x)):
        y[i] = np.sum(x[i-5:i+1]) / 6
    return y

# 6-point Differencing Filter
def diff_filter(x):
    y = np.zeros_like(x)
    for i in range(5, len(x)):
        y[i] = (x[i] - x[i-1] + x[i-2] - x[i-3] + x[i-4] - x[i-5]) / 6
    return y

# Plotting function
def plot_signal(x, y, title, color='b'):
    plt.plot(x, y, color=color)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

# Apply filters
y_avg = avg_filter(x)
y_diff = diff_filter(x)

# Plot results
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plot_signal(n, x, 'Original Analog-like Signal', 'gray')

plt.subplot(3, 1, 2)
plot_signal(n, y_avg, '6-Point Averaging Filter Output', 'green')

plt.subplot(3, 1, 3)
plot_signal(n, y_diff, '6-Point Differencing Filter Output', 'red')

plt.tight_layout()
plt.show()