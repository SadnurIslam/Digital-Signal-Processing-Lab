'''
Given x(n)=[1,3,−2,4] y(n)=[2,3,−1,3] z(n)=[2,−1,4,−2]
Find the correlation between x(n) & y(n) and y(n) & z(n). ⟹ observe the
realization

'''




import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 3, -2, 4])
y = np.array([2, 3, -1, 3])
z = np.array([2, -1, 4, -2])


# Cross-correlation
r_xy = np.correlate(x, y, mode='full')
r_yz = np.correlate(y, z, mode='full')

def correlation(x, y):
    x_len = len(x)
    y_len = len(y)
    y_rev = np.flip(y)  # Reverse y for correlation
    y = y_rev
    result = []
    for n in range(0, x_len + y_len - 1):
        sum = 0
        for k in range(x_len):
            if n - k >= 0 and n - k < y_len:
                sum += x[k] * y[n - k]
        result.append(sum)
    return np.array(result)

r_xy_custom = correlation(x, y)
r_yz_custom = correlation(y, z)

print("Cross-correlation r_xy:", r_xy)
print("Custom Cross-correlation r_xy:", r_xy_custom)
print("Cross-correlation r_yz:", r_yz)
print("Custom Cross-correlation r_yz:", r_yz_custom)

lags = np.arange(-len(x)+1, len(x))  # Lags for plotting

# Plot r_xy
plt.subplot(1, 2, 1)
plt.stem(lags, r_xy)
plt.title("Correlation r_xy[m]")
plt.xlabel("Lag (m)")
plt.ylabel("r_xy[m]")
plt.grid(True)

# Plot r_yz
plt.subplot(1, 2, 2)
plt.stem(lags, r_yz)
plt.title("Correlation r_yz[m]")
plt.xlabel("Lag (m)")
plt.ylabel("r_yz[m]")
plt.grid(True)

plt.tight_layout()
plt.show()
