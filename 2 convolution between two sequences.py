import numpy as np

# Define input sequences
x = np.array([1, 2, 3, 4])  # First sequence
h = np.array([1, 1, 0.5])    # Second sequence

# Perform convolution
y = np.convolve(x, h)

#y = [int(val) if val.is_integer() else float(val) for val in y]

# Display result
print("Input Sequence x(n):", x)
print("Impulse Response h(n):", h)
print("Convolution Result y(n):", y)
