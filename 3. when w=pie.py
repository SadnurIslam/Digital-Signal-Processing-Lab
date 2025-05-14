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
