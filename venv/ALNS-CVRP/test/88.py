import numpy as np

results = []

for t in np.arange(0, 2 * np.pi, 0.01):
    x = 50 * (np.sin(t) + 1)
    y = 50 * (np.sin(t) * np.cos(t) + 1)
    print(x)
    if np.round(x).is_integer():
      results.append((int(x), y))

print(results)