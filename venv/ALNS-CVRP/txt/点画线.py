import matplotlib.pyplot as plt
import numpy as np

data = np.array([(238, 106), (126, 178), (184, 291), (304, 329), (184, 291), (126, 178), (238, 106)])
x = [item[0] for item in data]
y = [item[1] for item in data]

plt.plot(x, y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
