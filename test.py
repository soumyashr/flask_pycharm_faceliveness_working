import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(1, 21, 200)
y = np.exp(-x)

fig, ax = plt.subplots()
ax.plot(x, y)
ax.hlines(y=0.2, xmin=0.01, xmax=30, linewidth=2, color='r')

plt.show()