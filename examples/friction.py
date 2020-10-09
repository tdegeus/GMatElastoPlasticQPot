import sys
import matplotlib.pyplot as plt
import numpy as np

ret = np.genfromtxt(sys.argv[1], delimiter=",")

fig, ax = plt.subplots()
ax.plot(ret[:, 0], ret[:, 1])
plt.show()
