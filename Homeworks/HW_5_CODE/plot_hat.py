import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import sys
plt.rcParams["figure.figsize"] = [16,8]
dx = 0.001
L = np.pi
x = L*np.arange(-1+dx, 1+dx, dx)
n = len(x)
nq = int(np.floor(n/4))
ub = np.pi/2

def fx(x, ub):
    if abs(x) < ub:
        return (np.pi/2)-abs(x)    
    else:
        return 0
f = np.array([fx(e,ub) for e in x])
fig, ax = plt.subplots()
ax.plot(x, f, '-', color='b', label="Original hat function")
ax.set_xlabel("$x$")
ax.set_ylabel("$f(x)$")
plt.savefig("hat_function.png")



