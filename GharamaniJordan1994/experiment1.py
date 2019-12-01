import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.colors import LogNorm
#from sklearn import mixture
from GaussUtils import GenerateGaussianData

"""
References:
[GJ93] Ghahramani, Jordan: "Supervised learning from incomplete data via an EM approach", NIPS 1993
"""

# Random
random = np.random.RandomState(0)

# dimension of data. 2 is for plotting...
D = 2

# expected number of samples
N = 1000

# number of Gaussians
M = 2

# center variance
center_variance = 0.1

ggd = GenerateGaussianData(D=D, N=N, M=M, random=random, center_variance=center_variance)


plt.plot(ggd.X[:, 0], ggd.X[:, 1], '.')
plt.show()





