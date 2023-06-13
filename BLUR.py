import numpy as np
import numpy.random as rnd


def blur(x, A, noise, scale):
    if noise:
        noise = np.array([i + rnd.normal(0., scale, size=None) for i in np.zeros(x.shape[0])])
    else:
        noise = np.zeros(x.shape[0])
    return np.array([i + rnd.normal(0, scale, size=None) for i in A @ x])
