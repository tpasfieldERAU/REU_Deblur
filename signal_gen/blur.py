import numpy as np
from numpy.random import normal
from math import sqrt, exp, pi
from scipy.linalg import toeplitz as tpz

def blur(sigma, x, scale):
    s = sigma
    k = []
    for i in range(len(x)):
        k.append((1/(sqrt(2*pi*s*s)) * exp(-i**2 / (2 * s * s))))

    k = np.transpose(np.array(k))
    A = tpz(k, k)

    b = np.matmul(A, x)
    b = [i + normal(0, scale, size=None) for i in b]
    return b, A
