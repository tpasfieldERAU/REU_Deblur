import numpy as np
from numpy.linalg import inv

def tkv_reconstruct(b, A, alpha):
    AT = np.transpose(A)

    L = np.identity(512)
    LT = np.transpose(L)

    x_re = inv(AT @ A + alpha * LT @ L) @ (AT @ b)

    diff = [abs(x_re[i] - b[i]) for i in range(len(x_re))]
    avgError = np.mean(diff)

    return x_re, avgError
