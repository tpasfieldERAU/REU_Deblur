import numpy as np
import scipy.linalg as la
import scipy.stats as st


def kernel(n, sigma):
    tmp = np.linspace(0, n/sigma, n+1)
    a = np.diff(st.norm.cdf(tmp))
    a = a/sum(a)/2
    return la.toeplitz(a)


def generateBlurMatrix(n, curve):
    a = []
    for i in curve:
        a.append(i)
    if len(a) < n:
        for i in range(n - len(a)):
            a.append(0)
    if len(a) > n:
        a = a[0:n]
    a = np.array(a)
    return la.toeplitz(a)

def genkernel(ys_size, sigma):
    tmp = np.linspace(0, ys_size/sigma, ys_size+1)
    a = np.diff(st.norm.cdf(tmp))
    a = a / a.sum()
    return a
