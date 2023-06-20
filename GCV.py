import numpy as np
from numpy.fft import fft
from math import sqrt
import matplotlib.pyplot as plt


def dft(x):
    m, n = np.shape(x)
    return fft(x) / (sqrt(m)*sqrt(n))


def run_gcv(A, b, func, n):
    d_hat = dft(b)
    alphavec = np.logspace(-6, 4, n)
    n = np.shape(b)[0]

    GCV = np.zeros(np.shape(alphavec))

    x_re = np.zeros(np.shape(b))
    ref = 100
    for i in range(max(np.shape(alphavec))):
        alpha = alphavec[i]
        A_hat = func(b, A, alpha)
        resid = abs(np.multiply((A_hat - 1), d_hat))
        rms_resid = sum(np.square(resid)) / n
        trace = sum(A_hat)
        Vdenom = (1 - trace/n)**2

        GCV[i] = rms_resid / Vdenom
        if GCV[i] < ref:
            ref = GCV[i]
            x_re = A_hat

        # Here for diagnostic purposes, especially for TV Regularization
        #print("STEP")

    plt.loglog(alphavec, GCV, '-.')
    plt.xlabel(r'Regularization Parameter \alpha')
    plt.title(r'GCV(\alpha)')
    plt.show()

    return x_re, min(GCV)