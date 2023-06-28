import numpy as np
from numpy.fft import fft
from math import sqrt, log10
import matplotlib.pyplot as plt

debug = False

def dft(x):
    m, n = np.shape(x)
    return fft(x) / (sqrt(m)*sqrt(n))

def toggleDebug(default=0):
    global debug
    if default != 0:
        if default > 0:
            debug = True
        else:
            debug = False
    else:
        debug = not debug

def run_gcv(A, b, func, n):
    lim = n
    d_hat = dft(b)
    n = np.shape(b)[0]
    x_re = np.zeros(np.shape(b))
    GCV = []


    ref = 100
    j = 0
    thresh = 100
    loc = -1
    low = -10
    high = 10
    out = []


    while j < lim:
        if j != 0:
            span = high - low
            low = loc - span/4
            high = loc + span/4
        alphavec = np.logspace(low, high, 10)

        for i in range(max(np.shape(alphavec))):
            alpha = alphavec[i]
            A_hat = func(b, A, alpha)
            resid = abs(np.multiply((A_hat - 1), d_hat))
            rms_resid = sum(np.square(resid)) / n
            trace = sum(A_hat)
            Vdenom = (1 - trace/n)**2

            GCV.append(rms_resid / Vdenom)
            if GCV[-1] < ref:
                ref = GCV[-1]
                x_re = A_hat
                loc = log10(alpha)
            out.append([alpha, GCV[-1]])
        j += 1

    if debug:
        plt.scatter(*zip(*out), 12)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(r'Regularization Parameter \alpha')
        plt.title(r'GCV(\alpha)')
        plt.show()


    return x_re, out