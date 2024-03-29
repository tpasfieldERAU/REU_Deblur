# TODO Use normal distribution for determining parameter. Switch to Monte Carlo sampling rather than uniform sampling.

import numpy as np
from numpy.fft import fft
from math import sqrt, log10
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

debug = False
pBar = True


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


def togglePbar(default=0):
    global pBar
    if default != 0:
        if default > 0:
            pBar = True
        else:
            pBar = False
    else:
        pBar = not pBar


def run_gcv(A, b, func, n):
    global debug
    global pBar

    lim = n
    d_hat = dft(b)
    n = b.shape[0]
    x_re = np.zeros(b.shape)
    GCV = []

    ref = 100
    j = 0
    loc = -1
    low = -4
    high = 1
    out = []

    if pBar:
        with tqdm(total=int(lim*10)) as pbar:
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
                    pbar.update(1)
                j += 1
    else:
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
        #plt.xlabel(r'Regularization Parameter \alpha')
        #plt.title(r'GCV( $\alpha$ )')
        plt.show()


    return x_re, out