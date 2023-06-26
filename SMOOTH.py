import numpy as np
from random import random

tension = 0
bias = 0


def hermite_interpolate(values, mu):
    y0, y1, y2, y3 = values
    mu2 = mu * mu
    mu3 = mu2 * mu

    m0 = (y1-y0) * (1+bias) * (1-tension) / 2
    m0 += (y2-y1) * (1-bias) * (1-tension) / 2
    m1 = (y2-y1) * (1+bias) * (1-tension) / 2
    m1 += (y3-y2) * (1-bias) * (1-tension) / 2

    a0 = 2*mu3 - 3*mu2 + 1
    a1 = mu3 - 2*mu2 + mu
    a2 = mu3 - mu2
    a3 = -2*mu3 + 3*mu2

    return a0*y1 + a1*m0 + a2*m1 + a3*y2


def generate_smooth_signal():
    vals = np.zeros(512, dtype=np.float32)
    n = 0
    data = [0, 0]
    for i in range(7):
        data.append(random())
    data.append(0)
    data.append(0)

    step = 0
    for i in range(1, len(data)-2):
        points = data[i-1:i+3]
        n = 0
        while n < 64:
            vals[step] = hermite_interpolate(points, n / 64.)
            n += 1
            step += 1

    return vals

def generate_separated_signal():
    vals = np.zeros(512)
    n = 0
    data = [0, 0]
    for i in range(7):
        if not i%2:
            data.append(random())
        else:
            data.append(0)
    data.append(0)
    data.append(0)

    step = 0
    for i in range(1, len(data)-2):
        points = data[i-1:i+3]
        n = 0
        while n < 64:
            vals[step] = hermite_interpolate(points, n/64.)
            n += 1
            step += 1

    return vals