import numpy as np
from random import randint, random, uniform
from SMOOTH import hermite_interpolate


def generated_combined_signal():
    out = np.zeros(512)
    dist = randint(64, 128)
    point = randint(128, 512-128)
    height = uniform(0.6, 1)

    for i in range(dist):
        out[(point-dist) + i] = hermite_interpolate([0, 0, height, 0], i/dist)

    for i in range(dist):
        out[point + i] = hermite_interpolate([0, height, 0, 0], i/dist)

    n = 32
    flag = 0
    while n < point - dist - 32:
        if randint(0, 75) > 0:
            out[n] = out[n-1]
        else:
            if flag:
                out[n] = 0
                flag = 0
            else:
                out[n] = random()
                flag = 1
        n += 1

    n = point + dist + 32
    flag = 0
    while n < 511:
        if randint(0, 75) > 0:
            out[n] = out[n-1]
        else:
            if flag:
                out[n] = 0
                flag = 0
            else:
                out[n] = random()
                flag = 1
        n += 1
    return out
