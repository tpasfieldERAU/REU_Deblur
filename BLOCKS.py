import numpy as np
from random import randint, random


def generate_block_signal():
    out = np.zeros(512)
    n = 1
    flag = 0
    while n < 512:
        if randint(0, 45) > 0:
            out[n] = out[n-1]
        else:
            if flag:
                out[n] = 0
                flag = 0
            else:
                out[n] = random()
                flag = 1

        if n == 511:
            out[n] = 0
        n += 1

    return out
