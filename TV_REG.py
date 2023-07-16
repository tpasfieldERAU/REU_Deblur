import numpy as np


def tv_regularize(b, A, delta, cutoff=1e-7):
    n = b.shape[0]
    y = b
    AT = A.transpose()
    D = np.eye(n) - np.eye(n, k=-1)

    def set_values(y, D):
        psi = 1 / np.sqrt(np.power(D @ y, 2) + 1e-12)
        psi = np.reshape(psi, (1, n))[0]
        DT = D.transpose()
        dpsi = np.diag(psi)
        return DT @ dpsi @ D

    buf = y
    err = 100
    while err > cutoff:
        L = set_values(buf, D)
        grad = AT @ (A @ buf - b) + delta*L@buf
        hess = AT @ A + delta*L
        step = np.linalg.solve(hess, -1.*grad)

        y = buf
        buf = buf + step
        err = (np.square(buf - y)).mean(axis=None)
    return buf
