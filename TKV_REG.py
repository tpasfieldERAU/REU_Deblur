import numpy as np

def tkv_regularize(b, A, delta):
    L = np.identity(np.shape(b)[0])
    AT = A.transpose()
    LT = L.transpose()
    return np.linalg.inv(AT@A + delta*LT@L)@AT@b