import numpy as np
from src.exercise2.sigmoid import *


def getDerivative(x, y, theta):
    """

    :param x:                       m×n
    :param y:                       m×1
    :param theta:                   n×1
    :return: Derivative             m×1
    """
    m, n = x.shape

    D = np.zeros((n, 1))
    h_minus_y = (sigmoid(x.dot(theta)) - y)
    for i in range(n):
        tmp = np.multiply(h_minus_y, x[:, i].reshape(m, 1))
        D[i] = np.sum(tmp) / m

    return D
