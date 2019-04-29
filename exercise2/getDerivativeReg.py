import numpy as np

from src.exercise2.sigmoid import *


def getDerivativeReg(theta, x, y, l):
    """

    :param x:           m×n
    :param y:           m×1
    :param theta:       n×1
    :param l:           lambda
    :return:
    """
    m, n = x.shape

    h = sigmoid(x.dot(theta).reshape(m, 1))
    h_minux_y = np.subtract(h, y)
    res = np.zeros((n, 1))
    for i in range(n):
        tmp = np.multiply(h_minux_y, x[:, i].reshape(m, 1))
        res[i] = np.sum(tmp) / m

    for i in range(1, n):
        res[i] += l * theta[i] / m

    return res
