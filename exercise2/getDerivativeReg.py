import numpy as np

from src.exercise2.sigmoid import *


def getDerivativeReg(theta, x, y, l):
    """
    compute all derivative with regularization
    :param x:           m×n
    :param y:           m×1
    :param theta:       n×1
    :param l:           lambda
    :return:            Derivative with regularization n×1
    """
    m, n = x.shape

    # service for vector compute
    h = sigmoid(x.dot(theta).reshape(m, 1))
    h_minux_y = np.subtract(h, y)

    # as before
    res = np.zeros((n, 1))
    for i in range(n):
        tmp = np.multiply(h_minux_y, x[:, i].reshape(m, 1))
        res[i] = np.sum(tmp) / m

    # the part of regularization
    for i in range(1, n):
        res[i] += l * theta[i] / m

    return res
