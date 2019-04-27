import numpy as np

from src.exercise2.sigmoid import *


def costFunctionReg(theta, x, y, l):
    """

    :param x:
    :param y:
    :param theta:
    :param l:           lambda
    :return:
    """
    m = x.shape[0]

    theta_square = theta[1:]**2
    res = np.sum(theta_square) * l / 2
    for i in range(m):
        h = sigmoid(x[i, :].dot(theta))
        res -= y[i] * np.log(h)
        res -= (1 - y[i]) * np.log(1 - h)

    res /= m

    return res