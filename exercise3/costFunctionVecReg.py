import numpy as np

from src.exercise3.sigmoid import *


def costFunctionVecReg(theta, x, y, l):
    """

    :param theta:
    :param x:
    :param y:
    :param l:               lambda
    :return:
    """
    m, n = x.shape

    h = sigmoid(x.dot(theta))
    theta_square = theta[1:]**2

    fore = np.multiply(np.log(h), y)
    post = np.multiply(np.log(1-h), 1-y)

    res = np.sum(fore + post) / -m
    res += np.sum(theta_square) * l / (2*m)

    return res
