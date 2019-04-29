import numpy as np

from src.exercise3.sigmoid import *


def getDerivativeVecReg(theta, x, y, l):
    """

    :param theta:
    :param x:
    :param y:
    :param l:               lambda
    :return:
    """
    m, n = x.shape

    h = sigmoid(x.dot(theta))
    h_minus_y = h - y

    D = x.T.dot(h_minus_y) / m
    theta[0] = 0
    D += l * theta / m

    return D
