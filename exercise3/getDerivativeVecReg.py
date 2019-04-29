import numpy as np

from src.exercise3.sigmoid import *


def getDerivativeVecReg(theta, x, y, l):
    """
    Get Derivative with regularization, vector form
    :param theta:           n×1
    :param x:               m×n
    :param y:               m×1
    :param l:               lambda
    :return:                The Derivative with regularization
    """
    m, n = x.shape

    h = sigmoid(x.dot(theta))
    h_minus_y = h - y

    D = x.T.dot(h_minus_y) / m
    theta[0] = 0
    D += l * theta / m

    return D
