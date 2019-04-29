import numpy as np

from src.exercise3.sigmoid import *


def getDerivativeVec(theta, x, y):
    """
    Get Derivative, vector form
    :param theta:           n×1
    :param x:               m×n
    :param y:               m×1
    :return:                The Derivative
    """
    m, n = x.shape

    h = sigmoid(x.dot(theta))
    h_minus_y = h - y

    D = x.T.dot(h_minus_y) / m

    return D