import numpy as np

from src.exercise3.sigmoid import *


def getDerivativeVec(theta, x, y):
    """

    :param theta:
    :param x:
    :param y:
    :return:
    """
    m, n = x.shape

    h = sigmoid(x.dot(theta))
    h_minus_y = h - y

    D = x.T.dot(h_minus_y) / m

    return D