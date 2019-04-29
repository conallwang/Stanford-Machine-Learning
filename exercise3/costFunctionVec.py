import numpy as np

from src.exercise3.sigmoid import *


def costFunctionVec(theta, x, y):
    """

    :param theta:
    :param x:
    :param y:
    :return:
    """
    m, n = x.shape

    h = sigmoid(x.dot(theta)).reshape(m, 1)

    fore = np.multiply(np.log(h), y).reshape(m, 1)
    post = np.multiply(np.log(1-h), 1-y).reshape(m, 1)

    res = sum(np.add(fore, post).reshape(m, 1))
    res /= -m

    return res
