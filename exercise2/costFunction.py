import numpy as np
from src.exercise2.sigmoid import *


def costFunction(theta, x, y):
    """
    computer cost function of logistic regression
    :param x: x         m×n
    :param y: y         m×1
    :param theta: theta n×1
    :return: J(θ)
    """
    m, n = x.shape

    # np.seterr(divide='ignore')
    res = 0
    for i in range(m):
        h = sigmoid(x[i, :].dot(theta))
        res -= (y[i] * np.log(h))
        res -= (1 - y[i]) * np.log(1 - h)

    return res / m


