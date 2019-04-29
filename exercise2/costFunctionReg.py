import numpy as np

from src.exercise2.sigmoid import *


def costFunctionReg(theta, x, y, l):
    """
    compute cost function with regularization
    :param x:           m×n
    :param y:           m×1
    :param theta:       n×1
    :param l:           lambda
    :return:            J(θ) with regularization
    """
    m = x.shape[0]

    # Just to computer the part of regularization
    theta_square = theta[1:]**2
    res = np.sum(theta_square) * l / 2

    # as before
    for i in range(m):
        h = sigmoid(x[i, :].dot(theta))
        res -= y[i] * np.log(h)
        res -= (1 - y[i]) * np.log(1 - h)

    res /= m

    return res