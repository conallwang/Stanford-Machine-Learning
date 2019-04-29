import numpy as np

from src.exercise2.getDerivativeReg import *
from src.exercise2.costFunctionReg import *


def gradientDescentReg(x, y, theta, alpha, l, iters):
    """
    Gradient Descent function with regularization
    :param x:               m×n
    :param y:               m×1
    :param theta:           n×1
    :param alpha:           learning rate
    :param iters:           the num of iters
    :return:                (theta, J_history) % (new parameters theta, all J value in a list)
    """

    J_history = [costFunctionReg(theta, x, y, l)]

    for i in range(iters):
        D = getDerivativeReg(theta, x, y, l)
        theta = theta - alpha * D

        J = costFunctionReg(theta, x, y, l)
        print(J)
        J_history.append(J)

    return theta, J_history