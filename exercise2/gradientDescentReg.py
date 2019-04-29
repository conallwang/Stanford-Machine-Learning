import numpy as np

from src.exercise2.getDerivativeReg import *
from src.exercise2.costFunctionReg import *


def gradientDescentReg(x, y, theta, alpha, l, iters):
    """

    :param x:
    :param y:
    :param theta:
    :param alpha:
    :param iters:
    :return:
    """

    J_history = [costFunctionReg(theta, x, y, l)]

    for i in range(iters):
        D = getDerivativeReg(theta, x, y, l)
        theta = theta - alpha * D

        J = costFunctionReg(theta, x, y, l)
        print(J)
        J_history.append(J)

    return theta, J_history