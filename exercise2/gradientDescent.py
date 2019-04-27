import numpy as np
from src.exercise2.costFunction import *
from src.exercise2.getDerivative import *


def gradientDescent(x, y, theta, alpha, iters):
    """

    :param x:
    :param y:
    :param theta:
    :param alpha:
    :param iters:
    :return:
    """
    m, n = x.shape

    J_history = [costFunction(x, y, theta)]
    for i in range(iters):
        D = getDerivative(x, y, theta)
        theta = theta - alpha * D

        J = costFunction(x, y, theta)
        print(J)
        J_history.append(J)

    return theta, J_history