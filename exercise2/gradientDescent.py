import numpy as np
from src.exercise2.costFunction import *
from src.exercise2.getDerivative import *


def gradientDescent(x, y, theta, alpha, iters):
    """
    Realize Gradient Descent, but it's to slow
    :param x:               m×n
    :param y:               m×1
    :param theta:           n×1
    :param alpha:           learing rate
    :param iters:           the num of iters
    :return:                (theta, J_history) % (new parameters theta, all J value in a list)
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