import numpy as np

from src.exercise1.GetDerivative import *
from src.exercise1.computeCost import *


def GradientDescent(X, y, theta, alpha, num_iters, debug=False):
    """
    Run Gradient Descent !
    :param X: X (plus X0)           m×n
    :param y: y                     m×1
    :param theta: the params        n×1
    :param alpha: learning rate     float
    :param num_iters: iter times    int
    :return: new theta              n×1
             his Cost
    """
    m = X.shape[0]
    n = X.shape[1]

    his_J = []
    his_J.append(computeCost(X, y, theta))
    for i in range(num_iters):
        D = GetDerivative(X, y, theta)
        theta = np.subtract(theta, alpha * D)

        his_J.append(computeCost(X, y, theta))
        if debug:
            print('iter: %d' % i)
            print('Current Cost: %s' % his_J[i])

    return theta, his_J

