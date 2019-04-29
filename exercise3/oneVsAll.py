import numpy as np

from src.exercise3.costFunctionVecReg import *
from src.exercise3.getDerivativeVecReg import *


def oneVsAll(x, y, K, alpha, iters, l):
    """
    To train multi classification models (Using several Logistic Regression)
    :param x:                   m×n
    :param y:                   m×1
    :param alpha:               learning rate
    :param iters:               num of iters
    :param l:                   lambda
    :return:                    The theta matrix (train result)
    """
    m, n = x.shape

    # all theta
    all_theta = np.zeros((K + 1, n))

    # Gradient Descent Training
    for i in range(1, K+1):
        theta = np.zeros((n, 1))
        tmp_y = y.copy()
        tmp_y[tmp_y != i] = 0
        tmp_y[tmp_y == i] = 1
        print('Classifier %d: ' % i)
        for j in range(iters):
            D = getDerivativeVecReg(theta, x, tmp_y, l)
            theta = theta - alpha * D

            J = costFunctionVecReg(theta, x, tmp_y, l)
            # print(J)
        all_theta[i, :] = theta.T
        print('Classifier %d: Trained Successfully!' % i)

    return all_theta
