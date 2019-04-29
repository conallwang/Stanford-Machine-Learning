import numpy as np

from src.exercise3.sigmoid import *


def predictOneVsAll(theta, x):
    """

    :param theta:
    :param x:
    :return:
    """
    m, n = x.shape
    res = sigmoid(x.dot(theta[1:, :].T))

    predict = np.argmax(res, axis=1).reshape(m, 1) + 1

    return predict