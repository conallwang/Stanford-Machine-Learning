import numpy as np

from src.exercise3.sigmoid import *


def predictOneVsAll(theta, x):
    """
    Using Logistic Regression to predict the classification
    :param theta:           n×1
    :param x:               m×n
    :return:                the result of prediction
    """
    m, n = x.shape
    res = sigmoid(x.dot(theta[1:, :].T))

    predict = np.argmax(res, axis=1).reshape(m, 1) + 1

    return predict