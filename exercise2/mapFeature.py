import numpy as np


def mapFeature(x1, x2, degree=6):
    """

    :param x1:
    :param x2:
    :param degree:
    :return:
    """
    m = x1.shape[0]
    res = np.ones((m, 1))
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            plus = np.multiply(x1**(i-j), x2**j).reshape(m, 1)
            res = np.c_[res, plus]

    return res
