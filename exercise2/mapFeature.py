import numpy as np


def mapFeature(x1, x2, degree=6):
    """
    Map Feature, to generate features in any dimension
    :param x1:          must be a matrix! if x1 is a num, just using numpy.reshape(1, 1) to convert
    :param x2:          must be a matrix! if x2 is a num, just using numpy.reshape(1, 1) to convert
    :param degree:      the dimension to generate
    :return:            new features (generally high dimension)
    """
    m = x1.shape[0]
    res = np.ones((m, 1))
    for i in range(1, degree + 1):
        for j in range(0, i + 1):
            plus = np.multiply(x1**(i-j), x2**j).reshape(m, 1)
            res = np.c_[res, plus]

    return res


