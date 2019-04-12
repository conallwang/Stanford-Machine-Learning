import numpy as np


def FeatureScale(x):
    """
    Scale the features to make Gradient Descent more fast
    :param x: Params Matrix
    :return: new Params Matrix (Feature Scaling)
    """
    m = x.shape[0]
    n = x.shape[1]

    new_x = np.ones((m, n))
    for i in range(1, n):
        max = np.max(x[:, i])
        min = np.min(x[:, i])
        diff = max - min

        mean = np.sum(x[:, i]) / m
        new_x[:, i] = (x[:, i] - mean) / diff

    return new_x
