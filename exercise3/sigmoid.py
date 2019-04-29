import numpy as np


def sigmoid(x):
    """

    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    print(sigmoid(-10))
