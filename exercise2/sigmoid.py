import math
import numpy as np


def sigmoid(x):
    """
    sigmoid function
    :param x: param
    :return: sigmoid(x)
    """
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    print(sigmoid(-1000))
