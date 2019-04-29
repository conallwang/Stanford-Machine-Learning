import numpy as np


def sigmoid(x):
    """
    The sigmoid function

    ** In neural network, It's called activation function
    :param x:           m×n
    :return:            The function value of x
    """
    return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    print(sigmoid(-10))
