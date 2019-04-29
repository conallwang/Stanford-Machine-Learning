import numpy as np

from src.exercise3.sigmoid import *


def predict(theta1, theta2, x):
    """

    :param theta1:
    :param theta2:
    :param x:
    :return:
    """
    hid_layer = sigmoid(theta1.dot(x.T))
    hid_layer_plus = np.r_[np.ones((1, x.shape[0])), hid_layer]
    output_layer = sigmoid(theta2.dot(hid_layer_plus))

    return output_layer
