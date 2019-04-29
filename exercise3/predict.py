import numpy as np

from src.exercise3.sigmoid import *


def predict(theta1, theta2, x):
    """
    Using theta (has been trained) to predict the classification
    :param theta1:              25×401  (in this example)
    :param theta2:              10×26   (in this example)
    :param x:                   5000×401(in this example)
    :return:                    output_layer in neural network
    """
    hid_layer = sigmoid(theta1.dot(x.T))
    hid_layer_plus = np.r_[np.ones((1, x.shape[0])), hid_layer]
    output_layer = sigmoid(theta2.dot(hid_layer_plus))

    return output_layer
