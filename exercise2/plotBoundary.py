import matplotlib.pyplot as plt
import numpy as np


def plotBoundary(x, theta):
    """

    :param theta:
    :return:
    """
    plot_x = [min(x[:, 1])-2, max(x[:, 1])+2]
    plot_y = np.multiply(np.multiply(theta[1], plot_x) + theta[0], (-1/theta[2]))

    plt.plot(plot_x, plot_y, 'b-')