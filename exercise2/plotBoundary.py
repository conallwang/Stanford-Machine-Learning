import matplotlib.pyplot as plt
import numpy as np

from src.exercise2.mapFeature import *


def plotBoundary(x, theta):
    """
    plot the boundary

    if boundary is a line, just choose two points to draw the line

    otherwise, boundary is non-linear
    we use pyplot.contour to draw, just as show beneath

    :param theta:       n×1
    :return:            None
    """
    n_dimension = x.shape[1]

    if n_dimension <= 3:
        plot_x = [min(x[:, 1]) - 2, max(x[:, 1]) + 2]
        plot_y = np.multiply(np.multiply(theta[1], plot_x) + theta[0], (-1 / theta[2]))

        plt.plot(plot_x, plot_y, 'b-')

    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = [np.array([mapFeature(np.array([u[i]]).reshape(1, 1), np.array((v[j])).reshape(1, 1)).dot(theta)
                       for i in range(len(u))]) for j in range(len(v))]

        y = np.array(z).reshape(len(u), len(v))
        plt.contour(u, v, y, levels=[0.0])
