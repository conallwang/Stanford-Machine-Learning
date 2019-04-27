import matplotlib.pyplot as plt
import numpy as np


def plotData(data, xlabel, ylabel):
    """
    Make a new figure to plot data for this exercise
    :param data: [x, y] raw data
    :param xlabel: x label
    :param ylabel: y label
    :return: None
    """
    neg = data[data[:, -1] == 0][:, :-1]
    pos = data[data[:, -1] == 1][:, :-1]

    # print(neg)
    # print(pos)

    plt.figure()
    plt.plot(neg[:, 0], neg[:, 1], 'ko')
    plt.plot(pos[:, 0], pos[:, 1], 'ro')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
