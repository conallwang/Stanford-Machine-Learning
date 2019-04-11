import matplotlib.pyplot as plt
import numpy as np


def plotData(data, xlabel=None, ylabel=None):
    """
    Plot the data using matplotlib
    :param data: two column [x, y]
    :param xlabel: str
    :param ylabel: str
    :return: None
    """
    x = data[:, 0]
    y = data[:, 1]
    plt.plot(x, y, 'rx')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


if __name__ == '__main__':
    data = np.loadtxt('ex1data1.txt', delimiter=',')

    ylabel = 'Profit in $10,000s'
    xlabel = 'Population of City in 10,000s'
    plotData(data, xlabel, ylabel)
    plt.show()

