import numpy as np

def computeCost(X, y, theta):
    """
    Compute Cost Function J
    :param X: X (plus X0)       m×n
    :param y: y                 m×1
    :param theta: the params    n×1
    :return: Cost               int
    """
    m = X.shape[0]
    n = X.shape[1]

    sum = 0
    for i in range(m):
        th = X[i, :].dot(theta)
        ty = y[i, :]

        h_minus_y = np.subtract(th, ty)
        sum += pow(h_minus_y, 2)

    res = sum / (2*m)
    return res


if __name__ == '__main__':
    data = np.loadtxt('ex1data1.txt', delimiter=',')

    m = data.shape[0]
    n = data.shape[1] - 1

    raw_x = np.array(data[:, 0]).reshape(m, 1)
    raw_y = np.array(data[:, 1]).reshape(m, 1)

    x = np.vstack(zip(np.ones(m), raw_x)).reshape(m, n + 1)
    y = raw_y

    theta = np.ones((n + 1, 1))
    costJ = computeCost(x, y, theta)
    print(costJ)
