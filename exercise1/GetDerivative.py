import numpy as np


# Get Derivative Of Cost Function J
def GetDerivative(X, y, theta):
    """
    Get All Partial Derivative
    :param X: X (plus x0)       m×n
    :param y: y                 m×1
    :param theta: the params    n×1
    :return: the partial detivative     n×1
    """
    m = X.shape[0]
    n = X.shape[1]

    # theta transpose
    # theta_t = theta.reshape(1, n)
    # Derivative
    D = np.zeros((n, 1))

    for i in range(n):
        # Compute D[i]
        sum = 0
        for j in range(m):
            th = X[j, :].dot(theta)
            ty = y[j, :]
            h_minux_y = np.subtract(th, ty)

            sum += h_minux_y * X[j][i]

        D[i] = sum / m

    return D


if __name__ == '__main__':
    data = np.loadtxt('ex1data1.txt', delimiter=',')

    m = data.shape[0]
    n = data.shape[1] - 1

    raw_x = np.array(data[:, 0]).reshape(m, 1)
    raw_y = np.array(data[:, 1]).reshape(m, 1)

    x = np.vstack(zip(np.ones(m), raw_x)).reshape(m, n + 1)
    y = raw_y

    theta = np.ones((n + 1, 1))
    D = GetDerivative(x, y, theta)
    print(D)
