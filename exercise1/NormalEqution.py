import numpy as np
from numpy.linalg import inv

def NormalEqution(x, y):

    m = x.shape[0]
    n = x.shape[1]

    theta = inv(x.T.dot(x)).dot(x.T).dot(y)

    return theta