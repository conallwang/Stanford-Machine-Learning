import numpy as np
from scipy.io import loadmat


from src.exercise3.plotData import *
from src.exercise3.oneVsAll import *
from src.exercise3.predictOneVsAll import *

# ===================== Part 1: Plotting Data ========================

data = loadmat('ex3data1.mat')

# print(type(data))
# print(data)

raw_x = data['X']
raw_y = data['y']

m = raw_x.shape[0]
n = raw_x.shape[1] + 1

x = np.c_[np.ones((m, 1)), raw_x]
y = raw_y

# print(raw_x)
# print(raw_y)

plotData(raw_x)
plt.show()

# ========================= Part 2: Training Multi Classifier ========================

# Some Arguments
K = 10
alpha = 0.01
num_iters = 5000
l = 1

theta = oneVsAll(x, y, K, alpha, num_iters, l)

print(theta)

# predict
predict = predictOneVsAll(theta, x)

res = y - predict
res[res != 0] = 1

print('Error rate: %f' % (np.sum(res) / m))

# Play with it
c = int(input('Please input a feature row: '))
while c:
    predict_x = x[c, :].reshape(1, n)
    predict_y = predictOneVsAll(theta, predict_x)
    print(predict_y)
    draw = predict_x[0, 1:].reshape(20, 20)
    plt.figure()
    plt.imshow(draw)
    plt.show()
    c = int(input('Please input a feature row: '))

print('Test Success!')