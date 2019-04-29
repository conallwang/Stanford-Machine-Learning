import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from src.exercise2.plotData import *
from src.exercise2.sigmoid import *
from src.exercise2.gradientDescent import *
from src.exercise2.plotBoundary import *

# ================ Part 1: Plotting =====================
data = np.loadtxt('./ex2data1.txt', delimiter=',')

m = data.shape[0]
n = data.shape[1]
print(m)
print(n)

raw_x = np.array(data[:, :-1]).reshape(m, n - 1)
raw_y = np.array(data[:, -1]).reshape(m, 1)
# print(raw_x)
# print(raw_y)

x = np.c_[np.ones((m, 1)), raw_x].reshape(m, n)
y = raw_y
print(x)
print(y)

# plotData
xlabel = 'Exam 1 score'
ylabel = 'Exam 2 score'
plotData(data, xlabel, ylabel)
plt.show()

# =================== Part 2: Optimizing ===============
# Gradient Descent has been realized in src/exercise2/gradientDescent.py
# But in this situation, Gradient Descent is too slow to converge
#
# If you want to know how to realize Gradient Descent, you can refer to src/exercise2/gradientDescent.py

initial_theta = np.zeros((n, 1))

# Using scipy.optimize.minimize to solve
res = minimize(costFunction, initial_theta, method='TNC', args=(x, y),
               options={'gtol': 1e-3, 'disp': True, 'maxiter': 1000})

print(res)
theta = res.x
cost = res.fun

# Print theta on screen
print('the result of Theta: ', theta)
print('The Res.fun: ', cost)

# plot data and decisionBoundary
plotData(data, xlabel, ylabel)
plotBoundary(x, theta)
plt.show()

# predict
predict_x = np.array([1, 45, 85])
predict = sigmoid(predict_x.dot(theta.reshape(n, 1)))
print('the result of predict: ', predict)
