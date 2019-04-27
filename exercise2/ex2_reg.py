import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from src.exercise2.plotData import *
from src.exercise2.mapFeature import *
from src.exercise2.costFunctionReg import *
from src.exercise2.getDerivativeReg import *
from src.exercise2.gradientDescentReg import *

# ================= Part 1: Visualizing the data ================

data = np.loadtxt('ex2data2.txt', delimiter=',')

m, n = data.shape

raw_x = np.array(data[:, :-1]).reshape(m, n - 1)
raw_y = np.array(data[:, -1]).reshape(m, 1)

x = np.c_[np.ones((m, 1)), raw_x]
y = raw_y

# print(x)
# print(y)

# plot data
xlabel = 'Microchip Test 1'
ylabel = 'Microchip Test 2'
plotData(data, xlabel, ylabel)
plt.show()

# ===================== Part 2: Optimize ======================

feature_x = mapFeature(x[:, 1], x[:, 2], 6)

initial_theta = np.ones((n, 1))
num_iters = 1000
alpha = 0.001
l = 0
print('The Cost of the initial theta: ', costFunctionReg(initial_theta, x, y, l))
print('The Gradient of the initial theta: ', getDerivativeReg(initial_theta, x, y, l))

# result = minimize(costFunctionReg, initial_theta, method='L-BFGS-B',
#               jac=getDerivativeReg, args=(x, y, l),
#               options={'gtol': 1e-4, 'disp': True, 'maxiter': 1000})

# theta = result.x
# cost = result.fun

# print('The result theta: ', theta)

# Gradient Descent
theta, j_history = gradientDescentReg(x, y, initial_theta, alpha, l, num_iters)

print('The result theta of gradient descent: ', theta)