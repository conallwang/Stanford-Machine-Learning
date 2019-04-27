from matplotlib import use
use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

from src.exercise1.FeatureScale import *
from src.exercise1.GradientDescent import *
from src.exercise1.NormalEqution import *

# ================ Part 1: Feature Normalization ================
print('Loading data ...')

# Load Data
data = np.loadtxt('ex1data2.txt', delimiter=',')

m = data.shape[0]
n = data.shape[1] - 1

raw_x = np.array(data[:, :2]).reshape(m, n)
raw_y = np.array(data[:, 2]).reshape(m, 1)

x = np.concatenate((np.ones((m, 1)), raw_x), axis=1)
y = raw_y

# Print out some data points
print('First 10 examples from the data set: ')
print(np.column_stack((x[:10, :], y[:10, :])))

# input('Program paused. Press Enter to continue...')

# Scale features and set them to zero
print('Normalizing Features ...')
new_x = FeatureScale(x)

print(new_x)

# ================ Part 2: Gradient Descent ================
print('Running Gradient Descent ...')

# Set some Gradient Descent Params
alpha = 1
num_iters = 1000

# Init theta and Run
theta = np.zeros((n + 1, 1))
theta, J_history = GradientDescent(new_x, y, theta, alpha, num_iters, True)

# Plot the convergence gragh
plt.plot(J_history, '-b')
plt.xlabel('Num of iterations')
plt.ylabel('Cost J')
plt.show()

# Display gradient descent's result
print('Theta computed from gradient descent: ')
print(theta)

# Estimate the price of house
price = np.array([[1, 3, 1650]]).dot(theta)
print('the prediction: ')
print(price)

# ================ Part 3: Normal Equations ================
x = raw_x
y = raw_y

x = np.concatenate((np.ones((m, 1)), x), axis=1)
theta = NormalEqution(x, y)

# display the result of Normal Eqution
print('Theta computed from the normal equations:')
print(theta)

# estimate the price of house
price = np.array([[1, 3, 1650]]).dot(theta)

# ============================================================

print('the result of prediction: ')
print(price)


# using sklearn
regr = linear_model.LinearRegression(fit_intercept=False, normalize='True')
regr.fit(new_x, y)

# display the result of sklearn
print('the result of sklearn: ')
print(regr.coef_)
