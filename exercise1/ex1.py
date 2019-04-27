import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

from src.exercise1.warmUpExercise import *
from src.exercise1.plotData import *
from src.exercise1.GradientDescent import *
from src.exercise1.computeCost import *

# ==================== Part 1: Basic Function ====================
print('Running warmUpExercise ...')
print('5×5 Identity Matrix: ')

warmUp = warmUpExercise()
print(warmUp)

# input('Program paused. Press Enter to continue...')

# ======================= Part 2: Plotting =======================
data = np.loadtxt('ex1data1.txt', delimiter=',')

m = data.shape[0]
n = data.shape[1] - 1

raw_x = np.array(data[:, 0]).reshape(m, 1)
raw_y = np.array(data[:, 1]).reshape(m, 1)

x = np.vstack(zip(np.ones(m), raw_x)).reshape(m, n + 1)
y = raw_y

# plotData
ylabel = 'Profit in $10,000s'
xlabel = 'Population of City in 10,000s'
plotData(data, xlabel, ylabel)
plt.show()

# input('Program paused. Press Enter to continue...')

# ======================= Part 3: Gradient Descent =======================
print('Running Gradient Descent ...')
theta = np.zeros((2, 1))

# compute and display initial cost
J = computeCost(x, y, theta)
print('cost: %.4f ' % J)

# Some gradient descent settings
iterations = 1000
alpha = 0.01

# run gradient descent
theta, J_history = GradientDescent(x, y, theta, alpha, iterations, True)

# print theta to screen
print('theta found by gradient descent: ')
print('%s %s \n' % (theta[0], theta[1]))

# Plot the linear fit
plt.figure()
plotData(data)
plt.plot(x[:, 1], x.dot(theta), '-', label='Linear Regression')
plt.show()

# input("Program paused. Press Enter to continue...")

# Plot the Cost Value
plt.figure()
plt.plot([x for x in range(iterations + 1)], J_history, 'b-')
plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.array([1, 3.5]).dot(theta)[0]
predict2 = np.array([1, 7]).dot(theta)[0]
print('For population = 35,000, we predict a profit of {:.4f}'.format(predict1*10000))
print('For population = 70,000, we predict a profit of {:.4f}'.format(predict2*10000))

# =============Use Scikit-learn =============
regr = linear_model.LinearRegression(fit_intercept=False, normalize=True)
regr.fit(x, y)

print('Theta found by scikit: ')
print('%s' % regr.coef_[0])

# predict1 = np.array([1, 3.5]).dot(regr.coef_)[0]
# predict2 = np.array([1, 7]).dot(regr.coef_)[0]
# print('For population = 35,000, we predict a profit of {:.4f}'.format(predict1*10000))
# print('For population = 70,000, we predict a profit of {:.4f}'.format(predict2*10000))

# plt.figure()
# plotData(data)
# plt.plot(x[:, 1],  x.dot(regr.coef_), '-', color='black', label='Linear regression wit scikit')
# plt.legend(loc='upper right', shadow=True, fontsize='x-large', numpoints=1)
# plt.show()
