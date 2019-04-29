import numpy as np
from scipy.io import loadmat

from src.exercise3.plotData import *
from src.exercise3.predict import *


# ====================== Part 1: Plotting Data =======================
data = loadmat('ex3data1.mat')

raw_x = data['X']
raw_y = data['y']

m = raw_x.shape[0]
n = raw_x.shape[1] + 1

x = np.c_[np.ones((m, 1)), raw_x]
y = raw_y

# print(x)
# print(y)

# Plot Data
plotData(raw_x)
plt.show()

# Load Theta Trained
theta = loadmat('ex3weights.mat')

# print(theta)
# Get Two Theta Matrix
# theta1:   25×401
# theta2:   10×26
theta1 = theta['Theta1']
theta2 = theta['Theta2']

# Output layer
output_layer = predict(theta1, theta2, x)

# Accuracy rate
res = np.argmax(output_layer, axis=0).reshape(m, 1)
res = res + 1

diff = res - y
diff[diff == 0] = 1000
diff[diff != 1000] = 0
diff[diff == 1000] = 1

print('Accuracy rate: ', np.sum(diff) / m)
