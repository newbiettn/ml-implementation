################################################################################
# backpropagation_1.py
# AUTHOR:           NGOC TRAN
# CREATED:          26 Mar 2021
# DESCRIPTION:      An implementation of neural net with 3 input nodes,
#                   0 hidden node and 1 output node to classify a toy dataset
#                   created by AND logical operator. The data is as follows:
#                   -------
#                   x1 x2 y
#                   -------
#                   1  1  1
#                   1  0  0
#                   0  1  0
#                   0  0  0
#                   NOTE: (1) I used row vector.
#                         (2) Besides, x1 and x2, I added x0 as a bias node
################################################################################
# Load libraries
import numpy as np
import pandas as pd

# Create data
d = {'x0': [1, 1, 1, 1],
     'x1': [1, 1, 0, 0],
     'x2': [1, 0, 1, 0],
     'y': [1, 0, 0, 0]}
data = pd.DataFrame(data=d)

x = np.matrix(data[['x0', 'x1', 'x2']])
y = np.matrix(data[['y']])
print("x:\n", x)
print("y:\n", y)

# Init values
eta = 0.1  # learning rate eta
theta = np.matrix([-.5, -.5, -.5])  # weights
print("w:\n", theta)


# Define sigmoid function
def sigmoid(z):
    """Sigmoid activation function of the form 1/[1+e^(-z)]"""
    return 1 / (1 + np.exp(-z))


# Backpropagation
epochs = 5000  # number of training epochs
n = len(data)  # number of data points
log = pd.DataFrame(columns=['Epoch', 'Error'])
for k in np.arange(0, epochs):
    e = 0  # init error
    # Batch processing of the data
    for i in np.arange(0, n):
        x_i = x[i]  # retrieve i-th training samples
        y_i = y[i]  # retrieve i-th actual prediction value

        s_j = np.dot(theta, np.transpose(x_i))  # sum of weights at neuron j,
                                                # which is also the output node
        a_j = sigmoid(s_j)  # prediction value made at neuron j

        # Delta is the error signal at neuron j
        delta_j = (y_i - a_j) * a_j * (1 - a_j)

        # Accumulate error values
        e = e + np.square(y_i - a_j)  # square error

        # Adjust weights
        theta = theta + eta * delta_j * x_i  # w = w + Δw

    # Mean squared error
    e = 1 / n * e

    # Update log
    l = {'Epoch': [k], 'Error': [e.item()]}
    log = log.append(pd.DataFrame(data=l))


# View error log
print("Error log: \n", log.tail())

# Examine prediction
for i in np.arange(0, n):
    x_i = x[i]
    pred = sigmoid(np.dot(theta, np.transpose(x_i)))
    print(pred)