################################################################################
# backpropagation_2.py
# AUTHOR:           NGOC TRAN
# CREATED:          27 Mar 2021
# DESCRIPTION:      An implementation of neural net with 3 input nodes,
#                   3 hidden node and 1 output node to classify a toy dataset
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
eta = 0.5  # learning rate eta
theta_1 = np.matrix([[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 0]])  # weights between input and hidden nodes
theta_2 = np.matrix([[1, 1, 1]])  # weights between hidden and output nodes
print("W1:\n", theta_1)
print("W2:\n", theta_2)


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
        # Feed forward
        x_i = x[i]  # retrieve i-th training samples
        d_i = y[i]  # retrieve i-th actual prediction value

        s_1 = np.dot(theta_1, np.transpose(x_i))  # sum of weights in hidden
        a_1 = sigmoid(s_1)  # output of hidden nodes

        s_2 = np.dot(theta_2, a_1)  # sum of weights in output
        a_2 = sigmoid(s_2)

        e = e + np.square(d_i - a_2)  # update squared error

        # Backpropagation
        # Error signal at the OUTPUT node j: δj = ej * φ′(sj) = (dj − yj) * φ′(sj)
        delta_2 = (d_i - a_2) * a_2 * (1 - a_2)

        # Error signal at the HIDDEN node j: δj = φ′(sj) * sum(δk ·wkj)
        delta_1 = np.multiply(np.dot(delta_2, theta_2),
                              np.transpose(np.multiply(a_1,1 - a_1))
                              )

        # Update weights ∆wji = η * δj * yi.
        theta_1 = theta_1 + eta * np.dot(np.transpose(delta_1), x_i)
        theta_2 = theta_2 + eta * np.transpose(np.dot(a_1, delta_2))

    # Mean squared error
    e = (1 / n) * e

    # Update log
    l = {'Epoch': [k], 'Error': [e.item()]}
    log = log.append(pd.DataFrame(data=l))

# View error log
print("Error log: \n", log.tail())

# Examine prediction
for i in np.arange(0, n):
    x_i = x[i]  # retrieve i-th training samples
    d_i = y[i]  # retrieve i-th actual prediction value

    s_1 = np.dot(theta_1, np.transpose(x_i))  # sum of weights in hidden
    a_1 = sigmoid(s_1)  # output of hidden nodes

    s_2 = np.dot(theta_2, a_1)  # sum of weights in output
    pred = sigmoid(s_2)
    print(pred)
