################################################################################
# backpropagation_3.py
# AUTHOR:           NGOC TRAN
# CREATED:          29 Mar 2021
# DESCRIPTION:      An implementation of neural net with 5 input nodes,
#                   5 hidden node and 3 output node to classify a toy UCI dataset
#                   called Iris (https://archive.ics.uci.edu/ml/datasets/iris)
#                   NOTE: (1) I used row vector.
################################################################################
# Load libraries
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import pandas as pd
import random

# Set seed
random.seed(23)


# Define softmax func
def softmax(vec):
    """Softmax function"""
    exp_sum = np.sum(np.exp(vec))
    return np.exp(vec) / exp_sum


# Define sigmoid func
def sigmoid(z):
    """Sigmoid/logistic function"""
    return 1 / (1 + np.exp(-z))


# Load Iris dataset
load_iris = datasets.load_iris()
x_df = pd.DataFrame(data=np.array(load_iris.data))
y_df = pd.DataFrame(data=load_iris.target)
iris = x_df
iris['class'] = y_df

# Define theta weight matrices
theta_1 = np.random.normal(loc=1, scale=1, size=(5, 5))  # Weights between inputs and hidden (6x5)
theta_2 = np.random.normal(loc=1, scale=1, size=(3, 5))  # Weights between hidden and outputs (3x5)
print("W1:\n", theta_1)
print("W2:\n", theta_2)

# Hyper-parameters
eta = 0.001  # learning rate eta
epochs = 1000  # number of training times

# Length
N = len(iris)

# Log
log = pd.DataFrame(columns=['Epoch', 'Error', 'Accuracy'])

# Learning data
for k in np.arange(0, epochs):
    # Shuffle data
    iris = shuffle(iris)

    # One hot encode the target column
    y = iris['class']
    temp_y = pd.DataFrame(data=y, columns=['class'])
    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_encoder.fit(temp_y)
    y_one_hot_encoded = one_hot_encoder.transform(temp_y)

    # Add bias column values
    X = iris[iris.columns[0:4]]
    col = np.repeat(1, len(y))
    X.insert(0, "Bias", col)
    X = np.array(X)

    # Batch processing all data samples
    x_i = X
    y_i = y_one_hot_encoded

    # Feed forward
    # Hidden nodes
    s_1 = np.dot(x_i, np.transpose(theta_1))  # 1x5 matrix
    a_1 = np.apply_along_axis(sigmoid, 1, s_1)  # hidden node output, 1x5 matrix

    # Output nodes
    s_2 = np.dot(a_1, np.transpose(theta_2))  # 1x3 matrix
    a_2 = np.apply_along_axis(softmax, 1, s_2)  # output nodes, 1x3 matrix
    pred = np.apply_along_axis(np.argmax, 1, a_2)

    # Measure error and accuracy
    corresponding_prob = np.array([a_2[i, pred[i]] for i in np.arange(0, len(pred))])
    cross_entropy_loss = -np.log(corresponding_prob)
    e = sum(cross_entropy_loss) / N
    accuracy = accuracy_score(y, pred)

    # Backpropagation
    # Output->Hidden weights
    dw2 = np.dot(np.transpose(a_2 - y_i),
                 a_1)

    # Hidden->Input weights
    dw1 = np.dot(np.transpose(np.multiply(np.dot(a_2 - y_i, theta_2),
                                          np.multiply(a_1, 1 - a_1))),
                 x_i)

    # Update weights
    theta_2 = theta_2 - eta * dw2
    theta_1 = theta_1 - eta * dw1

    # Update log
    l = {'Epoch': [k], 'Error': [e.item()], 'Accuracy': [accuracy]}
    log = log.append(pd.DataFrame(data=l))

# View error log
print("Error log: \n", log.tail())