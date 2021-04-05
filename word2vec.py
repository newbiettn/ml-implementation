####################################################################################################
# word2vec.py
# AUTHOR:           NGOC TRAN
# CREATED:          31 Mar 2021
# DESCRIPTION:      An implementation of word2vec algorithm introduced in the paper "Efficient
#                   Estimation of Word Representations in Vector Space"
#                   (https://arxiv.org/pdf/1301.3781.pdf).
####################################################################################################
# Import libraries
import numpy as np

# Sample data
doc = "We can only see a short distance ahead, but we can see plenty there that needs to be done"
corpus = [[w.lower() for w in doc.split()]]
# print(corpus)

# Hyper-parameters
window_size = 2  # context window
epochs = 1000  # training epochs
eta = 0.1  # learning rate


def softmax(vec):
    """Softmax function"""
    exp_sum = np.sum(np.exp(vec))
    return np.exp(vec) / exp_sum


def generate_training_date(tokens, window_size):
    """
    Generate one-hot encoding vectors for a list of tokens, e.g. ['w1', 'w2', 'w3'...]
    Return: the one-hot encoding data, size of the vocabulary
    """
    # Extract a list of unique words
    unique_words = []
    for w in tokens:
        if w not in unique_words:
            unique_words.append(w)

    # Count the number of unique words
    vocab_size = len(unique_words)

    # One-hot vector library for unique words
    one_hot = {}
    for i in np.arange(0, vocab_size):
        # Get corresponding word
        w = unique_words[i]

        # Define a one-hot encoding vector for w
        vec = np.repeat(0, vocab_size)
        vec[i] = 1

        one_hot[w] = vec

    # Generate vectors for current word and context words
    generated_data = []
    for j in np.arange(0, len(tokens)):
        sliding_window = []

        # Current word
        current_w = tokens[j]
        current_w_vec = one_hot[current_w]
        sliding_window.append(current_w_vec)

        # Target words
        target_ws = []
        for k in np.arange(-window_size, window_size + 1):
            if (k == 0) or ((j + k) < 0) or ((j + k) > len(tokens) - 1):
                continue
            else:
                target_w = tokens[j + k]
                target_w_vec = one_hot[target_w]
                target_ws.append(target_w_vec)
        sliding_window.append(target_ws)
        generated_data.append(sliding_window)
    return generated_data, vocab_size


def train(data, vocab_size, hidden_size):
    """
    Train a neural net with:
    (1) number of input nodes = V (i.e. vocab_size),
    (2) number of hidden nodes = N (i.e. heuristically chosen),
    (3) number of output nodes = V (i.e. again, vocabulary size)
    Thus,
    (1) W(VxN) is a matrix weight between input->hidden
    (2) W'(NxV) is a matrix weights between hidden->output
    """

    # Hidden->input weights, a (hidden_size x vector_size) matrix
    W1 = np.random.normal(loc=0, scale=1, size=(vocab_size, hidden_size))

    # Output->hidden weights, a (vector_size x hiddne size) matrix
    W2 = np.random.normal(loc=0, scale=1, size=(hidden_size, vocab_size))

    # Set input and output
    for i in range(epochs):
        for x, y in training_data:
            # Feed forward
            h = np.dot(W1.T, x)
            u = np.dot(W2.T, h)
            y_pred = softmax(u)

            # Backpropagation
            e = [y_pred - label for label in y]
            dW2 = np.outer(h, np.sum(e, axis=0))
            dW1 = np.outer(x, np.dot(W2, np.sum(e, axis=0)))

            # Update weights
            W1 = W1 - eta * dW1
            W2 = W2 - eta * dW2
    return W1, W2


training_data, vocab_size = generate_training_date(tokens=corpus[0],
                                                   window_size=window_size)

W1, W2 = train(data=training_data,
               vocab_size=vocab_size,
               hidden_size=10)
print(W1)