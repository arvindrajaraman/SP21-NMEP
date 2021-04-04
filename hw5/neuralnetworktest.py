import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from neuralnetwork import NeuralNet

digits = load_digits()

X = digits.data
y = digits.target

X = X[np.logical_or(y == 4, y == 9)]
y = y[np.logical_or(y == 4, y == 9)]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=200, test_size=100)

y_train = np.where(y_train == 4, 0, y_train)
y_train = np.where(y_train == 9, 1, y_train)
y_test = np.where(y_test == 4, 0, y_test)
y_test = np.where(y_test == 9, 1, y_test)

nn = NeuralNet(input_size=64, hidden_size=20, output_size=1)

def bce(y_hat, y):
    n = y_hat.shape[0]
    return (1 / n) * np.sum(np.square(y_hat - y))

def accuracy(y_hat, y):
    y_pred = y_hat >= 0.5
    n = y.shape[0]
    correct = 0
    for i in range(len(y)):
        if int(y_pred[i]) == y[i]:
            correct += 1
    return correct / n

for i in range(1, 100):
    nn.forward(X_train, y_train)
    print('Epoch', i, 'Training Acc', accuracy(nn.predict(X_train.T), y_train), 'Test Acc', accuracy(nn.predict(X_test.T), y_test))
