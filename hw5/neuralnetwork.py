import numpy as np
import math

class NeuralNet:

    #constructor, we will hardcode this to a 1 hidden layer network, for simplicity
    #the problem we will grade on is differentiating 0 and 1s
    #Some things/structure may need to be changed. What needs to stay consistent is us being able to call
    #forward with 2 arguments: a data point and a label. Strange architecture, but should be good for learning
    def __init__(self, input_size=784, hidden_size=100, output_size=1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        #YOUR CODE HERE, initialize appropriately sized weights/biases with random paramters
        self.weight1 = np.random.randn(hidden_size, input_size) * math.sqrt(6) / math.sqrt(hidden_size + input_size)
        self.bias1 = np.zeros((hidden_size, 1))
        self.weight2 = np.random.randn(output_size, hidden_size) * math.sqrt(6) / math.sqrt(output_size + hidden_size)
        self.bias2 = np.zeros((output_size, 1))

    #Potentially helpful, np.dot(a, b), also @ is the matrix product in numpy (a @ b)

    #loss function, implement L1 loss
    #YOUR CODE HERE
    def loss(self, y0, y1):
        return np.abs(y0 - y1)

    #relu and sigmoid, nonlinear activations
    #YOUR CODE HERE
    def relu(self, x):
        return np.maximum(np.zeros(x.shape), x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    #You also may want the derivative of Relu and sigmoid
    def drelu(self, x):
        return np.where(x <= 0, np.zeros(x.shape), np.ones(x.shape))

    #forward function, you may assume x is correct input size
    #have the activation from the input to hidden layer be relu, and from hidden to output be sigmoid
    #have your forward function call backprop: we won't be doing batch training, so for EVERY SINGLE input,
    #we will update our weights. This is not always (maybe not even here) possible or practical, why?
    #Also, normally forward doesn't take in labels. Since we'll have forward call backprop, it'll take in labels
    #YOUR CODE HERE
    def forward(self, x, label):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        elif x.shape[1] == self.input_size:
            x = x.T
        z1 = (self.weight1 @ x) + self.bias1
        a1 = self.relu(z1)
        z2 = (self.weight2 @ a1) + self.bias2
        a2 = self.sigmoid(z2)
        self.backprop(x, z1, a1, z2, a2, label)
    
    def predict(self, x):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        elif x.shape[1] == self.input_size:
            x = x.T
        z1 = (self.weight1 @ x) + self.bias1
        a1 = self.relu(z1)
        z2 = (self.weight2 @ a1) + self.bias2
        a2 = self.sigmoid(z2)
        return a2.T

    #implement backprop, might help to have a helper function update weights
    #Recommend you check out the youtube channel 3Blue1Brown and their video on backprop
    #YOUR CODE HERE
    def backprop(self, x, z1, a1, z2, a2, label): #What else might we need to take in as arguments? Modify as necessary
        #Compute the gradients first
        #First will have to do with combining derivative of sigmoid, output layer, and what else?
        #np.sum(x, axis, keepdims) may be useful
        m = self.input_size
        
        dz2 = a2 - label
        dw2 = (1 / m) * (dz2 @ a1.T)
        db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
        dz1 = (self.weight2.T @ dz2) * self.drelu(z1)
        dw1 = (1 / m) * (dz1 @ x.T)
        db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

        #Update your weights and biases. Use a learning rate of 0.1, and update on every call to backprop
        lr = .1
        self.weight2 -= lr * dw2
        self.bias2 -= lr * db2
        self.weight1 -= lr * dw1
        self.bias1 -= lr * db1
