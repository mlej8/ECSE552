import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

start = time.time()

# Training Data
x_train = pd.read_csv('data/training_set.csv', header=None).values
y_train = pd.read_csv('data/training_labels_bin.csv', header=None).values
x_val = pd.read_csv('data/validation_set.csv', header=None).values
y_val = pd.read_csv('data/validation_labels_bin.csv', header=None).values
N = len(x_train)

num_features = x_train.shape[1]
output_dim = y_train.shape[1] 

# hyperparameters (you may change these)
eta = 0.1 # intial learning rate
gamma = 0.1 # multiplier for the learning rate
stepsize = 200 # epochs before changing learning rate
threshold = 0.08 # stopping criterion
test_interval = 10 # number of epoch before validating
max_epoch = 3000

def sigmoid(z):
    """ 
    Returns the output of the sigmoid function on z.
    NOTE: All hidden and output layers have a sigmoid activation function. 

    :param z: Input to sigmoid function
    """
    return 1/(1+np.exp(-z))

# Define Architecture of NN
# [ TODO ] Intialize your network weights and biases here
class MLP(object):
    """ Class implementing a Multilayer Perceptron with two hidden layers from scratch. """
    def __init__(self, layer1_dim, layer2_dim, l=3):
        """ 
        Constructor for Multilayer Perceptron. 
        The network is contrained to the following specifications:
            - It only has 2 hidden layers (i.e. input -> hidden1 -> hidden2 -> out)
            - The output layer has 3 outputs (no softmax)
            - All hidden and output layers have a sigmoid activation function
            - Batch size is 1
            - The loss function is the sum of squared errors (SSE)
                - SSE across the 3 outputs
                - Use the mean SSE across all samples to measure model's performance
        
        The depth of this MLP is 3, because 2 hidden layers + 1 output layer. 

        :param l: Network depth
        :param layer1_dim: Width of first hidden layer
        :param layer1_dim: Width of second hidden layer
        :return: 
        """

        # storing the dimensions of the hidden layers
        self.layer1_dim = layer1_dim
        self.layer2_dim = layer2_dim

        # the size of the output layer is fixed and is comprised of 3 units (no softmax)
        self.output_dim = output_dim

        # storing number of layers in the network
        self.l = l

        # hidden layers parameters
        self.bias = [ # list of biases
            np.random.uniform(-1,1, (layer1_dim,1)),
            np.random.uniform(-1,1, (layer2_dim,1)),
            np.random.uniform(-1,1, (self.output_dim,1))
        ]
        self.weights = [ # list of weights
            np.random.uniform(-1,1, (layer1_dim,num_features)), 
            np.random.uniform(-1,1, (layer2_dim,layer1_dim)),
            np.random.uniform(-1,1, (self.output_dim,layer2_dim))
        ]
        
        # stack storing the input to the nonlinearity/activation of each hidden layer
        self.inputs = []

        # stack storing the output of the activation of each hidden layer
        self.outputs = []
        
    def forward(self, x, y_true):
        """ 
        Implementation of forward propagation to compute the loss function.
        The loss function used is the sum of squared errors (SSE) accross the 3 outputs of the output layer. 

        :param x: Input of the neural network.
        """
        # save outputs of the input layer (e.g. the inputs)
        self.outputs.append(x)

        # for each layer
        for i in range(self.l):

            # saving the input to the nonlinearity/activation of each hidden layer
            self.inputs.append(self.bias[i] + self.weights[i].dot(x))

            # update x
            x = sigmoid(self.inputs[i])

            # save the output of each layer for computing the gradients later
            self.outputs.append(x)
        
        return self.loss(x, y_true)

    def loss(self, y_pred, y_true):
        """ 
        Implementation of the sum of squared loss function (SSE).
        """
        return np.sum(np.square(y_true - y_pred))

    def backward(self, y):
        """ 
        Executing backward propagation to compute the gradients which will be used by the step function to update the weights. 
        """
        # create a queue to hold the gradients for weights and biases
        gradients_w = deque() # gradients for weights
        gradients_b = deque() # gradients for biases 

        # get the predictions
        y_pred = self.outputs.pop()
        
        # dJ/dy_hat, e.g. derivative of the loss function (SSE) w.r.t the predictions
        g = -2*(y - y_pred)

        for k in range(self.l-1, -1, -1): # going from the last layer to the first
            
            # dJ/da_k = g * f'(a_k), since we only use sigmoid activation function
            g = g * (sigmoid(self.inputs[k]) * (1 - sigmoid(self.inputs[k])))

            # dJ/b_k = g, derivative of the loss function w.r.t. biases of current layer
            gradients_b.appendleft(g)

            # dJ/d_W_k = g h_{k-1}.T, save gradients for weights
            gradients_w.appendleft(g.dot(self.outputs[k].T))

            # dJ/dh_{k-1} = W_k.T g
            g = self.weights[k].T.dot(g)

            # clean the input and output stack
            self.inputs.pop()
            self.outputs.pop()

        return (gradients_w, gradients_b)
        

    def step(self, gradients_w, gradients_b):
        """
        Method that updates the weights and biases of the MLP using: w <- w - eta * gradient_f(w) 
        """
        for i, g in enumerate(gradients_w):
            self.weights[i] -= eta * g
        
        for i, g in enumerate(gradients_b):
            self.bias[i] -= eta * g

# initialize the Multilayer Perceptron model and its weights
net = MLP(64, 16)

for epoch in range(0, max_epoch):
    
    order = np.random.permutation(N) # shuffle data
    
    sse = 0
    for n in range(0, N): # using a batch size of 1 as specified
        idx = order[n]

        # get a sample (batch size=1)
        x_in = np.array(x_train[idx]).reshape((num_features, 1))
        y = np.array(y_train[idx]).reshape((output_dim, 1))

        # [ TODO ] do the forward pass here
        # hint: you need to save the output of each layer to calculate the gradients later. NOTE: save input to the activation of each layer for calculating gradients later.
        squared_error = net.forward(x_in,y)
    
        # [ TODO ] compute error and gradients here
        # hint: don't forget the chain rule
        (gradients_w, gradients_b) = net.backward(y)

        # [ TODO ] update weights and biases here
        # update weights and biases in output layer 
        net.step(gradients_w, gradients_b)
    
        sse += squared_error

    train_mse = sse/len(x_train)

    if epoch % test_interval == 0: 
        # [TODO] test on validation set here

        # if termination condition is satisfied, exit
        if val_mse < threshold:
            break

    if epoch % stepsize == 0 and epoch != 0:
        eta = eta*gamma
        print('Changed learning rate to lr = ' + str(eta))