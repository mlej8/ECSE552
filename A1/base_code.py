import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt

start = time.time()

# Training Data
x_train = pd.read_csv('data/training_set.csv', header=None).values
y_train = pd.read_csv('data/training_labels_bin.csv', header=None).values
x_val = pd.read_csv('data/validation_set.csv', header=None).values
y_val = pd.read_csv('data/validation_labels_bin.csv', header=None).values
N = len(x_train)

num_feats = x_train.shape[1]
n_out = y_train.shape[1]

# hyperparameters (you may change these)
eta = 0.1 # intial learning rate
gamma = 0.1 # multiplier for the learning rate
stepsize = 200 # epochs before changing learning rate
threshold = 0.08 # stopping criterion
test_interval = 10 # number of epoch before validating
max_epoch = 3000

# Define Architecture of NN
# [ ] Intialize your network weights and biases here

for epoch in range(0, max_epoch):
    
    order = np.random.permutation(N) # shuffle data
    
    sse = 0
    for n in range(0, N):
        idx = order[n]

        # get a sample (batch size=1)
        x_in = np.array(x_train[idx]).reshape((num_feats, 1))
        y = np.array(y_train[idx]).reshape((n_out, 1))

        # [ ] do the forward pass here
        # hint: you need to save the output of each layer to calculate the gradients later

    
        # [ ] compute error and gradients here
        # hint: don't forget the chain rule
               

        # [ ] update weights and biases here
        # update weights and biases in output layer 

    
        sse += squared_error

    train_mse = sse/len(x_train)

    if epoch % test_interval == 0: 
        # [ ] test on validation set here

        # if termination condition is satisfied, exit
        if val_mse < threshold:
            break

    if epoch % stepsize == 0 and epoch != 0:
        eta = eta*gamma
        print('Changed learning rate to lr=' + str(eta))