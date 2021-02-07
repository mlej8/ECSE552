import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime

import torch

# Generate my own dataset
dataset_size = 10000
val_size = dataset_size//5
train_size = dataset_size*4//5 

# generate r and t from normal and uniform distribution respectively
r = np.random.normal(loc=0,scale=1, size=dataset_size) # mean = 0, std = 1
t = np.random.uniform(low=0, high=2*math.pi, size=dataset_size) # uniform [0, 2pi)

# construct x1 and x2
x1 = np.concatenate([
    (r[:dataset_size//2] * np.cos(t[:dataset_size//2])), # label = 0
    ((r[dataset_size//2:] + 5) * np.cos(t[dataset_size//2:])) # label = 1
    ],
    axis=0).reshape(-1,1)
x2 = np.concatenate([
    (r[:dataset_size//2] * np.sin(t[:dataset_size//2])), # label = 0
    ((r[dataset_size//2:] + 5) * np.sin(t[dataset_size//2:])) # label = 1
    ],
    axis=0).reshape(-1,1)

# concatenate the features
x = np.concatenate([x1,x2], axis=1)

# set first half of the labels to 0 and second half remains 1
y = np.ones(dataset_size).reshape(-1,1)
y[:dataset_size//2] = 0 

# draw a scatter plot of the data
plt.scatter(x[:dataset_size//2,0], x[:dataset_size//2,1], label='0')
plt.scatter(x[dataset_size//2:,0], x[dataset_size//2:,1], label='1', marker='s')
plt.legend()
plt.show()
plt.close()

# creating our dataset using PyTorch
dataset = torch.utils.data.TensorDataset(
            torch.Tensor(x), # convert x into tensors 
            torch.Tensor(y)
            )

# split into training val
train, val = torch.utils.data.random_split(dataset, lengths=[train_size, val_size])

# define hyperparameters 
lr = 1e-3
batch_size = 32
max_epoch = 50

# define dataloaders
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False)

# define the network architecture
class NeuralNetwork(torch.nn.Module):
    def __init__(self, n_in, n_hidden=30):
        """ 
        Constructor of the neural network. 
        The network architecture is limited to:
            - 1 hidden layer
            - number of nodes in hidden layer <= 30
            - hidden layer activation: ReLU

        :param n_in: input size
        :param n_hidden: number of nodes in hidden layer <= 30
        """
        super(NeuralNetwork, self).__init__()
        self.fc = torch.nn.Linear(n_in, n_hidden)
        self.fc2 = torch.nn.Linear(n_hidden, 1)
    
    def forward(self, x):
        x = self.fc(x)
        x = torch.relu(x)
        x = self.fc2(x)

        """
        We are using sigmoid as the activation function for the output layer, because it is a binary classification task. 
        Sigmoid is good, because it computes a probability between 0 and 1 of the target output being labeled 1
        """
        x = torch.sigmoid(x)

        return x

model = NeuralNetwork(n_in=2, n_hidden=30)

""" 
Since activation function of the output layer is sigmoid, we will use binary cross entropy as our loss function. 
This is due to the fact that we are doing a classificaiton task. 
"""
criterion = torch.nn.BCELoss()

# using stochastic gradient descent as the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

# lists to store train/val loss
train_losses = []
val_losses = []

# see if cuda available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# move network to appropriate device
model.to(device)

# time tracker
start_time = datetime.now()

# training loop
for epoch in range(max_epoch):

    # set model in training mode
    model.train()   

    # variable to keep track of training/validation loss per epoch 
    train_loss = 0
    val_loss = 0

    for data, labels in train_loader:
        
        # clear gradients
        optimizer.zero_grad()

        # send data and labels to correct device
        data = data.to(device)
        labels = labels.to(device)

        # forward pass
        y_hat = model(data)

        # compute the Binary Cross Entropy loss
        loss = criterion(y_hat, labels)

        # accumulate training loss
        train_loss += loss
        
        # compute gradients
        loss.backward()

        # update parameters
        optimizer.step() 
    
    # compute training loss per sample
    loss_per_sample = train_loss/train_size

    # append training loss per sample 
    train_losses.append(loss_per_sample)

    # set model in validation mode
    model.eval()

    # we don't need to compute the gradients during validation
    with torch.no_grad():
        for data, labels in val_loader: # using a batch size of 1 as specified

            # send data and labels to correct device
            data = data.to(device)
            labels = labels.to(device)

            # forward pass (predict)
            y_hat = model(data)

            # accumulate validation loss
            val_loss += criterion(y_hat, labels)

    # compute validation loss per sample
    val_loss_per_sample = val_loss/val_size

    # append validation loss per sample 
    val_losses.append(val_loss_per_sample)

    end_time = datetime.now()
    print(f"Epoch {epoch} in {end_time - start_time}\tTraining loss: {loss_per_sample}\tValidation loss: {val_loss_per_sample}")
    start_time = end_time  

# saving model
PATH = "model.pt"
torch.save(model.state_dict(), PATH)

# Code to plot the training and validation loss
var1 = np.linspace(0, max_epoch, len(train_losses))
var2 = np.linspace(0, max_epoch, len(val_losses))
plt.figure(figsize=(24,8))
plt.plot(var1, train_losses, label="Training Error")
plt.plot(var2, val_losses, label= "Validation Error")
plt.xticks([epoch_num for epoch_num in range(max_epoch)])
plt.xlabel('Epoch') # num of epoch in x-axis
plt.ylabel('Average loss per sample') 
plt.title("Binary Cross Entropy loss per sample over number of epochs")
plt.legend()
plt.savefig('error.pdf')
plt.savefig('error.png')
plt.close()

# get the data from class 0 and 1
class0 = []
class1 = []
for data, label in val: 
    if label == 1:
        class1.append(data)
    else:
        class0.append(data)

# helper function to extract that numpy values from the parameter object
get_values = lambda x: x.data.detach().numpy()

# get the weights and biases for the first hidden layer
param_gen = model.parameters()
weights = get_values(next(param_gen))
biases = get_values(next(param_gen)).reshape(-1, 1)
lines = np.concatenate([weights, biases], axis=1)

""" 
Since the ReLU function is defined as 

    y = max(0, w1x1 + w2x2 + b)

We want to know when does 

    0 = w1x1 + w2x2 + b

In order to find two points to plot the decision boundary, we must set x1 = 0 and x2 = 0 and find their opposite coordinate (find x2 for x1, vice versa)

    x1 = -b/w1          x2 = 0 
    x2 = -b/w2          x1 = 0
"""
point_pairs = []
for line in lines: 
    point_pairs.append(((-line[2] * line[0], 0), (0 ,-line[2] * line[1])))

# make a scatter plot of the validation data
plt.figure(figsize=(8,8))
plt.scatter([data[0] for data in class0], [data[1] for data in class0], label='Label 0')
plt.scatter([data[0] for data in class1], [data[1] for data in class1],  label='Label 1', marker='s')
for point in point_pairs:
    x1_values = [point[0][0], point[1][0]]
    x2_values = [point[0][1], point[1][1]]
    plt.plot(x1_values, x2_values)
plt.xlabel('x1') # num of epoch in x-axis
plt.ylabel('x2') 
plt.title("Validation dataset")
plt.legend()
plt.savefig('validation.pdf') 
plt.savefig('validation.png')
plt.show()
plt.close()
