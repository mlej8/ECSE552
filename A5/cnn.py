import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from train import train

from preprocess import features, targets

""" 
Using multivariate CNN model and following multiple input series paradigm. 
https://machinelearningmastery.com/how-to-develop-convolutional-neural-network-models-for-time-series-forecasting/
"""
class CNN(pl.LightningModule):
    def __init__(self, feature_size, target_size, kernel_size, dropout=0.2):
        super(CNN, self).__init__()
        self.target_size = target_size
        
        self.conv1 = nn.Conv1d(in_channels=feature_size, out_channels=32, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size)
        self.pooling = nn.MaxPool1d(2)
        self.fc = nn.Linear(64,32)
        self.fc2 = nn.Linear(32, target_size)
        
    def forward(self,x):
        # transform input into (batch_size, number of channels, seq_len) as original form is (batch_size, seq_len, number of channels)
        x = x.transpose(1,2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pooling(x)
        x = torch.flatten(x,1)
        x = F.relu(self.fc(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        # get data and target
        data, target = batch
        
        # forward pass
        preds = self(data)
        
        # compute MSE loss
        loss = F.mse_loss(preds, target)

        # log training loss
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """  
        Validation step.

        Note that model.eval() and torch.no_grad() are called automatically for validation
        """
        # get data and target
        data, target = batch
        
        # forward pass
        preds = self(data)
        
        # compute MSE loss
        loss = F.mse_loss(preds, target)

        # log validation loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        # get data and target
        data, target = batch
        
        # forward pass
        preds = self(data)
        
        # compute MSE loss
        loss = F.mse_loss(preds, target)

        # log validation loss
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


if __name__ == "__main__":
    train(CNN(feature_size=len(features), target_size=len(targets), kernel_size=2))