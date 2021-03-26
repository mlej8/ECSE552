import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_ranger import RangerQH
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from dict_logger import logger

from params import *

from preprocess import train_loader, val_loader, test_loader, features, targets

import matplotlib.pyplot as plt
from datetime import datetime
import os

""" 
Using multivariate CNN model and following multiple input series paradigm. 
https://machinelearningmastery.com/how-to-develop-convolutional-neural-network-models-for-time-series-forecasting/
"""

class CNN(pl.LightningModule):
    def __init__(self, feature_size, target_size, kernel_size, dropout=0.2):
        super(CNN, self).__init__()
        self.target_size = target_size
        
        self.conv1 = nn.Conv1d(in_channels=feature_size, out_channels=32, kernel_size=kernel_size)
        self.pooling = nn.MaxPool1d(2)
        self.fc = nn.Linear(32,16)
        self.fc2 = nn.Linear(16, target_size)
        
    def forward(self,x):
        # transform input into (batch_size, number of channels, time steps) as original form is (batch_size, number of channels, time steps)
        x = x.transpose(1,2)
        x = F.relu(self.conv1(x))
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
    
    # create folder for each run
    folder = "models/{}".format(datetime.now().strftime("%b-%d-%H-%M-%S"))
    if not os.path.exists(folder):
        os.makedirs(folder)

    # early stoppping
    early_stopping_callback = EarlyStopping(
      monitor='val_loss', # monitor validation loss
      verbose=True, # log early-stop events
      patience=patience,
      min_delta=0.00 # minimum change is 0
      )

    # update checkpoints based on validation loss by using ModelCheckpoint callback monitoring 'val_loss'
    checkpoint_callback = ModelCheckpoint(monitor='val_loss')

    # define trainer 
    trainer = pl.Trainer(
      default_root_dir=folder, # Lightning automates saving and loading checkpoints
      max_epochs=epochs, gpus=0,
      logger=logger, 
      progress_bar_refresh_rate=30, 
      callbacks=[early_stopping_callback, checkpoint_callback])

    # create model
    model = CNN(feature_size=len(features), target_size=len(targets), kernel_size=2)

    # train
    trainer.fit(model=model, train_dataloader=train_loader, val_dataloaders=val_loader)

    # plot training
    plt.plot(range(len(logger.metrics['train_loss'])), logger.metrics['train_loss'], lw=2, label='Training Loss')
    plt.plot(range(len(logger.metrics['val_loss'])), logger.metrics['val_loss'], lw=2, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.savefig(folder + "/cnn_training_validation.png")

    # test
    result = trainer.test(test_dataloaders=test_loader,verbose=True) # NOTE: loads the best checkpoint automatically
    print(f"Final test score: {result}")