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
Input is aranged following this format: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
"""

class LSTM(pl.LightningModule):
    def __init__(self, input_size, hidden_size, target_size, num_layers=2, dropout=0):
        super(LSTM, self).__init__()
        
        # store parameters 
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # network
        self.lstm = nn.LSTM(input_size=input_size,hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout) 
        self.fc = nn.Linear(hidden_size, hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, target_size)
    
    def forward(self, x, hidden):
        """ 
        We feed the entire sequence all at once - the first value returned by LSTM is all of the hidden states throughout the sequence.
        The second second is just the most recent hidden state. 
        The reason for this is that: output will give you access to all hidden states in the sequence and "hidden" will allow you to continue the sequence and backpropagate by passing it as an argument to the lstm at a later time

        Input size: # of features
        Sequence length: k
        
        Input shape: (batch_size, seq_len, num_features/input_size)
        """
        output, (h_n, c_n) = self.lstm(x, hidden)
        output = self.fc(output[:,-1,:])
        output = self.fc2(output)
        return output

    def training_step(self, batch, batch_idx):
        # get data and target
        data, target = batch
        
        # forward pass
        preds = self(data, None)
        
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
        preds = self(data, None)
        
        # compute MSE loss
        loss = F.mse_loss(preds, target)

        # log validation loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        # get data and target
        data, target = batch
        
        # forward pass
        preds = self(data, None)
        
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
      max_epochs=epochs, gpus=1,
      logger=logger, 
      progress_bar_refresh_rate=30, 
      callbacks=[early_stopping_callback, checkpoint_callback])

    # create model
    model = LSTM(input_size=len(features), hidden_size=32, target_size=len(targets), num_layers=4, dropout=0.1)

    # train
    trainer.fit(model=model, train_dataloader=train_loader, val_dataloaders=val_loader)

    # plot training
    plt.plot(range(len(logger.metrics['train_loss'])), logger.metrics['train_loss'], lw=2, label='Training Loss')
    plt.plot(range(len(logger.metrics['val_loss'])), logger.metrics['val_loss'], lw=2, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.savefig(folder + "/lstm_training_validation.png")

    # test
    result = trainer.test(test_dataloaders=test_loader,verbose=True) # NOTE: loads the best checkpoint automatically
    
    # save test result
    PATH =  folder + '/result'
    with open(PATH, "w") as f:
        f.write(f"Final test score: {result}")