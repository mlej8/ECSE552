import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from dict_logger import logger

from preprocess import features, targets

from train import train

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
        self.fc = nn.Linear(hidden_size, target_size)
    
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
        with torch.no_grad():
            self.log('train_p_loss', F.mse_loss(preds[:,0], target[:,0]) , on_step=False, on_epoch=True, prog_bar=False)
            self.log('train_T_loss', F.mse_loss(preds[:,1], target[:,1]), on_step=False, on_epoch=True, prog_bar=False)
            self.log('train_rh_loss', F.mse_loss(preds[:,2], target[:,2]), on_step=False, on_epoch=True, prog_bar=False)
            self.log('train_wv_loss', F.mse_loss(preds[:,3], target[:,3]), on_step=False, on_epoch=True, prog_bar=False)

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
        self.log('val_p_loss', F.mse_loss(preds[:,0], target[:,0]) , on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_T_loss', F.mse_loss(preds[:,1], target[:,1]), on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_rh_loss', F.mse_loss(preds[:,2], target[:,2]), on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_wv_loss', F.mse_loss(preds[:,3], target[:,3]), on_step=False, on_epoch=True, prog_bar=False)
        
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
        self.log('test_p_loss', F.mse_loss(preds[:,0], target[:,0]) , on_step=False, on_epoch=True, prog_bar=False)
        self.log('test_T_loss', F.mse_loss(preds[:,1], target[:,1]), on_step=False, on_epoch=True, prog_bar=False)
        self.log('test_rh_loss', F.mse_loss(preds[:,2], target[:,2]), on_step=False, on_epoch=True, prog_bar=False)
        self.log('test_wv_loss', F.mse_loss(preds[:,3], target[:,3]), on_step=False, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

if __name__ == "__main__":
    train(LSTM(input_size=len(features), hidden_size=64, target_size=len(targets), num_layers=4, dropout=0.1))