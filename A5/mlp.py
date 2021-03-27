import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from train import train

from dict_logger import logger

from params import *

from preprocess import features, targets

class MLP(pl.LightningModule):
    def __init__(self, feature_size, target_size, dropout=0.2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(k*feature_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, target_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # expand matrix into a 1D vector
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        output = self.fc4(x)
        return output

    def training_step(self, batch, batch_idx):
        # get data and target
        data, target = batch
        
        # forward pass
        preds = self(data)
        
        # compute MSE loss
        loss = F.mse_loss(preds, target)

        # log training loss
        with torch.no_grad():
            self.log('train_loss', torch.sqrt(loss), on_step=False, on_epoch=True, prog_bar=True)
            self.log('train_p_loss', torch.sqrt(F.mse_loss(preds[:,0], target[:,0])) , on_step=False, on_epoch=True, prog_bar=False)
            self.log('train_T_loss', torch.sqrt(F.mse_loss(preds[:,1], target[:,1])), on_step=False, on_epoch=True, prog_bar=False)
            self.log('train_rh_loss', torch.sqrt(F.mse_loss(preds[:,2], target[:,2])), on_step=False, on_epoch=True, prog_bar=False)
            self.log('train_wv_loss', torch.sqrt(F.mse_loss(preds[:,3], target[:,3])), on_step=False, on_epoch=True, prog_bar=False)

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
        self.log('val_loss', torch.sqrt(loss), on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_p_loss', torch.sqrt(F.mse_loss(preds[:,0], target[:,0])) , on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_T_loss', torch.sqrt(F.mse_loss(preds[:,1], target[:,1])), on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_rh_loss', torch.sqrt(F.mse_loss(preds[:,2], target[:,2])), on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_wv_loss', torch.sqrt(F.mse_loss(preds[:,3], target[:,3])), on_step=False, on_epoch=True, prog_bar=False)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        # get data and target
        data, target = batch
        
        # forward pass
        preds = self(data)
        
        # compute MSE loss
        loss = F.mse_loss(preds, target)

        # log validation loss
        self.log('test_loss', torch.sqrt(loss), on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_p_loss', torch.sqrt(F.mse_loss(preds[:,0], target[:,0])) , on_step=False, on_epoch=True, prog_bar=False)
        self.log('test_T_loss', torch.sqrt(F.mse_loss(preds[:,1], target[:,1])), on_step=False, on_epoch=True, prog_bar=False)
        self.log('test_rh_loss', torch.sqrt(F.mse_loss(preds[:,2], target[:,2])), on_step=False, on_epoch=True, prog_bar=False)
        self.log('test_wv_loss', torch.sqrt(F.mse_loss(preds[:,3], target[:,3])), on_step=False, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


if __name__ == "__main__":
    train(MLP(len(features), len(targets)))