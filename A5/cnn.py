import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from train import train

from params import k

from preprocess import features, targets, targets_idx

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
        with torch.no_grad():
            self.log('train_loss', torch.sqrt(loss), on_step=False, on_epoch=True, prog_bar=True)
            self.log('train_p_loss', torch.sqrt(F.mse_loss(preds[:,0], target[:,0])), on_step=False, on_epoch=True, prog_bar=False)
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
        self.log('val_p_loss', torch.sqrt(F.mse_loss(preds[:,0], target[:,0])), on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_T_loss', torch.sqrt(F.mse_loss(preds[:,1], target[:,1])), on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_rh_loss', torch.sqrt(F.mse_loss(preds[:,2], target[:,2])), on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_wv_loss', torch.sqrt(F.mse_loss(preds[:,3], target[:,3])), on_step=False, on_epoch=True, prog_bar=False)
   
        return loss
    
    # def test_step(self, batch, batch_idx):
    #     # get data and target
    #     data, target = batch
        
    #     # forward pass
    #     preds = self(data)
        
    #     # compute MSE loss
    #     loss = F.mse_loss(preds, target)

    #     # log test loss
    #     self.log('test_loss', torch.sqrt(loss), on_step=False, on_epoch=True, prog_bar=True)
    #     self.log('test_p_loss', torch.sqrt(F.mse_loss(preds[:,0], target[:,0])) , on_step=False, on_epoch=True, prog_bar=False)
    #     self.log('test_T_loss', torch.sqrt(F.mse_loss(preds[:,1], target[:,1])), on_step=False, on_epoch=True, prog_bar=False)
    #     self.log('test_rh_loss', torch.sqrt(F.mse_loss(preds[:,2], target[:,2])), on_step=False, on_epoch=True, prog_bar=False)
    #     self.log('test_wv_loss', torch.sqrt(F.mse_loss(preds[:,3], target[:,3])), on_step=False, on_epoch=True, prog_bar=False)
    
    def test_step(self, batch, batch_idx):
        """ Test step for testing error propagation """
        # get data and target
        data, targets = batch
        
        for i in range(0,k+1):
            # preprocess data
            features = data[:, i:i+k]
            target = targets[:,i]

            # forward pass
            preds = self(features)
        
            # compute MSE loss
            loss = F.mse_loss(preds, target)

            # use previous prediction as features for next prediction 
            if i+k < data.size(1):
                data[:,i+k,targets_idx['p (mbar)']] = preds[:,0]
                data[:,i+k,targets_idx['T (degC)']] = preds[:,1]
                data[:,i+k,targets_idx['rh (%)']] = preds[:,2]
                data[:,i+k,targets_idx['wv (m/s)']] = preds[:,3]
            
            t_loss = torch.sqrt(loss)
            p_loss = torch.sqrt(F.mse_loss(preds[:,0], target[:,0]))
            T_loss = torch.sqrt(F.mse_loss(preds[:,1], target[:,1]))
            rh_loss = torch.sqrt(F.mse_loss(preds[:,2], target[:,2]))
            wv_loss = torch.sqrt(F.mse_loss(preds[:,3], target[:,3]))

            # log test loss
            self.log(f'test_t_{i}_loss', t_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f'test_p_loss_t_{i}', p_loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f'test_T_loss_t_{i}', T_loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f'test_rh_loss_t_{i}', rh_loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log(f'test_wv_loss_t_{i}', wv_loss, on_step=False, on_epoch=True, prog_bar=False)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


if __name__ == "__main__":
    train(CNN(feature_size=len(features), target_size=len(targets), kernel_size=2))