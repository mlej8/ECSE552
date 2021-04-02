import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from dict_logger import logger

from preprocess import features, targets, targets_idx

from params import k

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
        with torch.no_grad():
            self.log('train_loss', torch.sqrt(loss), on_step=False, on_epoch=True, prog_bar=True)
            self.log('train_p_loss',torch.sqrt(F.mse_loss(preds[:,0], target[:,0])), on_step=False, on_epoch=True, prog_bar=False)
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
        preds = self(data, None)
        
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
    #     preds = self(data, None)
        
    #     # compute MSE loss
    #     loss = F.mse_loss(preds, target)

    #     # log test loss
    #     self.log('test_loss', torch.sqrt(loss), on_step=False, on_epoch=True, prog_bar=True)
    #     self.log('test_p_loss', torch.sqrt(F.mse_loss(preds[:,0], target[:,0])), on_step=False, on_epoch=True, prog_bar=False)
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
            preds = self(features, None)
        
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
    train(LSTM(input_size=len(features), hidden_size=32, target_size=len(targets), num_layers=2))