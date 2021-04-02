### Script studying error propagation 
import torch
from torch.utils.data import DataLoader, Subset

import pytorch_lightning as pl

from preprocess import targets, features, data_test, targets_test

from datetime import datetime
import os
import json

from params import * 

from dict_logger import logger

from mlp import MLP
from cnn import CNN
from lstm import LSTM

class WeatherDatasetErrorPropagation(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        """ Create a custom weather dataset """
        self.labels = labels
        self.data = data
        self.transform = transform

    def __len__(self):
        """ return size of the dataset """
        assert len(self.labels) == len(self.data), "Error: dataset targets and features are not of the same size."
        return len(self.labels)
    
    def __getitem__(self, idx):
        """ 
        The features of a datapoint consist of a history of weather attributes from time t-k to t-1.
        Given thes evalues, we have to predict the following values for time t:
            - p(mbar), atmospheric pressure
            - T (degC), air temperature
            - rh (%), relative humidity
            - wv(m/s), wind velocity
        
        In other words, the input features are the weather attributes of the previous k time steps. 
        You are required to use p(mbar), T (degC), rh (%), and wv(m/s) as features.
        We may include other attributes provided that these are values from the previous time steps.
        support indexing such that dataset[i] can be used to get the ith sample. """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if idx >= k:
            return (torch.tensor(self.data.iloc[idx-k:idx+k].to_numpy(), dtype=torch.float), torch.tensor(self.labels.iloc[idx:idx+k+1].to_numpy(), dtype=torch.float))
        else: # if this is one of the first k samples, return None
            return None

def write_results(model, result):
    # create folder for each run
    folder = "results/{}".format(datetime.now().strftime("%b-%d-%H-%M-%S"))
    if not os.path.exists(folder):
        os.makedirs(folder)
    PATH = folder + f'/{type(model).__name__}'
    with open(PATH, "w") as f:
        f.write(f"Model: {str(model)}\n")
        f.write(json.dumps(logger.metrics))
        f.write("\n")
        f.write(f"Test loss: {result}")

if __name__ == "__main__":

    # create test dataset for error propagation
    dataset = WeatherDatasetErrorPropagation(data_test, targets_test)

    # get indices to sample error propagation dataset
    indices = [i for i in range(k, len(data_test)-2*k, 2*k+1)]
    
    # create subset of dataset
    sub_dataset = Subset(dataset, indices)

    # checkpoint for each model
    mlp_path = "models/Apr-02-00-31-59/DictLogger/0.1/checkpoints/epoch=12-step=18225.ckpt"
    lstm_path = "models/Apr-02-00-37-59/DictLogger/0.1/checkpoints/epoch=35-step=50471.ckpt"
    cnn_path = "models/Apr-02-00-52-02/DictLogger/0.1/checkpoints/epoch=39-step=56079.ckpt"

    # loading models
    MLP = MLP.load_from_checkpoint(mlp_path, feature_size=len(features), target_size=len(targets))
    LSTM = LSTM.load_from_checkpoint(lstm_path, input_size=len(features), hidden_size=32, target_size=len(targets), num_layers=3)
    CNN = CNN.load_from_checkpoint(cnn_path, feature_size=len(features), target_size=len(targets), kernel_size=2) 

    # set models in eval mode
    MLP.eval()
    LSTM.eval()
    CNN.eval()

    # init trainer
    trainer = pl.Trainer(
        gpus=0,
        logger=logger, 
        progress_bar_refresh_rate=30
    )

    # create test dataloader
    test_dataloader = DataLoader(sub_dataset, batch_size=batch_size, shuffle=False, num_workers=0) # num_workers

    # test
    for model in [MLP,LSTM,CNN]:
        result = trainer.test(test_dataloaders=test_dataloader, verbose=True, model=model)
    
        # save test result
        write_results(model, result)

