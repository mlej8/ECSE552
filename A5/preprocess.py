
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import random_split, DataLoader
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import pandas as pd
import pickle
import datetime
import numpy as np

import math

from params import * 

# given a history of weather attributes from time t−k to t−1, predict the following values for time t.
targets = ['p (mbar)', 'T (degC)', 'rh (%)', 'wv (m/s)']
features = ['hour', 'p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)','rh (%)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)','H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)','wd (deg)']

class WeatherDataset(torch.utils.data.Dataset):
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
            return (torch.tensor(self.data.iloc[idx-k:idx].to_numpy()), torch.tensor(self.labels.iloc[idx]))
        else: # if this is one of the first k samples, return None
            return None

def weather_collate(batch):
    """ Custom collate function for dataloader to filter out bad samples from Weather Dataset """
    batch = list(filter(lambda x: x is not None, batch))
    data = torch.stack([item[0] for item in batch]).float()
    targets = torch.stack([item[1] for item in batch]).float()
    return data, targets

def preprocess(df):
    # time and angles have different notions of similarity compared to temperature

    # use the hour in the Date Time column since the time of the day can affect the temperature
    df["hour"] = pd.to_datetime(df.pop("Date Time"), format='%d.%m.%Y %H:%M:%S').apply(lambda x: x.hour)

    # apply standardization (x-mean/std)
    df = (df-df.mean())/df.std()

    # get targets and data
    labels = df[targets]
    data = df[features] # using all features as data

    return data, labels

data_train = pd.read_csv('weather_train.csv')
data_test = pd.read_csv('weather_test.csv')

# create weather dataset
data_train, targets_train = preprocess(data_train)
data_test, targets_test = preprocess(data_test)
dataset = WeatherDataset(data_train, targets_train)
dataset_train, dataset_val = random_split(dataset, [math.ceil(len(dataset)*0.8),math.floor(len(dataset)*0.2)])
dataset_test = WeatherDataset(data_test, targets_test)

# dataloaders
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, collate_fn=weather_collate, num_workers=num_workers) # datapoints shall not be shuffled since the order is important
val_loader  = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, collate_fn=weather_collate, num_workers=num_workers)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=weather_collate, num_workers=num_workers)