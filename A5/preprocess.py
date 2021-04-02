
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import random_split, DataLoader
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import datetime
import numpy as np

import math

from params import * 

# given a history of weather attributes from time t−k to t−1, predict the following values for time t.
targets = ['p (mbar)', 'T (degC)', 'rh (%)', 'wv (m/s)']
features = ['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)', 'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)', 'hour', 'month', 'wd_cos', 'wd_sin']

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

def preprocess(df, mean, std):
    # use the hour in the Date Time column since the time of the day can affect the temperature
    date_time = pd.to_datetime(df.pop("Date Time"), format='%d.%m.%Y %H:%M:%S')
    df["hour"] = date_time.dt.hour
    df["month"] = date_time.dt.month

    # transform degree into sin and cos
    wd_deg = df.pop("wd (deg)")
    df["wd_cos"] = np.cos(np.deg2rad(wd_deg))
    df["wd_sin"] = np.sin(np.deg2rad(wd_deg))

    # apply standardization (x-mean/std)
    df = (df-mean)/std

    # get targets and data
    labels = df[targets]
    data = df[features] # using all features as data

    return data, labels

def find_mean_std(train_df):
    """ Preprocess method that returns the mean and std of the dataframe as Pandas series. """
    # use the hour in the Date Time column since the time of the day can affect the temperature
    date_time = pd.to_datetime(train_df.pop("Date Time"), format='%d.%m.%Y %H:%M:%S')
    train_df["hour"] = date_time.dt.hour
    train_df["month"] = date_time.dt.month

    # transform degree into sin and cos
    wd_deg = train_df.pop("wd (deg)")
    train_df["wd_cos"] = np.cos(np.deg2rad(wd_deg))
    train_df["wd_sin"] = np.sin(np.deg2rad(wd_deg))

    return train_df.mean(), train_df.std()

data_train = pd.read_csv('weather_train.csv')
data_test = pd.read_csv('weather_test.csv')

# drop rows where outliers are present
data_test.drop(data_test.loc[data_test['wv (m/s)']==-9999].index, inplace=True)
data_test.drop(data_test.loc[data_test['max. wv (m/s)']==-9999].index, inplace=True)

# split training into train and val
data_train, data_val = train_test_split(data_train, shuffle=False, test_size=0.2)

# create weather dataset
train_mean, train_std = find_mean_std(data_train.copy(deep=True))
data_train, targets_train = preprocess(data_train, train_mean, train_std)
data_val, targets_val = preprocess(data_val, train_mean, train_std)
data_test, targets_test = preprocess(data_test, train_mean, train_std)
dataset_train = WeatherDataset(data_train, targets_train)
dataset_val = WeatherDataset(data_val, targets_val)
dataset_test = WeatherDataset(data_test, targets_test)

# create dictionary mapping column to indexing
targets_idx = {'p (mbar)':data_train.columns.get_loc('p (mbar)'), 'T (degC)':data_train.columns.get_loc('T (degC)'), 'rh (%)':data_train.columns.get_loc('rh (%)'), 'wv (m/s)':data_train.columns.get_loc('wv (m/s)')}

# dataloaders
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, collate_fn=weather_collate, num_workers=num_workers) # datapoints shall not be shuffled since the order is important
val_loader  = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, collate_fn=weather_collate, num_workers=num_workers)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=weather_collate, num_workers=num_workers)