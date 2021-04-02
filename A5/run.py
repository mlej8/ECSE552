from train import train
from mlp import MLP
from cnn import CNN
from lstm import LSTM

from preprocess import features, targets

if __name__== "__main__":
    train(MLP(len(features), len(targets)))
    train(CNN(feature_size=len(features), target_size=len(targets), kernel_size=2))
    train(LSTM(input_size=len(features), hidden_size=32, target_size=len(targets), num_layers=2))