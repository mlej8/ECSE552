from dict_logger import logger

from params import *

from preprocess import train_loader, val_loader, test_loader, features, targets

import matplotlib.pyplot as plt
from datetime import datetime
import os
import json

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_ranger import RangerQH


def train(model):
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

    # train
    trainer.fit(model=model, train_dataloader=train_loader, val_dataloaders=val_loader)

    # test
    result = trainer.test(test_dataloaders=test_loader,verbose=True) # NOTE: loads the best checkpoint automatically
    
    # save test result
    PATH = folder + '/result'
    with open(PATH, "w") as f:
        f.write(f"Model: {str(model)}\n")
        f.write(json.dumps(logger.metrics))
        f.write("\n")
        f.write(f"Lowest training loss: {str(min(logger.metrics['train_loss']))}\n")
        f.write(f"Lowest validation loss: {str(min(logger.metrics['val_loss']))}\n")
        f.write(f"Test loss: {result}")

    # plot training
    plt.plot(range(len(logger.metrics['train_loss'])), logger.metrics['train_loss'], lw=2, label='Training Loss')
    plt.plot(range(len(logger.metrics['val_loss'])), logger.metrics['val_loss'], lw=2, label='Validation Loss')
    plt.plot(range(len(logger.metrics['test_loss'])), logger.metrics['test_loss'], lw=2, label='Test Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.savefig(folder + f"/{type(model).__name__}_training_validation_test_loss.png")

    # plot p loss
    plt.plot(range(len(logger.metrics['train_p_loss'])), logger.metrics['train_p_loss'], lw=2, label='Training Loss')
    plt.plot(range(len(logger.metrics['val_p_loss'])), logger.metrics['val_p_loss'], lw=2, label='Validation Loss')
    plt.plot(range(len(logger.metrics['test_p_loss'])), logger.metrics['test_p_loss'], lw=2, label='Test Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.savefig(folder + f"/p_loss.png")
    
    # plot T loss
    plt.plot(range(len(logger.metrics['train_T_loss'])), logger.metrics['train_T_loss'], lw=2, label='Training Loss')
    plt.plot(range(len(logger.metrics['val_T_loss'])), logger.metrics['val_T_loss'], lw=2, label='Validation Loss')
    plt.plot(range(len(logger.metrics['test_T_loss'])), logger.metrics['test_T_loss'], lw=2, label='Test Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.savefig(folder + f"/T_loss.png")
    
    # plot T loss
    plt.plot(range(len(logger.metrics['train_rh_loss'])), logger.metrics['train_rh_loss'], lw=2, label='Training Loss')
    plt.plot(range(len(logger.metrics['val_rh_loss'])), logger.metrics['val_rh_loss'], lw=2, label='Validation Loss')
    plt.plot(range(len(logger.metrics['test_rh_loss'])), logger.metrics['test_rh_loss'], lw=2, label='Test Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.savefig(folder + f"/rh_loss.png")
    
    # plot wv loss
    plt.plot(range(len(logger.metrics['train_wv_loss'])), logger.metrics['train_wv_loss'], lw=2, label='Training Loss')
    plt.plot(range(len(logger.metrics['val_wv_loss'])), logger.metrics['val_wv_loss'], lw=2, label='Validation Loss')
    plt.plot(range(len(logger.metrics['test_wv_loss'])), logger.metrics['test_wv_loss'], lw=2, label='Test Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.savefig(folder + f"/wv_loss.png")