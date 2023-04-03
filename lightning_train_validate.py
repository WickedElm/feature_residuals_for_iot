#!/usr/bin/env python

import hydra
from omegaconf import DictConfig, OmegaConf 
from hydra.utils import instantiate
import argparse
import importlib
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

from metrics.utilities import *
from callbacks.sklearn_trainer_callback import SklearnTrainerCallback
import torch
import torchmetrics
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torch.nn.functional as F

import torchvision
from torchvision import transforms

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping
import ipdb
import sys
import dataset.Dataset
import glob
import re
import copy
import math
import pickle
import socket

def list_models():
    # Get all files in models directory
    model_files = glob.glob(f'{hydra.utils.get_original_cwd()}/models/*.py')

    print('')
    print('Models Available: (--model)')
    print('')

    # Get all classes in those files
    for mf in model_files:
        if '__init__' in mf:
            continue
        
        # Get module name from file name
        _, file_name = os.path.split(mf)
        file_name = file_name.replace('.py', '')

        # Get agent classes from file
        with open(mf, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if re.match(r'^class', line):
                model_class_name = (re.findall(r'^class\s+(\w+)\(.*', line))[0]
                print(f'        - {file_name}.{model_class_name}')
    print('')

@hydra.main(config_path='conf', config_name='config')
def main(cfg : DictConfig) -> None:
    # Seed everything for distributed processing
    pl.seed_everything()

    # Avoid too many open files error
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Restrict CPUs used
    torch.set_num_threads(7)

    num_gpus = 0
    if torch.cuda.is_available() and socket.gethostname() != 'system76-pc':
        num_gpus = 1

    if cfg.general.list:
        list_models()
        sys.exit(0)

    results_dir = os.getcwd()
    
    # wandb init
    ds_name = os.path.split(cfg.data_module.data_path)[-1].replace('.pkl', '')
    wandb.init(project=cfg.general.project, name=f'{cfg.general.experiment}-{ds_name}', group=cfg.general.group)

    # Loggers
    wandb_logger = WandbLogger(project=cfg.general.project, name=f'{cfg.general.experiment}-{ds_name}')
    csv_logger = CSVLogger(results_dir, name=f'{cfg.general.experiment}-{ds_name}') 

    # Instantiate data_module and model using hydra config
    dm = instantiate(cfg.data_module)
    ae = instantiate(cfg.model)

    ###
    # callbacks
    ###

    checkpoint_callback = ModelCheckpoint(
        dirpath=results_dir, 
        save_top_k=-1, 
        every_n_epochs=50, 
        save_last=True, 
        filename=f'AE-{cfg.general.project}-{cfg.general.experiment}-{ds_name}' + '-{epoch:06d}'
    )

    model_callbacks = ae.get_callbacks(cfg.general.num_epochs)
    all_callbacks = [checkpoint_callback] + model_callbacks

    max_epochs = cfg.general.num_epochs
    if cfg.model.use_pretrained_ae:
        max_epochs = max_epochs - cfg.model.pretraining_epochs
        print(f'LEWBUG:  REMOVING PRETRAINING EPOCHS.  TRAINING FOR {max_epochs} EPOCHS.')

    trainer = pl.Trainer(
        gpus=num_gpus, 
        logger=[wandb_logger, csv_logger], 
        max_epochs=max_epochs,
        callbacks=all_callbacks,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epoch,
    )

    # Train/validate and test RAE
    if not cfg.model.use_pretrained_ae or (cfg.model.use_pretrained_ae and cfg.general.continue_training):
        trainer.fit(ae, datamodule=dm)
        trainer.test(ae, datamodule=dm)

    # Print out info for child processes (hack)
    print(f'CHECKPOINT_DIR:  {results_dir}')

    # Exit here if we are only training a pytorch model
    if not cfg.general.train_external_model:
        wandb.finish()
        sys.exit(0)

    ###
    # CLASSIFIER
    ###

    ae.eval()

    if not cfg.external_classifier:
        print('No external classifier specified.  Exiting')
        sys.exit(0)

    # Instantiate external model
    if cfg.external_classifier:
        external_classifier = instantiate(cfg.external_classifier, _args_=[ae])

    # Set up our data
    if not dm.data_loaded:
        dm.prepare_data()
        dm.setup()

    train_dl = DataLoader(dm.ds_train, batch_size=len(dm.ds_train), shuffle=False, num_workers=11)
    val_dl = DataLoader(dm.ds_val, batch_size=len(dm.ds_val), shuffle=False, num_workers=11)
    test_dl = DataLoader(dm.ds_test, batch_size=len(dm.ds_test), shuffle=False, num_workers=11)

    ###
    # Perform training
    ###
    for X_train, y_train in train_dl:
        pass
    external_classifier.fit(X_train, y_train)

    ###
    # Perform validation
    ###
    for X_val, y_val in val_dl:
        pass
    external_classifier.validation_step(X_val, y_val)

    ###
    # Perform test
    ###
    for X_test, y_test in test_dl:
        pass
    external_classifier.test_step(X_test, y_test)

    wandb.finish()

if __name__ == '__main__':
    main()
