#!/usr/bin/env python

import argparse
import importlib
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

from metrics.utilities import *
from data_modules.transformers import NetflowToTensor
from data_modules.torch_datasets import StandardNetflowDataset
import torch
import torchmetrics
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torch.nn.functional as F

import torchvision
from torchvision import transforms

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
import hydra

class NetflowConferenceDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path='./',
        batch_size=32,
        total_rows_threshold=500000,
        weighted=False,
        load_from_disk=False,
        load_data_path='tmp_ds_data',
        prefix='',
        reserve_type='',
        oversampling_multiplier=1.0
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.total_rows_threshold = total_rows_threshold
        self.load_from_disk = load_from_disk
        self.prefix = prefix
        self.reserve_type = reserve_type
        self.load_data_path = f'{hydra.utils.get_original_cwd()}/{load_data_path}'
        self.transform = transforms.Compose([
            NetflowToTensor()
        ])
        self.data_loaded = False

        self.weighted = weighted
        self.sampler = None
        self.scaler = 1.

    def prepare_data(self):
        # TODO:  Download data is needed
        pass

    def setup(self, stage=None):

        if not self.data_loaded:
            # NOTE:  May need to think about this if more than 1 GPU is ever used
            # Load our dataset using data_path
            ds = dataset.Dataset.Dataset.load(self.data_path)
            self.dataset_name = ds.name
            ds.sample_data(total_rows_threshold=self.total_rows_threshold)

            # Insert new processors/updates here
            ds.append_processor(dataset.DatasetProcessors.AddBytesPerSecondFeatures(), process=False)
            ds.append_processor(dataset.DatasetProcessors.ApplyFunctionToColumns(target_cols=['total_bytes', 'total_source_bytes', 'total_destination_bytes', 'total_bytes_per_second', 'source_bytes_per_second', 'destination_bytes_per_second'], function=math.log), process=False)
            ds.cols_to_normalize.append('total_bytes_per_second')
            ds.cols_to_normalize.append('destination_bytes_per_second')
            ds.cols_to_normalize.append('source_bytes_per_second')

            ds.append_processor(dataset.DatasetProcessors.AddPacketsPerSecondFeatures(), process=False)
            ds.append_processor(dataset.DatasetProcessors.ApplyFunctionToColumns(target_cols=['total_packets', 'source_packets', 'destination_packets', 'packets_per_second', 'source_packets_per_second', 'destination_packets_per_second'], function=math.log), process=False)
            ds.cols_to_normalize.append('packets_per_second')
            ds.cols_to_normalize.append('destination_packets_per_second')
            ds.cols_to_normalize.append('source_packets_per_second')

            ds.append_processor(dataset.DatasetProcessors.SortColumnsAlphabetically(), process=False)
            ds.append_processor(dataset.DatasetProcessors.MakeTimestampColumnFirst(), process=False)

            if self.load_from_disk:
                if not ds.load_data_split_from_disk(self.load_data_path, self.prefix, self.reserve_type):
                    print('Generating data splits.')
                    ds.perform_processing()
                    ds.save_split_to_disk(self.load_data_path, self.prefix, 'ae_tuning', 1.0, 0.15, 0.3)
                    ds.load_data_split_from_disk(self.load_data_path, self.prefix, self.reserve_type)
            else:
                ds.perform_processing()
                ds.default_stratified_split(split_point_1=0.15, split_point_2=0.3)

            # Remove any duplicates that occurred as a result of encoding
            train_to_keep = ds.training_data.iloc[:,:-1].drop_duplicates().index
            ds.training_data = ds.training_data.loc[train_to_keep]

            validation_to_keep = ds.validation_data.iloc[:,:-1].drop_duplicates().index
            ds.validation_data = ds.validation_data.loc[validation_to_keep]

            test_to_keep = ds.test_data.iloc[:,:-1].drop_duplicates().index
            ds.test_data = ds.test_data.loc[test_to_keep]

            ds.save_indices()

            # Shuffle our data
            ds.training_data = ds.training_data.sample(frac=1)
            ds.validation_data = ds.validation_data.sample(frac=1)
            ds.test_data = ds.test_data.sample(frac=1)
            ds.normalize_columns_min_max(target_cols=ds.cols_to_normalize)
            self.scaler = ds.scaler
            ds.print_dataset_info()

            # Calculate standard proportional weights
            total_samples = ds.training_data.shape[0]
            num_benign = ds.training_data.loc[ds.training_data.label == 0].shape[0]
            num_attack = total_samples - num_benign
            weight_benign = num_attack / num_benign
            weight_attack = num_benign / num_attack
            class_weights = torch.tensor([weight_benign, weight_attack], dtype=float)

            print(f'num_benign = {num_benign}, num_attack = {num_attack}, weight_attack = {weight_attack}')

            self.data_loaded = True

            # Construct training and validation sets
            # - class_weights is not used for weighting in StandardNetflowDataset
            # - it is stored for reference later if needed
            self.ds_train = StandardNetflowDataset(ds.training_data, self.transform, class_weights=class_weights)
            self.ds_val = StandardNetflowDataset(ds.validation_data, self.transform)
            self.ds_test = StandardNetflowDataset(ds.test_data, self.transform)

        if stage == 'train' or stage == None:
            self.dims = tuple(self.ds_train.dims)

        if stage == 'test':
            self.dims = tuple(self.ds_test.dims)

    def train_dataloader(self):
        return DataLoader(self.ds_train, shuffle=True, batch_size=self.batch_size, num_workers=12)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, num_workers=12)

    def test_dataloader(self):
            return DataLoader(self.ds_test, batch_size=self.batch_size, num_workers=12)
