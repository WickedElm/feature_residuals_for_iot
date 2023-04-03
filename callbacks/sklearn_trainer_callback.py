#!/usr/bin/env python

import argparse
import importlib
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

from metrics.utilities import *
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

class SklearnTrainerCallback(Callback):
    def __init__(self, opt, clf_model_name, rae, feature_transformer, s_threshold=[0.001], threshold_name='default'):
        super().__init__()
        self.opt = opt
        self.clf_model_name = clf_model_name
        self.feature_transformer = feature_transformer
        self.rae = rae

        self.s_threshold = s_threshold
        if len(s_threshold) == 1:
            self.s_threshold = s_threshold[0]

        self.threshold_name = threshold_name

        # Create the sklearn model
        self.model_modulename, self.model_classname = clf_model_name.split('.')
        self.model_module = importlib.import_module(f'models.{self.model_modulename}')
        self.model_class = getattr(self.model_module, self.model_classname)

        self.data_loaded = False
        self.training_data = None
        self.validation_data = None

    def recalculate_training_thresholds(self):
        # Recalculate the thresholds based on autoencoder reconstruction
        # Using the full training set of data
        # This mimics the autoencoder's validation_step core functionality

        if self.threshold_name == 'no_threshold':
            return

        x = torch.tensor(self.training_data.iloc[:,1:-1].values).float().to(self.rae.device)
        label = torch.tensor(self.training_data.iloc[:,-1].values).float().to(self.rae.device)
        benign_x = x[label == 0]
        attack_x = x[label == 1]

        recon = self.rae.forward(x)
        loss, feature_loss = self.rae.loss_function(recon, x)

        benign_recon = self.rae.forward(benign_x)
        benign_loss, benign_feature_loss = self.rae.loss_function(benign_recon, benign_x)

        attack_recon = self.rae.forward(attack_x)
        attack_loss, attack_feature_loss = self.rae.loss_function(attack_recon, attack_x)

        threshold_options, feature_threshold_options = self.rae.calculate_threshold_options(benign_loss, attack_loss, benign_feature_loss, attack_feature_loss)

        # Update s_threshold to use new threshold
        if 'feature' in self.threshold_name:
            threshold_index = self.rae.feature_threshold_names.index(self.threshold_name)
            self.s_threshold = self.rae.feature_threshold_options[threshold_index]
        else:
            threshold_index = self.rae.threshold_names.index(self.threshold_name)
            self.s_threshold = self.rae.threshold_options[threshold_index]

    def on_validation_epoch_end(self, trainer, pl_module):
        # Get training and validation data
        if not self.data_loaded:
            if self.opt.use_all_training_data and hasattr(trainer.datamodule.ds_train, 'all_training_data'):
                print('Using ds_train.all_training_data')
                self.training_data = copy.deepcopy(trainer.datamodule.ds_train.all_training_data)
            else:
                self.training_data = copy.deepcopy(trainer.datamodule.ds_train.original_X)

            self.validation_data = copy.deepcopy(trainer.datamodule.ds_val.data_df)
            self.data_loaded = True

            self.recalculate_training_thresholds()

        xtrain, ytrain = self.feature_transformer.transform(trainer, self.rae, self.training_data, threshold=self.s_threshold) 
        xval, yval = self.feature_transformer.transform(trainer, self.rae, self.validation_data, threshold=self.s_threshold)

        # Save off LSO comparison
        if self.feature_transformer.__class__.__name__ == "OriginalLSFeatureTransformer" or self.feature_transformer.__class__.__name__ == "OriginalLSThresholdFeatureTransformer" :
            self.feature_transformer.plot_lso_comparison(self.opt.results_dir, f'val_{self.threshold_name}')

        # Save for reference
        with open(f'{self.opt.results_dir}/{self.feature_transformer.__class__.__name__}_{self.threshold_name}_sample_data.npy', 'wb') as f:
            np.save(f, xval)
            np.save(f, yval)

        # Used on_train_end
        self.ytrain = ytrain
        self.yval = yval

        # Create new model
        if 'knn' in self.model_modulename:
            self.model = self.model_class(n_neighbors=self.opt.n_neighbors)
        elif 'random_forest' in self.model_modulename:
            self.model = self.model_class(max_features=self.opt.rf_max_features)
        elif 'threshold' in self.model_modulename:
            self.model = self.model_class(threshold=self.s_threshold)
        else:
            self.model = self.model_class()

        # Call fit with training data
        self.model.fit(xtrain, ytrain)

        print('VALIDATION:')

        # Perform validation 
        self.predictions = self.model.forward(xval,)
        self.predictions_proba = self.model.predict_proba(xval,)
        self.val_acc = self.model.score(xval, yval)

        # Collect metrics
        self.prec = precision(
            predicted_labels=self.predictions,
            true_labels=yval,
            metrics_dir=f'{self.opt.results_dir}/{self.threshold_name}',
            is_training=False,
            log_to_disk=False
        )
        
        self.rec = recall(
            predicted_labels=self.predictions,
            true_labels=yval,
            metrics_dir=f'{self.opt.results_dir}/{self.threshold_name}',
            is_training=False,
            log_to_disk=False
        )
        
        self.f1 = f1_score(
            predicted_labels=self.predictions,
            true_labels=yval,
            metrics_dir=f'{self.opt.results_dir}/{self.threshold_name}',
            is_training=False,
            log_to_disk=False
        )
        
        # Log our metrics
        pl_module.log(f'classifier/val_{self.threshold_name}/{self.feature_transformer.__class__.__name__}_precision', self.prec, on_step=False, on_epoch=True)
        pl_module.log(f'classifier/val_{self.threshold_name}/{self.feature_transformer.__class__.__name__}_recall', self.rec, on_step=False, on_epoch=True)
        pl_module.log(f'classifier/val_{self.threshold_name}/{self.feature_transformer.__class__.__name__}_f1-score', self.f1, on_step=False, on_epoch=True)
        pl_module.log(f'classifier/val_{self.threshold_name}/{self.feature_transformer.__class__.__name__}_accuracy', self.val_acc, on_step=False, on_epoch=True)

        if hasattr(self.model.model, 'oob_score_'):
            self.val_oob_score = self.model.model.oob_score_
            pl_module.log(f'classifier/val_{self.threshold_name}/{self.feature_transformer.__class__.__name__}_oob_score', self.val_oob_score, on_step=False, on_epoch=True)

    def on_train_end(self, trainer, pl_module):
        EXPERIMENTS_DB = './output/experiments_db'

        if 'random_forest' in self.model_modulename:
            importances = self.model.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plot_feature_importance(self.model, self.feature_transformer.num_features, self.feature_transformer.feature_names, metrics_dir=self.opt.results_dir, prefix=f'{self.feature_transformer.__class__.__name__}_{self.threshold_name}')

        # VALIDATION - Get confusion matrix
        cm, cm_norm = plot_confusion_matrix(
            predicted_labels=self.predictions,
            true_labels=self.yval,
            metrics_dir=self.opt.results_dir,
            prefix=f'{self.feature_transformer.__class__.__name__}_{self.threshold_name}',
            is_training=False,
            title=f'{trainer.datamodule.dataset_name} Validation Set',
            target_names=['benign','attack']
        )

        add_metric_to_db(EXPERIMENTS_DB, f'{self.feature_transformer.__class__.__name__}_{self.threshold_name}_{self.opt.experiment}', trainer.datamodule.dataset_name, 'precision', self.prec)
        add_metric_to_db(EXPERIMENTS_DB, f'{self.feature_transformer.__class__.__name__}_{self.threshold_name}_{self.opt.experiment}', trainer.datamodule.dataset_name, 'recall', self.rec)
        add_metric_to_db(EXPERIMENTS_DB, f'{self.feature_transformer.__class__.__name__}_{self.threshold_name}_{self.opt.experiment}', trainer.datamodule.dataset_name, 'f1_score', self.f1)
        add_metric_to_db(EXPERIMENTS_DB, f'{self.feature_transformer.__class__.__name__}_{self.threshold_name}_{self.opt.experiment}', trainer.datamodule.dataset_name, 'accuracy', self.val_acc)

        if hasattr(self.model.model, 'oob_score_'):
            add_metric_to_db(EXPERIMENTS_DB, f'{self.feature_transformer.__class__.__name__}_{self.threshold_name}_{self.opt.experiment}', trainer.datamodule.dataset_name, 'oob_score', self.val_oob_score)

        # Save model
        self.model.feature_names = self.feature_transformer.feature_names
        self.model.save(path=f'{self.opt.results_dir}/{self.opt.save_prefix}_{self.threshold_name}_{self.feature_transformer.__class__.__name__}_{self.model_modulename}_{self.model_classname}-final.pkl')

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Save clf model
        self.model.feature_names = self.feature_transformer.feature_names
        self.model.save(path=f'{self.opt.results_dir}/{self.opt.save_prefix}_{self.threshold_name}_{self.feature_transformer.__class__.__name__}_{self.model_modulename}_{self.model_classname}-epoch={pl_module.current_epoch}.pkl')

    def on_test_epoch_start(self, trainer, pl_module):
        # Get training and validation data
        self.test_data = copy.deepcopy(trainer.datamodule.ds_test.data_df)

        # Get transformed features using RAE
        xtest, ytest = self.feature_transformer.transform(trainer, self.rae, self.test_data, threshold=self.s_threshold)

        # Save off LSO comparison
        if self.feature_transformer.__class__.__name__ == "OriginalLSFeatureTransformer" or self.feature_transformer.__class__.__name__ == "OriginalLSThresholdFeatureTransformer":
            self.feature_transformer.plot_lso_comparison(self.opt.results_dir, f'test_{self.threshold_name}')

        # Load RF model
        self.test_predictions = self.model.forward(xtest,)
        self.test_acc = self.model.score(xtest, ytest)

        print('FINAL TEST PREDICTIONS:')

        self.prec = precision(
            predicted_labels=self.test_predictions,
            true_labels=ytest,
            metrics_dir=f'{self.opt.results_dir}/{self.threshold_name}',
            is_training=False,
            log_to_disk=False
        )
        
        self.rec = recall(
            predicted_labels=self.test_predictions,
            true_labels=ytest,
            metrics_dir=f'{self.opt.results_dir}/{self.threshold_name}',
            is_training=False,
            log_to_disk=False
        )
        
        self.f1 = f1_score(
            predicted_labels=self.test_predictions,
            true_labels=ytest,
            metrics_dir=f'{self.opt.results_dir}/{self.threshold_name}',
            is_training=False,
            log_to_disk=False
        )
        
        # Log our metrics
        pl_module.log(f'classifier/test_{self.threshold_name}/{self.feature_transformer.__class__.__name__}_precision', self.prec, on_step=False, on_epoch=True)
        pl_module.log(f'classifier/test_{self.threshold_name}/{self.feature_transformer.__class__.__name__}_recall', self.rec, on_step=False, on_epoch=True)
        pl_module.log(f'classifier/test_{self.threshold_name}/{self.feature_transformer.__class__.__name__}_f1-score', self.f1, on_step=False, on_epoch=True)
        pl_module.log(f'classifier/test_{self.threshold_name}/{self.feature_transformer.__class__.__name__}_accuracy', self.test_acc, on_step=False, on_epoch=True)

        EXPERIMENTS_DB = './output/experiments_db'
        add_metric_to_db(EXPERIMENTS_DB, f'test/{self.feature_transformer.__class__.__name__}_{self.threshold_name}_{self.opt.experiment}', trainer.datamodule.dataset_name, 'precision', self.prec)
        add_metric_to_db(EXPERIMENTS_DB, f'test/{self.feature_transformer.__class__.__name__}_{self.threshold_name}_{self.opt.experiment}', trainer.datamodule.dataset_name, 'recall', self.rec)
        add_metric_to_db(EXPERIMENTS_DB, f'test/{self.feature_transformer.__class__.__name__}_{self.threshold_name}_{self.opt.experiment}', trainer.datamodule.dataset_name, 'f1_score', self.f1)
        add_metric_to_db(EXPERIMENTS_DB, f'test/{self.feature_transformer.__class__.__name__}_{self.threshold_name}_{self.opt.experiment}', trainer.datamodule.dataset_name, 'accuracy', self.test_acc)

