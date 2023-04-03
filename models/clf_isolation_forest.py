#!/usr/bin/env python

import os
import pickle
from sklearn.ensemble import IsolationForest
import ipdb
from metrics.utilities import *
import wandb
import copy

"""
Takes in data and performs a classification using
the sklearn Random Forest classsifier.
"""

class ClfIsolationForest():
    def __init__(
        self, 
        ae,
        sklearn_model=None, 
        results_dir='./', 
    ):
        self.ae = ae
        self.model = sklearn_model
        self.results_dir = results_dir
        self.metrics = dict()

    def forward(self, X):
        # Get our final features using AE
        clf_input, L, S = self.ae.forward_ae(X)
        clf_input = clf_input.detach().cpu().squeeze().numpy()

        y = self.model.predict(clf_input)
        y[y == 1] = 0
        y[y == -1] = 1
        return y

    def fit(self, X, y):
        # During training only fit using benign samples
        benign_x = copy.deepcopy(X[y.flatten() == 0.0])
        benign_y = copy.deepcopy(y[y.flatten() == 0.0])

        # Get final feature set using AE
        clf_input, L, S = self.ae.forward_ae(benign_x)
        clf_input = clf_input.detach().cpu().squeeze().numpy()
        benign_y = benign_y.detach().cpu().squeeze().numpy()

        # Perform Training
        self.model.fit(clf_input, benign_y)

        # Perform predict to get a score for training
        y_pred = self.forward(X)
        self.log_epoch_end('classifier/train/', y, y_pred)

    def validation_step(self, X, y):
        y = y.detach().cpu().squeeze().numpy()
        y_pred = self.forward(X)
        self.log_epoch_end('classifier/val/', y, y_pred)

    def test_step(self, X, y):
        y = y.detach().cpu().squeeze().numpy()
        y_pred = self.forward(X)
        self.log_epoch_end('classifier/test/', y, y_pred)

    def predict_proba(self, X):
        pass

    def score(self, X, y):
        pass

    def save(self, path=None):
        if path:
            with open(path, 'wb') as f:
                pickle.dump(self, f)

    @staticmethod
    def load(path=None):
        if path and os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)

    def log_epoch_end(self, epoch_type, truth, predictions):
        # Obtain and log metrics
        print(f'{epoch_type}:')

        self.metrics[f'{epoch_type}predictions'] = predictions
        self.metrics[f'{epoch_type}truth'] = truth

        # Collect metrics
        self.metrics[f'{epoch_type}prec'] = precision(
            predicted_labels=predictions,
            true_labels=truth,
            metrics_dir=self.results_dir,
            is_training=False,
            log_to_disk=False
        )
        
        self.metrics[f'{epoch_type}rec'] = recall(
            predicted_labels=predictions,
            true_labels=truth,
            metrics_dir=self.results_dir,
            is_training=False,
            log_to_disk=False
        )
        
        self.metrics[f'{epoch_type}f1'] = f1_score(
            predicted_labels=predictions,
            true_labels=truth,
            metrics_dir=self.results_dir,
            is_training=False,
            log_to_disk=False
        )

        self.metrics[f'{epoch_type}accuracy'] = accuracy(
            predicted_labels=predictions,
            true_labels=truth,
            metrics_dir=self.results_dir,
            is_training=False,
            log_to_disk=False
        )

        # Save confusion matrix
        cm, cm_norm = plot_confusion_matrix(
            predicted_labels=predictions,
            true_labels=truth,
            metrics_dir=self.results_dir,
            is_training=False,
            prefix=epoch_type,
            title=f'{epoch_type} Confusion Matrix',
            target_names=['benign','attack']
        )

        self.metrics[f'{epoch_type}false_alarm_rate'] = false_alarm_rate(
            predicted_labels=predictions,
            true_labels=truth,
            metrics_dir=self.results_dir,
            is_training=False,
            log_to_disk=False
        )
        
        ## Log our metrics
        wandb.log({
            f'{epoch_type}precision' : self.metrics[f'{epoch_type}prec'],
            f'{epoch_type}recall' : self.metrics[f'{epoch_type}rec'],
            f'{epoch_type}f1-score' : self.metrics[f'{epoch_type}f1'],
            f'{epoch_type}accuracy' : self.metrics[f'{epoch_type}accuracy'],
            f'{epoch_type}false_alarm_rate' : self.metrics[f'{epoch_type}false_alarm_rate'],
        })
