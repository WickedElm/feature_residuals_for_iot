import torch
import torchmetrics
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import numpy as np
import pandas as pd

class StandardNetflowDataset(torch.utils.data.Dataset):
    def __init__(self, data_df=None, transform=None, class_weights=None):
        self.data_df = data_df
        self.transform = transform
        self.dims = self.data_df.iloc[0, 1:-1].shape
        self.class_weights = class_weights

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # NOTE:  Timestamp removed, conversion to floats
        flow = self.data_df.iloc[idx, 1:-1]
        flow = np.array([flow], dtype=float)

        label = self.data_df.iloc[idx, -1]
        label = np.array([label])

        if self.transform:
            flow = self.transform(flow)
            label = self.transform(label)

        return flow, label

class SarhanFormatNetflowDataset(torch.utils.data.Dataset):
    def __init__(self, data_df=None, transform=None, class_weights=None):
        self.data_df = data_df
        self.transform = transform
        self.dims = self.data_df.iloc[0, :-1].shape
        self.class_weights = class_weights

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # NOTE:  Timestamp removed, conversion to floats
        flow = self.data_df.iloc[idx, :-1]
        flow = np.array([flow], dtype=float)

        label = self.data_df.iloc[idx, -1]
        label = np.array([label])

        if self.transform:
            flow = self.transform(flow)
            label = self.transform(label)

        return flow, label

class SarhanFormatWithCacheNetflowDataset(torch.utils.data.Dataset):
    def __init__(self, data_df=None, transform=None, class_weights=None):
        self.data_df = data_df
        self.transform = transform
        self.dims = self.data_df.iloc[0, :-1].shape
        self.class_weights = class_weights
        self.L_cache = None
        self.S_cache = None

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # NOTE:  Timestamp removed, conversion to floats
        flow = self.data_df.iloc[idx, :-1]
        flow = np.array([flow], dtype=float)

        label = self.data_df.iloc[idx, -1]
        label = np.array([label])

        if self.transform:
            flow = self.transform(flow)
            label = self.transform(label)

        if self.L_cache == None:
            L = torch.zeros_like(flow)
            S = torch.zeros_like(flow)
        else:
            L = self.L_cache[idx]
            S = self.S_cache[idx]

        return flow, L, S, label

class SScalingNetflowDataset(torch.utils.data.Dataset):
    def __init__(self, data_df=None, transform=None, class_weights=None):
        self.data_df = data_df
        self.transform = transform
        self.dims = self.data_df.iloc[0, 1:-1].shape
        self.class_weights = class_weights
        self.L_cache = None
        self.S_cache = None

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # NOTE:  Timestamp removed, conversion to floats
        flow = self.data_df.iloc[idx, 1:-1]
        flow = np.array([flow], dtype=float)

        label = self.data_df.iloc[idx, -1]
        label = np.array([label])

        if self.transform:
            flow = self.transform(flow)
            label = self.transform(label)

        if self.L_cache == None:
            L = torch.zeros_like(flow)
            S = torch.zeros_like(flow)
        else:
            L = self.L_cache[idx]
            S = self.S_cache[idx]

        return flow, L, S, label

class CachingNetflowDataset(torch.utils.data.Dataset):
    def __init__(self, data_df=None, transform=None, class_weights=None):
        self.data_df = data_df
        self.transform = transform
        self.dims = self.data_df.iloc[0, 1:-1].shape
        self.class_weights = class_weights
        self.L_cache = None
        self.S_cache = None

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # NOTE:  Timestamp removed, conversion to floats
        flow = self.data_df.iloc[idx, 1:-1]
        flow = np.array([flow], dtype=float)

        label = self.data_df.iloc[idx, -1]
        label = np.array([label])

        if self.transform:
            flow = self.transform(flow)
            label = self.transform(label)

        if self.L_cache == None:
            L = torch.zeros_like(flow)
            S = torch.zeros_like(flow)
        else:
            L = self.L_cache[idx]
            S = self.S_cache[idx]

        return flow, L, S, label

class TimeSeriesNetflowDataset(torch.utils.data.Dataset):
    def __init__(self, data_df=None, transform=None, sequence_length=10, class_weights=None):
        self.data_df = data_df
        self.transform = transform
        self.sequence_length = sequence_length
        self.dims = self.data_df.iloc[0, 1:-1].shape
        self.class_weights = class_weights

    def __len__(self):
        return len(self.data_df) - self.sequence_length - 1

    def __getitem__(self, idx):
        netflow_sequence = self.data_df.iloc[idx:idx + self.sequence_length, 1:-1]
        netflow_sequence = np.array([netflow_sequence], dtype=float)

        # Label is last entry in the sequence
        label_sequence = self.data_df.iloc[idx:idx + self.sequence_length, -1]
        label_sequence = np.array([label_sequence])

        if self.transform:
            netflow_sequence = self.transform(netflow_sequence)
            label_sequence = self.transform(label_sequence)

        return netflow_sequence, label_sequence

class NetflowRandomSampleTimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, dataset=None, transform=None, sequence_length=10):
        self.dataset = dataset
        self.transform = transform
        self.sequence_length = sequence_length
        self.dims = self.dataset.dims

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        netflow_sequence = self.dataset[idx - self.sequence_length:idx][0]

        # Label is last entry in the sequence
        label_sequence = self.dataset[idx - self.sequence_length:idx][1]

        if self.transform:
            netflow_sequence = self.transform(netflow_sequence)
            label_sequence = self.transform(label_sequence)

        return netflow_sequence, label_sequence
