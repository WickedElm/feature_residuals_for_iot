import torch 
import torchvision
from torchvision import transforms
import numpy as np

class NetflowToTensor(object):
    def __call__(self, sample):
        return torch.from_numpy(sample).float()
