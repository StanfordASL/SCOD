from torchvision.datasets import MNIST as torchMNIST
from torchvision.datasets import FashionMNIST
from torch.utils.data import Subset
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torch
import numpy as np
import os
import nn_ood

class BinaryMNIST(Subset):
    def __init__(self, split, N=None):
        dataset = torchMNIST
        root = os.path.join(nn_ood.DATASET_FOLDER, "MNIST")
        if split == "train":
            mnist_split = "train"
            target_criterion = lambda x: x < 2
        elif split == "val":
            mnist_split = "val"
            target_criterion = lambda x: x < 2
        elif split == "ood":
            mnist_split = "val"
            target_criterion = lambda x: x >= 2
        elif split == "fashion":
            dataset = FashionMNIST
            root = os.path.join(nn_ood.DATASET_FOLDER, "FashionMNIST")
            mnist_split = "val"
            target_criterion = lambda x: x == x
        
        self.mnist = dataset(root=root, train=(split=="train"), download=True)
        
        self.normalize = transforms.Normalize((0.1307,), (0.3081,))
        
        # filter out only 1s
        valid_idx = np.flatnonzero(target_criterion(self.mnist.targets))
        
        if N is not None:
            valid_idx = valid_idx[:N]
        
        super().__init__(self.mnist, valid_idx)
        
        
        
    def __getitem__(self, i):
        input, target = super(BinaryMNIST, self).__getitem__(i)
        target = torch.from_numpy(np.array([target % 2])).float()
        input = transforms.ToTensor()(input)
        input = self.normalize(input)
        return input, target

    
