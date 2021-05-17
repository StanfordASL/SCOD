from torchvision.datasets import MNIST
from torch.utils.data import Subset
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torch
import numpy as np
import os
import nn_ood

class RotatedMNIST(Subset):
    def __init__(self, split, N=None):
        self.mean_rot = 0
        self.std_rot = 3.14/4
        self.digit = 2
        if split == "train":
            gen = torch.random.manual_seed(1)
        elif split == "val":
            gen = torch.random.manual_seed(2)
        elif split == "ood":
            self.digit = 5
            gen = torch.random.manual_seed(3)
            self.mean_rot = 0
            self.std_rot = 3.14/4
        elif split == "ood_angle":
            self.digit = 2
            gen = torch.random.manual_seed(3)
            self.mean_rot = 3.14
            self.std_rot = 3.14/4
        
        root = os.path.join(nn_ood.DATASET_FOLDER, "MNIST")

        self.mnist = MNIST(root=root, train=(split=="train"), download=True)
        
        self.gen = gen
        self.normalize = transforms.Normalize((0.1307,), (0.3081,))
        
        # filter out only 1s
        valid_idx = np.flatnonzero(self.mnist.targets == self.digit)
        
        if N is not None:
            valid_idx = valid_idx[:N]
        
        super().__init__(self.mnist, valid_idx)
        
        
        
    def __getitem__(self, i):
        input, target = super(RotatedMNIST, self).__getitem__(i)
        target = torch.normal(self.mean_rot, self.std_rot, (1,), generator=self.gen)
        input = TF.rotate(input, target * (180 / 3.14))
        input = transforms.ToTensor()(input)
        input = self.normalize(input)
        return input, target

    
