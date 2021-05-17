from torch.utils.data import TensorDataset
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torch
import numpy as np
import pandas as pd
import os
import nn_ood


class WineData(TensorDataset):
    def __init__(self, split, N=None):
        root = os.path.join(nn_ood.DATASET_FOLDER, "wine")

        filename = "winequality-red.csv"
        if split == 'whites':
            filename = "winequality-white.csv"
        
        dataframe_raw = pd.read_csv(os.path.join(root,filename), sep=";")
        input_cols=list(dataframe_raw.columns)[:-1]
        output_cols = ['quality']
        
        inputs = dataframe_raw[input_cols].to_numpy()
        targets = dataframe_raw[output_cols].to_numpy()
        
        total_n = inputs.shape[0]
        
        np.random.seed(1)
        idx_list = np.random.permutation(total_n)
        if split == "train":
            idx_list = idx_list[:1000]
        elif split == "val":
            idx_list = idx_list[1000:]
        
        self.random_inputs = False
        self.low = torch.from_numpy(np.amin(inputs, axis=0)).float()
        self.high = torch.from_numpy(np.amax(inputs, axis=0)).float()
        if split == "random":
            self.random_inputs = True        
        
        if N is not None:
            N = min(N, len(idx_list))
            idx_list = idx_list[:N]
        
        x_t = torch.from_numpy(inputs[idx_list]).float()
        y_t = torch.from_numpy(targets[idx_list]).float()
                
        super().__init__(x_t,y_t)
        
        
        
    def __getitem__(self, i):
        input, target = super().__getitem__(i)
        if self.random_inputs:
            input = self.low + (self.high - self.low)*torch.rand(11)
        
        return input, target

