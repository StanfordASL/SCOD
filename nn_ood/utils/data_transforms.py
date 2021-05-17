import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np


class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    
    def __getitem__(self, i):
        x,y = self.dataset[i]
        x = self.transform(x)
        return x,y

    def __len__(self):
        return len(self.dataset)
    

## TRANSFORMS
# all take argument of severity, integer between 0 (small amount) to 4 (large deviation)
class GaussianNoise(object):
    def __init__(self, severity=0):
        std_options = [0.08, 0.12, 0.16, 0.18, 0.20]
        self.std = std_options[severity]
    
    def __call__(self, x):
        return x + self.std*torch.randn(x.size())
    
    def __repr__(self):
        return self.__class__.__name__ + '_std={1}'.format(self.std)
    
class SaltAndPepper(object):
    def __init__(self, severity=0, lowval=-1., highval=1.):
        thresholds = [0.01, 0.02, 0.03, 0.05, 0.07]
        self.threshold = thresholds[severity]
    
    def __call__(self):
        random_matrix = torch.rand(x.shape)
        x = x*(random_matrix < 1 - self.threshold)*(random_matrix > self.threshold) +\
            upper_val*(random_matrix >= 1 - self.threshold)+\
            lower_val*(random_matrix <= self.threshold)
        
        return x
    
    def __repr__(self):
        return self.__class__.__name__ + '_threshold={1}'.format(self.threshold)
    

class GaussianBlur(object):
    def __init__(self, severity=0):
        std_options = np.linspace(0.1,2.0,5)
        self.transform = transforms.GaussianBlur(5,sigma=std_options[severity])
    
    def __call__(self, x):
        return self.transform(x)
    
    def __repr__(self):
        return self.__class__.__name__ + '_std={1}'.format(self.std)