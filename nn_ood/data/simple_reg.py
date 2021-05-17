import torch
import numpy as np


class SineLike(torch.utils.data.TensorDataset):
    def __init__(self, split='train', N=128):
        self.N = N
        if split == 'train':
            np.random.seed(1),
        elif split == 'val':
            np.random.seed(5)
        elif split == 'test':
            np.random.seed(10)
        elif split == 'ood':
            np.random.seed(10)
        else:
            print('split must be train/test/val/ood')
            
    
        if split == 'ood':
            self.x = np.concatenate([np.random.rand(int(N/3)) - 1, 
                 2 + 0.5*np.random.rand(int(N/3)), 
                 3.5 + np.random.rand(int(N/3))])
        else:
            self.x = np.concatenate( [2*np.random.rand(int(3*N/4)),
                                 2.5+np.random.rand(int(N/4))])
        
        self.x = self.x[:,None]
        
        self.y = np.sin(self.x**2)[:,0]
        x_t = torch.from_numpy(self.x).float()
        y_t = torch.from_numpy(self.y[:,None]).float()
        
        super().__init__(x_t, y_t)

    
    
class Cubic(torch.utils.data.TensorDataset):
    def __init__(self, split='train', N=20):
        self.N = N
        if split == 'train':
            np.random.seed(1),
        elif split == 'val':
            np.random.seed(5)
        else:
            np.random.seed(10)
            
        self.fn = lambda x : x**3
            
        if split == 'unif':
            self.x = np.linspace(-10,10,100)
            self.y = self.fn(self.x)
        elif split == 'ood':
            self.x = np.concatenate([-4 - 2*np.random.rand(N//2), 
                 4 + 2*np.random.rand(N - N//2)])
            self.y = self.fn(self.x)
        else:
            self.x = -4 + 8*np.random.rand(N)
            self.y = self.fn(self.x) + 3*np.random.randn(N)
        
        x_t = torch.from_numpy(self.x[:,None]).float()
        y_t = torch.from_numpy(self.y[:,None]).float()
        
        super().__init__(x_t, y_t)