import torch
import numpy as np

class TwoMoons(torch.utils.data.TensorDataset):
    def __init__(self, split='train', N=128):
        self.N = N
        if split == 'train':
            np.random.seed(1),
        elif split == 'val':
            np.random.seed(5)
        elif split == 'test':
            np.random.seed(10)
        elif split == 'unif':
            np.random.seed(10)
        else:
            print('split must be train/test/val/unif')
            
        
        if split == 'unif':
            self.x = 3*np.random.rand(N,2) - 1
        else:
            radii = 1 + np.sqrt(0.01)*np.random.randn(N)
            theta = np.pi*np.random.rand(int(N/2))
            outer_circ_x = radii[:int(N/2)]*np.cos(theta)
            outer_circ_y = radii[:int(N/2)]*np.sin(theta)
            inner_circ_x = 1 - radii[int(N/2):]*np.cos(theta)
            inner_circ_y = 1 - radii[int(N/2):]*np.sin(theta) - .5

            self.x = np.vstack([np.append(outer_circ_x, inner_circ_x),
                                np.append(outer_circ_y, inner_circ_y)]).T
    
        
        self.y = np.sin(np.sum(self.x,axis=-1,keepdims=True))
        
        x_t = torch.from_numpy(self.x).float()
        y_t = torch.from_numpy(self.y).float()
        
        super().__init__(x_t, y_t)

        
