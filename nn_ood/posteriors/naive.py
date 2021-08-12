import torch
from torch import nn
import numpy as np

from copy import deepcopy


base_config = {
}

    
class Naive(nn.Module):
    """
    Wraps a trained model with functionality for adding epistemic uncertainty estimation.
    
    Only works with models which have output dimension of 1.
    """
    def __init__(self, model, dist_fam, args={}):
        super().__init__()
        
        self.config = deepcopy(base_config)
        self.config.update(args)
        
        self.model = model
        self.dist_fam = dist_fam
        self.gpu = self.config["device"] != "cpu"

        self.trainable_params = list(filter(lambda x: x.requires_grad, self.model.parameters()))
        self.n_params = int(sum(p.numel() for p in self.trainable_params))  
        
    def process_dataset(self, dataset):
        """
        summarizes information about training data by logging gradient directions
        seen during training, and then using gram schmidt of these to form
        an orthonormal basis. directions not seen during training are 
        taken to be irrelevant to data, and used for detecting generalization
        
        dataset - torch dataset of (input, target) pairs
        """
        pass
    
    def forward(self, inputs, verbose=False):
        """
        assumes inputs are of shape (N, input_dims...)
        where N is the batch dimension,
              input_dims... are the dimensions of a single input
                            
        returns 
            mu = model(inputs) -- shape (N, 1)
            unc = hessian based uncertainty estimates shape (N)
        """ 
        mu = self.model(inputs)
        output = self.dist_fam.output(mu)
        unc = self.dist_fam.uncertainty(output)
    
        return output, unc