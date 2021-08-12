import torch
from torch import nn
import numpy as np

from copy import deepcopy

base_config = {
    'device': 'cpu',
}


class Ensemble(nn.Module):
    """
    Wraps a list of trained model with functionality for adding epistemic uncertainty estimation.
    
    Only works with models which have output dimension of 1.
    """
    def __init__(self, models, dist_fam, args={}):
        super().__init__()
        
        self.config = deepcopy(base_config)
        self.config.update(args)
        
        self.model = deepcopy(models[0])
        self.dist_fam = dist_fam
        self.state_dicts = [ model.state_dict() for model in models ]
            
        self.gpu = self.config["device"] != "cpu"

        
        
    def process_dataset(self, dataset):
        """
        summarizes information about training data in terms of 
        variance of weights seen during sgd after training model
        """
        pass
            
    
    def forward(self, inputs, verbose=False):
        """
        assumes inputs are of shape (N, input_dims...)
        where N is the batch dimension,
              input_dims... are the dimensions of a single input
              
              
        returns 
            mu = mean( model(inputs) for model in models ) -- shape (N, d)
            unc =  unc measure from outputs -- shape (N)

        """
        n = len(self.state_dicts)
        outputs = []
        for state_dict in self.state_dicts:
            self.model.load_state_dict(state_dict)
            if self.gpu:
                self.model.cuda()
            with torch.no_grad():
                outputs.append( self.model(inputs) )
        
        outputs = torch.stack(outputs, dim=0)
        mu, unc = self.dist_fam.merge_ensemble(outputs)
        
        return mu, unc