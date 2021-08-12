import torch
from torch import nn
import numpy as np

from copy import deepcopy

from curvature.sampling import invert_factors, sample_and_replace_weights
from curvature import fisher

from tqdm import tqdm



base_config = {
    'input_shape': [1,28,28],
    'batch_size': 32,
    'device': 'cpu',
    'num_loss_samples': 30,
    'num_samples': 10,
    'norm': 1.,
    'scale': 1.,
}
    
class KFAC(nn.Module):
    """
    Wraps a trained model with functionality for adding epistemic uncertainty estimation.
    
    Only works with models which have output dimension of 1.
    """
    def __init__(self, model, dist_fam, args={}):
        super().__init__()
        
        self.config = deepcopy(base_config)
        self.config.update(args)
        
        self.model = deepcopy(model)
        self.posterior_mean = deepcopy(self.model.state_dict())
        self.dist_fam = dist_fam
        self.kfac = fisher.KFAC(self.model)
        
        zero_input = torch.zeros([1] + self.config['input_shape'])
        
        self.batch_size = self.config['batch_size']
        
#         self.prescale = 0.
        self.scale = self.config['scale']
        self.norm = self.config['norm']
        
        self.num_samples = self.config['num_samples']
        self.num_loss_samples = self.config['num_loss_samples']
        
        self.factors = nn.ModuleList()
        self.configured = nn.Parameter(torch.zeros(1, dtype=torch.bool), requires_grad=False)
        self.gpu = self.config["device"] != "cpu"
        
        # initialize factor list
        if self.gpu:
            zero_input = zero_input.cuda()
        output = self.model(zero_input)
        output.mean().backward()
        self.kfac.update(batch_size=1)
        
        factors = list(self.kfac.state.values())
        for i, fac in enumerate(factors):
            facModule = nn.ParameterList(nn.Parameter(p, requires_grad=False) for p in fac)
            self.factors.append(facModule)
            
        self.inv_factors_list = None
        
        
    def process_dataset(self, dataset):
        """
        summarizes information about training data by logging gradient directions
        seen during training, and then using gram schmidt of these to form
        an orthonormal basis. directions not seen during training are 
        taken to be irrelevant to data, and used for detecting generalization
        
        dataset - torch dataset of (input, target) pairs
        """
        def prep_vec(vec):
            if self.gpu:
                return vec.cuda()
            else:
                return vec    
        
        # loop through data as many times as we need to get 
        # num_samples of the weights with itr_between_samples
        print("computing basis")
        N = len(dataset)

        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size=self.batch_size, 
                                                 shuffle=True)
        
        n_data = len(dataloader)
        for i, (inputs,labels) in tqdm(enumerate(dataloader), total=n_data):
            inputs = prep_vec(inputs)

            thetas = self.model(inputs) # get params of output dist
            dist = self.dist_fam.dist(thetas)
            for _ in range(self.num_loss_samples):
                sampled_labels = dist.sample()

                loss = self.dist_fam.loss(thetas, sampled_labels).mean()
                self.model.zero_grad()
                loss.backward(retain_graph=True)

                self.kfac.update(batch_size=inputs.size(0))
                        
        factors = list(self.kfac.state.values())
        
        self.store_factors(factors)
        self.configured.data = torch.ones(1, dtype=torch.bool)
    
    def store_factors(self, factors):
        for i, fac in enumerate(factors):
            for j, val in enumerate(fac):
                self.factors[i][j].data = val
    
    def get_inv_factors(self, norm=None, scale=None):
        if norm is None:
            norm = self.norm
        if scale is None:
            scale = self.scale
            
        if self.inv_factors_list is None:
            factors = []
            for facModule in self.factors:
                factors.append(tuple(p for p in facModule))
            
            self.inv_factors_list = invert_factors(factors, norm=norm, scale=scale, estimator='kfac')
    
    def forward(self, inputs, verbose=False, norm=None, scale=None):
        """
        assumes inputs are of shape (N, input_dims...)
        where N is the batch dimension,
              input_dims... are the dimensions of a single input
                            
        returns 
            mu = model(inputs) -- shape (N, d)
            unc = hessian based uncertainty estimates shape (N)
        """
        if not self.configured:
            print("Must call process_dataset first before using model for predictions.")
            raise NotImplementedError
        
        self.get_inv_factors(norm, scale)
        outputs = []
        for sample in range(self.num_samples):
            sample_and_replace_weights(self.model, self.inv_factors_list, 'kfac')
            outputs.append(self.model(inputs))
            self.model.load_state_dict(self.posterior_mean)
        
        outputs = torch.stack(outputs, dim=0)
#         print(outputs)
        mu, unc = self.dist_fam.merge_ensemble(outputs)
    
        return mu, unc