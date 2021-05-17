import torch
from torch import nn
import numpy as np

from ..second_order import zero_grads
from copy import deepcopy

from tqdm import trange, tqdm

base_config = {
    'batch_size': 16,
    'device': 'cpu',
    'num_samples': 150,
    'itr_between_samples':100,
    'learning_rate':1e-4
}


class SWAG(nn.Module):
    """
    Wraps a trained model with functionality for adding epistemic uncertainty estimation.
    
    Only works with models which have output dimension of 1.
    """
    def __init__(self, model, args={}):
        super().__init__()
        
        self.config = deepcopy(base_config)
        self.config.update(args)
        
        self.model = deepcopy(model)
        self.gpu = self.config["device"] != "cpu"

        
        self.n_params = int(sum(p.numel() for p in model.parameters()))
        
        self.batch_size = self.config['batch_size']
        self.num_samples = self.config['num_samples']
        self.itr_between_samples = self.config['itr_between_samples']
        self.learning_rate = self.config['learning_rate']
        
        self.configured = nn.Parameter(torch.zeros(1, dtype=torch.bool), requires_grad=False)
        
        
        self.mean = nn.Parameter(torch.zeros(self.n_params), requires_grad = False)
        self.cov_diag = nn.Parameter(torch.zeros(self.n_params), requires_grad = False)
        self.cov_factor = nn.Parameter(torch.zeros(self.n_params, self.num_samples), requires_grad=False)
        
        
    def process_dataset(self, dataset, criterion):
        """
        summarizes information about training data in terms of 
        variance of weights seen during sgd after training model
        """
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size=self.batch_size, 
                                                 shuffle=True)
        
        def prep_vec(vec):
            if self.gpu:
                return vec.cuda()
            else:
                return vec
        
        optim = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.)
        
        
        mean = self._get_param_vec().detach().cpu()
        sq_mean = mean.data**2
        D = torch.zeros(self.n_params, self.num_samples, dtype=torch.float)
        D.data[:,0] = self._get_param_vec().detach().cpu()
        
        # loop through data as many times as we need to get 
        # num_samples of the weights with itr_between_samples
        N = len(dataloader)
        n_epochs = int( self.num_samples*self.itr_between_samples / N ) + 1
        i = 0
        idx = 0
        with tqdm(total=self.num_samples) as pbar:
            for _ in range(n_epochs):
                for inputs,labels in dataloader:
                    inputs = prep_vec(inputs)
                    labels = prep_vec(labels)

                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optim.step()

                    i += 1
                    idx = int(i / self.itr_between_samples)
                    if idx == self.num_samples:
                        break

                    if i % self.itr_between_samples == 0:
                        params = self._get_param_vec().detach().cpu()

                        mean = (idx*mean + params ) / (idx + 1)

                        sq_mean = (idx*sq_mean + params**2 ) / (idx + 1)

                        D.data[:, idx] = params - mean
                        
                        pbar.update(1)
                        
        self.mean.data = mean
                    
        self.cov_diag.data = torch.clamp( (sq_mean - mean**2), 0., np.inf )/2
        self.cov_factor.data = D / ( np.sqrt(self.num_samples - 1) * np.sqrt(2) )
        self.configured.data = torch.ones(1, dtype=torch.bool)

        
        self._set_params_from_vec(mean)
                
    def _get_param_vec(self):
        """
        returns parameters of model vectorized
        """
        return torch.cat([p.contiguous().view(-1) 
                          for p in self.model.parameters()]
                        )
    
    def _get_grad_vec(self):
        """
        returns gradient of NN parameters flattened into a vector
        assumes backward() has been called so each parameters grad attribute
        has been updated
        """
        return torch.cat([p.grad.contiguous().view(-1) 
                             for p in self.model.parameters()]
                        )
    
    def _set_params_from_vec(self, params):
        i = 0
        for p in self.model.parameters():
            n = p.numel()
            
            p.data = params[i:i+n].view(p.shape)
            
            i += n
    
        if self.gpu:
            self.model.cuda()
            
    def _sample_params(self, n):
        """
        returns n vectors of params sampled from posterior
        """
        sample1 = torch.randn(n, self.num_samples)
        sample2 = torch.randn(n, self.n_params)
        sample = self.mean + sample1 @ self.cov_factor.t() + torch.sqrt(self.cov_diag).unsqueeze(0) * sample2
        return sample
    
    def forward_sampling(self, inputs, n=50):
        """
        
        """
        weights = self._sample_params(n)
        outputs = []
        for j in trange(n):
            param_vec = weights[j,:]
            self._set_params_from_vec(param_vec)
            with torch.no_grad():
                outputs.append( self.model(inputs) )
        
        #reset params
        self._set_params_from_vec(self.mean)
        
        outputs = torch.stack(outputs, dim=0)
        mean_outputs = outputs.mean(dim=0)
        std_outputs = torch.sqrt( ((outputs - mean_outputs)**2).sum(dim=0)/(n-1) )[:,0]
        
        return mean_outputs, std_outputs
            
    
    def forward(self, inputs, verbose=False):
        """
        assumes inputs are of shape (N, input_dims...)
        where N is the batch dimension,
              input_dims... are the dimensions of a single input
              
        only works with models with output dimension of 1
              
        returns 
            mu = model(inputs) -- shape (N, 1)
            unc = hessian based uncertainty estimates shape (N)
        """
        if not self.configured:
            print("Must call process_dataset first before using model for predictions.")
            raise NotImplementedError
            
        N = inputs.shape[0]
        
        mu = self.model(inputs)
        unc = torch.zeros(N)
        
        # compute uncertainty by backpropping back into each sample
        for j in range(N):
            zero_grads(self.model)
            
            # need to retain graph since we call backward multiple times
            mu[j].backward(retain_graph=True) 
            g = self._get_grad_vec().cpu()
            
            with torch.no_grad():
                low_rank_term = ((g @ self.cov_factor)**2).sum()
                diag_term = ((g**2)*self.cov_diag).sum()
                unc[j] = torch.sqrt( low_rank_term + diag_term )
            
        return mu, unc