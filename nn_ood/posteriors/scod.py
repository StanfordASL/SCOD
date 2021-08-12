import torch
from torch import nn
import numpy as np

from ..sketching import sketch_args, Projector
from copy import deepcopy

from tqdm import tqdm

base_config = {
    'device': 'cpu', # device to use @TODO: replace with torch.device
    'num_samples': None, # sketch size T (T)
    'num_eigs': 10, # low rank estimate to recover (k)
    'weighted': False, # weight samples in sketch by loss
    'sketch_type': 'random', # sketch type 
    'proj_type': 'posterior_pred', # type of metric @TODO: move to forward_args
}

    
class SCOD(nn.Module):
    """
    Wraps a trained model with functionality for adding epistemic uncertainty estimation.
    """
    def __init__(self, model, dist_fam, args={}):
        """
        model: base DNN to equip with an uncertainty metric
        dist_fam: distributions.DistFam object representing how to interpret output of model
        args: configuration variables - defaults are in base_config
        """
        super().__init__()
        
        self.config = deepcopy(base_config)
        self.config.update(args)
        
        self.model = model
        self.dist_fam = dist_fam
        
        self.gpu = self.config["device"] != "cpu"

        # extract parameters to consider in sketch - keep all that yield valid gradients
        self.trainable_params = list(filter(lambda x: x.requires_grad, self.model.parameters()))
        self.n_params = int(sum(p.numel() for p in self.trainable_params))
        
        self.weighted = self.config['weighted']
        
        self.num_samples = self.config['num_samples']
        self.num_eigs = self.config['num_eigs']
        
        if self.num_samples is None:
            self.num_samples = 6*self.num_eigs + 4
            
        self.configured = nn.Parameter(torch.zeros(1, dtype=torch.bool), requires_grad=False)
        
        self.proj_type = self.config['proj_type']
        self.sketch_class = sketch_args[self.config['sketch_type']]   
        
        self.projector = Projector(
            N=self.n_params,
            r=2*max(self.num_eigs + 2, (self.num_samples-1)//3),
            T=self.n_params,
            gpu=self.gpu
        )
        
    def process_dataset(self, dataset):
        """
        summarizes information about training data by logging gradient directions
        seen during training, and then using gram schmidt of these to form
        an orthonormal basis. directions not seen during training are 
        taken to be irrelevant to data, and used for detecting generalization
        
        dataset - torch dataset of (input, target) pairs
        """
        # TODO: use .to(self.device) instead
        def prep_vec(vec):
            if self.gpu:
                return vec.cuda()
            else:
                return vec    
        
        # loop through data, one sample at a time
        print("computing basis")
            
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size=1, 
                                                 shuffle=True)
        
        sketch = self.sketch_class(N=self.n_params, 
                                   M=len(dataloader),
                                   r=self.num_eigs,
                                   T=self.num_samples,
                                   gpu=self.gpu)
        
        n_data = len(dataloader)
        for i, (inputs,labels) in tqdm(enumerate(dataloader), total=n_data):
            inputs = prep_vec(inputs)
            labels = prep_vec(labels)

            thetas = self.model(inputs) # get params of output dist
            weight = 1.
            if self.weighted:
                nll = self.dist_fam.loss(thetas, labels) # get nll of sample
                weight = torch.exp(-nll) # p(y|x)
            
            
            thetas = self.dist_fam.apply_sqrt_F(thetas) # pre-multipy by sqrt fisher
            thetas = thetas.mean(dim=0) # mean over batch dim
            jac = self._get_weight_jacobian(thetas) # then compute jacobian to get L^(i)_w
            sketch.low_rank_update(i,jac.t(),weight) # add 1/M jac jac^T to the sketch
        
        del jac
        
        eigs, basis = sketch.get_range_basis()
        del sketch
        
        self.projector.process_basis(eigs, basis)
            
        self.configured.data = torch.ones(1, dtype=torch.bool)
    
    def _get_weight_jacobian(self, vec):
        """
        returns d x nparam matrix, with each row being d(vec[i])/d(weights)
        """
        assert len(vec.shape) == 1
        grad_vecs = []
        for j in range(vec.shape[0]):
            self.model.zero_grad(set_to_none=True)
            vec[j].backward(retain_graph=True)
            g = self._get_grad_vec().detach()
            grad_vecs.append(g)
            
        return torch.stack(grad_vecs)
            
    def _get_grad_vec(self):
        """
        returns gradient of NN parameters flattened into a vector
        assumes backward() has been called so each parameters grad attribute
        has been updated
        """
        return torch.cat([p.grad.contiguous().view(-1) 
                             for p in self.trainable_params]
                        )
            
    
    def forward(self, inputs, n_eigs=None, Meps=5000):
        """
        assumes inputs are of shape (N, input_dims...)
        where N is the batch dimension,
              input_dims... are the dimensions of a single input
                            
        returns 
            mu = model(inputs) -- shape (N, 1)
            unc = hessian based uncertainty estimates shape (N)
        """
        if not self.configured:
            print("Must call process_dataset first before using model for predictions.")
            raise NotImplementedError
            
        if n_eigs is None:
            n_eigs = self.num_eigs
            
        N = inputs.shape[0]
        
        mu = self.model(inputs)
        unc = torch.zeros(N)
        
        # batch apply sqrt(I_th) to output
        theta = self.dist_fam.apply_sqrt_F(mu, exact=True)

        # compute uncertainty by backpropping back into each sample
        for j in range(N):
            jac = self._get_weight_jacobian(theta[j,:])    
            unc[j] = self.projector.compute_distance(jac.t(), self.proj_type, n_eigs=n_eigs, Meps=Meps)
    
        return self.dist_fam.output(mu), unc