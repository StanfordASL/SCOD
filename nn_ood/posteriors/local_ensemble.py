import torch
from torch import nn

from ..second_order import zero_grads
from hessian_eigenthings import compute_hessian_eigenthings
from copy import deepcopy

from tqdm import trange

base_config = {
    'batch_size': 128,
    'max_samples': 16,
    'num_eigs': 20,
    'device': 'cpu',
    'model_preprocess': None,
    'n_y_samp': 1,
    'full_data': False,
}

class LocalEnsemble(nn.Module):
    """
    Wraps a trained model with functionality for adding epistemic uncertainty estimation.
    """
    def __init__(self, model, dist_fam, args={}):
        super().__init__()
        
        self.config = deepcopy(base_config)
        self.config.update(args)
        
        self.model = model
        self.dist_fam = dist_fam
        if self.config['model_preprocess'] is not None:
            self.config['model_preprocess'](model)
            
        
        self.gpu = self.config["device"] != "cpu"
        
        self.trainable_params = list(filter(lambda x: x.requires_grad, self.model.parameters()))
        self.n_params = int(sum(p.numel() for p in self.trainable_params))
        self.batch_size = self.config['batch_size']
        self.max_samples = self.config['max_samples']
        self.num_eigs = self.config['num_eigs']
        self.n_y_samp = self.config['n_y_samp']

        self.device = torch.device('cpu')
        if self.gpu:
            self.device = torch.cuda.current_device()
        
        self.mean = nn.Parameter(torch.zeros(self.n_params), requires_grad = False)
        self.configured = nn.Parameter(torch.zeros(1, dtype=torch.bool), requires_grad=False)
        self.top_eigs = nn.Parameter(torch.zeros(self.n_params, self.num_eigs, device=self.device), requires_grad=False)
        self.eig_weights = nn.Parameter(torch.zeros(self.num_eigs))
        
        self.sampled_weights = None

        
        
    def process_dataset(self, dataset):
        """
        summarizes information about training data in terms of 
        curvature of loss in weight space. 
        here we save only the top eigenvectors of the hessian.
        """
#         MVP_op = get_hvp_op(self.model, dataset, criterion, use_gpu=self.gpu, n_samples=32)
        
#         eigvals, eigvecs = lanczos_eigenvecs(MVP_op,
#                                              N=self.n_params,
#                                              num_eigs=self.num_eigs)

        self.model.eval()
        self.mean.data = self._get_param_vec().detach().cpu()
    
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size=self.batch_size, 
                                                 shuffle=True)
        eigs, eigvecs = compute_hessian_eigenthings(
                            self.model,dataloader,
                            lambda x,y: self.dist_fam.loss(x,y).mean(),
                            self.num_eigs, mode='power_iter', full_dataset=self.config['full_data'], max_samples=self.max_samples, use_gpu=self.gpu,
        momentum=0.9,
        power_iter_steps=20)
        eigvecs=eigvecs.T
        
        self.eig_weights.data = torch.from_numpy(eigs.copy() ).float()
        self.top_eigs.data = torch.from_numpy( eigvecs.copy() ).float()
        self.configured.data = torch.ones(1, dtype=torch.bool)
        
    def _get_weight_jacobian(self, vec):
        """
        returns d x nparam matrix, with each row being d(vec[i])/d(weights)
        """
        assert len(vec.shape) == 1
        grad_vecs = []
        for j in range(vec.shape[0]):
            zero_grads(self.model)
            vec[j].backward(retain_graph=True)
            g = self._get_grad_vec().detach()
            grad_vecs.append(g)
            
        return torch.stack(grad_vecs)
    
    def _get_param_vec(self):
        """
        returns parameters of model vectorized
        """
        return torch.cat([p.contiguous().view(-1) 
                          for p in self.trainable_params]
                        )
    
    def _get_grad_vec(self):
        """
        returns gradient of NN parameters flattened into a vector
        assumes backward() has been called so each parameters grad attribute
        has been updated
        """
        return torch.cat([p.grad.contiguous().view(-1) 
                             for p in self.trainable_params]
                        )
    
    def _set_params_from_vec(self, params):
        i = 0
        for p in self.trainable_params:
            n = p.numel()
            
            p.data = params[i:i+n].view(p.shape)
            
            i += n
    
        if self.gpu:
            self.model.cuda()
            
    def _sample_params(self, n, n_eigs=None):
        """
        returns n vectors of params sampled from posterior
        doesn't really work in this case because we only know directions to *not* sample from
        """
        if n_eigs == None:
            n_eigs = self.num_eigs
            
        sample= 1e-5*torch.randn(n, self.n_params)
        proj_sample = ( (self.eig_weights > 0) * (sample @ self.top_eigs[:,:n_eigs]) ) @ self.top_eigs[:,:n_eigs].t()
        sample -= proj_sample # remove variation in top eigendirections
        sample += self.mean
        return sample
    
    def forward_sampling(self, inputs, n=50, n_eigs=None, reset=False):
        """
        uses _sample_params to sample from the posterior
        """
        if self.sampled_weights == None:
            self.sampled_weights = self._sample_params(n, n_eigs)
        weights = self.sampled_weights
        outputs = []
        for j in range(n):
            param_vec = weights[j,:]
            self._set_params_from_vec(param_vec)
            with torch.no_grad():
                outputs.append( self.model(inputs) )
        
        #reset params
        if reset:
            self._set_params_from_vec(self.mean)
        
        outputs = torch.stack(outputs, dim=0)
        mean_outputs = outputs.mean(dim=0)
        std_outputs = 100*torch.sqrt( ((outputs - mean_outputs)**2).sum(dim=0)/(n-1) )[:,0]
        
        return mean_outputs, std_outputs
    
    def forward_extrap(self, mu, n_eigs):
        N = mu.shape[0]
        unc = torch.zeros(N, self.n_y_samp)
        for j in range(N):
            if self.n_y_samp > 1:
                y_samp = self.dist_fam.sample_y(mu,self.n_y_samp).to(mu.device) # d
                mu = mu[j:j+1,:].expand([self.n_y_samp]+[mu.shape[-1]]) # d x d
                loss = self.dist_fam.loss(mu, y_samp) # d x N
            else:
                loss = mu[j:j+1,0:1]
                
            for k in range(self.n_y_samp):
                zero_grads(self.model)
                loss[k].backward(retain_graph=True)
                
                with torch.no_grad():
                    g = self._get_grad_vec()
                    proj_g = g @ self.top_eigs[:, :n_eigs]
                    proj_g = proj_g @ self.top_eigs[:,:n_eigs].t()
                    proj_g = g - proj_g
                    unc[j,k] = torch.norm(proj_g)
        
        unc = torch.amin(unc, 1)
        return unc
    
    def forward_fisher(self, mu, n_eigs):
        N = mu.shape[0]
        unc = torch.zeros(N)
        # batch apply sqrt(I_th) to output
        theta = self.dist_fam.apply_sqrt_F(mu, mu)
        
        # compute uncertainty by backpropping back into each sample
        for j in range(N):
            jac = self._get_weight_jacobian(theta[j,:])
            with torch.no_grad():
                proj_g =  jac @ self.top_eigs[:,:n_eigs]
                proj_g = proj_g @ self.top_eigs[:,:n_eigs].t()
                impactful_g = jac - proj_g
                unc[j] = torch.norm(impactful_g)
                
        return unc
    
    def forward(self, inputs, verbose=False, n_eigs=None, n_samples=1, online_metric='extrap'):
        """
        assumes inputs are of shape (N, input_dims...)
        where N is the batch dimension,
              input_dims... are the dimensions of a single input
              
        only works with models with output dimension of 1
              
        returns 
            mu = model(inputs) -- shape (N, 1)
            unc = hessian based uncertainty estimates shape (N, 1)
        """
        if not self.configured:
            print("Must call process_dataset first before using model for predictions.")
            raise NotImplementedError
            
        if n_samples > 1:
            return self.forward_sampling(inputs, n=n_samples, n_eigs=n_eigs)
        
        if n_eigs == None:
            n_eigs = self.num_eigs
        
        mu = self.model(inputs)
            
        if online_metric == 'extrap':
            unc = self.forward_extrap(mu, n_eigs)
        elif online_metric == 'fisher':
            unc = self.forward_fisher(mu, n_eigs)
            
        return self.dist_fam.output(mu), unc
    