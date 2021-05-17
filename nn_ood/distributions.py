import torch
import torch.nn as nn
import numpy as np

"""
this file implements different output distributional families, with their loss functions and Fisher matrices

specifically, for each family, if F(theta) = LL^T
we implement left multiplication by L^T by the function
apply_sqrt_F
"""

class DistFam(nn.Module):
    def loss(self, thetas, targets):
        """
        thetas (..., thetadim)
        targets (..., ydim)
        """
        return self._loss(thetas, targets)
    
    def metric(self, outputs, targets):
        """
        return a user facing metric based on
        outputs = self.output(theta)
        
        returns {'name':value}
        """
        return {}
        
    def apply_sqrt_F(self, theta):
        """
        F(theta) = LL^T

        returns y = L^T x
        """
        raise NotImplementedError
        
    def output(self, theta):
        """
        turns theta into user-facing output
        """
        return theta
    
    def uncertainty(self, outputs):
        """
        outputs: [..., outputdim]
        returns unc [...], uncertainty score per output
        """
        return 1. + 0.*outputs[...,0]
        
    def merge_ensemble(self, thetas):
        return NotImplementedError

class GaussianFixedDiagVar(DistFam):
    """
    P(theta) = N(theta, diag(sigma))
    
    Here, F(theta) = diag(sigma)^-1
    normalized F = diag(sigma)^(-1) / sum(sigma^-1)
    """
    def __init__(self, sigma_diag=np.array([1.]), min_val=-5, max_val=5):
        super().__init__()
        self.sigma_diag = nn.Parameter(torch.from_numpy(sigma_diag).float(), requires_grad=False)
    
    def dist(self, thetas):
        C = torch.diag_embed(torch.ones_like(thetas)*self.sigma_diag)
        return torch.distributions.MultivariateNormal(loc = thetas, covariance_matrix=C )
    
    def sample_y(self, mu, n):
        # TODO: implement this for general distributions
        shape = mu.shape[-1]
        ys = torch.from_numpy(np.linspace(min_val*np.ones(shape), max_val*np.ones(shape), n)).float()
        return ys
        
    def loss(self, thetas, targets):
        err = targets - thetas
        return 0.5 * torch.sum( err**2 / self.sigma_diag, dim=-1 ) +0.5*err.shape[-1]*np.log(2*np.pi) + 0.5*torch.sum(torch.log(self.sigma_diag))

    @torch.no_grad()
    def metric(self, outputs, targets):
        err = outputs - targets
        return 'Mahalanobis Error', torch.sum(err**2 / self.sigma_diag, dim=-1)
        
    def apply_sqrt_F(self, theta, exact=True):
        return theta / torch.sqrt( self.sigma_diag * torch.sum(self.sigma_diag**(-1)) ) # (normalizing for trace = 1)
    
    def uncertainty(self, outputs):
        return torch.sum(self.sigma_diag) + 0*outputs[...,0]
    
    def merge_ensemble(self, thetas):
        mu = torch.mean(thetas, dim=0)
#         diag_var = torch.mean(self.sigma_diag + thetas**2, dim=0) - mu**2
#         unc = 0.5*torch.sum( 1 + torch.log(2*np.pi*diag_var), dim=-1 )
        diag_var = torch.mean(thetas**2, dim=0) - mu**2
        unc = torch.sqrt(torch.sum(diag_var, dim=-1))
        return mu, unc
    
class Bernoulli(DistFam):
    """
    theta \in \R^1
    P(theta) = Bern( theta )
    
    Here, F(theta) = 1/(p(1-p))
    normalized F(theta) = 1
    """
    def __init__(self):
        super().__init__()
        self._loss = nn.BCELoss()
    
    def dist(self, thetas):
        return torch.distributions.Bernoulli(probs = thetas)
    
    def sample_y(self, mu, n):
        ys = torch.from_numpy(np.arange(n, dtype=np.int64) % 2)
        return ys
    
    def loss(self, thetas, targets):
        return y*torch.log(theta) + (1-y)*torch.log(1-theta)
    
    @torch.no_grad()
    def metric(self, outputs, targets):
        accuracy = (targets*(outputs > 0.5) + (1-targets)*(outputs < 0.5))[...,0]
        return 'Accuracy', accuracy
        
    def apply_sqrt_F(self, theta):
        t = theta.detach()
        L = torch.sqrt( t*(1-t) ) + 1e-10 # for stability
        return theta / L
    
    def merge_ensemble(self, thetas):
        mu = torch.mean(thetas, dim=0)
        unc = (- mu*torch.log(mu) - (1-mu)*torch.log(1-mu))[...,0]
        return mu, unc
    
class BernoulliLogit(DistFam):
    """
    theta \in \R^1
    P(theta) = Bern( 1/(1 + exp(-theta)) )
    
    Here, F(theta) = p(1-p)
    Normalize F(theta) = 1
    """
    def __init__(self):
        super().__init__()
        self._loss = nn.BCEWithLogitsLoss(reduction='none')
        
    def dist(self, thetas):
        return torch.distributions.Bernoulli(logits = thetas)
    
    def sample_y(self, mu, n):
        ys = torch.from_numpy(np.arange(n, dtype=np.int64) % 2)
        return ys.unsqueeze(-1).float()
        
    @torch.no_grad()
    def metric(self, outputs, targets):
        accuracy = (targets*(outputs > 0.5) + (1-targets)*(outputs < 0.5))[...,0]
        return 'accuracy', accuracy
#         return '-log(p(y))', -torch.log(outputs[...,0])
        
    def apply_sqrt_F(self, theta, exact=True):
        t = theta.detach()
        
        p = torch.sigmoid(t)
        L = torch.sqrt(p*(1-p))
        return L*theta
    
    def output(self, theta):
        return torch.sigmoid(theta)
    
    def uncertainty(self, outputs):
        return (- outputs*torch.log(outputs) - (1-outputs)*torch.log(1-outputs))[...,0]
    
    def merge_ensemble(self, thetas):
        ps = torch.sigmoid(thetas)
        mu = torch.mean(ps, dim=0)
        unc = (- mu*torch.log(mu) - (1-mu)*torch.log(1-mu))[...,0]
        if torch.isnan(unc):
            unc = torch.zeros_like(mu)[...,0]
        return mu, unc
    
class Categorical(DistFam):
    """
    theta \in \R^k, 1^T theta = 1, theta > 0
    P(theta) = Categorical( p = theta )
    
    Here, F(theta) = (diag(p^{-1})) = LL^T
    L = diag(p^{-1/2})
    normalized F = diag(p^{-1}) / sum(p^{-1})
    """
    def __init__(self):
        super().__init__()
        self.nll_loss = nn.NLLLoss(reduction='none')
    
    def dist(self, thetas):
        return torch.distributions.Categorical(probs = thetas)
    
    def sample_y(self, mu, n):
        k = mu.shape[-1]
        ys = torch.from_numpy(np.arange(n, dtype=np.int64) % k)
        return ys
        
    def loss(self, thetas, targets):
        return self.nll_loss(torch.log(thetas), targets)
    
    @torch.no_grad()
    def metric(self, outputs, targets):
#         pred_label = torch.argmax(outputs, dim=-1)
#         accuracy = 1.*(targets == pred_label)
#         return 'Accuracy', accuracy
        prob_y = torch.gather(outputs,-1,targets[:,None])[...,0]
        return '-log(p(y))', -torch.log(prob_y)
        
    def apply_sqrt_F(self, theta):
        t = theta.detach()
        L_diag = torch.sqrt(t) + 1e-7 # for stability
        return theta / L_diag 
    
    def uncertainty(self, outputs):
        return -torch.sum(outputs*torch.log(outputs), dim=-1)
    
    def merge_ensemble(self, thetas):
        mu = torch.mean(thetas, dim=0)
        unc = -torch.sum(mu*torch.log(mu), dim=-1)
        return mu, unc
    
class CategoricalLogit(DistFam):
    """
    theta \in \R^k
    P(theta) = Categorical( p = SoftMax(theta) )
    
    Here, F(theta) = (diag(p) - pp^T) = LL^T
    L = (I - p1^T) diag(p^{1/2})
    L^T = diag(p^{1/2}) (I - 1p^T)
    """
    def __init__(self):
        super().__init__()
        self._loss = nn.CrossEntropyLoss(reduction='none')
        
    def dist(self, thetas):
        return torch.distributions.Categorical(logits = thetas)
    
    def sample_y(self, mu, n):
        k = mu.shape[-1]
        ys = torch.from_numpy(np.arange(n, dtype=np.int64) % k)
        return ys
        
    @torch.no_grad()
    def metric(self, outputs, targets):
#         pred_label = torch.argmax(outputs, dim=-1)
#         accuracy = 1.*(targets == pred_label)
#         return 'Accuracy', accuracy
        prob_y = torch.gather(outputs,-1,targets[:,None])[...,0]
        return '-log(p(y))', -torch.log(prob_y)
        
    def apply_sqrt_F(self, theta, exact=True):
        t = theta.detach()
        
        # exact computation
        if exact:
            p = torch.softmax(t, dim=-1)
            theta_bar = torch.sum(p*theta, dim=-1)[...,None]
            result = torch.sqrt(p)*(theta - theta_bar)
        
        # or, just sample a couple outputs from p(y) and then compute gradients
        else:
            logp = torch.log_softmax(theta, dim=-1)
            vals, idx = logp.topk(min(5,logp.shape[-1]))
            result = -torch.exp(vals).detach()*torch.gather(logp,-1,idx)
#             i = torch.argmax(p)
        return result
    
    def output(self, theta):
        return torch.softmax(theta, dim=-1)
    
    def uncertainty(self, outputs):
        return -torch.sum(outputs*torch.log(outputs), dim=-1)
    
    def merge_ensemble(self, thetas):
        ps = torch.softmax(thetas, dim=-1)
        mu = torch.mean(ps, dim=0)
        unc = -torch.sum(mu*torch.log(mu), dim=-1)
        return mu, unc