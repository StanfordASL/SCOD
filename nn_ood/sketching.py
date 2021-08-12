import numpy as np
import torch
import torch.nn as nn

def idct(X, norm=None):
    """
    based on https://github.com/zh217/torch-dct/blob/master/torch_dct/_dct.py
    updated to work with more recent versions of pytorch which moved fft functionality to 
    the torch.fft module
    """
    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)

    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape) 

class SketchOperator(nn.Module):
    """
    implements linear operator for sketching
    """
    def __init__(self,d,N):
        """
        d x N operator
        """
        self.d = d
        self.N = N
        super().__init__()
    
    def forward(self, M):
        """
        implements \tilde{M} = S M, left multiplication by M
        """
        raise NotImplementedError

class GaussianSketchOp(SketchOperator):
    def __init__(self, d, N, device=torch.device('cpu')):
        super().__init__(d,N)
        self.test_matrix = nn.Parameter(torch.randn(d, N, dtype=torch.float, device=device), requires_grad=False)
    
    @torch.no_grad()
    def forward(self,M, transpose=False):
        if transpose:
            return M @ self.test_matrix.t()
        return self.test_matrix @ M

class SRFTSketchOp(SketchOperator):
    def __init__(self, d, N, device=torch.device('cpu')):
        super().__init__(d,N)
        self.D = nn.Parameter(2*(torch.rand(N, device=device)>0.5).float()- 1, requires_grad=False)
        self.P = np.random.choice(N,d) # choose d elements from n 
    
    @torch.no_grad()
    def forward(self,M, transpose=False):
        if transpose:
            M = M.t()
            
        if M.dim() == 2:
            result = idct((self.D[:,None]*M).t()).t()[self.P,:]
        elif M.dim() == 1:
            result = idct(self.D*M)[self.P]
        else:
            raise InvalidArgumentError
        
        if transpose:
            result = result.t()
            
        return result
    

class LinearSketch():
    """
    Abstract class implementing several methods useful for extracting
    low rank approximations from a sketch of a matrix. The actual sketching mechanics
    are not implemented, and must be implemented by subclasses.
    """
    def __init__(self):
        """
        All subclasses must have the following defined after
        the sketch is constructed of matrix A
        
        self.Y (N x k) = A @ Om # torch.Tensor
        self.W (l x M) = Psi @ A # torch.Tensor
        self.Om_fn (implements right multiplication by linear operator Om)
        self.Psi_fn (implements left multiplication by linear operator Psi)
        """
        
    def Om_fn(self,M):
        return self.Om(M, transpose=True)
    
    def Psi_fn(self,M):
        return self.Psi(M)
    
    @torch.no_grad()
    def low_rank_approx(self):
        """
        returns Q (N x k), X (k x M) such that A ~= QX
        """
        Q,_ = torch.linalg.qr(self.Y,'reduced') # (N, k)
        U,T = torch.linalg.qr( self.Psi_fn( Q ), 'reduced' ) # (l, k), (k, k)
        X,_ = torch.triangular_solve(U.t() @ self.W, T) # (k, N)

        return Q, X

    @torch.no_grad()
    def fixed_rank_svd_approx(self, r):
        """
        returns U (N x r), S (r,), V (M x r) such that A ~= U diag(S) V.T
        """
        Q,X = self.low_rank_approx()
        U, S, V = torch.svd_lowrank(X, r)
        U = Q @ U
        
        return U,S,V
        
    @torch.no_grad()
    def sym_low_rank_approx(self):
        """
        returns U (N x 2k), S (2k x 2k) such that A ~= U S U^T
        """
        Q,X = self.low_rank_approx()
        U,T = torch.linalg.qr(torch.cat([Q, X.t()], dim=1), 'reduced') # (N, 2k), (2k, 2k)
        del Q, X
        T1 = T[:,:self.k] # (2k, k)
        T2 = T[:,self.k:2*self.k] # (2k, k)
        S = (T1 @ T2.t() + T2 @ T1.t()) / 2 # (2k, 2k)
        
        return U,S

    @torch.no_grad()
    def fixed_rank_eig_approx(self, r):
        """
        returns U (N x r), D (r) such that A ~= U diag(D) U^T
        """
        U, S = self.sym_low_rank_approx()
        D, V = torch.linalg.eigh(S) # (2k), (2k, 2k)
        D = D[-r:]
        V = V[:,-r:] # (2k, r)
        U = U @ V # (N, r)
        
        return U,D

class MiniBatchSketch():
    """
    uses a single pass through columns of A, a (N x M) matrix 
    to estimate r top left singular vectors)
    """
    def __init__(self, N, M, r, T=None, gpu=False):
        self.N = N
        self.M = M
        self.r = r
        self.T = T
        if T is None:
            self.T = r
        
        self.device = torch.device('cpu')
        if gpu:
            self.device = torch.cuda.current_device()
        
        self.itr_between_samples = max(1, int(np.floor( self.M / self.T )))
        
        # sketch data
        self.Y = torch.zeros(self.N, self.T, dtype=torch.float, device=self.device)
        print(self.Y.shape)

    @torch.no_grad()
    def low_rank_update(self, i, v):
        """
        processes v, the ith batch of columns of matrix A
        @TODO: this is broken for multivariate distfams
        """
        v = v.to(self.device)
        
        idx = int(i / self.itr_between_samples)
        if idx >= self.T:
            return
        
        self.Y[:,idx] += v / self.itr_between_samples
    
    @torch.no_grad()
    def get_range_basis(self):
        """
        returns a basis for the range of the top r left singular vectors
        if center is not None,
        returns a basis for [G - gbar][G - gbar]^T = GG^T - Ggbar^T - gbar^TG
        """ 
        if self.r == self.T:
            U,_ = torch.qr(self.Y)
        else:
            U,S,V = torch.svd_lowrank(self.Y, self.r)
        
        return S,U
    
class RandomSymSketch(LinearSketch):
    """
    computes a sketch of AA^T when presented columns of A sequentially
    then, uses eigenvalue decomp of sketch to compute 
    rank r range basis
    """
    def __init__(self, N, M, r, T=None, gpu=False, sketch_op_class=GaussianSketchOp):
        self.N = N
        self.M = M
        self.r = r
        self.T = T
        if T is None: #or T < 6*r + 4:
            self.T = 6*r + 4
        print("using T =", self.T)
        
        self.device = torch.device('cpu')
        if gpu:
            self.device = torch.cuda.current_device()
            
        self.k = max(self.r + 2, (self.T-1)//3)
        self.l = self.T - self.k
                
        # random test matrices
        self.Om = sketch_op_class(self.k, self.N, device=self.device)
        self.Psi = sketch_op_class(self.l, self.N, device=self.device)
        
        # sketch data
        self.Y = torch.zeros(self.N, self.k, dtype=torch.float, device=self.device)
        self.W = torch.zeros(self.l, self.N, dtype=torch.float, device=self.device)
        
        super().__init__()

    @torch.no_grad()
    def low_rank_update(self, i, v, weight):
        """
        processes v (nparam x d) the a batch of columns of matrix A
        """
        v = v.to(self.device)
        torch.addmm(self.Y, weight*v, self.Om_fn(v.t()), alpha=1/self.M, out=self.Y)
        # Y = Y + 1/M* V V^T Om 
        torch.addmm(self.W, weight*self.Psi_fn(v), v.t(), alpha=1/self.M, out=self.W)
        
    @torch.no_grad()
    def get_range_basis(self):
        """
        returns a basis for the range of the top r left singular vectors
        """
        self.total_weight = 1.
        self.device = torch.device("cpu")
        self.Y = self.Y.cpu()
        self.W = self.W.cpu()
        self.Om.cpu()
        self.Psi.cpu()
        U,D = self.fixed_rank_eig_approx(2*self.k)#self.r)
        return D,U
    
class RandomOneSidedSymSketch(LinearSketch):
    """
    computes a sketch of AA^T when presented columns of A sequentially
    then, uses eigenvalue decomp of sketch to compute 
    rank r range basis. only sketches one direction as AA^T is symmetric
    """
    def __init__(self, N, M, r, T=None, gpu=False, sketch_op_class=GaussianSketchOp):
        self.N = N
        self.M = M
        self.r = r
        self.T = T
        if T is None: #or T < 3*r + 2:
            self.T = 3*r + 2
        print("using T =", self.T)
        self.k = self.T
        
        self.device = torch.device('cpu')
        if gpu:
            self.device = torch.cuda.current_device()
            
        # random test matrices
        self.Psi = sketch_op_class(self.T, self.N, device=self.device)
        
        # sketch data
        self.W = torch.zeros(self.T, self.N, dtype=torch.float, device=self.device)
        
        super().__init__()
        
    @property
    def Y(self):
        return self.W.t()
    
    def Om_fn(self, M):
        return self.Psi_fn(M.t()).t()
        
    @torch.no_grad()
    def low_rank_update(self, i, v, weight):
        """
        processes v (nparam x d) the a batch of columns of matrix A
        """
        v = v.to(self.device)
                
        torch.addmm(self.W, self.Psi_fn(v), v.t(), alpha=1./self.M, out=self.W)
    
    @torch.no_grad()
    def get_range_basis(self):
        """
        returns a basis for the range of the top r left singular vectors
        """
        self.device = torch.device("cpu")
        self.W = self.W.cpu()
        self.Psi.cpu()
        U,D = self.fixed_rank_eig_approx(self.r)
        return D,U
    
class RandomSVDSketch(LinearSketch):
    """
    computes a sketch of A when presented columns of A sequentially
    then, uses svd decomp of sketch to compute 
    rank r range basis
    """
    def __init__(self, N, M, r, T=None, gpu=False):
        self.N = N
        self.M = M
        self.r = r
        self.T = T
        if T is None or T < 6*r + 4:
            self.T = 6*r + 4
        print("using T =", self.T)
        
        self.device = torch.device('cpu')
        if gpu:
            self.device = torch.cuda.current_device()

        self.k = max(self.r + 2, (self.T-1)//3)
        self.l = self.T - self.k
        
        # random test matrices
        self.Om = torch.randn(self.M, self.k, dtype=torch.float, device=self.device)
        self.Psi = torch.randn(self.l, self.N, dtype=torch.float, device=self.device)
            
        # sketch data
        self.Y = torch.zeros(self.N, self.k, dtype=torch.float, device=self.device)
        self.W = torch.zeros(self.l, self.M, dtype=torch.float, device=self.device)
    
    @torch.no_grad()
    def low_rank_update(self, i, v):
        """
        processes v (nparam x d) the ith batch of columns of matrix A
        """
        v = v[:, 0] # this shouldn't receive more than 1 column
        v = v.to(self.device)
            
        self.Y += (v[:, None] @ self.Om[i:i+1,:])
        self.W[:,i] = self.Psi @ v
    
    @torch.no_grad()
    def get_range_basis(self):
        """
        returns a basis for the range of the top r left singular vectors
        """
        U,S,V = self.fixed_rank_svd_approx(self.r)
        
        return S,U
    
class SRFTSymSketch(RandomSymSketch):
    """
    computes a subsampled randomized fourier transform sketch 
    of AA^T when presented columns of A sequentially.
    
    then, uses eigen decomp of sketch to compute 
    rank r range basis
    """
    def __init__(self, N, M, r, T=None, gpu=True):
        super().__init__(N, M, r, T, gpu, sketch_op_class=SRFTSketchOp)
        
    
class SRFTOneSidedSymSketch(RandomOneSidedSymSketch):
    """
    computes a subsampled randomized fourier transform sketch 
    of AA^T when presented columns of A sequentially.
    
    then, uses eigen decomp of sketch to compute 
    rank r range basis
    """
    def __init__(self, N, M, r, T=None, gpu=True):
        super().__init__(N, M, r, T, gpu, sketch_op_class=SRFTSketchOp)


sketch_args = {
    'minibatch': MiniBatchSketch,
    'random': RandomSymSketch,
    'random_onesided': RandomOneSidedSymSketch,
    'random_svd': RandomSVDSketch,
    'srft': SRFTSymSketch,
    'srft_onesided': SRFTOneSidedSymSketch,
}


class Projector(nn.Module):
    def __init__(self, N, r, T, gpu=False):
        super().__init__()
        self.N = N
        self.r = r
        
        self.device = torch.device('cpu')
        if gpu:
            self.device = torch.cuda.current_device()
        
        self.eigs = nn.Parameter(torch.zeros(self.r, device=self.device), requires_grad=False)
        self.basis = nn.Parameter(torch.zeros(self.N, self.r, device=self.device), requires_grad=False)
        
    @torch.no_grad()
    def process_basis(self, eigs, basis):
        self.eigs.data = eigs.to(self.device)
        self.basis.data = basis.to(self.device)

    def ortho_proj(self, L, n_eigs):
        """
        we have U = basis,
        computes ||(I-UU^T)L||^2_F
        """
        basis = self.basis[:,-n_eigs:]
        proj_L = basis.t() @ L 
        proj_L = basis @ proj_L
        return torch.norm(L - proj_L) 
    
    def sampled_ortho_proj(self, L, n_eigs):
        """
        we have U = basis,
        computes mean( ||(I-UU^T)l_i||^2_2 )
        """
        basis = self.basis[:,-n_eigs:]
        proj_L = basis.t() @ L 
        proj_L = basis @ proj_L
        return torch.mean( torch.norm(L - proj_L, dim=0) ) 
    
    def posterior_pred(self, L, n_eigs, Meps):
        """
        we have U = basis,
        computes ||(I-UU^T)L||^2_F
        """
        basis = self.basis[:,-n_eigs:]
        eigs = torch.clamp( self.eigs[-n_eigs:], min=0.)

        scaling = torch.sqrt( eigs / ( eigs + 1./(2*Meps) ) )
        proj_L = scaling[:,None] * (basis.t() @ L)

        return torch.sqrt( torch.sum(L**2) - torch.sum(proj_L**2) )
    
    def ortho_2norm(self, L, n_eigs):
        """
        we have U = basis,
        computes ||(I-UU^T)L||^2_F
        """
        basis = self.basis[:,-n_eigs:]
        proj_L = basis.t() @ L 
        proj_L = basis @ proj_L
        
        return torch.linalg.norm(L - proj_L, ord=2)**2
    
    def fisher_norm(self, L):
        return torch.norm(L.t() @ L)
    
    def cosine_dist(self, L, n_eigs):
        """
        if A = U eigs U^T
        and B = LL^T
        return arccos( Tr(AB) / ||A||_F ||B||_F)
        """
        basis = self.basis[:,-n_eigs:]
        eigs = self.eigs[-n_eigs:]
        trAB = torch.sum(eigs[:,None]*(basis.t() @ L)**2)
        normA = torch.norm(eigs) # 2 norm of eigenvalues
        normB = self.fisher_norm(L) # frobenius norm of B
        
        return 2*torch.acos(trAB/normA/normB) / np.pi
    
    def ortho_cosine_sim(self, L, n_eigs):
        """
        if A = U eigs U^T
        and B = LL^T
        """
        trAB = self.ortho_proj(L, n_eigs)
        normB = self.fisher_norm(L)
        
        return trAB/normB
    
    def scaled_ortho_proj(self, L, n_eigs):
        """
        if A = U eigs U^T
        and B = LL^T
        """
        trAB = self.ortho_proj(L, n_eigs)
        normB = torch.sum(L**2, dim=(-1,-2))
        
        return trAB/torch.sqrt(normB)
    
    @torch.no_grad()
    def compute_distance(self, L, proj_type, n_eigs=None, Meps=5000.):
        if n_eigs is None:
            n_eigs = self.r
            
        L.to(self.device)
        if proj_type == 'ortho':
            return self.ortho_proj(L, n_eigs)
        elif proj_type == 'sampled_ortho':
            return self.sampled_ortho_proj(L, n_eigs)
        elif proj_type == 'posterior_pred':
            return self.posterior_pred(L, n_eigs, Meps)
        elif proj_type == 'ortho_2norm':
            return self.ortho_2norm(L, n_eigs)
        elif proj_type == "scaled_ortho":
            return self.scaled_ortho_proj(L, n_eigs) 
        elif proj_type == "ortho_cosine_sim":
            return self.ortho_cosine_sim(L, n_eigs)
        elif proj_type == "cosine_dist":
            return self.cosine_dist(L, n_eigs)
        elif proj_type == "norm":
            return self.fisher_norm(L)
        else:
            raise ValueError(proj_type +" is not an understood projection type.")