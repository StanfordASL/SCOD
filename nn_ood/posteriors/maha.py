import torch
from torch import nn
import numpy as np

from copy import deepcopy

from tqdm import tqdm

def get_features_and_classifier(model):
    """works for nn.Sequential with last linear layer"""
    features = model[:-1]
    classifier = model[-1]
    
    return features, classifier
    
base_config = {
    "device": "cpu",
    "features_and_classifier": get_features_and_classifier,
    "num_classes": 1,
    "feat_dim": 1024,
    "eps": 1e-5,
}


    
class Mahalanobis(nn.Module):
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
        
        self.device = torch.device("cpu")
        if self.config["device"] != "cpu":
            self.device = torch.cuda.current_device()

        features, classifier = self.config["features_and_classifier"](model)
        self.features = features
        self.classifier = classifier
        
        self.N = self.config["feat_dim"]
        self.K = self.config["num_classes"]
        
        self.means = nn.Parameter( 
            torch.zeros(self.K,self.N, device=self.device), requires_grad=False)
        self.precisions = nn.Parameter( 
            torch.zeros(self.K,self.N,self.N, device=self.device), requires_grad=False)
        
        self.eps = self.config["eps"]
        
        
    def process_dataset(self, dataset):
        """
        summarizes information about training data by logging gradient directions
        seen during training, and then using gram schmidt of these to form
        an orthonormal basis. directions not seen during training are 
        taken to be irrelevant to data, and used for detecting generalization
        
        dataset - torch dataset of (input, target) pairs
        """
        def prep_vec(vec):
            return vec.to(self.device)
        # loop through data, one sample at a time
        print("processing data")
            
        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size=1, 
                                                 shuffle=True)
        
        n_data = len(dataloader)
        
        
        mean_squares = torch.zeros(self.K, self.N, self.N, device=self.device)
        counts = torch.zeros(self.K,device=self.device)
        for i, (inputs,labels) in tqdm(enumerate(dataloader), total=n_data):
            inputs = prep_vec(inputs)
            labels = prep_vec(labels)

            feat = self.features(inputs).detach()
            
            i = 0
            if self.K > 1:
                i = labels[0].long()
                        
            self.means.data[i,:] += feat[0]
            mean_squares[i,:,:] += feat.t() @ feat
            counts[i] += 1
        
        self.means.data = self.means.data/counts[:,None]
        mean_squares = mean_squares/counts[:,None,None]
        
        self.precisions.data = torch.inverse(mean_squares - self.means[:,:,None] @ self.means[:,None,:] + self.eps*torch.eye(self.means.shape[-1], device=self.device))
            
    
    def forward(self, inputs):
        """
        assumes inputs are of shape (N, input_dims...)
        where N is the batch dimension,
              input_dims... are the dimensions of a single input
                            
        returns 
            mu = model(inputs) -- shape (N, 1)
            unc = hessian based uncertainty estimates shape (N)
        """    
        feats = self.features(inputs)
        mu = self.classifier(feats)

        errs = feats[:,None,:] - self.means[None,:,:]
        maha_dists = (errs[:,:,None,:] @ self.precisions[None,:,:,:] @ errs[:,:,:,None])[:,:,0,0]
        
        unc, _ = torch.min(maha_dists,dim=1)
    
        return self.dist_fam.output(mu), unc