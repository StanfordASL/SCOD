import numpy as np
import torch
from torchvision import models
import torch.nn as nn
from nn_ood.data.two_moons import TwoMoons
from nn_ood.posteriors import ExtrapUncertainty, SWAG, Ensemble, SWAGS
    
# WHERE TO SAVE THE MODEL
FILENAME = "model"

## HYPERPARAMS
N_MODELS = 10

LEARNING_RATE = 0.001
SGD_MOMENTUM = 0.9

LR_DROP_FACTOR = 0.8
EPOCHS_PER_DROP = 25

BATCH_SIZE = 32

N_EPOCHS = 150

## SET UP DATASETS
dataset_class = TwoMoons
test_dataset_args = ['train', 'val', 'unif']

## DEFINE VISUALIZATION FUNCTIONS
   
def viz_dataset_sample(ax, dataset, idx=0, model=None, unc_model=None):
    xx = np.linspace(-3,3)
    yy = np.linspace(-3,3)
    X,Y = np.meshgrid(xx,yy)
    inps = torch.from_numpy(np.stack([X.flatten(),Y.flatten()], axis=-1)).float()
    if unc_model is not None:
        inps = inps.to(device)
        pred, unc = unc_model(inps)
        pred = pred.cpu().detach().numpy()
        unc = unc.cpu().detach().numpy()
        Unc = unc.reshape(X.shape)
        ax.contourf(X,Y,Unc)
        ax.set_xlabel('Uncertainty Estimate')
    elif model is not None:
        inps = inps.to(device)
        pred = model(inps)
        pred = pred.cpu().detach().numpy()
        Pred = pred.reshape(X.shape)
        ax.contourf(X,Y,Pred)
        ax.set_xlabel('Prediction')
        
    ax.scatter(dataset.x[:,0], dataset.x[:,1], color='C1')


## USE CUDA IF POSSIBLE
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## MODEL SET UP
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=5./3)
        
og_req_grads = []
def make_model():
    global og_req_grads
    x_dim = 2
    y_dim = 1
    hid_dim=64
    model = nn.Sequential(
        nn.Linear(x_dim, hid_dim),
        nn.Tanh(),
        nn.Linear(hid_dim, hid_dim),
        nn.Tanh(),    
        nn.Linear(hid_dim, hid_dim),
        nn.Tanh(),
        nn.Linear(hid_dim, y_dim)
    )
    og_req_grads = [p.requires_grad for p in model.parameters()]

    
    model.apply(weight_init)
    
    return model

def freeze_model(model):
    # freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # make last layer trainable
    for p in model.children[-1].parameters():
        p.requires_grad = True
        
def unfreeze_model(model):
    global og_req_grads
    # unfreeze everything
    for p,v in zip( model.parameters(), og_req_grads):
        p.requires_grad = v
    
    
## OPTIMIZATION 
criterion = nn.MSELoss()
opt_class = torch.optim.SGD
opt_kwargs = {
    'lr': LEARNING_RATE,
    'momentum': SGD_MOMENTUM
}
sched_class = torch.optim.lr_scheduler.StepLR
sched_kwargs = {
    'step_size': EPOCHS_PER_DROP,
    'gamma': LR_DROP_FACTOR
}    
    

from nn_ood.posteriors.swags import ErrorCriterion    
prep_unc_models = {
#     'extrap': {
#         'class': ExtrapUncertainty,
#         'kwargs': {
#             'num_eigs': 20,
#         }
#     },
#     'swag': {
#         'class': SWAG,
#         'kwargs': {
#             'num_samples': 50,
#             'batch_size': 4,
#             'learning_rate': 1e-5
#         }
#     },
    'swags': {
        'class': SWAGS,
        'kwargs': {
            'num_samples': 50,
            'num_eigs': 50,
            'batch_size': 4,
            'learning_rate': 1e-5,
            'device': 'gpu',
            'normalize': False,
        },
        'criterion': ErrorCriterion(beta=0.01)
    },
    'swags_10': {
        'class': SWAGS,
        'kwargs': {
            'num_samples': 50,
            'num_eigs': 10,
            'batch_size': 4,
            'learning_rate': 1e-5,
            'device': 'gpu',
            'normalize': False,
        },
        'criterion': ErrorCriterion(beta=0.01)
    },
    'swags_20': {
        'class': SWAGS,
        'kwargs': {
            'num_samples': 50,
            'num_eigs': 20,
            'batch_size': 4,
            'learning_rate': 1e-5,
            'device': 'gpu',
            'normalize': False,
        },
        'criterion': ErrorCriterion(beta=0.01)
    }
}

test_unc_models = {
#     'extrap_sample': {
#         'class': ExtrapUncertainty,
#         'kwargs': {
#             'num_eigs': 20,
#         },
#         'load_name': 'extrap',
#         'forward_kwargs': {
#             'n_samples': 50
#         }
#     },
#     'extrap': {
#         'class': ExtrapUncertainty,
#         'kwargs': {
#             'num_eigs': 20,
#         },
#         'load_name': 'extrap',
#         'forward_kwargs': {
#             'n_samples': 1
#         }
#     },
#     'swag': {
#         'class': SWAG,
#         'kwargs': {
#             'num_samples': 50,
#             'batch_size': 4,
#             'learning_rate': 1e-5
#         },
#         'load_name': 'swag',
#         'forward_kwargs': {}
#     },
    'swags': {
        'class': SWAGS,
        'kwargs': {
            'num_samples': 50,
            'num_eigs': 50,
            'batch_size': 32,
            'learning_rate': 1e-5,
            'device':'gpu',
            'normalize': False,
        },
        'load_name': 'swags',
        'forward_kwargs': {}
    },
    'swags_10': {
        'class': SWAGS,
        'kwargs': {
            'num_samples': 50,
            'num_eigs': 10,
            'batch_size': 32,
            'learning_rate': 1e-5,
            'device':'gpu',
            'normalize': False,
        },
        'load_name': 'swags_10',
        'forward_kwargs': {}
    },
    'swags_20': {
        'class': SWAGS,
        'kwargs': {
            'num_samples': 50,
            'num_eigs': 20,
            'batch_size': 32,
            'learning_rate': 1e-5,
            'device':'gpu',
            'normalize': False,
        },
        'load_name': 'swags_20',
        'forward_kwargs': {}
    },
#     'swags_norm': {
#         'class': SWAGS,
#         'kwargs': {
#             'num_samples': 50,
#             'batch_size': 32,
#             'learning_rate': 1e-5,
#             'device':'gpu',
#             'normalize':True,
#         },
#         'load_name': 'swags',
#         'forward_kwargs': {}
#     },
#     'ensemble': {
#         'class': Ensemble,
#         'kwargs': {},
#         'load_name': None,
#         'multi_model': True,
#         'forward_kwargs': {}
#     }
}

# OOD PERFORMANCE TESTS
splits_to_use = test_dataset_args
err_thresh = 1e-2

in_dist_splits = ["val"]
out_dist_splits = ["unif"]