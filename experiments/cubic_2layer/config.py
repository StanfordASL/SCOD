import torch
import torch.nn as nn
from nn_ood.data.simple_reg import Cubic
from nn_ood.posteriors import LocalEnsemble, SCOD, Ensemble, Naive, KFAC
from nn_ood.distributions import GaussianFixedDiagVar
import numpy as np
import matplotlib.pyplot as plt

    
# WHERE TO SAVE THE MODEL
FILENAME = "model"

## HYPERPARAMS
N_MODELS = 3

LEARNING_RATE = 0.1

LR_DROP_FACTOR = 1.
EPOCHS_PER_DROP = 20

BATCH_SIZE = 5

N_EPOCHS = 30

## SET UP DATASETS
dataset_class = Cubic
test_dataset_args = ['train', 'val', 'ood']

def viz_datasets(idx=0, unc_model=None, model=None, unc_scale=1.):
    dataset = dataset_class('unif')
    xx = dataset.tensors[0]
    yy = dataset.tensors[1]
    
    fig, ax = plt.subplots(figsize=[4,4])
#     ax.plot(xx[:,0], yy[:,0], color='k', linestyle=':')
    dataset = dataset_class('train')
    ax.scatter(dataset.x, dataset.y, color='C1', marker='x')
#     dataset = dataset_class('val')
#     ax.scatter(dataset.x, dataset.y, color='C2')
    
    if unc_model is not None:
        y_pred, unc = unc_model(xx.to(device))
        unc = unc.detach().cpu().numpy()*unc_scale
        y_pred = y_pred.detach().cpu().numpy()[:,0]
        ax.plot(xx[:,0], y_pred, color='C0')
        
        ax.fill_between(xx[:,0], y_pred-unc, y_pred+unc, color='C0', alpha=0.4)
        ax.fill_between(xx[:,0], y_pred-2*unc, y_pred+2*unc, color='C0', alpha=0.2)
        
    elif model is not None:
        y_pred = model(xx.to(device))
        y_pred = y_pred.detach().cpu().numpy()[:,0]
        ax.plot(xx[:,0], y_pred, color='C0')
        
    ax.set_xlim([-6,6])
    ax.set_ylim([-200,200])
        

## USE CUDA IF POSSIBLE
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## MODEL SET UP
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=5./3)
    if m.__class__.__name__.find('Conv') != -1:
        nn.init.normal_(m.weight, 0.0, 0.02)

def make_model():
    model = nn.Sequential(
        nn.Linear(1, 100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.ReLU(),
        nn.Linear(100, 1)
    )
    model.apply(weight_init)
    
    return model

def freeze_model(model):
    # freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # make linear layers trainable
    for p in model.children[-2].parameters():
        p.requires_grad = True
    for p in model.children[-1].parameters():
        p.requires_grad = True
    
def unfreeze_model(model):
    # unfreeze everything
    for p in model.parameters():
        p.requires_grad = True

dist_fam = GaussianFixedDiagVar(sigma_diag=np.array([3])).to(device)
opt_class = torch.optim.Adam
opt_kwargs = {
    'lr': LEARNING_RATE,
}
sched_class = torch.optim.lr_scheduler.StepLR
sched_kwargs = {
    'step_size': EPOCHS_PER_DROP,
    'gamma': LR_DROP_FACTOR
}    
    
prep_unc_models = {
    'local_ensemble': {
        'class': LocalEnsemble,
        'kwargs': {
            'num_eigs': 50,
            'device':'gpu',
            'full_data': True,
        },
    },
    'scod_SRFT_s304_n50': {
        'class': SCOD,
        'kwargs': {
            'num_samples': 304,
            'num_eigs': 50,
            'device':'gpu',
            'sketch_type': 'srft'
        },
    },
    'scod_SRFT_s304_n50_freeze': {
        'class': SCOD,
        'kwargs': {
            'num_samples': 304,
            'num_eigs': 50,
            'device':'gpu',
            'sketch_type': 'srft'
        },
        'freeze': True,
    },
    'kfac': {
        'class': KFAC,
        'kwargs': {
            'device':'gpu',
            'input_shape': [1],
        },
    },
}

test_unc_models = {
    'local_ensemble_n50': {
        'class': LocalEnsemble,
        'kwargs': {
            'num_eigs': 50,
            'n_y_samp': 2,
            'device':'gpu'
        },
        'load_name': 'local_ensemble',
        'forward_kwargs': {}
    },
    'SCOD': {
        'class': SCOD,
        'kwargs': {
            'num_eigs': 50,
            'device':'gpu'
        },
        'load_name': 'scod_SRFT_s304_n50',
        'forward_kwargs': {}
    },
    'SCOD_freeze': {
        'class': SCOD,
        'kwargs': {
            'num_eigs': 50,
            'device':'gpu'
        },
        'freeze':True,
        'load_name': 'scod_SRFT_s304_n50_freeze',
        'forward_kwargs': {}
    },
    'naive': {
        'class': Naive,
        'kwargs': {
            'device':'gpu'
        },
        'load_name': None,
        'forward_kwargs': {}
    },
    'kfac_100_10': {
        'class': KFAC,
        'kwargs': {
            'device':'gpu',
            'input_shape': [1, 28, 28]
        },
        'load_name': 'kfac',
        'forward_kwargs': {
            'norm': 100.,
            'scale': 10.,
        }
    },
    'ensemble': {
        'class': Ensemble,
        'kwargs': {
            'device':'gpu'
        },
        'load_name': None,
        'multi_model': True,
        'forward_kwargs': {}
    },
}

# OOD PERFORMANCE TESTS
splits_to_use = test_dataset_args
err_thresh = 1.

in_dist_splits = ['val']
out_dist_splits = ['ood']


# Visualization
from nn_ood.utils.viz import summarize_ood_results, plot_perf_vs_runtime

keys_to_compare = [
                   'SCOD',
                   'SCOD_freeze',
                   'ensemble', 
                   'local_ensemble_n50',
                   'kfac_100_10',
                   'naive',
                   'maha',
]

colors= [
         'xkcd:azure',
         'xkcd:electric blue',
         'xkcd:adobe',
         'xkcd:mango',
         'xkcd:blood orange',
         'xkcd:scarlet',
         'xkcd:indigo'
]

plots_to_generate = {
    'auroc_vs_runtime.pdf': {
        'summary_fn': summarize_ood_results,
        'summary_fn_args': [
            in_dist_splits,
            out_dist_splits
        ],
        'summary_fn_kwargs': {
            'keys_to_compare': keys_to_compare,
        },
        'plot_fn': plot_perf_vs_runtime,
        'plot_fn_args': [],
        'plot_fn_kwargs': {
            'colors': colors,
            'figsize': [4,2.5],
            'dpi': 150,
            'normalize_x': True,
        },
        'legend': {},
        'title': "Cubic",
    },
}